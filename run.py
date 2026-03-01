import os
import numpy as np
import pandas as pd
import warnings
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

warnings.filterwarnings('ignore')
np.random.seed(42)

TRAIN_PATH = '//Users/yash/algothon-26/man-imperial-algothon-2026/data/2025-05-30'
TEST_PATH  = '/Users/yash/algothon-26/man-imperial-algothon-2026/data/2025-09-30'

instruments = [f'INSTRUMENT_{i}' for i in range(1, 11)]
N = len(instruments)
FORWARD = 63
IC_VAL_DAYS = 252


def load_data(path):
    prices    = pd.read_csv(os.path.join(path, 'prices.csv'),    parse_dates=['date']).sort_values('date').set_index('date')
    signals   = pd.read_csv(os.path.join(path, 'signals.csv'),   parse_dates=['date']).sort_values('date').set_index('date')
    volumes   = pd.read_csv(os.path.join(path, 'volumes.csv'),   parse_dates=['date']).sort_values('date').set_index('date')
    cash_rate = pd.read_csv(os.path.join(path, 'cash_rate.csv'), parse_dates=['date']).sort_values('date').set_index('date')
    returns   = prices[instruments].pct_change().dropna()
    return returns, signals, volumes, cash_rate


def build_features(ret_df, sig_df, vol_df, cash_df, forward_days=63):
    frames = []
    cash_cols  = ['3mo', '6mo', '1yr', '2yr', '5yr', '10yr']
    cash_avail = [c for c in cash_cols if c in cash_df.columns]
    cash_ri    = cash_df[cash_avail].reindex(ret_df.index).ffill()
    for inst in instruments:
        f = pd.DataFrame(index=ret_df.index)
        r = ret_df[inst]
        f['ret_1d']           = r.shift(1)
        f['ret_5d']           = r.shift(1).rolling(5).mean()
        f['ret_10d']          = r.shift(1).rolling(10).mean()
        f['ret_21d']          = r.shift(1).rolling(21).mean()
        f['ret_63d']          = r.shift(1).rolling(63).mean()
        f['ret_126d']         = r.shift(1).rolling(126).mean()
        f['ret_252d']         = r.shift(1).rolling(252).mean()
        f['vol_10d']          = r.shift(1).rolling(10).std()
        f['vol_21d']          = r.shift(1).rolling(21).std()
        f['vol_63d']          = r.shift(1).rolling(63).std()
        f['vol_ratio_10_63']  = f['vol_10d']  / (f['vol_63d'] + 1e-8)
        f['vol_ratio_21_63']  = f['vol_21d']  / (f['vol_63d'] + 1e-8)
        f['mom_21']           = (1 + r.shift(1)).rolling(21).apply(np.prod, raw=True) - 1
        f['mom_63']           = (1 + r.shift(1)).rolling(63).apply(np.prod, raw=True) - 1
        f['mom_126']          = (1 + r.shift(1)).rolling(126).apply(np.prod, raw=True) - 1
        f['mom_252']          = (1 + r.shift(1)).rolling(252).apply(np.prod, raw=True) - 1
        f['skew_21']          = r.shift(1).rolling(21).skew()
        f['skew_63']          = r.shift(1).rolling(63).skew()
        f['kurt_63']          = r.shift(1).rolling(63).kurt()
        for h in [4, 8, 16, 32]:
            col = f'{inst}_trend{h}'
            if col in sig_df.columns:
                f[f'trend{h}'] = sig_df[col].reindex(ret_df.index).shift(1)
        f['trend_avg']        = f[[f'trend{h}' for h in [4, 8, 16, 32]]].mean(axis=1)
        f['trend_std']        = f[[f'trend{h}' for h in [4, 8, 16, 32]]].std(axis=1)
        f['trend_short_long'] = f['trend4'] - f['trend32']
        vol_col = f'{inst}_vol'
        if vol_col in vol_df.columns:
            v = vol_df[vol_col].reindex(ret_df.index).shift(1)
            f['vol_rel_5']  = v / (v.rolling(5).mean()  + 1)
            f['vol_rel_21'] = v / (v.rolling(21).mean() + 1)
            f['vol_rel_63'] = v / (v.rolling(63).mean() + 1)
        for cc in cash_avail:
            f[f'rate_{cc}'] = cash_ri[cc].shift(1)
        if '2yr' in cash_avail and '10yr' in cash_avail:
            f['yield_spread'] = (cash_ri['10yr'] - cash_ri['2yr']).shift(1)
        if '3mo' in cash_avail and '10yr' in cash_avail:
            f['term_spread']  = (cash_ri['10yr'] - cash_ri['3mo']).shift(1)
        f['target']     = r.rolling(forward_days).mean().shift(-forward_days)
        f['instrument'] = inst
        frames.append(f)
    out       = pd.concat(frames)
    feat_cols = [c for c in out.columns if c not in ['target', 'instrument']]
    return out, feat_cols


def fit_models(feat_df, feat_cols):
    xgb_models, rf_models = {}, {}
    for inst in instruments:
        tr  = feat_df[feat_df['instrument'] == inst]
        X   = tr[feat_cols].values
        y   = tr['target'].values
        xm  = xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.7,
            reg_alpha=0.5, reg_lambda=2.0,
            min_child_weight=10, random_state=42, verbosity=0)
        xm.fit(X, y)
        rm  = RandomForestRegressor(
            n_estimators=500, max_depth=5, min_samples_leaf=30,
            max_features=0.5, random_state=42, n_jobs=-1)
        rm.fit(X, y)
        xgb_models[inst] = xm
        rf_models[inst]  = rm
    return xgb_models, rf_models


def compute_ic(xgb_models, rf_models, val_feat, feat_cols):
    xgb_p, rf_p, act = [], [], []
    for inst in instruments:
        sub = val_feat[val_feat['instrument'] == inst].dropna(subset=feat_cols + ['target'])
        if len(sub) < 20:
            continue
        X = sub[feat_cols].values
        y = sub['target'].values
        xgb_p.extend(xgb_models[inst].predict(X).tolist())
        rf_p.extend(rf_models[inst].predict(X).tolist())
        act.extend(y.tolist())
    if len(act) < 10:
        return 0.5, 0.5
    ic_xgb = spearmanr(xgb_p, act)[0]
    ic_rf  = spearmanr(rf_p,  act)[0]
    return float(ic_xgb), float(ic_rf)


def ic_weights(ic_xgb, ic_rf):
    w_x = max(ic_xgb, 0)
    w_r = max(ic_rf,  0)
    tot = w_x + w_r
    if tot == 0:
        return 0.5, 0.5
    return w_x / tot, w_r / tot


def predict_ensemble(xgb_models, rf_models, w_xgb, w_rf, feat_df, feat_cols, cutoff):
    xgb_p, rf_p, ens_p = {}, {}, {}
    for inst in instruments:
        mask    = (feat_df['instrument'] == inst) & (feat_df.index <= pd.Timestamp(cutoff))
        last    = feat_df[mask][feat_cols].dropna().iloc[-1:]
        p_x     = xgb_models[inst].predict(last.values)[0]
        p_r     = rf_models[inst].predict(last.values)[0]
        xgb_p[inst] = p_x
        rf_p[inst]  = p_r
        ens_p[inst] = w_xgb * p_x + w_rf * p_r
    return xgb_p, rf_p, ens_p


def optimize_mvo(mu, cov, max_weight=0.40):
    n   = len(mu)
    w0  = np.ones(n) / n
    bounds = [(0.0, max_weight)] * n
    cons   = [{'type': 'eq', 'fun': lambda w: w.sum() - 1.0}]
    def obj(w):
        pv = np.sqrt(w @ cov @ w)
        return -(w @ mu) / pv if pv > 1e-10 else 0
    res = minimize(obj, w0, method='SLSQP', bounds=bounds, constraints=cons)
    w = np.maximum(res.x, 0)
    return w / w.sum()


def black_litterman(mu_views, cov, ret_df, max_weight=0.40):
    last_px  = ret_df.iloc[-1].values
    w_mkt    = last_px / last_px.sum()
    delta    = 2.5
    tau      = 0.05
    pi       = delta * cov @ w_mkt
    P        = np.eye(N)
    Q        = mu_views
    sigma_pi = tau * cov
    omega    = np.diag(np.diag(sigma_pi))
    inv_sp   = np.linalg.inv(sigma_pi)
    inv_om   = np.linalg.inv(omega)
    bl_cov   = np.linalg.inv(inv_sp + P.T @ inv_om @ P)
    bl_mu    = bl_cov @ (inv_sp @ pi + P.T @ inv_om @ Q)
    return optimize_mvo(bl_mu, cov, max_weight=max_weight)


def portfolio_stats(w, ret_df):
    pr      = ret_df[instruments].values @ w
    cum     = np.cumprod(1 + pr)
    n       = len(pr)
    ann_ret = (cum[-1] ** (252 / n)) - 1
    ann_vol = np.std(pr) * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0
    mdd     = ((cum / np.maximum.accumulate(cum)) - 1).min()
    return ann_ret, ann_vol, sharpe, mdd, cum, pr


# ---------- load data ----------
train_ret, train_sig, train_vol, train_cash = load_data(TRAIN_PATH)
test_ret_all, test_sig, test_vol, test_cash  = load_data(TEST_PATH)

signal_cols = [c for c in train_sig.columns if 'trend' in c]
print(f'train: {train_ret.index[0].date()} to {train_ret.index[-1].date()} ({len(train_ret)} days)')
print(f'test_path: {test_ret_all.index[0].date()} to {test_ret_all.index[-1].date()} ({len(test_ret_all)} days)')
print(f'signals: {len(signal_cols)} columns')

train_end  = train_ret.index[-1]
oos_ret    = test_ret_all[test_ret_all.index > train_end]
print(f'oos period: {oos_ret.index[0].date()} to {oos_ret.index[-1].date()} ({len(oos_ret)} days, {len(oos_ret)/21:.1f} months)')

# ---------- features ----------
feat_all, feat_cols = build_features(train_ret, train_sig, train_vol, train_cash, FORWARD)
print(f'features: {len(feat_cols)}')
print(feat_cols)

# ---------- IC computation on held-out slice of training data ----------
ic_split      = train_ret.index[-IC_VAL_DAYS - 1]
feat_ic_train = feat_all[feat_all.index <= ic_split].dropna(subset=feat_cols + ['target'])
feat_ic_val   = feat_all[feat_all.index  > ic_split].dropna(subset=feat_cols + ['target'])
print(f'ic train rows: {len(feat_ic_train)}, ic val rows: {len(feat_ic_val)}')

xgb_ic_models, rf_ic_models = fit_models(feat_ic_train, feat_cols)
ic_xgb, ic_rf = compute_ic(xgb_ic_models, rf_ic_models, feat_ic_val, feat_cols)
w_xgb, w_rf   = ic_weights(ic_xgb, ic_rf)
print(f'information coefficients: xgb={ic_xgb:.4f}, rf={ic_rf:.4f}')
print(f'ic-weighted ensemble: xgb={w_xgb:.3f}, rf={w_rf:.3f}')

# ---------- train on full TRAIN_PATH, predict ----------
feat_train = feat_all[feat_all.index <= train_end].dropna(subset=feat_cols + ['target'])
print(f'train features: {len(feat_train)}')
print(f'train returns: {len(train_ret)} days')
print(f'test returns: {len(oos_ret)} days ({len(oos_ret)/21:.1f} months)')

xgb_models, rf_models = fit_models(feat_train, feat_cols)
xgb_p, rf_p, ens_p    = predict_ensemble(xgb_models, rf_models, w_xgb, w_rf, feat_all, feat_cols, train_end)

print(f'{"instrument":>15s} {"xgb":>10s} {"rf":>10s} {"ensemble":>10s}')
for inst in instruments:
    print(f'{inst:>15s} {xgb_p[inst]:>10.6f} {rf_p[inst]:>10.6f} {ens_p[inst]:>10.6f}')

mu_ens    = np.array([ens_p[i] for i in instruments]) * 252
cov_train = LedoitWolf().fit(train_ret[instruments].values).covariance_ * 252

w_mvo = optimize_mvo(mu_ens, cov_train, max_weight=0.40)
w_bl  = black_litterman(mu_ens, cov_train, train_ret[instruments], max_weight=0.40)

# ---------- OOS comparison: pick best method ----------
_, _, sharpe_mvo_oos, _, _, _ = portfolio_stats(w_mvo, oos_ret)
_, _, sharpe_bl_oos,  _, _, _ = portfolio_stats(w_bl,  oos_ret)
print(f'\noos validation ({oos_ret.index[0].date()} to {oos_ret.index[-1].date()}):')
print(f'  mvo: sharpe={sharpe_mvo_oos:+.3f}')
print(f'  black-litterman: sharpe={sharpe_bl_oos:+.3f}')

if sharpe_bl_oos >= sharpe_mvo_oos:
    best_method = 'black-litterman'
    w_val       = w_bl
    print(f'  selected: black-litterman (sharpe={sharpe_bl_oos:+.3f})')
else:
    best_method = 'mvo'
    w_val       = w_mvo
    print(f'  selected: mvo (sharpe={sharpe_mvo_oos:+.3f})')

ann_ret_oos, ann_vol_oos, sharpe_oos, mdd_oos, _, _ = portfolio_stats(w_val, oos_ret)
print(f'validation')
print(f'return: {ann_ret_oos:+.2%}, ann vol: {ann_vol_oos:.2%}, sharpe: {sharpe_oos:+.3f}, max dd: {mdd_oos:+.2%}')
for inst, w in zip(instruments, w_val):
    print(f'{inst}: {w:.4f}')

# ---------- walk-forward backtest on training data ----------
wf_window  = 504
wf_step    = 63
wf_results = []

for end_idx in range(wf_window + 252, len(train_ret) - wf_step, wf_step):
    wf_tr_start   = max(0, end_idx - wf_window)
    wf_tr_end     = end_idx
    wf_te_start   = end_idx
    wf_te_end     = min(end_idx + wf_step, len(train_ret))
    wf_tr_dates   = train_ret.index[wf_tr_start:wf_tr_end]
    wf_te_dates   = train_ret.index[wf_te_start:wf_te_end]
    wf_xgb_p, wf_rf_p = {}, {}
    for inst in instruments:
        inst_feat = feat_all[(feat_all['instrument'] == inst) & (feat_all.index.isin(wf_tr_dates))]
        inst_feat = inst_feat.dropna(subset=feat_cols + ['target'])
        if len(inst_feat) < 50:
            wf_xgb_p[inst] = 0.0
            wf_rf_p[inst]  = 0.0
            continue
        X_wf = inst_feat[feat_cols].values
        y_wf = inst_feat['target'].values
        xm   = xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.7,
            reg_alpha=0.5, reg_lambda=2.0,
            min_child_weight=10, random_state=42, verbosity=0)
        xm.fit(X_wf, y_wf)
        rm   = RandomForestRegressor(
            n_estimators=500, max_depth=5, min_samples_leaf=30,
            max_features=0.5, random_state=42, n_jobs=-1)
        rm.fit(X_wf, y_wf)
        last          = inst_feat[feat_cols].iloc[-1:]
        wf_xgb_p[inst] = xm.predict(last.values)[0]
        wf_rf_p[inst]  = rm.predict(last.values)[0]
    wf_ens     = {inst: w_xgb * wf_xgb_p[inst] + w_rf * wf_rf_p[inst] for inst in instruments}
    wf_mu      = np.array([wf_ens[i] for i in instruments]) * 252
    wf_ret_tr  = train_ret[instruments].iloc[wf_tr_start:wf_tr_end]
    wf_cov     = LedoitWolf().fit(wf_ret_tr.values).covariance_ * 252
    if best_method == 'black-litterman':
        wf_w = black_litterman(wf_mu, wf_cov, wf_ret_tr, max_weight=0.40)
    else:
        wf_w = optimize_mvo(wf_mu, wf_cov, max_weight=0.40)
    wf_te_ret  = train_ret[instruments].iloc[wf_te_start:wf_te_end]
    wf_pr      = wf_te_ret.values @ wf_w
    wf_sharpe  = wf_pr.mean() / wf_pr.std() * np.sqrt(252) if wf_pr.std() > 0 else 0
    wf_ann_ret = (np.prod(1 + wf_pr) ** (252 / len(wf_pr))) - 1
    wf_results.append({
        'period_start': wf_te_dates[0],
        'period_end':   wf_te_dates[-1],
        'sharpe':       wf_sharpe,
        'ann_ret':      wf_ann_ret,
        'weights':      wf_w,
    })
    print(f'{wf_te_dates[0].date()} to {wf_te_dates[-1].date()}: sharpe={wf_sharpe:+.3f} ret={wf_ann_ret:+.1%}')

wf_sharpes = [r['sharpe'] for r in wf_results]
print(f'mean sharpe: {np.mean(wf_sharpes):.3f}', 'std sharpe', np.std(wf_sharpes), 'min sharpe', np.min(wf_sharpes), 'max sharpe', np.max(wf_sharpes), '%positive', np.mean(np.array(wf_sharpes) > 0))

all_dates_list, all_ret_list = [], []
for r in wf_results:
    period_ret = train_ret[instruments].loc[r['period_start']:r['period_end']]
    pr         = period_ret.values @ r['weights']
    all_dates_list.extend(period_ret.index.tolist())
    all_ret_list.extend(pr.tolist())

bt      = pd.Series(all_ret_list, index=all_dates_list)
bt      = bt[~bt.index.duplicated(keep='first')]
bt_cum  = (1 + bt).cumprod()
tot_days   = len(bt)
total_ret  = bt_cum.iloc[-1] - 1
ann_ret_bt = (1 + total_ret) ** (252 / tot_days) - 1
ann_vol_bt = bt.std() * np.sqrt(252)
sharpe_bt  = ann_ret_bt / ann_vol_bt
mdd_bt     = ((bt_cum / bt_cum.cummax()) - 1).min()
print(f"period: {bt.index[0].date()} to {bt.index[-1].date()} ({tot_days} days), total return {total_ret:+.2%}, annualized return: {ann_ret_bt:+.2%}, annualized vol: {ann_vol_bt:.2%}, sharpe ratio: {sharpe_bt:+.3f}, max drawdown: {mdd_bt:+.2%}")

ew_ret     = train_ret[instruments].loc[bt.index[0]:bt.index[-1]].mean(axis=1)
ew_cum     = (1 + ew_ret).cumprod()
ew_ann_ret = (ew_cum.iloc[-1] ** (252 / len(ew_ret))) - 1
ew_ann_vol = ew_ret.std() * np.sqrt(252)
ew_sharpe  = ew_ann_ret / ew_ann_vol
print(f'{ew_sharpe:+.3f} ret={ew_ann_ret:+.2%}')

# ---------- final: train on full TEST_PATH, apply best method ----------
print(f'\nfull dataset: {test_ret_all.index[0].date()} to {test_ret_all.index[-1].date()} ({len(test_ret_all)} days)')

feat_full, feat_cols_full = build_features(test_ret_all, test_sig, test_vol, test_cash, FORWARD)
train_full = feat_full.dropna(subset=feat_cols_full + ['target'])
print(f'train samples: {len(train_full)}')

xgb_final, rf_final         = fit_models(train_full, feat_cols_full)
final_end                   = test_ret_all.index[-1]
xgb_fp, rf_fp, ens_fp       = predict_ensemble(xgb_final, rf_final, w_xgb, w_rf, feat_full, feat_cols_full, final_end)

print(f'\n{"instrument":>15s} {"xgb":>10s} {"rf":>10s} {"ens":>10s}')
for inst in instruments:
    print(f'{inst:>15s} {xgb_fp[inst]:>10.6f} {rf_fp[inst]:>10.6f} {ens_fp[inst]:>10.6f}')

mu_final  = np.array([ens_fp[i] for i in instruments]) * 252
cov_final = LedoitWolf().fit(test_ret_all[instruments].values).covariance_ * 252

if best_method == 'black-litterman':
    w_final = black_litterman(mu_final, cov_final, test_ret_all[instruments], max_weight=0.40)
else:
    w_final = optimize_mvo(mu_final, cov_final, max_weight=0.40)

w_final  = np.round(w_final, 4)
residual = round(1.0 - w_final.sum(), 4)
w_final[np.argmax(w_final)] += residual

assert abs(w_final.sum() - 1.0) < 1e-8
assert all(w_final >= 0)

for inst, w in zip(instruments, w_final):
    print(f'  {inst}: {w:.4f}')
print(f'sum: {w_final.sum():.4f}')
print(f'method: {best_method}')

alloc = pd.DataFrame({'asset': instruments, 'weight': w_final})
alloc.to_csv('/Users/yash/algothon-26/labubu-money/allocation.csv', index=False)
print(alloc.to_string(index=False))
print(f'\nsaved to /Users/yash/algothon-26/labubu-money/allocation.csv')
