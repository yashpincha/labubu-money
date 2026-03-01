import os
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
warnings.filterwarnings('ignore')
np.random.seed(42)
DATA_PATH = '/Users/yash/algothon-26/man-imperial-algothon-2026/data/2025-09-30'
prices = pd.read_csv(os.path.join(DATA_PATH, 'prices.csv'), parse_dates=['date']).sort_values('date').set_index('date')
signals = pd.read_csv(os.path.join(DATA_PATH, 'signals.csv'), parse_dates=['date']).sort_values('date').set_index('date')
instruments = [f'INSTRUMENT_{i}' for i in range(1, 11)]
N = len(instruments)
returns = prices[instruments].pct_change().dropna()
signal_cols = [c for c in signals.columns if 'trend' in c]

TRAIN_END = '2025-05-30'
TEST_START = '2025-09-30'
train_ret = returns.loc[:TRAIN_END]
test_ret = returns.loc[TEST_START:]
train_sig = signals.loc[:TRAIN_END]

print(f"loaded: {len(instruments)} instruments.")
print(f"period: {train_ret.index[0].date()} to {train_ret.index[-1].date()}")
print(f"test period:  {test_ret.index[0].date()} to {test_ret.index[-1].date()}\n")

def portfolio_stats(w, ret_df):
    pr = (ret_df[instruments].values @ w)
    cum = np.cumprod(1 + pr)
    n = len(pr)
    ann_ret = (cum[-1] ** (252/n)) - 1
    ann_vol = np.std(pr) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    mdd = ((cum / np.maximum.accumulate(cum)) - 1).min()
    return ann_ret, ann_vol, sharpe, mdd, cum

def print_stats(name, w, train, test):
    tr_ret, tr_vol, tr_sh, tr_mdd, tr_cum = portfolio_stats(w, train)
    te_ret, te_vol, te_sh, te_mdd, te_cum = portfolio_stats(w, test)
    print(f'{name:30s},  is sharpe={tr_sh:+.3f} ret={tr_ret:+.1%} vol={tr_vol:.1%} mod={tr_mdd:+.1%}')
    print(f'{"":30s}, oos sharpe={te_sh:+.3f} ret={te_ret:+.1%} vol={te_vol:.1%} mod={te_mdd:+.1%}')
    return te_sh, te_cum

# big up chatgpt
def optimize_mvo(mu, cov, method='max_sharpe'):
    n = len(mu)
    w0 = np.ones(n) / n
    bounds = [(0.0, 0.4)] * n # allocation limits: 0% to 40% per asset
    cons = [{'type': 'eq', 'fun': lambda w: w.sum() - 1.0}]
    
    if method == 'max_sharpe':
        def obj(w):
            pv = np.sqrt(w @ cov @ w)
            return -(w @ mu) / pv if pv > 0 else 0
    elif method == 'min_var':
        def obj(w):
            return w @ cov @ w
    elif method == 'risk_parity':
        def obj(w):
            pv = np.sqrt(w @ cov @ w)
            rc = w * (cov @ w) / pv
            return np.sum((rc - pv/n)**2)
    res = minimize(obj, w0, method='SLSQP', bounds=bounds, constraints=cons)
    w = np.maximum(res.x, 0)
    return w / w.sum()

mu_train = train_ret[instruments].mean().values * 252
cov_train = LedoitWolf().fit(train_ret[instruments].values).covariance_ * 252
w_mvo_sharpe = optimize_mvo(mu_train, cov_train, 'max_sharpe')
print_stats('mvo max-sharpe', w_mvo_sharpe, train_ret, test_ret)

w_mvo_minvar = optimize_mvo(mu_train, cov_train, 'min_var')
print_stats('mvo min-variance', w_mvo_minvar, train_ret, test_ret)

w_mvo_rp = optimize_mvo(mu_train, cov_train, 'risk_parity')
print_stats('mvo risk-parity', w_mvo_rp, train_ret, test_ret)

#market equilibrium
market_caps = prices[instruments].loc[:TRAIN_END].iloc[-1].values
w_mkt = market_caps / market_caps.sum()
delta = 2.5 # risk aversion pulled this one straight outta my ass
pi = delta * cov_train @ w_mkt
latest_sig = signals[signal_cols].loc[:TRAIN_END].iloc[-1]
P = np.eye(N)
Q = np.zeros(N)
omega_diag = np.zeros(N)

for i, inst in enumerate(instruments):
    inst_cols = [c for c in signal_cols if c.startswith(inst + '_')]
    avg_sig = latest_sig[inst_cols].mean()
    if not np.isnan(avg_sig):
        Q[i] = -avg_sig * 0.05
        omega_diag[i] = 1.0 / (abs(avg_sig) + 0.01) # uncertainty on view

tau = 0.05
omega = np.diag(omega_diag) * tau
sigma_pi = tau * cov_train
inv_sigma = np.linalg.inv(sigma_pi)
inv_omega = np.linalg.inv(omega)
bl_cov = np.linalg.inv(inv_sigma + P.T @ inv_omega @ P)
bl_mu = bl_cov @ (inv_sigma @ pi + P.T @ inv_omega @ Q)
w_bl = optimize_mvo(bl_mu, cov_train, 'max_sharpe')
print_stats('black-litterman', w_bl, train_ret, test_ret)