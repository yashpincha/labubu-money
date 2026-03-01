import time
import math
import requests
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from datetime import datetime, timezone, timedelta
from bot_template import BaseBot, OrderBook, OrderRequest, Side, Trade

# ==========================================
# CONFIGURATION
# ==========================================
EXCHANGE_URL = "http://ec2-52-19-74-159.eu-west-1.compute.amazonaws.com/" 
USERNAME = "NewPret"
PASSWORD = "NewPret123456"
AERODATABOX_KEY = "YOUR_RAPIDAPI_KEY"

# How wide around our theoretical fair value we want to quote (in ticks)
SPREAD_PERCENTAGE = {
    "WX_SPOT": 0.005,
    "WX_SUM": 0.005,
    "TIDE_SPOT": 0.01,
    "TIDE_SWING": 0.05,
    "LHR_COUNT": 0.02,
    "LHR_INDEX": 0.05,
    "LON_ETF": 0.01,
    "LON_FLY": 0.05,
}
ORDER_VOLUME = 10
MAX_POSITION = 100 # Rule: +-100 position limit
TIER_OFFSETS = {
    "WX_SPOT": [50.0, 5.0, 2.5, 1.0],
    "WX_SUM": [50.0, 5.0, 2.5, 1.0],
    "TIDE_SPOT": [50.0, 5.0, 2.5, 1.0],
    "TIDE_SWING": [10.0, 5.0, 2.5, 1.0],
    "LHR_COUNT": [10.0, 5.0, 2.5, 1.0],
    "LHR_INDEX": [5.0, 3.5, 2.0, 1.0],
    "LON_ETF": [25.0, 5.0, 2.5, 1.0],
    "LON_FLY": [10.0, 5.0, 2.5, 1.0],
}  # Multipliers for dynamic_width
TIER_VOLUMES = [1, 20, 10, 10]

# ==========================================
# PRICING ENGINE
# ==========================================
class PricingEngine:
    """
    Calculates theoretical fair values ('theos') for all 8 products.
    Note: These are 'naive' theos based on current spot data. For the competition,
    you will need to upgrade these to forecast the values at 12pm London time.
    """
    LONDON_LAT, LONDON_LON = 51.5074, -0.1278
    THAMES_MEASURE = "0006-level-tidal_level-i-15_min-mAOD"

    def __init__(self):
        self.theos = {}
        
    def celsius_to_fahrenheit(self, c: float) -> float:
        """Convert Open-Meteo Celsius to Fahrenheit for settlement."""
        return (c * 9/5) + 32
    
    def get_weather(self, past_steps=96, forecast_steps=96):
        """15-min weather for London. 96 steps = 24 hours.

        Returns DataFrame with: time, temperature, wind_speed, humidity,
        precipitation, cloud_cover, visibility, apparent_temperature.
        """
        variables = "temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,wind_speed_10m,cloud_cover,visibility"
        resp = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": self.LONDON_LAT, "longitude": self.LONDON_LON,
            "minutely_15": variables,
            "past_minutely_15": past_steps,
            "forecast_minutely_15": forecast_steps,
            "timezone": "Europe/London",
        })
        resp.raise_for_status()
        m = resp.json()["minutely_15"]
        return pd.DataFrame({
            "time": pd.to_datetime(m["time"]).tz_localize("Europe/London"),
            "temperature": m["temperature_2m"],
            "apparent_temperature": m["apparent_temperature"],
            "humidity": m["relative_humidity_2m"],
            "precipitation": m["precipitation"],
            "wind_speed": m["wind_speed_10m"],
            "cloud_cover": m["cloud_cover"],
            "visibility": m["visibility"],
        })
    
    def get_thames(self, limit=200):
        """Fetch recent Thames tidal readings at Westminster.

        Returns DataFrame with: time, level (mAOD).
        Use limit=400 for ~4 days of history.
        """
        resp = requests.get(
            f"https://environment.data.gov.uk/flood-monitoring/id/measures/{self.THAMES_MEASURE}/readings",
            params={"_sorted": "", "_limit": limit},
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
        df = pd.DataFrame(items)[["dateTime", "value"]].rename(columns={"dateTime": "time", "value": "level"})
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Europe/London")
        return df.sort_values("time").reset_index(drop=True)

    def update_weather_theos(self):
        try:
            df = self.get_weather(96, 96)

            # 2. WX_SPOT: Target Sunday 12:00 PM specifically
            target_time = pd.Timestamp("2026-03-01 12:00:00", tz="Europe/London")
            
            # Find the row closest to our target settlement time
            settlement_row = df.iloc[(df['time'] - target_time).abs().argsort()[:1]]
            
            if not settlement_row.empty:
                temp_c = settlement_row['temperature'].values[0]
                humidity = settlement_row['humidity'].values[0]
                temp_f = self.celsius_to_fahrenheit(temp_c)
                
                # Settlement formula: temp_F * humidity_%
                self.theos["WX_SPOT"] = temp_f * humidity
            
            # 3. WX_SUM: Sum of (temp_F * humidity_%) / 100 over the 24h session
            # Define the 24h session window (e.g., Saturday 12pm to Sunday 12pm)
            session_start = target_time - pd.Timedelta(hours=24)
            session_df = df[(df['time'] >= session_start) & (df['time'] <= target_time)].copy()
            
            if not session_df.empty:
                # Apply formula to each 15-min interval in the session
                session_df['interval_val'] = session_df.apply(
                    lambda row: self.celsius_to_fahrenheit(row['temperature']) * row['humidity'], 
                    axis=1
                )
                self.theos["WX_SUM"] = session_df['interval_val'].sum() / 100.0
                
        except Exception as e:
            print(f"Error fetching weather: {e}")

    def update_tide_theos(self):
        try:
            # 1. Fetch historical data (use a large limit for a strong fit)
            # 500 readings is ~5 days of data
            df = self.get_thames(limit=500)
            if df.empty:
                return

            # Target: Sunday March 1st, 12:00 PM
            target_time = pd.Timestamp("2026-03-01 12:00:00", tz="Europe/London")
            session_start = target_time - pd.Timedelta(hours=24)

            # 2. Multi-Constituent Tidal Model
            # Tides are composed of M2 (12.42h) and S2 (12.0h) cycles primarily.
            def advanced_tidal_func(t, A1, phi1, A2, phi2, C):
                # M2 constituent (~12.42h period)
                m2 = A1 * np.sin(2 * np.pi * t / 12.42 + phi1)
                # S2 constituent (12.0h period)
                s2 = A2 * np.sin(2 * np.pi * t / 12.0 + phi2)
                return m2 + s2 + C

            df['hours'] = (df['time'] - df['time'].min()).dt.total_seconds() / 3600.0
            
            # Initial guesses: [Amp1, Phase1, Amp2, Phase2, Offset]
            initial_guess = [2.0, 0, 0.5, 0, df['level'].mean()]
            
            params, _ = curve_fit(advanced_tidal_func, df['hours'], df['level'], p0=initial_guess)
            
            # 3. Predict future levels
            last_time = df['time'].max()
            future_times = pd.date_range(start=last_time + pd.Timedelta(minutes=15), 
                                        end=target_time, freq='15min')
            future_hours = (future_times - df['time'].min()).total_seconds() / 3600.0
            future_levels = advanced_tidal_func(future_hours, *params)
            
            df_future = pd.DataFrame({'time': future_times, 'level': future_levels})
            df_full = pd.concat([df[['time', 'level']], df_future]).sort_values('time')

            # 4. TIDE_SPOT Theo
            target_level = df_full.iloc[-1]['level']
            self.theos["TIDE_SPOT"] = abs(target_level) * 1000

            # 5. TIDE_SWING Theo
            session_df = df_full[(df_full['time'] >= session_start) & (df_full['time'] <= target_time)].copy()
            if len(session_df) > 1:
                session_df['diff_m'] = session_df['level'].diff().abs()
                def strangle(d): return max(0, 0.20 - d) + max(0, d - 0.25)
                self.theos["TIDE_SWING"] = session_df['diff_m'].apply(strangle).sum() * 100
                
            print(f"✅ TIDE_SPOT: {self.theos['TIDE_SPOT']:.1f} | TIDE_SWING: {self.theos['TIDE_SWING']:.1f}")

        except Exception as e:
            print(f"❌ Error in harmonic fit: {e}")

    def update_flight_theos(self):
        try:
            # 1. Load the data
            # Assuming columns: scheduled_arrival_times, revised_arrival_times
            arrivals = pd.read_csv('data/arrivals.csv')
            # Assuming columns: scheduled_departure_times, revised_departure_times
            departures = pd.read_csv('data/departures.csv')

            # 2. Define the 24h Window (Sunday 12:00 PM back to Saturday 12:00 PM)
            target_time = pd.Timestamp("2026-03-01 12:00:00", tz="Europe/London")
            session_start = target_time - pd.Timedelta(hours=24)

            def to_london_dt(series):
                # parse_dates doesn't always handle offsets well, so we convert manually
                dt = pd.to_datetime(series, utc=True)
                return dt.dt.tz_convert("Europe/London")

            arr_times = to_london_dt(arrivals["final_arrival_time"])
            dep_times = to_london_dt(departures["final_departure_time"])

            # Filter for the session window
            arr_session = arr_times[(arr_times >= session_start) & (arr_times <= target_time)]
            dep_session = dep_times[(dep_times >= session_start) & (dep_times <= target_time)]

            # 3. LHR_COUNT: Total arrivals + departures
            self.theos["LHR_COUNT"] = len(arr_session) + len(dep_session)

            # 4. LHR_INDEX: Imbalance metric per 30-min interval
            # Formula: abs(sum(100 * (arr - dep) / (arr + dep)))
            
            # Create a time range of 30-minute bins for the session
            bins = pd.date_range(start=session_start, end=target_time, freq='30min')
            
            interval_metrics = []
            for i in range(len(bins) - 1):
                start, end = bins[i], bins[i+1]
                
                n_arr = len(arr_session[(arr_session >= start) & (arr_session < end)])
                n_dep = len(dep_session[(dep_times >= start) & (dep_times < end)])
                
                if (n_arr + n_dep) > 0:
                    # Calculate imbalance for this specific 30m block
                    imbalance = 100 * (n_arr - n_dep) / (n_arr + n_dep)
                    interval_metrics.append(imbalance)
                else:
                    interval_metrics.append(0)

            # Final settlement is the absolute value of the sum of these metrics
            self.theos["LHR_INDEX"] = abs(sum(interval_metrics))

            print(f"✈️ LHR_COUNT Theo: {self.theos['LHR_COUNT']}")
            print(f"✈️ LHR_INDEX Theo: {self.theos['LHR_INDEX']:.2f}")

        except Exception as e:
            print(f"❌ Error processing flight CSVs: {e}")

    def update_derived_theos(self):
        # LON_ETF = TIDE_SPOT + WX_SPOT + LHR_COUNT
        tide = self.theos.get("TIDE_SPOT", 0)
        wx = self.theos.get("WX_SPOT", 0)
        flights = self.theos.get("LHR_COUNT", 0)
        etf_theo = tide + wx + flights
        self.theos["LON_ETF"] = etf_theo
        
        # LON_FLY = 2*Put(6200) + Call(6200) - 2*Call(6600) + 3*Call(7000)
        def put(k, s): return max(0, k - s)
        def call(k, s): return max(0, s - k)
        
        fly_theo = (2 * put(6200, etf_theo)) + call(6200, etf_theo) - (2 * call(6600, etf_theo)) + (3 * call(7000, etf_theo))
        self.theos["LON_FLY"] = fly_theo

    def get_all_theos(self):
        self.update_weather_theos()
        self.update_tide_theos()
        self.update_flight_theos()
        self.update_derived_theos()
        print(f"Theoretical TIDE_SPOT = {self.theos["TIDE_SPOT"]}")
        print(f"Theoretical TIDE_SWING = {self.theos["TIDE_SWING"]}")
        print(f"Theoretical WX_SPOT = {self.theos["WX_SPOT"]}")
        print(f"Theoretical WX_SUM = {self.theos["WX_SUM"]}")
        print(f"Theoretical LHR_COUNT = {self.theos["LHR_COUNT"]}")
        print(f"Theoretical LHR_INDEX = {self.theos["LHR_INDEX"]}")
        print(f"Theoretical LON_ETF = {self.theos["LON_ETF"]}")
        print(f"Theoretical LON_FLY = {self.theos["LON_FLY"]}")
        return self.theos

# ==========================================
# MARKET MAKER BOT
# ==========================================
class MarketMakerBot(BaseBot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pricer = PricingEngine()
        self.theos = {}
        self.positions = {}

    def on_orderbook(self, ob: OrderBook):
        pass # We use a timed loop for quoting instead of reacting to every tick to save requests

    def on_trades(self, trade: Trade):
        side = "BOUGHT" if trade.buyer == self.username else "SOLD"
        # print(f"\n🔔 FILL: {side} {trade.volume}x {trade.product} @ {trade.price}")

    def run(self):
        self.start() # Starts the SSE streams
        print(f"Connected to {EXCHANGE_URL} as {self.username}")
        
        products = self.get_products()
        tick_sizes = {p.symbol: p.tickSize for p in products}
        
        loop_counter = 0

        while True:
            try:
                # 1. Update Theos periodically
                if loop_counter % 6 == 0:
                    print("\n🔄 Updating theoretical values...")
                    self.theos = self.pricer.get_all_theos()
                    # print(f"Current Positions: {self.positions}")

                # 2. Cancel old orders & refresh positions
                self.cancel_all_orders()
                time.sleep(0.5) # Prevent rate limiting
                self.positions = self.get_positions()

                # 3. Calculate and build the tiered order ladder
                new_orders = []
                for symbol in tick_sizes.keys():
                    theo = self.theos.get(symbol)
                    if theo is None or math.isnan(theo):
                        continue

                    current_pos = self.positions.get(symbol, 0)
                    
                    # A. Dynamic Spread & Skew Calculation
                    base_width = theo * SPREAD_PERCENTAGE.get(symbol, 0.01)
                    base_width = max(base_width, 2)  # Minimum 2 tick spread
                    
                    # Skew shifts the entire ladder based on current inventory
                    skew_factor = current_pos / MAX_POSITION 
                    skew = skew_factor * base_width

                    # B. Track remaining capacity for the +-100 limit
                    remaining_buy_room = MAX_POSITION - current_pos
                    remaining_sell_room = MAX_POSITION + current_pos

                    # C. Generate Tiers (Market Making + Ridiculous Stub)
                    for offset_mult, vol in zip(TIER_OFFSETS[symbol], TIER_VOLUMES):
                        # Calculate prices for this specific tier
                        tier_bid = math.floor(theo - (base_width * offset_mult) - skew)
                        tier_ask = math.ceil(theo + (base_width * offset_mult) - skew)

                        # Determine sizes that fit within remaining limit room
                        this_buy_vol = max(0, min(vol, remaining_buy_room))
                        this_sell_vol = max(0, min(vol, remaining_sell_room))

                        # Build Buy Order
                        if this_buy_vol > 0 and tier_bid > 0:
                            new_orders.append(OrderRequest(symbol, tier_bid, Side.BUY, int(this_buy_vol)))
                            remaining_buy_room -= this_buy_vol  # Consume room for next tier

                        # Build Sell Order (ensure spread is not crossed)
                        if this_sell_vol > 0 and tier_ask > tier_bid:
                            new_orders.append(OrderRequest(symbol, tier_ask, Side.SELL, int(this_sell_vol)))
                            remaining_sell_room -= this_sell_vol # Consume room for next tier

                if new_orders:
                    self.send_orders(new_orders)

                loop_counter += 1
                time.sleep(10)  # Respect API limits (max 1 request/sec)

            except Exception as e:
                print(f"Error in trading loop: {e}")
                time.sleep(10)


if __name__ == "__main__":
    bot = MarketMakerBot(EXCHANGE_URL, USERNAME, PASSWORD)
    try:
        bot.run()
    except KeyboardInterrupt:
        print("\nStopping bot and cancelling all orders...")
        bot.cancel_all_orders()
        bot.stop()
