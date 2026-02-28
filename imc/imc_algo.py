import time
import math
import requests
import pandas as pd
from datetime import datetime, timezone
from bot_template import BaseBot, OrderBook, OrderRequest, Side, Trade

# ==========================================
# CONFIGURATION
# ==========================================
EXCHANGE_URL = "http://ec2-52-49-69-152.eu-west-1.compute.amazonaws.com/" 
USERNAME = "Pret"
PASSWORD = "Pret123456"
AERODATABOX_KEY = "YOUR_RAPIDAPI_KEY"

# How wide around our theoretical fair value we want to quote (in ticks)
BASE_SPREAD_WIDTH = 5.0
ORDER_VOLUME = 5

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
    
    def get_weather(past_steps=96, forecast_steps=96):
        """15-min weather for London. 96 steps = 24 hours.

        Returns DataFrame with: time, temperature, wind_speed, humidity,
        precipitation, cloud_cover, visibility, apparent_temperature.
        """
        variables = "temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,wind_speed_10m,cloud_cover,visibility"
        resp = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": LONDON_LAT, "longitude": LONDON_LON,
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
    
    def get_thames(limit=200):
        """Fetch recent Thames tidal readings at Westminster.

        Returns DataFrame with: time, level (mAOD).
        Use limit=400 for ~4 days of history.
        """
        resp = requests.get(
            f"https://environment.data.gov.uk/flood-monitoring/id/measures/{THAMES_MEASURE}/readings",
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
            target_time = pd.Timestamp("2026-03-01 12:00:00")
            
            # Find the row closest to our target settlement time
            settlement_row = df.iloc[(df['time'] - target_time).abs().argsort()[:1]]
            
            if not settlement_row.empty:
                temp_c = settlement_row['temperature_2m'].values[0]
                humidity = settlement_row['relative_humidity_2m'].values[0]
                temp_f = self.celsius_to_fahrenheit(temp_c)
                
                # Settlement formula: temp_F * humidity_%
                self.theos["WX_SPOT"] = temp_f * humidity
            
            # 3. WX_SUM: Sum of (temp_F * humidity_%) / 100 over the 24h session
            # Define the 24h session window (e.g., Saturday 12pm to Sunday 12pm)
            session_start = target_time - pd.Timedelta(hours=24)
            session_df = df[(df['time'] > session_start) & (df['time'] <= target_time)]
            
            if not session_df.empty:
                # Apply formula to each 15-min interval in the session
                session_df['interval_val'] = session_df.apply(
                    lambda row: self.celsius_to_fahrenheit(row['temperature_2m']) * row['relative_humidity_2m'], 
                    axis=1
                )
                self.theos["WX_SUM"] = session_df['interval_val'].sum() / 100.0
                
        except Exception as e:
            print(f"Error fetching weather: {e}")

    def update_tide_theos(self):
        try:
            # Fetch enough history to cover the 24h session
            # 96 readings = 24h of 15-min intervals
            df = self.get_thames(limit=120)
            if df.empty:
                return

            # Define settlement target: Sunday 12:00 PM
            # Current time is Saturday; settlement is tomorrow at noon.
            target_time = pd.Timestamp("2026-03-01 12:00:00", tz="Europe/London")
            session_start = target_time - pd.Timedelta(hours=24)

            # 1. TIDE_SPOT Theo
            # Logic: Absolute value of tidal height in mm AOD at 12pm
            # Since we are currently in the session, the latest reading is our best proxy
            # until you implement a sinusoidal curve-fitting model.
            latest_reading = df.iloc[-1]
            latest_level_maod = latest_reading['level'] # API returns meters
            
            # Calculation: |-1.48| * 1000 = 1480
            self.theos["TIDE_SPOT"] = abs(latest_level_maod) * 1000

            # 2. TIDE_SWING Theo
            # Logic: Sum of strangle payoffs on 15-min absolute changes
            # Filter for data within the active 24h session window
            session_df = df[(df['time'] > session_start) & (df['time'] <= target_time)].copy()
            
            if len(session_df) > 1:
                # Calculate absolute differences in meters
                session_df['diff_m'] = session_df['level'].diff().abs()
                
                def calculate_strangle(diff_m):
                    if pd.isna(diff_m): return 0
                    # Strikes are 0.20m and 0.25m (20cm and 25cm)
                    put_payoff = max(0, 0.20 - diff_m)
                    call_payoff = max(0, diff_m - 0.25)
                    return put_payoff + call_payoff

                # Realized swing in the session so far
                realized_swing_m = session_df['diff_m'].apply(calculate_strangle).sum()
                
                # Projection: Estimate remaining volatility
                # Calculate how many 15-min intervals remain until Sunday 12pm
                time_left = target_time - session_df['time'].iloc[-1]
                intervals_left = max(0, int(time_left.total_seconds() / 900))
                
                # Use the average realized payoff per interval to forecast the rest of the session
                avg_payoff = realized_swing_m / len(session_df)
                forecasted_swing_m = intervals_left * avg_payoff
                
                # Total Theo: (Realized + Forecasted) * 100 multiplier
                self.theos["TIDE_SWING"] = (realized_swing_m + forecasted_swing_m) * 100
                
            print(f"✅ TIDE_SPOT Theo: {self.theos.get('TIDE_SPOT'):.2f}")
            print(f"✅ TIDE_SWING Theo: {self.theos.get('TIDE_SWING'):.2f}")

        except Exception as e:
            print(f"❌ Error generating Thames theos: {e}")

    def update_flight_theos(self):
        # NOTE: RapidAPI is limited to ~150 requests/month. 
        # In a real bot, cache this and only call it once an hour!
        # For safety in this template, we will hardcode a naive static estimate unless you plug in the API.
        self.theos["LHR_COUNT"] = 1200 # Placeholder: typical daily LHR flights
        self.theos["LHR_INDEX"] = 500  # Placeholder

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
        print(f"\n🔔 FILL: {side} {trade.volume}x {trade.product} @ {trade.price}")

    def run(self):
        self.start() # Starts the SSE streams
        print(f"Connected to {EXCHANGE_URL} as {self.username}")
        
        products = self.get_products()
        tick_sizes = {p.symbol: p.tickSize for p in products}
        
        loop_counter = 0

        while True:
            try:
                # 1. Update Theos (every ~60 seconds to avoid spamming public APIs)
                if loop_counter % 12 == 0:
                    print("\n🔄 Updating theoretical values...")
                    self.theos = self.pricer.get_all_theos()
                    print(f"Current Positions: {self.positions}")

                # 2. Cancel old orders
                self.cancel_all_orders()
                time.sleep(0.5) # Prevent rate limiting
                self.positions = self.get_positions()

                # 3. Calculate and send new quotes
                new_orders = []
                for symbol in tick_sizes.keys():
                    theo = self.theos.get(symbol)
                    if theo is None or math.isnan(theo):
                        continue

                    
                    # Inventory risk management: shift quotes based on our position
                    # If we are long (positive pos), we drop our prices to sell easier and buy less.
                    pos = self.positions.get(symbol, 0)
                    skew = pos * 0.5 

                    # Calculate Bid and Ask
                    bid = math.floor((theo - BASE_SPREAD_WIDTH - skew))
                    ask = math.ceil((theo + BASE_SPREAD_WIDTH - skew))

                    # Ensure bids are strictly positive and bid < ask
                    if bid > 0 and bid < ask:
                        new_orders.append(OrderRequest(symbol, bid, Side.BUY, ORDER_VOLUME))
                        new_orders.append(OrderRequest(symbol, ask, Side.SELL, ORDER_VOLUME))
                        print(f"Quoting {symbol: <10} | Theo: {theo:.1f} | Bid: {bid} | Ask: {ask}")

                # Send orders in bulk if your API supports it, otherwise loop.
                # Assuming bot.send_orders takes a list:
                if new_orders:
                    self.send_orders(new_orders)

                # 4. Sleep to respect API limits (max 1 request/sec)
                loop_counter += 1
                time.sleep(5) 

            except Exception as e:
                print(f"Error in trading loop: {e}")
                time.sleep(5)

if __name__ == "__main__":
    bot = MarketMakerBot(EXCHANGE_URL, USERNAME, PASSWORD)
    try:
        bot.run()
    except KeyboardInterrupt:
        print("\nStopping bot and cancelling all orders...")
        bot.cancel_all_orders()
        bot.stop()
