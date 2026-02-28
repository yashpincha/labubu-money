import time
import math
from bot_template import BaseBot, OrderBook, Trade, OrderRequest, Side

import requests

AERODATABOX_KEY = "123"
AERODATABOX_HOST = "aerodatabox.p.rapidapi.com"
AIRPORT = "LHR"

def fetch_flights(airport=AIRPORT, offset_minutes=-360, duration_minutes=720, filters: dict | None = None):
    """Fetch flights by relative time window (offset from now)."""
    
    params = f"?offsetMinutes={offset_minutes}&durationMinutes={duration_minutes}&direction=Both"
    
    if filters:
        for k, v in filters.items():
            params += f"&{k}={str(v).lower()}"

    url = f"https://{AERODATABOX_HOST}/flights/airports/iata/{airport}{params}"

    headers = {
        "X-RapidAPI-Key": AERODATABOX_KEY,
        "X-RapidAPI-Host": AERODATABOX_HOST
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    return response.json()


class IntegratedBot(BaseBot):
    def __init__(self, url, user, pw):
        super().__init__(url, user, pw)
        self.books = {}
        self.is_active = True
        self.MAX_POS = 40  # Stay under 50 to avoid risk halts
        self.last_action_time = 0
        
    def on_orderbook(self, ob: OrderBook):
        self.books[ob.product] = ob
        # Run arb check immediately on book update
        if self.is_active:
            self.run_arbitrage()
    def manage_stop_loss(self, symbol):
        """Checks if a specific product's position needs to be liquidated."""
        positions = self.get_positions()
        pos_qty = positions.get(symbol, 0)
        
        if pos_qty == 0:
            return

        mid = self.get_mid(symbol)
        if not mid:
            return

        # You would ideally track 'avg_price' in on_trades, 
        # but for simplicity, we compare Mid to the market price.
        # If Long and Mid drops too low, or Short and Mid rises too high:
        
        # NOTE: A more robust version tracks self.avg_entry_price[symbol]
        # logic: if current_pnl_of_position < -threshold: Sell/Buy to Close.
        
        if abs(pos_qty) > self.MAX_POS:
            print(f"Position Limit Triggered for {symbol}. Reducing...")
            self.liquidate_position(symbol, pos_qty)

    def liquidate_position(self, symbol, qty):
        """Sends an aggressive order to get out of a position immediately."""
        self.throttle()
        side = Side.SELL if qty > 0 else Side.BUY
        # Use a slightly worse price to ensure an immediate fill (IOC-like behavior)
        book = self.books.get(symbol)
        if not book: return
        
        exit_price = book.buy_orders[0].price if qty > 0 else book.sell_orders[0].price
        
        print(f"!!! STOP LOSS/LIMIT REACHED: Closing {qty} of {symbol} !!!")
        self.send_order(OrderRequest(symbol, exit_price, side, abs(qty)))
    def on_trades(self, trade: Trade):
        if trade.buyer == self.username or trade.seller == self.username:
            print(f"  FILL: {trade.volume}x {trade.product} @ {trade.price}")

    def get_mid(self, symbol):
        book = self.books.get(symbol)
        if book and book.buy_orders and book.sell_orders:
            return (book.buy_orders[0].price + book.sell_orders[0].price) / 2
        return None

    def throttle(self):
        """Ensures we stay within the 1 req/sec limit."""
        elapsed = time.time() - self.last_action_time
        if elapsed < 1.1:
            time.sleep(1.1 - elapsed)
        self.last_action_time = time.time()

    def run_arbitrage(self):
        """
        Strategy: LON_ETF Arb.
        Formula: ETF = TIDE_SPOT + WX_SPOT + LHR_COUNT
        """
        try:
            p_tide = self.get_mid("TIDE_SPOT")
            p_wx = self.get_mid("WX_SPOT")
            p_lhr = self.get_mid("LHR_COUNT")
            p_etf = self.get_mid("LON_ETF")

            if None in [p_tide, p_wx, p_lhr, p_etf]:
                return

            fair_etf = p_tide + p_wx + p_lhr
            
            # If ETF is cheap, buy it and sell the parts (Reverse if expensive)
            if p_etf < (fair_etf - 10):
                self.throttle()
                self.send_order(OrderRequest("LON_ETF", p_etf + 1, Side.BUY, 2))
                print(f"Arb: Buying cheap ETF (FV: {fair_etf}, Price: {p_etf})")
        except Exception as e:
            pass

    def run_market_maker(self):
        """Quotes all 8 markets with risk-adjusted widths."""
        products = self.get_products()
        positions = self.get_positions()
        
        for p in products:
            mid = self.get_mid(p.symbol) or p.startingPrice
            pos = positions.get(p.symbol, 0)
            
            # Alpha Bias: If we are long, we lower our ask to sell; if short, we raise bid to buy
            inventory_skew = (pos / self.MAX_POS) * 5.0 
            
            bid = math.floor(mid - 5 - inventory_skew)
            ask = math.ceil(mid + 5 - inventory_skew)

            if abs(pos) < self.MAX_POS:
                self.throttle()
                self.send_orders([
                    OrderRequest(p.symbol, bid, Side.BUY, 5),
                    OrderRequest(p.symbol, ask, Side.SELL, 5)
                ])
    def update_flight_alpha(self):
        """Fetch real data and update our internal 'Fair Value'."""
        try:
            # Fetch last 12 hours of flight data
            data = fetch_flights(offset_minutes=-720, duration_minutes=720)
            arr = len(data.get('arrivals', []))
            dep = len(data.get('departures', []))
            
            # Simple alpha: count + expected remaining (e.g., 50 flights/hr)
            # This becomes your internal 'Price Target'
            self.flight_theo = arr + dep + 200 
            print(f"New LHR_COUNT Theo: {self.flight_theo}")
        except Exception as e:
            print(f"Alpha Fetch Error: {e}")


    def trade_flights(self, volume = 5, width = 2):
        """Place quotes based on our Alpha rather than market Mid."""
        if not self.flight_theo:
            return

        # Place a Bid slightly below our Theo and an Ask slightly above
        bid = math.floor(self.flight_theo - width)
        ask = math.ceil(self.flight_theo + width)

        # Send orders to the exchange
        self.send_orders([
            OrderRequest("LHR_COUNT", bid, Side.BUY, volume),
            OrderRequest("LHR_COUNT", ask, Side.SELL, volume)
        ])
# --- EXECUTION ---
if __name__ == "__main__":
    URL = "http://ec2-52-49-69-152.eu-west-1.compute.amazonaws.com/"
    bot = IntegratedBot(URL, "Pret", "Pret123456")
    
    try:
        bot.start()
        print("Integrated Bot Online...")
        while True:
            # Main loop for Market Making refresh
            bot.cancel_all_orders()
            bot.run_market_maker()
            time.sleep(10) # Refresh quotes every 10s
    except KeyboardInterrupt:
        bot.stop()