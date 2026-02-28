import time
import math
import json
import requests
from bot_template import BaseBot, OrderBook, Trade, OrderRequest, Side

# EXACT path from your system log
FILE_PATH = "data/flight_data.json"

class IntegratedBot(BaseBot):
    def __init__(self, cmi_url, username, password):
        super().__init__(cmi_url, username, password)
        self.books = {}
        self.lhr_index_theo = 50.0  # Default neutral starting point
        self.MAX_POS = 50          # Risk limit
        self.ORDER_SIZE = 5        # Size per order

    def on_orderbook(self, ob: OrderBook):
        self.books[ob.product] = ob

    def on_trades(self, trade: Trade):
        """Mandatory implementation to prevent TypeError."""
        pass

    def update_alpha(self):
        """Reads local file and filters for operating carriers only."""
        try:
            with open(FILE_PATH, 'r') as f:
                data = json.load(f)
                
                # Filter to only count actual aircraft movements (IsOperator)
                arr_list = [f for f in data.get('arrivals', []) if f.get('codeshareStatus') == 'IsOperator']
                dep_list = [f for f in data.get('departures', []) if f.get('codeshareStatus') == 'IsOperator']
                
                arr = len(arr_list)
                dep = len(dep_list)

                # Settlement formula from Screenshot: abs(sum(arr-dep)/(arr+dep)) * 100
                if (arr + dep) > 0:
                    imbalance = (arr - dep) / (arr + dep)
                    # We scale this to a theoretical price. 
                    # If arrivals > departures, we expect the index to be higher.
                    self.lhr_index_theo = abs(imbalance) * 100
                    print(f"Alpha Update | Arr: {arr} Dep: {dep} | Theo Price: {self.lhr_index_theo:.2f}")
        except Exception as e:
            print(f"File Error: {e}")

    def trade(self):
        """Places orders around the theoretical price with inventory skew."""
        symbol = "LHR_INDEX"
        book = self.books.get(symbol)
        if not book:
            return

        # 1. Get current position to manage risk
        positions = self.get_positions()
        current_pos = positions.get(symbol, 0)

        # 2. Inventory Skew: If we are long (+), we lower our prices to sell. 
        # If we are short (-), we raise our prices to buy.
        skew = (current_pos / self.MAX_POS) * 2.0  # Adjust the '2.0' to be more/less aggressive
        
        fair_price = self.lhr_index_theo - skew
        
        # 3. Define Spread (Profit margin)
        spread = 2.0 
        bid_price = math.floor(fair_price - spread)
        ask_price = math.ceil(fair_price + spread)

        # 4. Send Orders if within position limits
        orders = []
        if current_pos < self.MAX_POS:
            orders.append(OrderRequest(symbol, bid_price, Side.BUY, self.ORDER_SIZE))
        
        if current_pos > -self.MAX_POS:
            orders.append(OrderRequest(symbol, ask_price, Side.SELL, self.ORDER_SIZE))

        if orders:
            print(f"Trading {symbol} | Pos: {current_pos} | Fair: {fair_price:.2f} | Bid: {bid_price} Ask: {ask_price}")
            self.send_orders(orders)

if __name__ == "__main__":
    EXCHANGE_URL = "http://ec2-52-49-69-152.eu-west-1.compute.amazonaws.com/"
    bot = IntegratedBot(EXCHANGE_URL, "usertest", "test123456")
    
    bot.start()
    try:
        print("Bot active. Trading LHR_INDEX...")
        while True:
            # Refresh data and trade
            bot.cancel_all_orders()
            bot.update_alpha()
            bot.trade()
            time.sleep(5) # Throttle to 5 seconds to avoid spamming the exchange
    except KeyboardInterrupt:
        bot.stop()