import time
import math
import pandas as pd
from bot_template import BaseBot, OrderBook, Trade

class MicropriceSignalBot(BaseBot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.market_stats = {} 
        self.last_trade_prices = {}
        self.current_microprices = {}

    def calculate_microprice(self, ob: OrderBook):
        if not ob.buy_orders or not ob.sell_orders:
            return None
        
        best_bid = ob.buy_orders[0]
        best_ask = ob.sell_orders[0]
        
        total_vol = best_bid.volume + best_ask.volume
        if total_vol == 0: 
            return None        
        micro = (best_bid.price * best_ask.volume + best_ask.price * best_bid.volume) / total_vol
        return micro

    def on_orderbook(self, ob: OrderBook):
        symbol = ob.product
        micro = self.calculate_microprice(ob)        
        if micro:
            self.current_microprices[symbol] = micro
            if symbol not in self.market_stats:
                self.market_stats[symbol] = {"correct": 0, "total": 0}

    def on_trades(self, trade: Trade):
        symbol = trade.product
        new_price = trade.price
        last_p = self.last_trade_prices.get(symbol)
        signal_p = self.current_microprices.get(symbol)
        
        if last_p is not None and signal_p is not None:
            actual_move = new_price - last_p
            predicted_direction = signal_p - last_p
        
            if actual_move != 0:
                self.market_stats[symbol]["total"] += 1
                if (actual_move > 0 and predicted_direction > 0) or \
                   (actual_move < 0 and predicted_direction < 0):
                    self.market_stats[symbol]["correct"] += 1
        
        self.last_trade_prices[symbol] = new_price

    def run_analysis(self):    
        try:
            self.start()
            while True:
                time.sleep(15)
                self.print_report()
        except KeyboardInterrupt:
            print("shut down...")
        finally:
            self.stop()

    def print_report(self):
        print(f"{'PRODUCT':<12} | {'ACCURACY':<10} | {'SAMPLES':<8}")        
        for symbol, stats in sorted(self.market_stats.items()):
            if stats["total"] > 0:
                acc = (stats["correct"] / stats["total"]) * 100
                print(f"{symbol:<12} | {acc:>8.2f}% | {stats['total']:>8}")
            else:
                print(f"{symbol:<12} | waiting.| 0")
        print("="*50)

if __name__ == "__main__":
    URL = "http://ec2-52-19-74-159.eu-west-1.compute.amazonaws.com/"
    bot = MicropriceSignalBot(URL, "NewPret", "NewPret123456")
    bot.run_analysis()