import time
from datetime import datetime
import threading
import websocket
import json
import requests
from collections import defaultdict
import redis
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class EnhancedBinanceTracker:
    def __init__(self):
        self.ws_url = "wss://stream.binance.com:9443/ws"
        self.prices = defaultdict(dict)
        self.lock = threading.Lock()
        self.ws = None
        self.ws_thread = None
        self.running = True
        self.last_24h_update = 0
        self.update_interval = 4  # 更新间隔（秒）
        
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
            self.redis_client.ping()
            logging.info("Redis 连接已建立")
        except Exception as e:
            logging.error(f"Redis 连接失败: {e}")
            raise

    def handle_error(self, error_type, details, e):
        """统一错误处理"""
        error_message = f"{error_type}: {details} - {e}"
        logging.error(error_message)

    def safe_redis_update(self, symbol, data_dict):
        """安全地更新 Redis 数据"""
        try:
            self.redis_client.hmset(symbol, {k: str(v) for k, v in data_dict.items()})
        except Exception as e:
            self.handle_error("Redis更新错误", f"更新{symbol}失败", e)

    def update_price(self, symbol, price):
        """更新价格数据"""
        try:
            last_update = datetime.now().isoformat()
            with self.lock:
                if symbol in self.prices:
                    # 更新本地缓存
                    self.prices[symbol].update({
                        'price': price,
                        'last_update': last_update
                    })
                    # 更新Redis
                    self.safe_redis_update(symbol, {
                        'price': price,
                        'last_update': last_update
                    })
        except Exception as e:
            self.handle_error("价格更新错误", f"更新{symbol}价格失败", e)

    def update_statistics(self, symbol, volume, price_change):
        """更新统计数据"""
        try:
            with self.lock:
                if symbol in self.prices:
                    # 更新本地缓存
                    self.prices[symbol].update({
                        'volume_24h': volume,
                        'price_change_24h': price_change
                    })
                    # 更新Redis
                    self.safe_redis_update(symbol, {
                        'volume_24h': volume,
                        'price_change_24h': price_change
                    })
        except Exception as e:
            self.handle_error("统计更新错误", f"更新{symbol}统计数据失败", e)

    def initialize_prices(self):
        """初始化所有交易对的价格数据"""
        try:
            price_response = requests.get("https://api.binance.com/api/v3/ticker/price")
            h24_response = requests.get("https://api.binance.com/api/v3/ticker/24hr")
            price_response.raise_for_status()
            h24_response.raise_for_status()
            
            price_data = price_response.json()
            h24_data = {item['symbol']: item for item in h24_response.json()}
            
            initialized_count = 0
            with self.lock:
                for item in price_data:
                    try:
                        symbol = item['symbol']
                        if symbol.endswith('USDT'):
                            h24_info = h24_data.get(symbol, {})
                            
                            self.prices[symbol] = {
                                'price': float(item['price']),
                                'volume_24h': float(h24_info.get('volume', 0)),
                                'price_change_24h': float(h24_info.get('priceChangePercent', 0)),
                                'last_update': datetime.now().isoformat()
                            }
                            
                            self.safe_redis_update(symbol, self.prices[symbol])
                            initialized_count += 1
                            
                    except Exception as e:
                        self.handle_error("初始化错误", f"初始化{symbol}失败", e)
                        continue
            
            logging.info(f"已初始化 {initialized_count} 个交易对的数据")
            
        except Exception as e:
            self.handle_error("初始化错误", "初始化价格数据失败", e)
            raise

    def update_24h_data(self):
        """更新24小时统计数据"""
        current_time = time.time()
        if current_time - self.last_24h_update < self.update_interval:
            return
            
        try:
            response = requests.get("https://api.binance.com/api/v3/ticker/24hr")
            response.raise_for_status()
            data = response.json()
            
            for item in data:
                try:
                    symbol = item['symbol']
                    if symbol in self.prices:
                        self.update_statistics(
                            symbol,
                            float(item['volume']),
                            float(item['priceChangePercent'])
                        )
                except Exception as e:
                    self.handle_error("24h更新错误", f"更新{symbol}的24h数据失败", e)
                    continue
            
            self.last_24h_update = current_time
            logging.info("24小时数据更新完成")
            
        except Exception as e:
            self.handle_error("24h更新错误", "更新24小时数据失败", e)

    def on_message(self, ws, message):
        """处理WebSocket消息"""
        try:
            data = json.loads(message)
            symbol = data['s']
            price = float(data['p'])
            self.update_price(symbol, price)
        except Exception as e:
            self.handle_error("WebSocket消息错误", "处理消息失败", e)

    def on_error(self, ws, error):
        logging.error(f"WebSocket错误: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        logging.warning(f"WebSocket连接关闭: {close_status_code} - {close_msg}")
        if self.running:
            logging.info("尝试重新连接...")
            time.sleep(5)  # 等待5秒后重连
            self.start_websocket()

    def on_open(self, ws):
        logging.info("WebSocket连接已开启")
        try:
            response = requests.get("https://api.binance.com/api/v3/exchangeInfo")
            response.raise_for_status()
            symbols = response.json()['symbols']
            
            usdt_pairs = [s['symbol'].lower() for s in symbols if s['symbol'].endswith('USDT')]
            
            subscribe_message = {
                "method": "SUBSCRIBE",
                "params": [f"{pair}@trade" for pair in usdt_pairs],
                "id": 1
            }
            
            ws.send(json.dumps(subscribe_message))
            logging.info(f"已订阅 {len(usdt_pairs)} 个交易对")
            
        except Exception as e:
            self.handle_error("WebSocket订阅错误", "订阅交易对失败", e)
            raise

    def start_websocket(self):
        """启动WebSocket连接"""
        websocket.enableTrace(True)
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()
        return self.ws_thread

    def stop(self):
        """停止跟踪器"""
        self.running = False
        if self.ws:
            self.ws.close()
        logging.info("跟踪器已停止")

    def print_summary(self):
        """打印价格摘要"""
        try:
            with self.lock:
                data = [{
                    'symbol': symbol,
                    'price': info['price'],
                    'volume_24h': info['volume_24h'],
                    'price_change_24h': info['price_change_24h']
                } for symbol, info in self.prices.items()]
                
                if data:
                    print("\n" + "="*50)
                    print(f"按24小时交易量排序的前20名 - {datetime.now()}")
                    sorted_data = sorted(data, key=lambda x: x['volume_24h'], reverse=True)[:20]
                    for item in sorted_data:
                        print(f"{item['symbol']}: "
                              f"价格={item['price']:.8f}, "
                              f"24小时交易量={item['volume_24h']:.2f}, "
                              f"24小时价格变化={item['price_change_24h']:.2f}%")
                    print("="*50)
                    print(f"总计跟踪的交易对数量: {len(data)}")
                else:
                    print("等待数据更新...")
        except Exception as e:
            self.handle_error("打印错误", "打印摘要失败", e)

    def start_price_monitoring(self):
        """开始价格监控"""
        logging.info("开始监控加密货币价格...")
        
        self.initialize_prices()
        self.start_websocket()
        
        time.sleep(5)  # 等待初始化完成
        
        try:
            while self.running:
                self.update_24h_data()
                self.print_summary()
                time.sleep(self.update_interval)
        except KeyboardInterrupt:
            logging.info("\n正在停止跟踪器...")
            self.stop()

def main():
    try:
        tracker = EnhancedBinanceTracker()
        tracker.start_price_monitoring()
    except KeyboardInterrupt:
        logging.info("\n程序被用户中断")
    except Exception as e:
        logging.error(f"程序运行出错: {e}")
    finally:
        logging.info("程序已终止")

if __name__ == "__main__":
    main()