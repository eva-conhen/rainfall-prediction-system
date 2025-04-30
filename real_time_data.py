import os
import requests
import pandas as pd
import numpy as np
import json
import logging
import time
from datetime import datetime, timedelta
import schedule
import threading

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("real_time_data.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RealTimeData")

class RealTimeDataCollector:
    def __init__(self, api_key=None, location="beijing", data_dir="data"):
        self.api_key = api_key or os.environ.get("WEATHER_API_KEY", "demo_key")
        self.location = location
        self.data_dir = data_dir
        self.latest_data = None
        self.scheduler_thread = None
        self.is_running = False
        
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        logger.info(f"实时数据收集器初始化完成，数据将保存到 {self.data_dir}")
    
    def fetch_weather_data(self):
        """从气象API获取实时天气数据"""
        try:
            # 这里使用的是模拟API调用，实际应用中应替换为真实的API
            # 例如 OpenWeatherMap, 天气API等
            logger.info(f"正在获取 {self.location} 的实时天气数据")
            
            # 模拟API调用
            # 实际代码应类似于:
            # url = f"https://api.openweathermap.org/data/2.5/weather?q={self.location}&appid={self.api_key}"
            # response = requests.get(url)
            # data = response.json()
            
            # 模拟数据
            current_time = datetime.now()
            simulated_data = {
                "location": self.location,
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "day": current_time.timetuple().tm_yday,  # 一年中的第几天
                "pressure": np.random.normal(1013.0, 5.0),  # 气压，单位hPa
                "maxtemp": np.random.normal(25.0, 3.0),  # 最高温度
                "temparature": np.random.normal(22.0, 2.0),  # 当前温度
                "mintemp": np.random.normal(18.0, 2.0),  # 最低温度
                "dewpoint": np.random.normal(15.0, 2.0),  # 露点温度
                "humidity": np.random.normal(70.0, 10.0),  # 湿度
                "cloud": np.random.normal(50.0, 20.0),  # 云量
                "sunshine": np.random.normal(6.0, 2.0),  # 日照时间
                "winddirection": np.random.normal(180.0, 45.0),  # 风向
                "windspeed": np.random.normal(10.0, 5.0)  # 风速
            }
            
            self.latest_data = simulated_data
            logger.info(f"成功获取天气数据: {json.dumps(simulated_data, indent=2)}")
            return simulated_data
        except Exception as e:
            logger.error(f"获取天气数据失败: {str(e)}")
            return None
    
    def save_data(self, data):
        """保存天气数据到CSV文件"""
        if data is None:
            logger.warning("没有数据可保存")
            return False
        
        try:
            # 创建DataFrame
            df = pd.DataFrame([data])
            
            # 生成文件名
            today = datetime.now().strftime("%Y%m%d")
            file_path = os.path.join(self.data_dir, f"weather_data_{today}.csv")
            
            # 检查文件是否存在
            file_exists = os.path.isfile(file_path)
            
            # 保存数据
            if file_exists:
                # 追加到现有文件
                df.to_csv(file_path, mode='a', header=False, index=False)
                logger.info(f"数据已追加到 {file_path}")
            else:
                # 创建新文件
                df.to_csv(file_path, index=False)
                logger.info(f"数据已保存到新文件 {file_path}")
            
            return True
        except Exception as e:
            logger.error(f"保存数据失败: {str(e)}")
            return False
    
    def collect_and_save(self):
        """获取并保存天气数据"""
        data = self.fetch_weather_data()
        if data:
            self.save_data(data)
            return data
        return None
    
    def get_latest_data(self):
        """获取最新的天气数据"""
        if self.latest_data is None:
            self.collect_and_save()
        return self.latest_data
    
    def start_scheduler(self, interval_minutes=60):
        """启动定时任务，定期收集数据"""
        if self.is_running:
            logger.warning("调度器已在运行")
            return False
        
        def run_scheduler():
            self.is_running = True
            logger.info(f"启动数据收集调度器，间隔 {interval_minutes} 分钟")
            
            # 立即执行一次
            self.collect_and_save()
            
            # 设置定时任务
            schedule.every(interval_minutes).minutes.do(self.collect_and_save)
            
            # 运行调度器
            while self.is_running:
                schedule.run_pending()
                time.sleep(1)
        
        # 在新线程中启动调度器
        self.scheduler_thread = threading.Thread(target=run_scheduler)
        self.scheduler_thread.daemon = True  # 设为守护线程
        self.scheduler_thread.start()
        
        logger.info("数据收集调度器已在后台启动")
        return True
    
    def stop_scheduler(self):
        """停止定时任务"""
        if not self.is_running:
            logger.warning("调度器未运行")
            return False
        
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)  # 等待线程结束
        
        logger.info("数据收集调度器已停止")
        return True
    
    def update_data(self):
        """更新实时数据"""
        return self.collect_and_save()
        
    def get_historical_data(self, days=7):
        """获取历史天气数据"""
        try:
            # 计算日期范围
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # 收集所有数据文件
            all_data = []
            for i in range(days + 1):
                date = start_date + timedelta(days=i)
                date_str = date.strftime("%Y%m%d")
                file_path = os.path.join(self.data_dir, f"weather_data_{date_str}.csv")
                
                if os.path.isfile(file_path):
                    df = pd.read_csv(file_path)
                    all_data.append(df)
            
            if all_data:
                # 合并所有数据
                combined_data = pd.concat(all_data, ignore_index=True)
                logger.info(f"成功获取 {len(combined_data)} 条历史数据")
                return combined_data
            else:
                logger.warning(f"未找到过去 {days} 天的历史数据")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"获取历史数据失败: {str(e)}")
            return pd.DataFrame()

# 使用示例
if __name__ == "__main__":
    collector = RealTimeDataCollector(location="beijing")
    
    # 获取并保存一次数据
    data = collector.collect_and_save()
    print(f"获取的数据: {data}")
    
    # 启动定时任务（每10分钟收集一次数据）
    collector.start_scheduler(interval_minutes=10)
    
    # 让程序运行一段时间
    try:
        print("按 Ctrl+C 停止程序...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        collector.stop_scheduler()
        print("程序已停止")