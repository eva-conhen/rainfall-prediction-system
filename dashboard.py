import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import logging
from real_time_data import RealTimeDataCollector

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dashboard.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Dashboard")

class DashboardManager:
    def __init__(self, data_dir="data", prediction_db="instance/rainfall_prediction.db"):
        self.data_dir = data_dir
        self.prediction_db = prediction_db
        self.real_time_collector = RealTimeDataCollector(data_dir=data_dir)
        logger.info("仪表盘管理器初始化完成")
    
    def get_weather_stats(self, days=7):
        """获取天气数据统计信息"""
        try:
            # 获取历史天气数据
            df = self.real_time_collector.get_historical_data(days=days)
            
            if df.empty:
                logger.warning(f"未找到过去 {days} 天的历史数据")
                return None
            
            # 计算统计信息
            stats = {
                "data_count": len(df),
                "date_range": {
                    "start": df["timestamp"].min() if "timestamp" in df.columns else None,
                    "end": df["timestamp"].max() if "timestamp" in df.columns else None
                },
                "temperature": {
                    "avg": df["temparature"].mean() if "temparature" in df.columns else None,
                    "max": df["maxtemp"].max() if "maxtemp" in df.columns else None,
                    "min": df["mintemp"].min() if "mintemp" in df.columns else None
                },
                "humidity": {
                    "avg": df["humidity"].mean() if "humidity" in df.columns else None
                },
                "pressure": {
                    "avg": df["pressure"].mean() if "pressure" in df.columns else None
                }
            }
            
            logger.info(f"成功获取天气统计数据: {json.dumps(stats, default=str)}")
            return stats
        except Exception as e:
            logger.error(f"获取天气统计数据失败: {str(e)}")
            return None
    
    def generate_time_series_plot(self, days=7, feature="temparature"):
        """生成时间序列图表"""
        try:
            # 获取历史天气数据
            df = self.real_time_collector.get_historical_data(days=days)
            
            if df.empty:
                logger.warning(f"未找到过去 {days} 天的历史数据")
                return None
            
            # 确保时间戳列存在并格式正确
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp")
            else:
                logger.warning("数据中没有时间戳列")
                return None
            
            # 检查特征是否存在
            if feature not in df.columns:
                logger.warning(f"数据中没有 {feature} 列")
                return None
            
            # 创建时间序列图
            fig = px.line(df, x="timestamp", y=feature, 
                          title=f"过去 {days} 天的 {feature} 变化趋势",
                          labels={"timestamp": "时间", feature: feature})
            
            # 转换为JSON
            plot_json = fig.to_json()
            
            logger.info(f"成功生成 {feature} 时间序列图")
            return plot_json
        except Exception as e:
            logger.error(f"生成时间序列图失败: {str(e)}")
            return None
    
    def generate_correlation_heatmap(self, days=7):
        """生成相关性热力图"""
        try:
            # 获取历史天气数据
            df = self.real_time_collector.get_historical_data(days=days)
            
            if df.empty:
                logger.warning(f"未找到过去 {days} 天的历史数据")
                return None
            
            # 选择数值型列
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # 计算相关性
            corr_matrix = df[numeric_cols].corr()
            
            # 创建热力图
            fig = px.imshow(corr_matrix, 
                           text_auto=True,
                           title=f"过去 {days} 天天气数据相关性热力图",
                           color_continuous_scale="RdBu_r")
            
            # 转换为JSON
            plot_json = fig.to_json()
            
            logger.info("成功生成相关性热力图")
            return plot_json
        except Exception as e:
            logger.error(f"生成相关性热力图失败: {str(e)}")
            return None
    
    def get_prediction_stats(self):
        """获取预测结果统计信息"""
        try:
            # 连接到预测数据库
            import sqlite3
            conn = sqlite3.connect(self.prediction_db)
            
            # 查询预测结果
            query = """
            SELECT prediction_type, result, timestamp 
            FROM prediction 
            ORDER BY timestamp DESC
            LIMIT 1000
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                logger.warning("未找到预测记录")
                return None
            
            # 解析预测结果
            df["will_rain"] = df["result"].str.contains("会下雨")
            df["probability"] = df["result"].str.extract(r"概率: (\d+\.\d+)")[0].astype(float)
            
            # 计算统计信息
            stats = {
                "total_predictions": len(df),
                "single_predictions": len(df[df["prediction_type"] == "single"]),
                "batch_predictions": len(df[df["prediction_type"] == "batch"]),
                "rain_predictions": len(df[df["will_rain"] == True]),
                "no_rain_predictions": len(df[df["will_rain"] == False]),
                "avg_probability": df["probability"].mean(),
                "recent_predictions": df.head(10).to_dict("records")
            }
            
            logger.info("成功获取预测统计数据")
            return stats
        except Exception as e:
            logger.error(f"获取预测统计数据失败: {str(e)}")
            return None
    
    def generate_prediction_trend_plot(self, days=30):
        """生成预测趋势图"""
        try:
            # 连接到预测数据库
            import sqlite3
            conn = sqlite3.connect(self.prediction_db)
            
            # 查询预测结果
            query = """
            SELECT prediction_type, result, timestamp 
            FROM prediction 
            WHERE timestamp >= date('now', '-30 day')
            ORDER BY timestamp
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                logger.warning(f"未找到过去 {days} 天的预测记录")
                return None
            
            # 解析预测结果
            df["will_rain"] = df["result"].str.contains("会下雨")
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # 按日期分组
            df["date"] = df["timestamp"].dt.date
            daily_counts = df.groupby(["date", "will_rain"]).size().unstack(fill_value=0)
            
            if True not in daily_counts.columns:
                daily_counts[True] = 0
            if False not in daily_counts.columns:
                daily_counts[False] = 0
                
            daily_counts.columns = ["不会下雨", "会下雨"]
            daily_counts = daily_counts.reset_index()
            
            # 创建堆叠柱状图
            fig = px.bar(daily_counts, x="date", y=["会下雨", "不会下雨"],
                         title=f"过去 {days} 天的降雨预测趋势",
                         labels={"date": "日期", "value": "预测次数"},
                         barmode="stack")
            
            # 转换为JSON
            plot_json = fig.to_json()
            
            logger.info("成功生成预测趋势图")
            return plot_json
        except Exception as e:
            logger.error(f"生成预测趋势图失败: {str(e)}")
            return None
    
    def export_dashboard_data(self, format="json"):
        """导出仪表盘数据"""
        try:
            # 收集所有数据
            data = {
                "weather_stats": self.get_weather_stats(),
                "prediction_stats": self.get_prediction_stats(),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "export_format": format
            }
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dashboard_export_{timestamp}"
            
            if format.lower() == "json":
                # 导出为JSON
                file_path = os.path.join(self.data_dir, f"{filename}.json")
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=4, default=str)
            elif format.lower() == "csv":
                # 导出为CSV
                file_path = os.path.join(self.data_dir, f"{filename}.csv")
                
                # 将嵌套字典扁平化
                flat_data = {}
                for key, value in data.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, dict):
                                for sub_sub_key, sub_sub_value in sub_value.items():
                                    flat_data[f"{key}_{sub_key}_{sub_sub_key}"] = sub_sub_value
                            else:
                                flat_data[f"{key}_{sub_key}"] = sub_value
                    else:
                        flat_data[key] = value
                
                # 创建DataFrame并保存
                pd.DataFrame([flat_data]).to_csv(file_path, index=False)
            else:
                logger.error(f"不支持的导出格式: {format}")
                return None
            
            logger.info(f"仪表盘数据已导出到 {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"导出仪表盘数据失败: {str(e)}")
            return None

# 使用示例
if __name__ == "__main__":
    dashboard = DashboardManager()
    
    # 获取天气统计数据
    weather_stats = dashboard.get_weather_stats()
    print(f"天气统计数据: {weather_stats}")
    
    # 生成时间序列图
    temp_plot = dashboard.generate_time_series_plot(feature="temparature")
    print(f"温度时间序列图生成{'成功' if temp_plot else '失败'}")
    
    # 生成相关性热力图
    corr_plot = dashboard.generate_correlation_heatmap()
    print(f"相关性热力图生成{'成功' if corr_plot else '失败'}")
    
    # 获取预测统计数据
    pred_stats = dashboard.get_prediction_stats()
    print(f"预测统计数据: {pred_stats}")
    
    # 导出仪表盘数据
    export_path = dashboard.export_dashboard_data(format="json")
    print(f"仪表盘数据已导出到: {export_path}")