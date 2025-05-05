import requests
import json
from datetime import datetime
import pandas as pd
import logging
import time
try:
    from geopy.geocoders import Nominatim
except ImportError:
    Nominatim = None

class WeatherAPIClient:
    def __init__(self):
        self.base_url = "https://api.open-meteo.com/v1/forecast"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='weather_api.log'
        )
        self.logger = logging.getLogger(__name__)
        
    def get_weather_data(self, latitude=52.52, longitude=13.41):
        """
        获取天气预报数据
        :param latitude: 纬度
        :param longitude: 经度
        :return: 返回天气数据字典
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "daily": "temperature_2m_max,sunshine_duration,temperature_2m_mean,cloud_cover_mean,dew_point_2m_mean,relative_humidity_2m_mean,wind_speed_10m_mean,winddirection_10m_dominant,temperature_2m_min,pressure_msl_mean,weather_code"
        }
        
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                self.logger.error(f"获取天气数据时出错(尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    self.logger.info(f"将在{retry_delay}秒后重试...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    self.logger.error("达到最大重试次数，放弃获取天气数据")
                    return None
        
    def save_to_file(self, data, filename=None, file_format='json'):
        """
        将天气数据保存到文件
        :param data: 天气数据
        :param filename: 文件名，默认为当前日期
        :param file_format: 文件格式，支持'json'或'xlsx'
        """
        if not filename:
            filename = f"weather_data_{datetime.now().strftime('%Y%m%d')}"
            
        try:
            if file_format == 'json':
                with open(f"data/{filename}.json", "w") as f:
                    json.dump(data, f, indent=4)
                self.logger.info(f"数据已保存到JSON文件: data/{filename}.json")
            elif file_format == 'xlsx':
                df = self.json_to_dataframe(data)
                df.to_excel(f"data/{filename}.xlsx", index=False)
                self.logger.info(f"数据已保存到Excel文件: data/{filename}.xlsx")
            else:
                raise ValueError(f"不支持的文件格式: {file_format}")
        except Exception as e:
            self.logger.error(f"保存文件时出错: {e}")
            
    def json_to_dataframe(self, json_data):
        """
        将API返回的JSON数据转换为DataFrame
        :param json_data: API返回的JSON数据
        :return: 转换后的DataFrame
        """
        daily_data = json_data['daily']
        # 先转换为Series再访问dt属性，或直接使用pandas的dayofyear函数
        time_series = pd.Series(daily_data['time'])
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(daily_data['time']),
            'day': pd.to_datetime(time_series).dt.dayofyear,
            'pressure': daily_data['pressure_msl_mean'],
            'maxtemp': daily_data['temperature_2m_max'],
            'temparature': daily_data['temperature_2m_mean'],
            'mintemp': daily_data['temperature_2m_min'],
            'dewpoint': daily_data['dew_point_2m_mean'],
            'humidity': daily_data['relative_humidity_2m_mean'],
            'cloud': daily_data['cloud_cover_mean'],
            'sunshine': [round(duration / 3600, 2) for duration in daily_data['sunshine_duration']],
            'winddirection': daily_data['winddirection_10m_dominant'],
            'windspeed': daily_data['wind_speed_10m_mean']
        })
        return df
        
    def get_formatted_weather_data(self, location):
        """
        获取指定位置的天气数据并返回格式化DataFrame
        :param location: 城市名称或地址
        :return: 格式化后的DataFrame
        """
        if Nominatim is None:
            print(f"警告: geopy模块未安装，请手动输入{location}的经纬度坐标")
            return None
            
        geolocator = Nominatim(user_agent="geoapi")
        loc = geolocator.geocode(location)
        if not loc:
            print(f"无法找到位置: {location}")
            return None
            
        latitude, longitude = loc.latitude, loc.longitude
        print(f"{location}的坐标: ({latitude}, {longitude})")
        
        weather_data = self.get_weather_data(latitude, longitude)
        if weather_data:
            df = self.json_to_dataframe(weather_data)
            print("\n天气数据表格:")
            try:
                print(df.to_markdown(index=False))
            except ImportError:
                print("缺少tabulate库，使用默认显示方式")
                print(df)
            return df
        return None

if __name__ == "__main__":
    client = WeatherAPIClient()
    location = input("请输入城市名称或地址: ")
    weather_df = client.get_formatted_weather_data(location)
    if weather_df is not None:
        client.save_to_file(weather_df, file_format='xlsx')

