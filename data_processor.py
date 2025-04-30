import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os

class DataProcessor:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        
    def load_data(self, data_path=None):
        """加载数据集"""
        if data_path:
            self.data_path = data_path
        
        if self.data_path is None:
            raise ValueError("请提供数据路径")
            
        self.df = pd.read_csv(self.data_path)
        if 'id' in self.df.columns:
            self.df.drop(columns='id', inplace=True)
        print(f"数据加载完成，共 {self.df.shape[0]} 条记录，{self.df.shape[1]} 个特征")
        return self.df
    
    def explore_data(self):
        """数据探索分析"""
        if self.df is None:
            raise ValueError("请先加载数据")
            
        print("数据基本信息：")
        print(self.df.columns)  # 检查列名
        print(self.df.shape)    # 检查数据形状
        print(self.df.info())   # 检查数据类型
        print(self.df.describe()) # 检查数据统计信息
        print(f"重复值数量: {self.df.duplicated().sum()}") # 检查重复值
        print(f"缺失值情况:\n{self.df.isnull().sum()}") # 检查缺失值
        
        # 目标变量分布
        if 'rainfall' in self.df.columns:
            plt.figure(figsize=(10, 6))
            sns.countplot(x='rainfall', data=self.df)
            plt.title('降雨分布情况')
            plt.savefig('rainfall_distribution.png')
            plt.close()
            
            # 计算类别比例
            rainfall_counts = self.df['rainfall'].value_counts()
            print(f"降雨分布: \n{rainfall_counts}")
            print(f"降雨比例: \n{rainfall_counts / len(self.df)}")
        
        return self.df
    
    def clean_data(self):
        """数据清洗"""
        if self.df is None:
            raise ValueError("请先加载数据")
            
        # 处理缺失值
        self.df = self.df.fillna(self.df.mean())
        
        # 处理异常值
        def handle_outliers(series):
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + 1.5 * IQR
            lower_bound = Q1 - 1.5 * IQR
            return series.clip(lower_bound, upper_bound)
        
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != 'rainfall':  # 不处理目标变量
                self.df[col] = handle_outliers(self.df[col])
                
        print("数据清洗完成")
        return self.df
    
    def engineer_features(self):
        """特征工程"""
        if self.df is None:
            raise ValueError("请先加载数据")
            
        # 计算复合特征
        self.df['hci'] = self.df['humidity'] * self.df['cloud']  # 湿度云量指数
        self.df['hsi'] = self.df['humidity'] * self.df['sunshine']  # 湿度日照指数
        
        # 添加温湿指数
        if 'temparature' in self.df.columns and 'humidity' in self.df.columns:
            self.df['thi'] = 0.8 * self.df['temparature'] + self.df['humidity'] / 500 * \
                           (0.99 * self.df['temparature'] - 14.3) + 46.4  # 温湿指数
        
        # 添加季节性特征
        if 'day' in self.df.columns:
            self.df['season'] = pd.cut(self.df['day'] % 365, bins=[0, 90, 180, 270, 365], 
                                     labels=['spring', 'summer', 'fall', 'winter'])
            self.df = pd.get_dummies(self.df, columns=['season'])
            
        print("特征工程完成，新特征列表：")
        print(self.df.columns)
        return self.df
    
    def prepare_data(self, test_size=0.2, random_state=42, balance=True):
        """准备训练和测试数据"""
        if self.df is None:
            raise ValueError("请先加载数据")
            
        if 'rainfall' not in self.df.columns:
            raise ValueError("数据中缺少目标变量'rainfall'")
            
        # 分离特征和目标变量
        self.y = self.df['rainfall']
        self.X = self.df.drop(columns=['rainfall'])
        
        # 划分训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        # 特征标准化
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        # 保存标准化器
        joblib.dump(self.scaler, 'scaler.joblib')
        
        # 处理类别不平衡
        if balance:
            smote = SMOTE(random_state=random_state)
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
            print(f"应用SMOTE后的训练集形状: {self.X_train.shape}")
            
        print("数据准备完成")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def visualize_features(self, save_dir='./visualizations'):
        """可视化特征分析"""
        if self.df is None:
            raise ValueError("请先加载数据")
            
        # 创建保存目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 相关性热图
        plt.figure(figsize=(12, 10))
        corr = self.df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('特征相关性热图')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/correlation_heatmap.png')
        plt.close()
        
        # 特征与目标变量的关系
        if 'rainfall' in self.df.columns:
            features = [col for col in self.df.columns if col != 'rainfall']
            for feature in features[:10]:  # 只展示前10个特征，避免图太多
                plt.figure(figsize=(10, 6))
                sns.boxplot(x='rainfall', y=feature, data=self.df)
                plt.title(f'{feature} vs 降雨')
                plt.savefig(f'{save_dir}/{feature}_vs_rainfall.png')
                plt.close()
                
        print(f"特征可视化完成，图表保存在 {save_dir} 目录")
        
    def save_processed_data(self, output_path='processed_data.csv'):
        """保存处理后的数据"""
        if self.df is None:
            raise ValueError("请先加载数据")
            
        self.df.to_csv(output_path, index=False)
        print(f"处理后的数据已保存到 {output_path}")

# 使用示例
if __name__ == "__main__":
    processor = DataProcessor('01train.csv')
    processor.load_data()
    processor.explore_data()
    processor.clean_data()
    processor.engineer_features()
    X_train, X_test, y_train, y_test = processor.prepare_data(balance=True)
    processor.visualize_features()
    processor.save_processed_data()