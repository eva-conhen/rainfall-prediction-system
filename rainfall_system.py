import os
import argparse
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging
import time
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rainfall_system.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("RainfallSystem")

class RainfallPredictionSystem:
    def __init__(self, hdfs_enabled=False, spark_enabled=False):
        self.data_processor = None
        self.model_trainer = None
        self.model = None
        self.scaler = None
        self.hdfs_enabled = hdfs_enabled
        self.spark_enabled = spark_enabled
        self.metrics_history = []
        self.start_time = time.time()
        logger.info("初始化降雨预测系统")
        
    def process_data(self, data_path, visualize=True):
        """处理数据"""
        logger.info(f"开始数据处理，数据路径: {data_path}")
        print(f"\n=== 开始数据处理 ===")
        self.data_processor = DataProcessor(data_path)
        self.data_processor.load_data()
        self.data_processor.explore_data()
        self.data_processor.clean_data()
        self.data_processor.engineer_features()
        
        if visualize:
            self.data_processor.visualize_features()
            
        X_train, X_test, y_train, y_test = self.data_processor.prepare_data(balance=True)
        logger.info(f"数据处理完成，训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")
        print(f"数据处理完成，训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def collect_data(self, sources=None):
        """从多个来源收集数据"""
        logger.info("开始数据收集")
        if sources is None:
            sources = ['local']
            
        combined_data = None
        
        for source in sources:
            if source == 'local':
                # 从本地文件加载数据
                if os.path.exists('01train.csv'):
                    df = pd.read_csv('01train.csv')
                    logger.info(f"从本地加载数据: {df.shape[0]} 条记录")
                    if combined_data is None:
                        combined_data = df
                    else:
                        combined_data = pd.concat([combined_data, df], ignore_index=True)
            elif source == 'hdfs' and self.hdfs_enabled:
                # 从HDFS加载数据 (模拟实现)
                logger.info("从HDFS加载数据")
                print("从HDFS加载数据 (功能尚未实现)")
                # 实际实现中，这里会使用pyarrow或hdfs库从HDFS读取数据
            elif source == 'api':
                # 从API获取数据 (模拟实现)
                logger.info("从API获取数据")
                print("从API获取数据 (功能尚未实现)")
                # 实际实现中，这里会使用requests库从API获取数据
        
        if combined_data is not None:
            # 保存合并后的数据
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"combined_data_{timestamp}.csv"
            combined_data.to_csv(output_path, index=False)
            logger.info(f"合并数据已保存到 {output_path}")
            return output_path
        else:
            logger.warning("未能收集到任何数据")
            return None
    
    def store_to_hdfs(self, data_path):
        """将数据存储到HDFS (模拟实现)"""
        if not self.hdfs_enabled:
            logger.warning("HDFS功能未启用")
            return False
            
        logger.info(f"将数据 {data_path} 存储到HDFS")
        print(f"将数据 {data_path} 存储到HDFS (功能尚未实现)")
        # 实际实现中，这里会使用pyarrow或hdfs库将数据写入HDFS
        return True
    
    def process_with_spark(self, data_path):
        """使用Spark处理大数据 (模拟实现)"""
        if not self.spark_enabled:
            logger.warning("Spark功能未启用")
            return data_path
            
        logger.info(f"使用Spark处理数据 {data_path}")
        print(f"使用Spark处理数据 {data_path} (功能尚未实现)")
        # 实际实现中，这里会使用PySpark处理数据
        # 例如：特征工程、数据清洗等
        
        # 处理后的数据路径
        processed_path = f"spark_processed_{os.path.basename(data_path)}"
        logger.info(f"Spark处理完成，结果保存到 {processed_path}")
        return processed_path
    
    def train_models(self, X_train, X_test, y_train, y_test, optimize=True):
        """训练模型"""
        logger.info("开始模型训练")
        print(f"\n=== 开始模型训练 ===")
        self.model_trainer = ModelTrainer(X_train, X_test, y_train, y_test)
        
        # 训练随机森林
        print("\n训练随机森林模型...")
        rf_model = self.model_trainer.train_random_forest()
        
        # 训练XGBoost
        print("\n训练XGBoost模型...")
        xgb_model = self.model_trainer.train_xgboost()
        
        if optimize:
            # 优化XGBoost
            print("\n优化XGBoost模型...")
            try:
                best_xgb, best_params = self.model_trainer.optimize_xgboost(n_trials=50)
                logger.info(f"XGBoost优化完成，最佳参数: {best_params}")
            except Exception as e:
                logger.error(f"XGBoost优化失败: {str(e)}")
        
        # 训练深度学习模型
        print("\n训练深度学习模型...")
        try:
            dl_model, history = self.model_trainer.train_deep_learning(epochs=30)
            logger.info("深度学习模型训练完成")
        except Exception as e:
            logger.error(f"深度学习模型训练失败: {str(e)}")
        
        # 比较模型
        print("\n比较模型性能...")
        comparison = self.model_trainer.compare_models()
        
        # 保存最终模型
        print("\n保存最终模型...")
        final_model = self.model_trainer.save_final_model()
        self.model = final_model
        
        # 保存标准化器
        if os.path.exists('scaler.joblib'):
            self.scaler = joblib.load('scaler.joblib')
        
        logger.info("模型训练完成")
        print("模型训练完成")
        return final_model
    
    def predict(self, features):
        """使用训练好的模型进行预测"""
        if self.model is None:
            if os.path.exists('rainfall_model.joblib'):
                self.model = joblib.load('rainfall_model.joblib')
                logger.info("从文件加载模型")
            elif os.path.exists('rainfall_model.h5'):
                import tensorflow as tf
                self.model = tf.keras.models.load_model('rainfall_model.h5')
                logger.info("从文件加载深度学习模型")
            else:
                logger.error("模型未训练，无法进行预测")
                raise ValueError("模型未训练，请先训练模型")
        
        if self.scaler is None and os.path.exists('scaler.joblib'):
            self.scaler = joblib.load('scaler.joblib')
            logger.info("从文件加载标准化器")
        
        # 转换为numpy数组
        if isinstance(features, list):
            features = np.array(features).reshape(1, -1)
        
        # 应用标准化
        if self.scaler is not None:
            features = self.scaler.transform(features)
            logger.info("应用特征标准化")
        
        # 进行预测
        start_time = time.time()
        if hasattr(self.model, 'predict_proba'):
            # 机器学习模型
            prediction = self.model.predict(features)[0]
            probability = self.model.predict_proba(features)[0][1]
        else:
            # 深度学习模型
            # 重塑为3D格式 [样本数, 时间步, 特征数]
            features_reshaped = features.reshape(features.shape[0], 1, features.shape[1])
            probability = self.model.predict(features_reshaped)[0][0]
            prediction = 1 if probability > 0.5 else 0
        
        prediction_time = time.time() - start_time
        logger.info(f"预测完成，耗时: {prediction_time:.4f}秒，结果: {prediction}, 概率: {probability:.4f}")
        
        return prediction, probability
    
    def evaluate_model(self, X_test, y_test):
        """评估模型性能"""
        if self.model is None:
            logger.error("模型未训练，无法评估")
            raise ValueError("模型未训练，请先训练模型")
        
        logger.info("开始评估模型性能")
        # 进行预测
        if hasattr(self.model, 'predict_proba'):
            # 机器学习模型
            y_pred = self.model.predict(X_test)
            y_pred_prob = self.model.predict_proba(X_test)[:, 1]
        else:
            # 深度学习模型
            # 重塑为3D格式 [样本数, 时间步, 特征数]
            X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
            y_pred_prob = self.model.predict(X_test_reshaped).flatten()
            y_pred = (y_pred_prob > 0.5).astype(int)
        
        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_prob)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
        
        # 记录评估结果
        self.metrics_history.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'metrics': metrics
        })
        
        logger.info(f"模型评估结果: 准确率={accuracy:.4f}, 精确率={precision:.4f}, 召回率={recall:.4f}, F1={f1:.4f}, ROC AUC={roc_auc:.4f}")
        print(f"模型评估结果:")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        
        return metrics
    
    def run_full_pipeline(self, data_path=None, visualize=True, optimize=True, use_big_data=False):
        """运行完整的数据处理和模型训练流程"""
        self.start_time = time.time()
        logger.info("开始运行完整流程")
        print(f"\n=== 开始降雨预测系统完整流程 ===")
        
        # 大数据处理流程
        if use_big_data:
            # 1. 数据收集
            logger.info("启动大数据处理流程")
            collected_data_path = self.collect_data(['local', 'hdfs', 'api'])
            if collected_data_path is None:
                logger.error("数据收集失败，使用默认数据路径")
                collected_data_path = data_path
            
            # 2. 存储到HDFS
            if self.hdfs_enabled:
                self.store_to_hdfs(collected_data_path)
            
            # 3. Spark处理
            if self.spark_enabled:
                processed_data_path = self.process_with_spark(collected_data_path)
                data_path = processed_data_path
        
        if data_path is None:
            data_path = '01train.csv'
            
        logger.info(f"使用数据路径: {data_path}")
        print(f"数据路径: {data_path}")
        
        # 处理数据
        X_train, X_test, y_train, y_test = self.process_data(data_path, visualize)
        
        # 训练模型
        model = self.train_models(X_train, X_test, y_train, y_test, optimize)
        
        # 评估模型
        print(f"\n=== 最终模型评估 ===")
        metrics = self.evaluate_model(X_test, y_test)
        
        # 计算总耗时
        total_time = time.time() - self.start_time
        logger.info(f"完整流程完成，总耗时: {total_time:.2f}秒")
        print(f"\n=== 降雨预测系统流程完成，总耗时: {total_time:.2f}秒 ===")
        return model, metrics

def main():
    parser = argparse.ArgumentParser(description='降雨预测系统')
    parser.add_argument('--data', type=str, default='01train.csv', help='训练数据路径')
    parser.add_argument('--no-visualize', action='store_false', dest='visualize', help='不生成可视化图表')
    parser.add_argument('--no-optimize', action='store_false', dest='optimize', help='不进行超参数优化')
    parser.add_argument('--predict', action='store_true', help='进行预测模式')
    parser.add_argument('--features', nargs='+', type=float, help='用于预测的特征值')
    parser.add_argument('--big-data', action='store_true', help='启用大数据处理流程')
    parser.add_argument('--hdfs', action='store_true', help='启用HDFS存储')
    parser.add_argument('--spark', action='store_true', help='启用Spark处理')
    
    args = parser.parse_args()
    
    system = RainfallPredictionSystem(hdfs_enabled=args.hdfs, spark_enabled=args.spark)
    
    if args.predict:
        if args.features is None:
            print("请提供特征值进行预测")
            return
        
        prediction, probability = system.predict(args.features)
        print(f"预测结果: {'有降雨' if prediction == 1 else '无降雨'}")
        print(f"降雨概率: {probability:.4f}")
    else:
        system.run_full_pipeline(args.data, args.visualize, args.optimize, use_big_data=args.big_data)

if __name__ == "__main__":
    main()