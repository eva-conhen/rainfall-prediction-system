import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, LSTM, Bidirectional
from tensorflow.keras.layers import MultiHeadAttention, Add, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import os
import optuna
import shap

class ModelTrainer:
    def __init__(self, X_train=None, X_test=None, y_train=None, y_test=None):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = {}
        self.results = {}
        
    def set_data(self, X_train, X_test, y_train, y_test):
        """设置训练和测试数据"""
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
    def train_random_forest(self, params=None, cv=5):
        """训练随机森林模型"""
        if self.X_train is None or self.y_train is None:
            raise ValueError("请先设置训练数据")
            
        print("训练随机森林模型...")
        
        # 默认参数
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        
        # 使用提供的参数或默认参数
        if params:
            default_params.update(params)
            
        # 创建并训练模型
        rf_model = RandomForestClassifier(**default_params)
        rf_model.fit(self.X_train, self.y_train)
        
        # 交叉验证
        cv_scores = cross_val_score(rf_model, self.X_train, self.y_train, cv=cv, scoring='accuracy')
        print(f"交叉验证准确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # 保存模型
        self.models['random_forest'] = rf_model
        
        # 评估模型
        self._evaluate_model(rf_model, 'random_forest')
        
        return rf_model
    
    def train_xgboost(self, params=None, cv=5):
        """训练XGBoost模型"""
        if self.X_train is None or self.y_train is None:
            raise ValueError("请先设置训练数据")
            
        print("训练XGBoost模型...")
        
        # 默认参数
        default_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': 42
        }
        
        # 使用提供的参数或默认参数
        if params:
            default_params.update(params)
            
        # 创建并训练模型
        xgb_model = XGBClassifier(**default_params)
        xgb_model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # 交叉验证
        cv_scores = cross_val_score(xgb_model, self.X_train, self.y_train, cv=cv, scoring='accuracy')
        print(f"交叉验证准确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # 保存模型
        self.models['xgboost'] = xgb_model
        joblib.dump(xgb_model, 'xgboost_model.joblib')
        
        # 评估模型
        self._evaluate_model(xgb_model, 'xgboost')
        
        # 特征重要性
        self._plot_feature_importance(xgb_model, 'xgboost')
        
        return xgb_model
    
    def train_deep_learning(self, input_shape=None, epochs=50, batch_size=32):
        """训练深度学习模型 (CNN-BiLSTM-Attention)"""
        if self.X_train is None or self.y_train is None:
            raise ValueError("请先设置训练数据")
            
        print("训练深度学习模型...")
        
        # 准备输入数据形状
        if input_shape is None:
            # 将数据重塑为3D格式 [样本数, 时间步, 特征数]
            # 这里我们假设每个样本是一个时间点，所以时间步为1
            n_features = self.X_train.shape[1]
            input_shape = (1, n_features)
            X_train_reshaped = self.X_train.reshape(self.X_train.shape[0], 1, n_features)
            X_test_reshaped = self.X_test.reshape(self.X_test.shape[0], 1, n_features)
        else:
            # 如果提供了输入形状，则按照提供的形状重塑数据
            X_train_reshaped = self.X_train.reshape(self.X_train.shape[0], *input_shape)
            X_test_reshaped = self.X_test.reshape(self.X_test.shape[0], *input_shape)
        
        # 构建模型
        input_layer = Input(shape=input_shape)
        
        # CNN层
        x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=1)(x)
        
        # BiLSTM层
        x = Bidirectional(LSTM(units=64, return_sequences=True))(x)
        x = Bidirectional(LSTM(units=32, return_sequences=True))(x)
        
        # Attention层
        attention_out = MultiHeadAttention(
            num_heads=4, key_dim=32, dropout=0.1
        )(x, x)
        x = Add()([x, attention_out])
        x = LayerNormalization()(x)
        
        # 全连接层
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        output_layer = Dense(1, activation='sigmoid')(x)
        
        # 编译模型
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        # 回调函数
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
        ]
        
        # 训练模型
        history = model.fit(
            X_train_reshaped, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test_reshaped, self.y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # 保存模型
        self.models['deep_learning'] = model
        model.save('deep_learning_model.h5')
        
        # 评估模型
        y_pred_prob = model.predict(X_test_reshaped)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_prob)
        
        print(f"深度学习模型评估结果:")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        
        # 保存评估结果
        self.results['deep_learning'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': confusion_matrix(self.y_test, y_pred),
            'classification_report': classification_report(self.y_test, y_pred)
        }
        
        # 绘制训练历史
        self._plot_training_history(history)
        
        return model, history
    
    def optimize_xgboost(self, n_trials=100):
        """使用Optuna优化XGBoost模型超参数"""
        if self.X_train is None or self.y_train is None:
            raise ValueError("请先设置训练数据")
            
        print("优化XGBoost模型超参数...")
        
        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 1),
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'random_state': 42
            }
            
            model = XGBClassifier(**param)
            
            # 使用交叉验证评估模型
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, self.X_train, self.y_train, cv=kfold, scoring='roc_auc')
            
            return scores.mean()
        
        # 创建Optuna研究
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        print("最佳超参数:")
        print(study.best_params)
        print(f"最佳ROC AUC: {study.best_value:.4f}")
        
        # 使用最佳参数训练模型
        best_model = self.train_xgboost(params=study.best_params)
        
        # 绘制超参数重要性
        param_importance = optuna.visualization.plot_param_importances(study)
        # 保存为图片
        plt.figure(figsize=(12, 8))
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.title('超参数重要性')
        plt.tight_layout()
        plt.savefig('param_importance.png')
        plt.close()
        
        return best_model, study.best_params
    
    def _evaluate_model(self, model, model_name):
        """评估模型性能"""
        # 预测
        y_pred = model.predict(self.X_test)
        y_pred_prob = model.predict_proba(self.X_test)[:, 1]
        
        # 计算评估指标
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_prob)
        
        print(f"{model_name} 模型评估结果:")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        
        # 保存评估结果
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': confusion_matrix(self.y_test, y_pred),
            'classification_report': classification_report(self.y_test, y_pred)
        }
        
        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name} 混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.savefig(f'{model_name}_confusion_matrix.png')
        plt.close()
        
        # 绘制ROC曲线
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title(f'{model_name} ROC曲线')
        plt.legend(loc="lower right")
        plt.savefig(f'{model_name}_roc_curve.png')
        plt.close()
        
    def _plot_feature_importance(self, model, model_name):
        """绘制特征重要性"""
        if hasattr(model, 'feature_importances_'):
            # 获取特征重要性
            importances = model.feature_importances_
            
            # 获取特征名称
            if hasattr(self, 'X_train') and hasattr(self.X_train, 'columns'):
                feature_names = self.X_train.columns
            else:
                feature_names = [f'feature_{i}' for i in range(len(importances))]
                
            # 创建特征重要性DataFrame
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # 绘制特征重要性
            plt.figure(figsize=(12, 8))
            sns.barplot(x='importance', y='feature', data=feature_importance_df)
            plt.title(f'{model_name} 特征重要性')
            plt.tight_layout()
            plt.savefig(f'{model_name}_feature_importance.png')
            plt.close()
            
            print(f"特征重要性已保存为 {model_name}_feature_importance.png")
            
            # 使用SHAP值解释模型
            try:
                explainer = shap.Explainer(model)
                shap_values = explainer(self.X_test)
                
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, self.X_test, feature_names=feature_names, show=False)
                plt.title(f'{model_name} SHAP值摘要')
                plt.tight_layout()
                plt.savefig(f'{model_name}_shap_summary.png')
                plt.close()
                
                print(f"SHAP值摘要已保存为 {model_name}_shap_summary.png")
            except Exception as e:
                print(f"SHAP值计算失败: {str(e)}")
    
    def _plot_training_history(self, history):
        """绘制深度学习模型训练历史"""
        plt.figure(figsize=(12, 5))
        
        # 绘制损失
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='训练损失')
        plt.plot(history.history['val_loss'], label='验证损失')
        plt.title('模型损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend()
        
        # 绘制准确率
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='训练准确率')
        plt.plot(history.history['val_accuracy'], label='验证准确率')
        plt.title('模型准确率')
        plt.xlabel('Epoch')
        plt.ylabel('准确率')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('deep_learning_training_history.png')
        plt.close()
        
        print("训练历史已保存为 deep_learning_training_history.png")
    
    def compare_models(self):
        """比较不同模型的性能"""
        if not self.results:
            raise ValueError("请先训练模型")
            
        # 提取评估指标
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        comparison_data = []
        
        for model_name, result in self.results.items():
            model_metrics = [model_name] + [result[metric] for metric in metrics]
            comparison_data.append(model_metrics)
            
        # 创建比较DataFrame
        comparison_df = pd.DataFrame(comparison_data, columns=['模型'] + metrics)
        print("模型性能比较:")
        print(comparison_df)
        
        # 绘制比较图表
        plt.figure(figsize=(15, 10))
        
        for i, metric in enumerate(metrics):
            plt.subplot(2, 3, i+1)
            sns.barplot(x='模型', y=metric, data=comparison_df)
            plt.title(f'模型{metric}比较')
            plt.xticks(rotation=45)
            
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        plt.close()
        
        print("模型比较结果已保存为 model_comparison.png")
        
        return comparison_df
    
    def save_final_model(self, model_name=None):
        """保存最终模型"""
        if not self.models:
            raise ValueError("请先训练模型")
            
        # 如果没有指定模型名称，选择性能最好的模型
        if model_name is None:
            if not self.results:
                raise ValueError("请先评估模型")
                
            # 根据ROC AUC选择最佳模型
            best_model_name = max(self.results.items(), key=lambda x: x[1]['roc_auc'])[0]
        else:
            if model_name not in self.models:
                raise ValueError(f"模型 {model_name} 不存在")
                
            best_model_name = model_name
            
        best_model = self.models[best_model_name]
        
        # 保存模型
        if best_model_name == 'deep_learning':
            best_model.save('rainfall_model.h5')
            print(f"最终模型 (深度学习) 已保存为 rainfall_model.h5")
        else:
            joblib.dump(best_model, 'rainfall_model.joblib')
            print(f"最终模型 ({best_model_name}) 已保存为 rainfall_model.joblib")
            
        return best_model

# 使用示例
if __name__ == "__main__":
    # 加载数据
    from data_processor import DataProcessor
    
    processor = DataProcessor('01train.csv')
    processor.load_data()
    processor.clean_data()
    processor.engineer_features()
    X_train, X_test, y_train, y_test = processor.prepare_data(balance=True)
    
    # 训练模型
    trainer = ModelTrainer(X_train, X_test, y_train, y_test)
    
    # 训练随机森林
    rf_model = trainer.train_random_forest()
    
    # 训练XGBoost
    xgb_model = trainer.train_xgboost()
    
    # 优化XGBoost
    best_xgb, best_params = trainer.optimize_xgboost(n_trials=50)
    
    # 训练深度学习模型
    dl_model, history = trainer.train_deep_learning(epochs=30)
    
    # 比较模型
    comparison = trainer.compare_models()
    
    # 保存最终模型
    final_model = trainer.save_final_model()