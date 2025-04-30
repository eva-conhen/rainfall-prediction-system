from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from data_processor import DataProcessor
from model_trainer import ModelTrainer

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# 用户模型
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password, method='pbkdf2:sha256')

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# 创建数据库表
with app.app_context():
    db.create_all()

# 加载预测模型
try:
    model = joblib.load('rainfall_model.joblib')
    scaler = joblib.load('scaler.joblib') if os.path.exists('scaler.joblib') else None
    print("模型加载成功")
except Exception as e:
    print(f"模型加载失败: {str(e)}")
    model = None
    scaler = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    
    data = request.get_json()
    user = User.query.filter_by(username=data['username']).first()
    
    if user and user.check_password(data['password']):
        session['user_id'] = user.id
        session['username'] = user.username
        return jsonify({'success': True})
    
    return jsonify({'success': False, 'message': '用户名或密码错误'}), 401

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': '无效的请求数据'}), 400
        
        if User.query.filter_by(username=data['username']).first():
            return jsonify({'success': False, 'message': '用户名已存在'}), 400
        
        user = User(username=data['username'])
        user.set_password(data['password'])
        
        db.session.add(user)
        db.session.commit()
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': f'注册失败: {str(e)}'}), 500

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})

@app.route('/check_login')
def check_login():
    if 'user_id' in session:
        return jsonify({
            'logged_in': True,
            'username': session.get('username')
        })
    return jsonify({'logged_in': False})

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return jsonify({'error': '请先登录'}), 401
    
    data = request.get_json()
    try:
        # 准备输入数据
        features = [
            data['day'],
            data['pressure'],
            data['temparature'],
            data['humidity'],
            data['cloud'],
            data['sunshine'],
            data['windspeed']
        ]
        
        # 转换为numpy数组
        features_array = np.array(features).reshape(1, -1)
        
        # 应用标准化
        if scaler is not None:
            features_array = scaler.transform(features_array)
        
        # 进行预测
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0][1]
        
        return jsonify({
            'prediction': bool(prediction),
            'probability': float(probability)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/data_analysis')
def data_analysis():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    return render_template('data_analysis.html')

@app.route('/api/data_summary')
def data_summary():
    if 'user_id' not in session:
        return jsonify({'error': '请先登录'}), 401
    
    try:
        # 加载数据
        df = pd.read_csv('01train.csv')
        if 'id' in df.columns:
            df.drop(columns='id', inplace=True)
        
        # 基本统计信息
        summary = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': int(df.duplicated().sum()),
            'rainfall_distribution': df['rainfall'].value_counts().to_dict()
        }
        
        # 生成相关性热图
        plt.figure(figsize=(10, 8))
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('特征相关性热图')
        
        # 将图转换为base64字符串
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        
        correlation_plot = base64.b64encode(image_png).decode('utf-8')
        
        # 生成降雨分布图
        plt.figure(figsize=(8, 6))
        sns.countplot(x='rainfall', data=df)
        plt.title('降雨分布情况')
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        
        rainfall_plot = base64.b64encode(image_png).decode('utf-8')
        
        return jsonify({
            'summary': summary,
            'correlation_plot': correlation_plot,
            'rainfall_plot': rainfall_plot
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/feature_importance')
def feature_importance():
    if 'user_id' not in session:
        return jsonify({'error': '请先登录'}), 401
    
    try:
        # 检查特征重要性图是否存在
        if os.path.exists('xgboost_feature_importance.png'):
            with open('xgboost_feature_importance.png', 'rb') as f:
                image_data = f.read()
                feature_importance_plot = base64.b64encode(image_data).decode('utf-8')
        else:
            feature_importance_plot = None
        
        # 检查SHAP值图是否存在
        if os.path.exists('xgboost_shap_summary.png'):
            with open('xgboost_shap_summary.png', 'rb') as f:
                image_data = f.read()
                shap_plot = base64.b64encode(image_data).decode('utf-8')
        else:
            shap_plot = None
        
        # 检查参数重要性图是否存在
        if os.path.exists('param_importance.png'):
            with open('param_importance.png', 'rb') as f:
                image_data = f.read()
                param_importance_plot = base64.b64encode(image_data).decode('utf-8')
        else:
            param_importance_plot = None
        
        return jsonify({
            'feature_importance_plot': feature_importance_plot,
            'shap_plot': shap_plot,
            'param_importance_plot': param_importance_plot
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_performance')
def model_performance():
    if 'user_id' not in session:
        return jsonify({'error': '请先登录'}), 401
    
    try:
        # 检查模型比较图是否存在
        if os.path.exists('model_comparison.png'):
            with open('model_comparison.png', 'rb') as f:
                image_data = f.read()
                model_comparison_plot = base64.b64encode(image_data).decode('utf-8')
        else:
            model_comparison_plot = None
        
        # 检查ROC曲线图是否存在
        if os.path.exists('xgboost_roc_curve.png'):
            with open('xgboost_roc_curve.png', 'rb') as f:
                image_data = f.read()
                roc_curve_plot = base64.b64encode(image_data).decode('utf-8')
        else:
            roc_curve_plot = None
        
        # 检查混淆矩阵图是否存在
        if os.path.exists('xgboost_confusion_matrix.png'):
            with open('xgboost_confusion_matrix.png', 'rb') as f:
                image_data = f.read()
                confusion_matrix_plot = base64.b64encode(image_data).decode('utf-8')
        else:
            confusion_matrix_plot = None
        
        return jsonify({
            'model_comparison_plot': model_comparison_plot,
            'roc_curve_plot': roc_curve_plot,
            'confusion_matrix_plot': confusion_matrix_plot
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train_model', methods=['POST'])
def train_model():
    if 'user_id' not in session:
        return jsonify({'error': '请先登录'}), 401
    
    global model, scaler
    
    try:
        from rainfall_system import RainfallPredictionSystem
        
        # 创建系统实例
        system = RainfallPredictionSystem()
        
        # 运行数据处理和模型训练流程
        trained_model, metrics = system.run_full_pipeline('01train.csv', visualize=True, optimize=True)
        
        # 更新全局模型
        model = trained_model
        if os.path.exists('scaler.joblib'):
            scaler = joblib.load('scaler.joblib')
        
        return jsonify({
            'success': True,
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)