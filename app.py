import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash, send_file
from functools import wraps
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
import joblib
import json
import logging
import psutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 导入自定义模块
from real_time_data import RealTimeDataCollector

def stop_data_collection():
    """停止实时数据收集"""
    global data_collection_active
    data_collection_active = False
    print("数据收集已停止")
from dashboard import DashboardManager
from app_integration import integrate_with_app
from threading import Thread
import time

# 数据更新控制标志
data_collection_active = True

app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///rainfall_prediction.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

db = SQLAlchemy(app)

# 数据库模型
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    prediction_type = db.Column(db.String(20), nullable=False)  # 'single' or 'batch'
    result = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# 创建数据库表
try:
    with app.app_context():
        db.create_all()
        print("数据库表创建成功")
        # 检查数据库连接是否正常
        from sqlalchemy import text
        db.session.execute(text('SELECT 1'))
        db.session.commit()
        print("数据库连接测试成功")
        
        # 检查并创建默认测试账号
        if not User.query.first():
            default_user = User(username='admin', password=generate_password_hash('admin123', method='pbkdf2:sha256'))
            db.session.add(default_user)
            db.session.commit()
            print("已创建默认测试账号: admin/admin123")
except Exception as e:
    print(f"数据库初始化失败: {str(e)}")
    # 确保应用程序不会因数据库错误而完全崩溃
    # 但会记录错误信息以便调试

# 加载模型
model_path = 'rainfall_model.pkl'
scaler_path = 'scaler.pkl'

# 如果模型不存在，则训练模型
def train_model():
    try:
        # 加载训练数据
        train_data = pd.read_csv('01train.csv')
        
        # 分离特征和目标变量
        X = train_data.drop(['rainfall', 'id'], axis=1)
        y = train_data['rainfall']
        
        # 处理缺失值
        X = X.fillna(X.mean())
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 训练随机森林模型
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        # 保存模型和标准化器
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        print("模型训练完成并保存")
        return model, scaler
    except Exception as e:
        print(f"模型训练失败: {e}")
        return None, None

# 检查模型是否存在，不存在则训练
if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    model, scaler = train_model()
else:
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    except:
        model, scaler = train_model()

# 辅助函数
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_data(data):
    # 检查日期格式并转换
    if 'date' in data.columns:
        try:
            data['date'] = pd.to_datetime(data['date'])
            data['day_of_year'] = data['date'].dt.dayofyear
            data['month'] = data['date'].dt.month
            data = data.drop(columns=['date'])
        except Exception as e:
            raise ValueError(f"日期格式转换错误: {str(e)}")
    
    # 检查天气代码并转换
    if 'weather_code' in data.columns:
        try:
            # 将天气代码转换为数值
            weather_mapping = {'N':0, 'E':1, 'W':2, 'S':3}
            data['weather_code'] = data['weather_code'].apply(lambda x: sum([weather_mapping.get(c, 0) for c in x]))
        except Exception as e:
            raise ValueError(f"天气代码转换错误: {str(e)}")
    
    # 处理缺失值
    data = data.fillna(data.mean())
    
    # 标准化特征
    data_scaled = scaler.transform(data)
    
    return data_scaled

def make_prediction(data):
    # 预处理数据
    processed_data = preprocess_data(data)
    
    # 进行预测
    predictions = model.predict(processed_data)
    probabilities = model.predict_proba(processed_data)[:, 1]
    
    results = []
    for i, pred in enumerate(predictions):
        if pred == 1:
            result = f"预测结果: 明天会下雨 (概率: {probabilities[i]:.2f})"
        else:
            result = f"预测结果: 明天不会下雨 (概率: {1-probabilities[i]:.2f})"
        results.append(result)
    
    return results

# 配置管理后台日志
admin_logger = logging.getLogger('admin')
admin_logger.setLevel(logging.INFO)
handler = logging.FileHandler('admin.log')
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
admin_logger.addHandler(handler)

# 管理员权限检查装饰器
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or session.get('username') != 'admin':
            return jsonify({'error': '权限不足'}), 403
        return f(*args, **kwargs)
    return decorated_function

# 管理后台仪表盘视图
@app.route('/admin/dashboard_view')
def admin_dashboard_view():
    """管理员仪表盘页面视图"""
    if 'user_id' not in session or session.get('username') != 'admin':
        flash('请先以管理员身份登录', 'warning')
        return redirect(url_for('login'))
    
    return render_template('admin/dashboard.html', username=session.get('username'))

# 修复管理员用户视图函数，确保路由不冲突
@app.route('/admin/users_view')
def admin_users_view():
    """管理员用户管理页面视图"""
    if 'user_id' not in session or session.get('username') != 'admin':
        flash('请先以管理员身份登录', 'warning')
        return redirect(url_for('login'))
    
    return render_template('admin/users.html', username=session.get('username'))

@app.route('/admin/system_logs_view')
def admin_system_logs_view():
    """管理员系统日志页面视图"""
    if 'user_id' not in session or session.get('username') != 'admin':
        flash('请先以管理员身份登录', 'warning')
        return redirect(url_for('login'))
    
    return render_template('admin/system_logs.html', username=session.get('username'))

@app.route('/admin/data_files_view')
def admin_data_files_view():
    """管理员数据文件页面视图"""
    if 'user_id' not in session or session.get('username') != 'admin':
        flash('请先以管理员身份登录', 'warning')
        return redirect(url_for('login'))
    
    return render_template('admin/data_files.html', username=session.get('username'))

# 注册管理后台蓝图
from admin_backend.admin import admin_bp
app.register_blueprint(admin_bp)

# 更新导航栏中的管理后台链接
@app.context_processor
def inject_admin_status():
    """向所有模板注入管理员状态"""
    is_admin = False
    if 'user_id' in session and session.get('username') == 'admin':
        is_admin = True
    return dict(is_admin=is_admin)

# 修改现有admin_dashboard函数的签名，避免与新添加的视图函数冲突
@app.route('/admin/dashboard_data')
@admin_required
def admin_dashboard_data():
    """仪表盘数据API"""
    # 获取系统统计数据
    data = {
        'user_count': User.query.count(),
        'prediction_count': Prediction.query.count(),
        'active_users': User.query.filter(User.predictions.any()).count(),
        'system_status': '正常',
        'last_24h_predictions': Prediction.query.filter(Prediction.timestamp > datetime.utcnow() - timedelta(hours=24)).count(),
        'disk_usage': psutil.disk_usage('/').percent,
        'memory_usage': psutil.virtual_memory().percent,
        'cpu_usage': psutil.cpu_percent()
    }
    admin_logger.info('管理员访问仪表盘数据')
    return jsonify(data)

@app.route('/admin/users_data')
@admin_required
def admin_users_data():
    """获取用户列表数据API"""
    # 获取用户列表
    users = User.query.all()
    user_list = [{
        'id': user.id,
        'username': user.username,
        'predictions_count': len(user.predictions),
        'last_active': max([p.timestamp for p in user.predictions], default=None) if user.predictions else None
    } for user in users]
    
    admin_logger.info('管理员访问用户列表数据')
    return jsonify({'users': user_list})

@app.route('/admin/data_files')
@admin_required
def admin_data_files():
    """获取数据文件列表API"""
    # 获取上传目录中的文件列表
    upload_dir = app.config['UPLOAD_FOLDER']
    files = []
    if os.path.exists(upload_dir):
        for filename in os.listdir(upload_dir):
            filepath = os.path.join(upload_dir, filename)
            if os.path.isfile(filepath):
                stat = os.stat(filepath)
                files.append({
                    'name': filename,
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
    
    admin_logger.info('管理员访问数据文件列表')
    return jsonify({'files': files})

@app.route('/admin/system_logs')
@admin_required
def admin_system_logs():
    """获取系统日志API"""
    # 读取系统日志文件
    log_file = 'admin.log'
    logs = []
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.split(' - ', 3)
                    if len(parts) == 4:
                        logs.append({
                            'timestamp': parts[0],
                            'level': parts[2],
                            'message': parts[3].strip()
                        })
    
    admin_logger.info('管理员访问系统日志')
    return jsonify({'logs': logs[-100:]})  # 返回最近的100条日志

@app.route('/admin/predictions')
@admin_required
def admin_predictions():
    """获取预测记录API"""
    predictions = Prediction.query.order_by(Prediction.timestamp.desc()).all()
    pred_list = [{
        'id': p.id, 
        'user_id': p.user_id,
        'timestamp': p.timestamp.isoformat(),
        'result': p.result
    } for p in predictions]
    admin_logger.info('管理员访问预测记录')
    return jsonify({'predictions': pred_list})

# 路由
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/health')
def health_check():
    try:
        # 检查数据库连接
        from sqlalchemy import text
        db.session.execute(text('SELECT 1'))
        return jsonify({'status': 'ok', 'message': '系统正常运行中'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'数据库连接错误: {str(e)}'}), 500

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            # 获取请求数据
            data = request.get_json()
            print(f"收到注册请求数据: {data}")
            
            # 检查是否收到数据
            if not data:
                print("注册失败: 未接收到数据")
                return jsonify({'success': False, 'message': '未接收到数据，请重试'})
                
            username = data.get('username')
            password = data.get('password')
            
            # 验证用户名和密码
            if not username or not password:
                print("注册失败: 用户名或密码为空")
                return jsonify({'success': False, 'message': '用户名和密码不能为空'})
            
            # 检查用户名长度
            if len(username) < 4 or len(username) > 20:
                print(f"注册失败: 用户名长度无效 {len(username)}")
                return jsonify({'success': False, 'message': '用户名长度必须在4-20个字符之间'})
                
            # 检查密码强度
            if len(password) < 8:
                print(f"注册失败: 密码长度不足 {len(password)}")
                return jsonify({'success': False, 'message': '密码长度至少需要8个字符'})
            
            # 检查用户名是否已存在
            try:
                existing_user = User.query.filter_by(username=username).first()
                if existing_user:
                    print(f"注册失败: 用户名 {username} 已存在")
                    return jsonify({'success': False, 'message': '已注册'})
            except Exception as e:
                print(f"查询用户时发生数据库错误: {str(e)}")
                return jsonify({'success': False, 'message': '数据库查询错误，请稍后再试'})
            
            # 创建新用户
            try:
                hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
                new_user = User(username=username, password=hashed_password)
                    
                # 添加到数据库并提交
                db.session.add(new_user)
                db.session.commit()
                print(f"用户 {username} 注册成功")
                return jsonify({'success': True, 'message': '注册成功！'})
            except Exception as e:
                db.session.rollback()
                error_msg = str(e)
                print(f"创建用户时发生数据库错误: {error_msg}")
                
                # 提供更具体的错误信息
                if "UNIQUE constraint failed" in error_msg:
                    return jsonify({'success': False, 'message': '已注册'})
                elif "no such table" in error_msg:
                    return jsonify({'success': False, 'message': '数据库表不存在，请联系管理员'})
                elif "database is locked" in error_msg:
                    # 添加重试机制
                    try:
                        time.sleep(1)
                        db.session.add(new_user)
                        db.session.commit()
                        print(f"重试后提交成功")
                        return jsonify({'success': True, 'message': '注册成功！'})
                    except Exception as e2:
                        db.session.rollback()
                        print(f"重试失败: {str(e2)}")
                        return jsonify({'success': False, 'message': '数据库繁忙，请稍后再试'})
                else:
                    # 记录详细错误信息
                    error_details = {
                        'error': str(e),
                        'timestamp': datetime.now().isoformat(),
                        'operation': '用户注册'
                    }
                    with open('database_errors.log', 'a') as f:
                        f.write(json.dumps(error_details) + '\n')
                    return jsonify({'success': False, 'message': '数据库错误，请稍后再试'})
        
        except Exception as e:
            error_msg = str(e)
            print(f"注册过程中发生未处理的错误: {error_msg}")
            return jsonify({'success': False, 'message': '注册过程中发生错误，请稍后再试'})
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            data = request.get_json()
            
            # 检查是否收到数据
            if not data:
                print("登录失败: 未接收到数据")
                return jsonify({'success': False, 'message': '未接收到数据，请重试'})
                
            username = data.get('username')
            password = data.get('password')
            
            # 验证用户名和密码
            if not username or not password:
                print("登录失败: 用户名或密码为空")
                return jsonify({'success': False, 'message': '用户名和密码不能为空'})
            
            try:
                user = User.query.filter_by(username=username).first()
                print(f"查询用户: {username}, 结果: {'找到' if user else '未找到'}")
                
                if user and check_password_hash(user.password, password):
                    session.clear()  # 清除之前的会话数据
                    session['user_id'] = user.id
                    session['username'] = user.username
                    print(f"用户 {username} 登录成功, 会话ID: {session.get('user_id')}")
                    return jsonify({'success': True})
                else:
                    print(f"用户 {username} 登录失败：用户名或密码错误")
                    return jsonify({'success': False, 'message': '用户名或密码错误'})
            except Exception as e:
                print(f"数据库查询错误: {str(e)}")
                return jsonify({'success': False, 'message': '数据库错误，请稍后再试'})
        except Exception as e:
            print(f"登录过程中发生错误: {str(e)}")
            return jsonify({'success': False, 'message': '登录过程中发生错误，请稍后再试'})
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/admin/login', methods=['POST'])
def admin_login():
    """管理员登录API"""
    try:
        data = request.get_json()
        
        # 检查是否收到数据
        if not data:
            return jsonify({'success': False, 'message': '未接收到数据，请重试'})
            
        username = data.get('username')
        password = data.get('password')
        
        # 验证用户名和密码
        if not username or not password:
            return jsonify({'success': False, 'message': '用户名和密码不能为空'})
        
        # 检查是否是管理员账号
        if username != 'admin':
            return jsonify({'success': False, 'message': '非管理员账号，无法登录'})
        
        try:
            user = User.query.filter_by(username=username).first()
            
            if user and check_password_hash(user.password, password):
                session['user_id'] = user.id
                session['username'] = user.username
                admin_logger.info(f'管理员 {username} 登录成功')
                return jsonify({'success': True, 'redirect': '/admin/dashboard_view'})
            else:
                admin_logger.warning(f'管理员登录失败：用户名或密码错误')
                return jsonify({'success': False, 'message': '用户名或密码错误'})
        except Exception as e:
            admin_logger.error(f'管理员登录数据库查询错误: {str(e)}')
            return jsonify({'success': False, 'message': '数据库错误，请稍后再试'})
    except Exception as e:
        admin_logger.error(f'管理员登录过程中发生错误: {str(e)}')
        return jsonify({'success': False, 'message': '登录过程中发生错误，请稍后再试'})

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    username = session['username']
    
    # 获取用户的预测历史
    predictions = Prediction.query.filter_by(user_id=user_id).order_by(Prediction.timestamp.desc()).all()
    prediction_data = [(p.prediction_type, p.result, p.timestamp.strftime('%Y-%m-%d %H:%M:%S')) for p in predictions]
    
    return render_template('dashboard.html', username=username, predictions=prediction_data)

@app.route('/single_predict', methods=['GET', 'POST'])
def single_predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        try:
            # 获取表单数据
            data = {
                'day': float(request.form.get('day', 1)),
                'pressure': float(request.form.get('pressure')),
                'maxtemp': float(request.form.get('maxtemp')),
                'temparature': float(request.form.get('temparature')),
                'mintemp': float(request.form.get('mintemp')),
                'dewpoint': float(request.form.get('dewpoint')),
                'humidity': float(request.form.get('humidity')),
                'cloud': float(request.form.get('cloud')),
                'sunshine': float(request.form.get('sunshine')),
                'winddirection': float(request.form.get('winddirection')),
                'windspeed': float(request.form.get('windspeed')),
            }
            
            # 转换为DataFrame
            df = pd.DataFrame([data])
            
            # 进行预测
            results = make_prediction(df)
            result = results[0]
            
            # 保存预测结果
            prediction = Prediction(
                user_id=session['user_id'],
                prediction_type='single',
                result=result
            )
            db.session.add(prediction)
            db.session.commit()
            
            return render_template('single_predict.html', result=result)
        
        except Exception as e:
            return render_template('single_predict.html', error=f"预测失败: {str(e)}")
    
    return render_template('single_predict.html')

@app.route('/batch_predict', methods=['GET', 'POST'])
def batch_predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        # 检查是否有文件
        if 'file' not in request.files:
            return render_template('batch_predict.html', error='没有选择文件')
        
        file = request.files['file']
        
        # 检查文件名
        if file.filename == '':
            return render_template('batch_predict.html', error='没有选择文件')
        
        if file and allowed_file(file.filename):
            try:
                # 保存文件
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                # 确保上传目录存在
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                
                # 保存文件
                file.save(filepath)
                print(f"文件 {filename} 已保存到 {filepath}")
                
                # 读取CSV文件
                try:
                    data = pd.read_csv(filepath)
                    print(f"成功读取文件 {filename}, 共 {len(data)} 行数据")
                except Exception as e:
                    error_msg = f"读取CSV文件失败: {str(e)}"
                    print(error_msg)
                    return render_template('batch_predict.html', error=error_msg)
                
                # 确保数据格式正确
                required_columns = ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp', 
                                  'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']
                
                # 检查必要列是否存在
                missing_cols = [col for col in required_columns if col not in data.columns]
                if missing_cols:
                    error_msg = f"文件缺少必要列: {', '.join(missing_cols)}"
                    print(error_msg)
                    return render_template('batch_predict.html', error=error_msg)
                
                # 移除不需要的列
                data = data.drop(columns=['id', 'rainfall'], errors='ignore')
                
                # 进行预测
                try:
                    results = make_prediction(data)
                    print(f"预测完成，共 {len(results)} 条结果")
                except Exception as e:
                    error_msg = f"预测过程中出错: {str(e)}"
                    print(error_msg)
                    return render_template('batch_predict.html', error=error_msg)
                
                # 保存预测结果
                result_summary = f"批量预测完成，共 {len(results)} 条记录"
                prediction = Prediction(
                    user_id=session['user_id'],
                    prediction_type='batch',
                    result=result_summary
                )
                
                try:
                    db.session.add(prediction)
                    db.session.commit()
                    print("预测结果已保存到数据库")
                except Exception as e:
                    db.session.rollback()
                    print(f"保存预测结果失败: {str(e)}")
                
                return render_template('batch_predict.html', results=results, filename=filename)
            
            except Exception as e:
                error_msg = f"处理文件时出错: {str(e)}"
                print(error_msg)
                return render_template('batch_predict.html', error=error_msg)
    
    return render_template('batch_predict.html')

if __name__ == '__main__':
    # 集成实时数据和仪表盘功能
    app_integration = integrate_with_app(app, db, User, Prediction)
    
    # 启动实时数据更新线程
    def run_data_updater():
        while True:
            try:
                RealTimeDataCollector().update_data()
                time.sleep(300)  # 每5分钟更新一次
            except Exception as e:
                print(f"实时数据更新失败: {str(e)}")
    
    updater_thread = Thread(target=run_data_updater, daemon=True)
    updater_thread.start()
    
    # 启动应用程序
    app.run(debug=True, port=5001)