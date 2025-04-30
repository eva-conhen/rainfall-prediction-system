import os
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
import time
import threading
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash, send_file
from werkzeug.utils import secure_filename
import sqlite3
import csv
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# 导入自定义模块
from real_time_data import RealTimeDataCollector
from dashboard import DashboardManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AppIntegration")

class AppIntegration:
    def __init__(self, app, db, User, Prediction):
        self.app = app
        self.db = db
        self.User = User
        self.Prediction = Prediction
        self.data_dir = "data"
        
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 初始化实时数据收集器
        self.real_time_collector = RealTimeDataCollector(data_dir=self.data_dir)
        
        # 初始化仪表盘管理器
        self.dashboard_manager = DashboardManager(data_dir=self.data_dir)
        
        # 启动实时数据收集
        self.start_real_time_data_collection()
        
        # 注册路由
        self.register_routes()
        
        logger.info("应用程序集成模块初始化完成")
    
    def start_real_time_data_collection(self, interval_minutes=30):
        """启动实时数据收集"""
        try:
            # 启动定时任务
            success = self.real_time_collector.start_scheduler(interval_minutes=interval_minutes)
            if success:
                logger.info(f"实时数据收集已启动，间隔 {interval_minutes} 分钟")
            else:
                logger.warning("实时数据收集启动失败")
        except Exception as e:
            logger.error(f"启动实时数据收集失败: {str(e)}")
    
    def register_routes(self):
        """注册路由"""
        # 仪表盘路由
        self.app.add_url_rule('/dashboard/data', 'dashboard_data', self.dashboard_data, methods=['GET'])
        self.app.add_url_rule('/dashboard/weather_plot', 'weather_plot', self.weather_plot, methods=['GET'])
        self.app.add_url_rule('/dashboard/prediction_plot', 'prediction_plot', self.prediction_plot, methods=['GET'])
        self.app.add_url_rule('/dashboard/correlation_plot', 'correlation_plot', self.correlation_plot, methods=['GET'])
        
        # 用户管理路由
        self.app.add_url_rule('/admin/users', 'admin_users', self.admin_users, methods=['GET'])
        self.app.add_url_rule('/admin/add_user', 'add_user', self.add_user, methods=['POST'])
        self.app.add_url_rule('/admin/update_user/<int:user_id>', 'update_user', self.update_user, methods=['POST'])
        self.app.add_url_rule('/admin/delete_user/<int:user_id>', 'delete_user', self.delete_user, methods=['POST'])
        
        # 预测结果导出路由
        self.app.add_url_rule('/export/predictions', 'export_predictions', self.export_predictions, methods=['GET'])
        self.app.add_url_rule('/export/dashboard', 'export_dashboard', self.export_dashboard, methods=['GET'])
        
        logger.info("路由注册完成")
    
    # 仪表盘相关路由处理函数
    def dashboard_data(self):
        """获取仪表盘数据"""
        try:
            # 检查用户是否登录
            if 'user_id' not in session:
                return jsonify({'success': False, 'message': '请先登录'}), 401
            
            # 获取天气统计数据
            weather_stats = self.dashboard_manager.get_weather_stats()
            
            # 获取预测统计数据
            prediction_stats = self.dashboard_manager.get_prediction_stats()
            
            # 组合数据
            data = {
                'weather_stats': weather_stats,
                'prediction_stats': prediction_stats,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return jsonify({'success': True, 'data': data})
        except Exception as e:
            logger.error(f"获取仪表盘数据失败: {str(e)}")
            return jsonify({'success': False, 'message': f'获取仪表盘数据失败: {str(e)}'}), 500
    
    def weather_plot(self):
        """获取天气时间序列图"""
        try:
            # 检查用户是否登录
            if 'user_id' not in session:
                return jsonify({'success': False, 'message': '请先登录'}), 401
            
            # 获取参数
            feature = request.args.get('feature', 'temparature')
            days = int(request.args.get('days', 7))
            
            # 生成图表
            plot_json = self.dashboard_manager.generate_time_series_plot(days=days, feature=feature)
            
            if plot_json:
                return jsonify({'success': True, 'plot': json.loads(plot_json)})
            else:
                return jsonify({'success': False, 'message': '生成图表失败'}), 500
        except Exception as e:
            logger.error(f"获取天气图表失败: {str(e)}")
            return jsonify({'success': False, 'message': f'获取天气图表失败: {str(e)}'}), 500
    
    def prediction_plot(self):
        """获取预测趋势图"""
        try:
            # 检查用户是否登录
            if 'user_id' not in session:
                return jsonify({'success': False, 'message': '请先登录'}), 401
            
            # 获取参数
            days = int(request.args.get('days', 30))
            
            # 生成图表
            plot_json = self.dashboard_manager.generate_prediction_trend_plot(days=days)
            
            if plot_json:
                return jsonify({'success': True, 'plot': json.loads(plot_json)})
            else:
                return jsonify({'success': False, 'message': '生成图表失败'}), 500
        except Exception as e:
            logger.error(f"获取预测趋势图失败: {str(e)}")
            return jsonify({'success': False, 'message': f'获取预测趋势图失败: {str(e)}'}), 500
    
    def correlation_plot(self):
        """获取相关性热力图"""
        try:
            # 检查用户是否登录
            if 'user_id' not in session:
                return jsonify({'success': False, 'message': '请先登录'}), 401
            
            # 获取参数
            days = int(request.args.get('days', 7))
            
            # 生成图表
            plot_json = self.dashboard_manager.generate_correlation_heatmap(days=days)
            
            if plot_json:
                return jsonify({'success': True, 'plot': json.loads(plot_json)})
            else:
                return jsonify({'success': False, 'message': '生成图表失败'}), 500
        except Exception as e:
            logger.error(f"获取相关性热力图失败: {str(e)}")
            return jsonify({'success': False, 'message': f'获取相关性热力图失败: {str(e)}'}), 500
    
    # 用户管理相关路由处理函数
    def admin_users(self):
        """获取所有用户"""
        try:
            # 检查用户是否登录且是管理员
            if 'user_id' not in session:
                return jsonify({'success': False, 'message': '请先登录'}), 401
            
            # 检查是否是管理员（假设ID为1的用户是管理员）
            if session['user_id'] != 1:
                return jsonify({'success': False, 'message': '权限不足'}), 403
            
            # 查询所有用户
            users = self.User.query.all()
            
            # 转换为字典列表
            user_list = [{
                'id': user.id,
                'username': user.username,
                'is_admin': user.id == 1  # 假设ID为1的用户是管理员
            } for user in users]
            
            return jsonify({'success': True, 'users': user_list})
        except Exception as e:
            logger.error(f"获取用户列表失败: {str(e)}")
            return jsonify({'success': False, 'message': f'获取用户列表失败: {str(e)}'}), 500
    
    def add_user(self):
        """添加新用户"""
        try:
            # 检查用户是否登录且是管理员
            if 'user_id' not in session:
                return jsonify({'success': False, 'message': '请先登录'}), 401
            
            # 检查是否是管理员
            if session['user_id'] != 1:
                return jsonify({'success': False, 'message': '权限不足'}), 403
            
            # 获取请求数据
            data = request.get_json()
            username = data.get('username')
            password = data.get('password')
            
            # 验证数据
            if not username or not password:
                return jsonify({'success': False, 'message': '用户名和密码不能为空'}), 400
            
            # 检查用户名是否已存在
            if self.User.query.filter_by(username=username).first():
                return jsonify({'success': False, 'message': '用户名已存在'}), 400
            
            # 创建新用户
            from werkzeug.security import generate_password_hash
            new_user = self.User(username=username, password=generate_password_hash(password, method='pbkdf2:sha256'))
            
            # 保存到数据库
            self.db.session.add(new_user)
            self.db.session.commit()
            
            logger.info(f"管理员创建了新用户: {username}")
            return jsonify({'success': True, 'message': '用户创建成功', 'user_id': new_user.id})
        except Exception as e:
            logger.error(f"创建用户失败: {str(e)}")
            return jsonify({'success': False, 'message': f'创建用户失败: {str(e)}'}), 500
    
    def update_user(self, user_id):
        """更新用户信息"""
        try:
            # 检查用户是否登录且是管理员
            if 'user_id' not in session:
                return jsonify({'success': False, 'message': '请先登录'}), 401
            
            # 检查是否是管理员或本人
            if session['user_id'] != 1 and session['user_id'] != user_id:
                return jsonify({'success': False, 'message': '权限不足'}), 403
            
            # 获取请求数据
            data = request.get_json()
            password = data.get('password')
            
            # 查找用户
            user = self.User.query.get(user_id)
            if not user:
                return jsonify({'success': False, 'message': '用户不存在'}), 404
            
            # 更新密码
            if password:
                from werkzeug.security import generate_password_hash
                user.password = generate_password_hash(password, method='pbkdf2:sha256')
            
            # 保存到数据库
            self.db.session.commit()
            
            logger.info(f"用户 {user.username} 的信息已更新")
            return jsonify({'success': True, 'message': '用户信息更新成功'})
        except Exception as e:
            logger.error(f"更新用户信息失败: {str(e)}")
            return jsonify({'success': False, 'message': f'更新用户信息失败: {str(e)}'}), 500
    
    def delete_user(self, user_id):
        """删除用户"""
        try:
            # 检查用户是否登录且是管理员
            if 'user_id' not in session:
                return jsonify({'success': False, 'message': '请先登录'}), 401
            
            # 检查是否是管理员
            if session['user_id'] != 1:
                return jsonify({'success': False, 'message': '权限不足'}), 403
            
            # 不能删除管理员账号
            if user_id == 1:
                return jsonify({'success': False, 'message': '不能删除管理员账号'}), 400
            
            # 查找用户
            user = self.User.query.get(user_id)
            if not user:
                return jsonify({'success': False, 'message': '用户不存在'}), 404
            
            # 删除用户
            username = user.username
            self.db.session.delete(user)
            self.db.session.commit()
            
            logger.info(f"管理员删除了用户: {username}")
            return jsonify({'success': True, 'message': '用户删除成功'})
        except Exception as e:
            logger.error(f"删除用户失败: {str(e)}")
            return jsonify({'success': False, 'message': f'删除用户失败: {str(e)}'}), 500
    
    # 预测结果导出相关路由处理函数
    def export_predictions(self):
        """导出预测结果"""
        try:
            # 检查用户是否登录
            if 'user_id' not in session:
                return jsonify({'success': False, 'message': '请先登录'}), 401
            
            # 获取参数
            format_type = request.args.get('format', 'csv')
            days = int(request.args.get('days', 30))
            
            # 连接到数据库
            conn = sqlite3.connect('instance/rainfall_prediction.db')
            
            # 查询预测结果
            query = f"""
            SELECT p.id, u.username, p.prediction_type, p.result, p.timestamp 
            FROM prediction p
            JOIN user u ON p.user_id = u.id
            WHERE p.timestamp >= date('now', '-{days} day')
            ORDER BY p.timestamp DESC
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                return jsonify({'success': False, 'message': '没有可导出的预测结果'}), 404
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prediction_export_{timestamp}"
            
            if format_type.lower() == 'csv':
                # 导出为CSV
                output = io.StringIO()
                df.to_csv(output, index=False)
                output.seek(0)
                
                return send_file(
                    io.BytesIO(output.getvalue().encode('utf-8')),
                    mimetype='text/csv',
                    as_attachment=True,
                    download_name=f"{filename}.csv"
                )
            
            elif format_type.lower() == 'json':
                # 导出为JSON
                output = io.StringIO()
                df.to_json(output, orient='records', force_ascii=False, indent=4)
                output.seek(0)
                
                return send_file(
                    io.BytesIO(output.getvalue().encode('utf-8')),
                    mimetype='application/json',
                    as_attachment=True,
                    download_name=f"{filename}.json"
                )
            
            elif format_type.lower() == 'pdf':
                # 导出为PDF
                # 创建一个临时文件
                temp_file = os.path.join(self.data_dir, f"{filename}.pdf")
                
                # 创建PDF文件
                with PdfPages(temp_file) as pdf:
                    # 创建表格图
                    plt.figure(figsize=(12, 8))
                    plt.axis('off')
                    plt.title('降雨预测结果导出')
                    
                    # 限制显示的行数
                    display_df = df.head(50)  # 只显示前50行
                    
                    # 创建表格
                    table = plt.table(
                        cellText=display_df.values,
                        colLabels=display_df.columns,
                        loc='center',
                        cellLoc='center',
                        colWidths=[0.1, 0.2, 0.15, 0.35, 0.2]
                    )
                    
                    # 调整表格样式
                    table.auto_set_font_size(False)
                    table.set_fontsize(8)
                    table.scale(1, 1.5)
                    
                    # 保存页面
                    pdf.savefig()
                    plt.close()
                    
                    # 添加统计信息页面
                    plt.figure(figsize=(12, 8))
                    plt.axis('off')
                    plt.title('预测结果统计')
                    
                    # 计算统计信息
                    stats_text = [
                        f"总预测次数: {len(df)}",
                        f"单次预测: {len(df[df['prediction_type'] == 'single'])}",
                        f"批量预测: {len(df[df['prediction_type'] == 'batch'])}",
                        f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        f"导出用户: {self.User.query.get(session['user_id']).username}"
                    ]
                    
                    # 添加统计文本
                    plt.text(0.1, 0.5, '\n'.join(stats_text), fontsize=12)
                    
                    # 保存页面
                    pdf.savefig()
                    plt.close()
                
                # 发送文件
                return send_file(
                    temp_file,
                    mimetype='application/pdf',
                    as_attachment=True,
                    download_name=f"{filename}.pdf"
                )
            
            else:
                return jsonify({'success': False, 'message': f'不支持的导出格式: {format_type}'}), 400
        
        except Exception as e:
            logger.error(f"导出预测结果失败: {str(e)}")
            return jsonify({'success': False, 'message': f'导出预测结果失败: {str(e)}'}), 500
    
    def export_dashboard(self):
        """导出仪表盘数据"""
        try:
            # 检查用户是否登录
            if 'user_id' not in session:
                return jsonify({'success': False, 'message': '请先登录'}), 401
            
            # 获取参数
            format_type = request.args.get('format', 'json')
            
            # 导出数据
            file_path = self.dashboard_manager.export_dashboard_data(format=format_type)
            
            if not file_path or not os.path.exists(file_path):
                return jsonify({'success': False, 'message': '导出仪表盘数据失败'}), 500
            
            # 发送文件
            return send_file(
                file_path,
                as_attachment=True,
                download_name=os.path.basename(file_path)
            )
        
        except Exception as e:
            logger.error(f"导出仪表盘数据失败: {str(e)}")
            return jsonify({'success': False, 'message': f'导出仪表盘数据失败: {str(e)}'}), 500

# 集成到主应用程序的函数
def integrate_with_app(app, db, User, Prediction):
    """将实时数据和仪表盘功能集成到主应用程序"""
    try:
        # 创建集成对象
        integration = AppIntegration(app, db, User, Prediction)
        
        # 添加模板上下文处理器
        @app.context_processor
        def inject_real_time_data():
            """向模板注入实时数据"""
            def get_latest_weather():
                return integration.real_time_collector.get_latest_data()
            
            return dict(get_latest_weather=get_latest_weather)
        
        logger.info("实时数据和仪表盘功能已成功集成到主应用程序")
        return integration
    except Exception as e:
        logger.error(f"集成到主应用程序失败: {str(e)}")
        return None