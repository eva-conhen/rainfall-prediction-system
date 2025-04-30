from flask import Blueprint, jsonify, request
from datetime import datetime
import pandas as pd
import joblib
import numpy as np
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("admin.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("AdminBackend")

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

# 加载模型
try:
    model = joblib.load('rainfall_model.pkl')
    logger.info("成功加载模型")
except Exception as e:
    logger.error(f"加载模型失败: {str(e)}")
    model = None

@admin_bp.route('/dashboard')
def dashboard():
    """
    获取管理后台仪表盘数据
    """
    try:
        # 模拟数据 - 实际应用中应从数据库获取
        data = {
            'userCount': 100,
            'dataCount': 50,
            'predictionCount': 200,
            'visitCount': 500
        }
        return jsonify(data)
    except Exception as e:
        logger.error(f"获取仪表盘数据失败: {str(e)}")
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/users')
def get_users():
    """
    获取用户列表
    """
    try:
        # 模拟数据 - 实际应用中应从数据库获取
        users = [
            {'id': 1, 'username': 'admin', 'role': 'admin'},
            {'id': 2, 'username': 'user1', 'role': 'user'}
        ]
        return jsonify({'users': users})
    except Exception as e:
        logger.error(f"获取用户列表失败: {str(e)}")
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/data')
def get_data_files():
    """
    获取数据文件列表
    """
    try:
        # 模拟数据 - 实际应用中应从数据库获取
        files = [
            {'id': 1, 'filename': 'data1.csv', 'size': '1MB', 'upload_time': '2023-01-01'},
            {'id': 2, 'filename': 'data2.csv', 'size': '2MB', 'upload_time': '2023-01-02'}
        ]
        return jsonify({'files': files})
    except Exception as e:
        logger.error(f"获取数据文件列表失败: {str(e)}")
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/logs')
def get_system_logs():
    """
    获取系统日志
    """
    try:
        # 模拟数据 - 实际应用中应从日志文件获取
        logs = [
            {'id': 1, 'level': 'INFO', 'message': 'System started', 'time': '2023-01-01 10:00:00'},
            {'id': 2, 'level': 'ERROR', 'message': 'Failed to load model', 'time': '2023-01-01 10:01:00'}
        ]
        return jsonify({'logs': logs})
    except Exception as e:
        logger.error(f"获取系统日志失败: {str(e)}")
        return jsonify({'error': str(e)}), 500