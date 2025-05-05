# 大数据智能降雨预测平台

## 项目概述
基于机器学习和深度学习的降雨预测系统，集成数据收集、处理、模型训练和Web应用功能。支持单次预测、批量预测和7天降雨预测，提供可视化分析界面，并具备完善的用户注册登录系统和后台管理功能。本系统采用现代化UI设计，提供直观的操作体验和安全可靠的用户验证机制。

## 技术栈
- **前端框架**: Flask + Bootstrap
- **后端框架**: Python Flask
- **机器学习和深度学习**:  XGBoost, SHAP、CNN、BiLSTM、Attention机制
- **数据处理**: Pandas, NumPy
- **数据库**: SQLite
- **可视化**: Matplotlib, Plotly
- **api调用**: Geopy与Open-Meteo API
- **部署**: 阿里云ECS
## 项目地址
- **GitHub仓库**: https://github.com/eva-conhen/rainfall-prediction-system
- **系统公网地址**: http://8.137.144.251:5001/login

## 核心功能
| 功能模块 | 对应文件 | 描述 |
|---------|---------|------|
| 主系统逻辑 | rainfall_system.py | 包含数据收集、处理、模型训练和预测的核心业务流程 |
| Web应用 | app.py | Flask实现的预测平台，提供用户注册登录、数据上传和结果展示功能 |
| 数据处理 | data_processor.py | 数据清洗、特征工程和可视化处理模块 |
| 模型训练 | model_trainer.py | 包含随机森林、XGBoost和深度学习模型的训练与优化 |
| 实时数据 | real_time_data.py | 模拟实时数据收集和存储功能 |
| 7天预测 | api_predict.html | 提供输入行政区划地址来api调用获取未来7天气象数据并且进行降雨预测功能 |
| 用户系统 | register.html/login.html | 完善的用户注册登录系统 |
| 管理后台 | admin_backend/admin.py | 用户管理和预测记录查看的后台系统 |
| 增强应用 | app_enhanced.py | 增强版应用入口，提供更多高级功能 |
| 集成应用 | app_integration.py | 集成版应用，整合所有功能模块 |
| 数据看板 | dashboard.py | 数据可视化看板功能 |

## 项目结构
```
降雨系统源码/
├── app.py                  # Flask应用入口
├── app_enhanced.py         # 增强版应用入口
├── app_integration.py      # 集成版应用
├── data_processor.py       # 数据处理模块
├── model_trainer.py        # 模型训练模块
├── rainfall_system.py      # 核心业务逻辑
├── real_time_data.py       # 实时数据处理模块
├── dashboard.py            # 数据看板功能
├── admin_backend/          # 管理后台
│   ├── __init__.py         # 包初始化
│   └── admin.py            # 管理员功能
├── static/                 # 静态资源
│   ├── js/                 # 前端交互脚本
│   └── img/                # 图片和可视化图表
├── templates/              # 模板文件
│   ├── admin/              # 管理员页面
│   ├── dashboard.html      # 数据看板
│   ├── login.html          # 登录页面
│   ├── register.html       # 注册页面
│   ├── single_predict.html # 单次预测页面
│   ├── batch_predict.html  # 批量预测页面
│   ├── data_analysis.html  # 数据分析页面
│   └── export.html         # 导出功能页面
├── data/                   # 原始数据存储
│   └── weather_data_*.csv  # 气象数据文件
├── uploads/                # 用户上传文件
├── instance/               # 实例数据
│   ├── rainfall_platform.db # 平台数据库
│   ├── rainfall_prediction.db # 预测数据库
│   └── users.db            # 用户数据库
├── rainfall_model.pkl      # 训练好的模型
├── scaler.pkl              # 数据标准化器
└── requirements.txt        # 依赖库列表
```

## 安装与使用
```bash
# 克隆仓库
git clone https://github.com/eva-conhen/rainfall-prediction-system
# 进入项目目录
cd rainfall-prediction-system
# 安装依赖
pip install -r requirements.txt
# 运行Web应用
python app.py
# 打开浏览器访问 # 打开浏览器访问 URL_ADDRESS:5000
```

## 功能详解
### 数据处理流程
1. **数据加载**：从CSV文件或HDFS加载原始气象数据
2. **特征工程**：生成时间序列特征、天气指标交叉特征
3. **数据平衡**：通过SMOTE算法处理样本不平衡问题
4. **数据可视化**：提供多种图表展示数据分布和特征重要性

### 模型服务
- **随机森林**：基础预测模型（rainfall_system.py:157-163）
- **XGBoost优化**：贝叶斯优化超参数（model_trainer.py:89-104）
- **深度学习**：LSTM神经网络模型（model_trainer.py:132-158）
- **模型集成**：多模型投票和加权集成

### Web功能
- **用户系统**：注册、登录和权限管理
- **管理后台**：用户管理和系统监控
- **预测功能**：单次预测和批量预测
- **数据分析**：数据统计和可视化分析
- **结果导出**：支持多种格式导出预测结果
- **实时监控**：系统运行状态和资源使用监控

## 部署情况
- 服务器环境：阿里云ECS

## 贡献指南
欢迎贡献代码或提出问题！请遵循以下步骤：
1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开一个 Pull Request

## 许可证
MIT License