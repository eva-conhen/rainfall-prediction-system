/**
 * 降雨预测系统 - 仪表盘交互脚本
 * 用于处理仪表盘页面的数据加载、图表渲染和交互功能
 */

// 页面加载完成后初始化仪表盘
document.addEventListener('DOMContentLoaded', function() {
    // 检查用户登录状态
    checkLoginStatus();
    
    // 初始化仪表盘
    initDashboard();
    
    // 设置自动刷新（每5分钟刷新一次）
    setInterval(refreshDashboard, 5 * 60 * 1000);
    
    // 监听天气特征选择变化
    document.getElementById('weather-feature').addEventListener('change', function() {
        fetchWeatherPlot(this.value);
    });
    
    // 监听登出按钮
    document.getElementById('logout-btn')?.addEventListener('click', function() {
        logoutUser();
    });
});

/**
 * 初始化仪表盘
 */
function initDashboard() {
    // 获取仪表盘数据
    fetchDashboardData();
    
    // 获取天气趋势图
    fetchWeatherPlot('temparature');
    
    // 获取相关性热力图
    fetchCorrelationPlot();
    
    // 获取预测趋势图
    fetchPredictionPlot();
}

/**
 * 刷新仪表盘
 */
function refreshDashboard() {
    console.log('刷新仪表盘数据...');
    fetchDashboardData();
}

/**
 * 获取仪表盘数据
 */
function fetchDashboardData() {
    fetch('/dashboard/data')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateWeatherStats(data.data.weather_stats);
                updatePredictionStats(data.data.prediction_stats);
                
                // 更新最后更新时间
                const lastUpdated = document.getElementById('last-updated');
                if (lastUpdated) {
                    lastUpdated.textContent = `最后更新: ${data.data.timestamp}`;
                }
            } else {
                console.error('获取仪表盘数据失败:', data.message);
                showError('获取仪表盘数据失败');
            }
        })
        .catch(error => {
            console.error('获取仪表盘数据出错:', error);
            showError('获取仪表盘数据出错');
        });
}

/**
 * 更新天气统计数据
 */
function updateWeatherStats(stats) {
    if (!stats) return;
    
    // 更新天气统计卡片
    updateStatValue('current-temp', stats.temperature?.avg, 1);
    updateStatValue('current-humidity', stats.humidity?.avg, 1);
    updateStatValue('current-pressure', stats.pressure?.avg, 1);
    updateStatValue('data-count', stats.data_count);
    
    // 更新日期范围
    const dateRange = document.getElementById('date-range');
    if (dateRange && stats.date_range) {
        const start = stats.date_range.start || '未知';
        const end = stats.date_range.end || '未知';
        dateRange.textContent = `${start} 至 ${end}`;
    }
}

/**
 * 更新预测统计数据
 */
function updatePredictionStats(stats) {
    if (!stats) return;
    
    // 更新预测统计卡片
    updateStatValue('total-predictions', stats.total_predictions);
    updateStatValue('rain-predictions', stats.rain_predictions);
    updateStatValue('no-rain-predictions', stats.no_rain_predictions);
    
    // 更新平均概率（百分比格式）
    const avgProb = document.getElementById('avg-probability');
    if (avgProb && stats.avg_probability) {
        avgProb.textContent = `${(stats.avg_probability * 100).toFixed(1)}%`;
    } else if (avgProb) {
        avgProb.textContent = '--';
    }
    
    // 更新最近预测列表
    updateRecentPredictions(stats.recent_predictions);
}

/**
 * 更新最近预测列表
 */
function updateRecentPredictions(predictions) {
    const container = document.getElementById('recent-predictions');
    if (!container || !predictions || predictions.length === 0) return;
    
    const html = predictions.map(pred => {
        const isRain = pred.result.includes('会下雨');
        const resultClass = isRain ? 'text-primary' : 'text-success';
        
        return `
            <div class="recent-prediction-item">
                <div class="d-flex justify-content-between">
                    <span class="prediction-time">${pred.timestamp}</span>
                    <span class="prediction-type badge ${pred.prediction_type === 'single' ? 'bg-info' : 'bg-warning'}">${
                        pred.prediction_type === 'single' ? '单次预测' : '批量预测'
                    }</span>
                </div>
                <div class="prediction-result ${resultClass}">${pred.result}</div>
            </div>
        `;
    }).join('');
    
    container.innerHTML = html;
}

/**
 * 更新统计值
 */
function updateStatValue(elementId, value, decimals = 0) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    if (value !== undefined && value !== null) {
        element.textContent = typeof value === 'number' ? value.toFixed(decimals) : value;
    } else {
        element.textContent = '--';
    }
}

/**
 * 获取天气趋势图
 */
function fetchWeatherPlot(feature) {
    // 显示加载中
    document.getElementById('weather-plot').innerHTML = '<div class="text-center py-5"><div class="spinner-border text-primary"></div><p class="mt-2">加载中...</p></div>';
    
    fetch(`/dashboard/weather_plot?feature=${feature}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                Plotly.newPlot('weather-plot', data.plot.data, data.plot.layout);
            } else {
                console.error('获取天气趋势图失败:', data.message);
                document.getElementById('weather-plot').innerHTML = `<div class="text-center py-5 text-danger">获取天气趋势图失败</div>`;
            }
        })
        .catch(error => {
            console.error('获取天气趋势图出错:', error);
            document.getElementById('weather-plot').innerHTML = `<div class="text-center py-5 text-danger">获取天气趋势图出错</div>`;
        });
}

/**
 * 获取相关性热力图
 */
function fetchCorrelationPlot() {
    // 显示加载中
    document.getElementById('correlation-plot').innerHTML = '<div class="text-center py-5"><div class="spinner-border text-primary"></div><p class="mt-2">加载中...</p></div>';
    
    fetch('/dashboard/correlation_plot')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                Plotly.newPlot('correlation-plot', data.plot.data, data.plot.layout);
            } else {
                console.error('获取相关性热力图失败:', data.message);
                document.getElementById('correlation-plot').innerHTML = `<div class="text-center py-5 text-danger">获取相关性热力图失败</div>`;
            }
        })
        .catch(error => {
            console.error('获取相关性热力图出错:', error);
            document.getElementById('correlation-plot').innerHTML = `<div class="text-center py-5 text-danger">获取相关性热力图出错</div>`;
        });
}

/**
 * 获取预测趋势图
 */
function fetchPredictionPlot() {
    // 显示加载中
    document.getElementById('prediction-plot').innerHTML = '<div class="text-center py-5"><div class="spinner-border text-primary"></div><p class="mt-2">加载中...</p></div>';
    
    fetch('/dashboard/prediction_plot')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                Plotly.newPlot('prediction-plot', data.plot.data, data.plot.layout);
            } else {
                console.error('获取预测趋势图失败:', data.message);
                document.getElementById('prediction-plot').innerHTML = `<div class="text-center py-5 text-danger">获取预测趋势图失败</div>`;
            }
        })
        .catch(error => {
            console.error('获取预测趋势图出错:', error);
            document.getElementById('prediction-plot').innerHTML = `<div class="text-center py-5 text-danger">获取预测趋势图出错</div>`;
        });
}

/**
 * 检查用户登录状态
 */
function checkLoginStatus() {
    fetch('/auth/status')
        .then(response => response.json())
        .then(data => {
            if (!data.logged_in) {
                window.location.href = '/login';
            }
            
            // 更新用户信息显示
            const userInfo = document.getElementById('user-info');
            if (userInfo && data.username) {
                userInfo.textContent = `当前用户: ${data.username}`;
            }
        })
        .catch(error => {
            console.error('检查登录状态失败:', error);
            window.location.href = '/login';
        });
}

/**
 * 用户登出
 */
function logoutUser() {
    fetch('/logout', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                window.location.href = '/login';
            }
        })
        .catch(error => {
            console.error('登出失败:', error);
            showError('登出失败');
        });
}

/**
 * 显示错误消息
 */
function showError(message) {
    // 如果页面上有toast容器，则显示toast消息
    const toastContainer = document.getElementById('toast-container');
    if (toastContainer) {
        const toastId = 'error-toast-' + Date.now();
        const toastHtml = `
            <div id="${toastId}" class="toast align-items-center text-white bg-danger border-0" role="alert" aria-live="assertive" aria-atomic="true">
                <div class="d-flex">
                    <div class="toast-body">
                        ${message}
                    </div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
                </div>
            </div>
        `;
        
        toastContainer.insertAdjacentHTML('beforeend', toastHtml);
        const toastElement = document.getElementById(toastId);
        const toast = new bootstrap.Toast(toastElement, { delay: 5000 });
        toast.show();
    } else {
        // 否则使用alert
        alert(message);
    }
}