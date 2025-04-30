/**
 * 大数据智能降雨预测平台 - 管理后台使用统计功能
 */

// 全局变量
let statisticsData = {};
let currentPeriod = 'monthly';

/**
 * 加载统计数据
 */
async function loadStatistics(period = 'monthly') {
    try {
        showLoading();
        
        // 保存当前周期
        currentPeriod = period;
        
        // 获取统计数据
        const response = await fetch(`/api/admin/statistics?period=${period}`);
        
        if (!response.ok) {
            throw new Error('获取统计数据失败');
        }
        
        statisticsData = await response.json();
        
        // 更新统计图表
        updateStatCharts();
        
        // 更新周期按钮状态
        updatePeriodButtons();
        
        hideLoading();
    } catch (error) {
        console.error('加载统计数据失败:', error);
        hideLoading();
        showToast('加载统计数据失败: ' + error.message, 'error');
    }
}

/**
 * 更新周期按钮状态
 */
function updatePeriodButtons() {
    // 移除所有按钮的激活状态
    document.querySelectorAll('.period-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // 激活当前周期按钮
    const activeBtn = document.querySelector(`.period-btn[data-period="${currentPeriod}"]`);
    if (activeBtn) {
        activeBtn.classList.add('active');
    }
}

/**
 * 更新统计图表
 */
function updateStatCharts() {
    // 确保数据存在且格式正确
    if (!statisticsData || typeof statisticsData !== 'object') {
        console.error('没有有效的统计数据对象:', statisticsData);
        showToast('统计数据格式无效', 'error');
        return;
    }
    
    // 确保数据中包含必要的属性
    if (!statisticsData.labels || !Array.isArray(statisticsData.labels)) {
        console.error('统计数据缺少有效的标签数组:', statisticsData);
        // 创建默认标签
        statisticsData.labels = ['无数据'];
    }
    
    // 确保各项统计数据存在，如果不存在则创建空数组
    if (!statisticsData.visits || !Array.isArray(statisticsData.visits)) {
        statisticsData.visits = new Array(statisticsData.labels.length).fill(0);
    }
    
    if (!statisticsData.predictions || !Array.isArray(statisticsData.predictions)) {
        statisticsData.predictions = new Array(statisticsData.labels.length).fill(0);
    }
    
    if (!statisticsData.uploads || !Array.isArray(statisticsData.uploads)) {
        statisticsData.uploads = new Array(statisticsData.labels.length).fill(0);
    }
    
    if (!statisticsData.registrations || !Array.isArray(statisticsData.registrations)) {
        statisticsData.registrations = new Array(statisticsData.labels.length).fill(0);
    }
    
    console.log('更新图表，使用数据:', statisticsData);
    
    // 系统访问量图表
    updateVisitChart();
    
    // 预测次数图表
    updatePredictionChart();
    
    // 数据上传图表
    updateUploadChart();
    
    // 用户注册图表
    updateRegistrationChart();
}

/**
 * 更新系统访问量图表
 */
function updateVisitChart() {
    const visitChartCtx = document.getElementById('visitChart');
    if (!visitChartCtx) return;
    
    if (!window.visitChart) {
        window.visitChart = new Chart(visitChartCtx, {
            type: 'line',
            data: {
                labels: statisticsData.labels,
                datasets: [{
                    label: '系统访问量',
                    data: statisticsData.visits,
                    borderColor: 'rgba(255, 159, 64, 1)',
                    backgroundColor: 'rgba(255, 159, 64, 0.2)',
                    tension: 0.1,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: '系统访问量趋势'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            precision: 0
                        }
                    }
                }
            }
        });
    } else {
        window.visitChart.data.labels = statisticsData.labels;
        window.visitChart.data.datasets[0].data = statisticsData.visits;
        window.visitChart.update();
    }
}

/**
 * 更新预测次数图表
 */
function updatePredictionChart() {
    const predictionChartCtx = document.getElementById('predictionChart');
    if (!predictionChartCtx) return;
    
    if (!window.predictionChart) {
        window.predictionChart = new Chart(predictionChartCtx, {
            type: 'line',
            data: {
                labels: statisticsData.labels,
                datasets: [{
                    label: '预测次数',
                    data: statisticsData.predictions,
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    tension: 0.1,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: '预测使用趋势'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            precision: 0
                        }
                    }
                }
            }
        });
    } else {
        window.predictionChart.data.labels = statisticsData.labels;
        window.predictionChart.data.datasets[0].data = statisticsData.predictions;
        window.predictionChart.update();
    }
}

/**
 * 更新数据上传图表
 */
function updateUploadChart() {
    const uploadChartCtx = document.getElementById('uploadChart');
    if (!uploadChartCtx) return;
    
    if (!window.uploadChart) {
        window.uploadChart = new Chart(uploadChartCtx, {
            type: 'bar',
            data: {
                labels: statisticsData.labels,
                datasets: [{
                    label: '上传数据量',
                    data: statisticsData.uploads,
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: '数据上传趋势'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            precision: 0
                        }
                    }
                }
            }
        });
    } else {
        window.uploadChart.data.labels = statisticsData.labels;
        window.uploadChart.data.datasets[0].data = statisticsData.uploads;
        window.uploadChart.update();
    }
}

/**
 * 更新用户注册图表
 */
function updateRegistrationChart() {
    const registrationChartCtx = document.getElementById('registrationChart');
    if (!registrationChartCtx) return;
    
    if (!window.registrationChart) {
        window.registrationChart = new Chart(registrationChartCtx, {
            type: 'bar',
            data: {
                labels: statisticsData.labels,
                datasets: [{
                    label: '用户注册数',
                    data: statisticsData.registrations,
                    backgroundColor: 'rgba(75, 192, 192, 0.5)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: '用户注册趋势'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            precision: 0
                        }
                    }
                }
            }
        });
    } else {
        window.registrationChart.data.labels = statisticsData.labels;
        window.registrationChart.data.datasets[0].data = statisticsData.registrations;
        window.registrationChart.update();
    }
}

/**
 * 导出统计数据
 */
function exportStatistics() {
    if (!statisticsData || !statisticsData.date_labels) {
        showToast('没有可导出的统计数据', 'warning');
        return;
    }
    
    try {
        // 将统计数据转换为CSV格式
        let csvContent = "日期,系统访问量,预测次数,数据上传,用户注册\n";
        
        for (let i = 0; i < statisticsData.date_labels.length; i++) {
            const row = [
                statisticsData.date_labels[i],
                statisticsData.visits_data[i],
                statisticsData.predictions_data[i],
                statisticsData.uploads_data[i],
                statisticsData.registrations_data[i]
            ];
            
            csvContent += row.join(',') + "\n";
        }
        
        // 创建下载链接
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        
        // 设置下载文件名（当前日期_statistics_period.csv）
        const date = new Date().toISOString().split('T')[0];
        link.setAttribute('href', url);
        link.setAttribute('download', `${date}_statistics_${currentPeriod}.csv`);
        link.style.visibility = 'hidden';
        
        // 触发下载
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        showToast('统计数据导出成功', 'success');
    } catch (error) {
        console.error('导出统计数据失败:', error);
        showToast('导出统计数据失败: ' + error.message, 'error');
    }
}

/**
 * 初始化统计页面
 */
function initStatisticsPage() {
    // 创建统计图表容器
    createStatisticsCharts();
    
    // 创建周期按钮和导出按钮
    createStatisticsControls();
    
    // 添加事件监听器
    addStatisticsEventListeners();
}

/**
 * 创建统计图表容器
 */
function createStatisticsCharts() {
    const statisticsContainer = document.querySelector('#statistics');
    if (!statisticsContainer || document.getElementById('statisticsCharts')) return;
    
    const chartsHtml = `
        <div id="statisticsCharts" class="mt-4">
            <div class="row">
                <div class="col-md-6 mb-4">
                    <div class="card">
                        <div class="card-header">系统访问量趋势</div>
                        <div class="card-body">
                            <canvas id="visitChart"></canvas>
                        </div>
                    </div>
                </div>
                <div class="col-md-6 mb-4">
                    <div class="card">
                        <div class="card-header">预测使用趋势</div>
                        <div class="card-body">
                            <canvas id="predictionChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
            <div class="row">
                <div class="col-md-6 mb-4">
                    <div class="card">
                        <div class="card-header">数据上传趋势</div>
                        <div class="card-body">
                            <canvas id="uploadChart"></canvas>
                        </div>
                    </div>
                </div>
                <div class="col-md-6 mb-4">
                    <div class="card">
                        <div class="card-header">用户注册趋势</div>
                        <div class="card-body">
                            <canvas id="registrationChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    statisticsContainer.insertAdjacentHTML('beforeend', chartsHtml);
}

/**
 * 创建统计控制按钮
 */
function createStatisticsControls() {
    const statisticsHeader = document.querySelector('#statistics .card-header');
    if (!statisticsHeader || document.getElementById('statisticsControls')) return;
    
    const controlsHtml = `
        <div id="statisticsControls" class="float-end">
            <div class="btn-group btn-group-sm me-2">
                <button type="button" class="btn btn-outline-primary period-btn" data-period="weekly">周</button>
                <button type="button" class="btn btn-outline-primary period-btn active" data-period="monthly">月</button>
                <button type="button" class="btn btn-outline-primary period-btn" data-period="yearly">年</button>
            </div>
            <button id="exportStatBtn" class="btn btn-sm btn-outline-success">
                <i class="bi bi-download"></i> 导出数据
            </button>
        </div>
    `;
    
    statisticsHeader.insertAdjacentHTML('beforeend', controlsHtml);
}

/**
 * 添加统计页面事件监听器
 */
function addStatisticsEventListeners() {
    // 周期按钮
    document.querySelectorAll('.period-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const period = this.getAttribute('data-period');
            loadStatistics(period);
        });
    });
    
    // 导出按钮
    const exportBtn = document.getElementById('exportStatBtn');
    if (exportBtn) {
        exportBtn.addEventListener('click', exportStatistics);
    }
}

// 初始化统计页面
document.addEventListener('DOMContentLoaded', function() {
    initStatisticsPage();
}); 