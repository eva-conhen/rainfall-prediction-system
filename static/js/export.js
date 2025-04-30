/**
 * 降雨预测系统 - 导出功能脚本
 * 用于处理导出页面的交互功能
 */

// 页面加载完成后初始化导出功能
document.addEventListener('DOMContentLoaded', function() {
    // 初始化导出历史
    loadExportHistory();
    
    // 绑定导出按钮事件
    bindExportButtons();
});

/**
 * 绑定导出按钮事件
 */
function bindExportButtons() {
    // 预测结果导出按钮
    const predictionFormatButtons = document.querySelectorAll('.prediction-export-btn');
    predictionFormatButtons.forEach(button => {
        button.addEventListener('click', function() {
            const format = this.getAttribute('data-format');
            exportPredictions(format);
        });
    });
    
    // 仪表盘数据导出按钮
    const dashboardFormatButtons = document.querySelectorAll('.dashboard-export-btn');
    dashboardFormatButtons.forEach(button => {
        button.addEventListener('click', function() {
            const format = this.getAttribute('data-format');
            exportDashboard(format);
        });
    });
}

/**
 * 导出预测结果
 */
function exportPredictions(format) {
    const daysSelect = document.getElementById(`${format}-days`);
    const days = daysSelect ? daysSelect.value : 30;
    
    // 显示加载提示
    showLoading(`正在导出预测结果为 ${format.toUpperCase()} 格式...`);
    
    // 创建下载链接
    const downloadUrl = `/export/predictions?format=${format}&days=${days}`;
    
    // 创建一个隐藏的a标签并触发点击
    const a = document.createElement('a');
    a.href = downloadUrl;
    a.download = `prediction_export_${new Date().toISOString().slice(0, 10)}.${format}`;
    document.body.appendChild(a);
    
    // 模拟点击下载
    setTimeout(() => {
        a.click();
        document.body.removeChild(a);
        hideLoading();
        
        // 记录导出历史
        addExportHistory('预测结果', format);
        
        // 显示成功消息
        showSuccess(`预测结果已成功导出为 ${format.toUpperCase()} 格式`);
    }, 1000);
}

/**
 * 导出仪表盘数据
 */
function exportDashboard(format) {
    // 显示加载提示
    showLoading(`正在导出仪表盘数据为 ${format.toUpperCase()} 格式...`);
    
    // 创建下载链接
    const downloadUrl = `/export/dashboard?format=${format}`;
    
    // 创建一个隐藏的a标签并触发点击
    const a = document.createElement('a');
    a.href = downloadUrl;
    a.download = `dashboard_export_${new Date().toISOString().slice(0, 10)}.${format}`;
    document.body.appendChild(a);
    
    // 模拟点击下载
    setTimeout(() => {
        a.click();
        document.body.removeChild(a);
        hideLoading();
        
        // 记录导出历史
        addExportHistory('仪表盘数据', format);
        
        // 显示成功消息
        showSuccess(`仪表盘数据已成功导出为 ${format.toUpperCase()} 格式`);
    }, 1000);
}

/**
 * 加载导出历史
 */
function loadExportHistory() {
    // 从本地存储加载导出历史
    const exportHistory = getExportHistory();
    
    // 更新导出历史表格
    updateExportHistoryTable(exportHistory);
}

/**
 * 获取导出历史
 */
function getExportHistory() {
    const historyJson = localStorage.getItem('exportHistory');
    return historyJson ? JSON.parse(historyJson) : [];
}

/**
 * 添加导出历史
 */
function addExportHistory(type, format) {
    const exportHistory = getExportHistory();
    
    // 添加新的导出记录
    exportHistory.unshift({
        timestamp: new Date().toISOString(),
        type: type,
        format: format.toUpperCase()
    });
    
    // 只保留最近20条记录
    if (exportHistory.length > 20) {
        exportHistory.pop();
    }
    
    // 保存到本地存储
    localStorage.setItem('exportHistory', JSON.stringify(exportHistory));
    
    // 更新导出历史表格
    updateExportHistoryTable(exportHistory);
}

/**
 * 更新导出历史表格
 */
function updateExportHistoryTable(history) {
    const tableBody = document.getElementById('export-history');
    if (!tableBody) return;
    
    if (!history || history.length === 0) {
        tableBody.innerHTML = `
            <tr>
                <td colspan="4" class="text-center">暂无导出历史</td>
            </tr>
        `;
        return;
    }
    
    const rows = history.map((record, index) => {
        // 格式化时间
        const date = new Date(record.timestamp);
        const formattedDate = `${date.getFullYear()}-${(date.getMonth() + 1).toString().padStart(2, '0')}-${date.getDate().toString().padStart(2, '0')} ${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}`;
        
        // 格式化类型
        const typeClass = record.type.includes('预测') ? 'text-primary' : 'text-success';
        
        // 格式化格式
        const formatClass = getFormatClass(record.format);
        
        return `
            <tr>
                <td>${formattedDate}</td>
                <td><span class="${typeClass}">${record.type}</span></td>
                <td><span class="badge ${formatClass}">${record.format}</span></td>
                <td>
                    <button class="btn btn-sm btn-outline-danger" onclick="removeExportHistory(${index})">
                        删除
                    </button>
                </td>
            </tr>
        `;
    }).join('');
    
    tableBody.innerHTML = rows;
}

/**
 * 获取格式对应的样式类
 */
function getFormatClass(format) {
    switch (format.toLowerCase()) {
        case 'csv':
            return 'bg-success';
        case 'json':
            return 'bg-primary';
        case 'pdf':
            return 'bg-danger';
        default:
            return 'bg-secondary';
    }
}

/**
 * 删除导出历史
 */
function removeExportHistory(index) {
    const exportHistory = getExportHistory();
    
    // 删除指定索引的记录
    exportHistory.splice(index, 1);
    
    // 保存到本地存储
    localStorage.setItem('exportHistory', JSON.stringify(exportHistory));
    
    // 更新导出历史表格
    updateExportHistoryTable(exportHistory);
}

/**
 * 显示加载提示
 */
function showLoading(message) {
    // 检查是否已存在加载提示
    let loadingEl = document.getElementById('loading-overlay');
    
    if (!loadingEl) {
        // 创建加载提示元素
        loadingEl = document.createElement('div');
        loadingEl.id = 'loading-overlay';
        loadingEl.className = 'position-fixed top-0 start-0 w-100 h-100 d-flex justify-content-center align-items-center';
        loadingEl.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
        loadingEl.style.zIndex = '9999';
        
        loadingEl.innerHTML = `
            <div class="bg-white p-4 rounded shadow-lg text-center">
                <div class="spinner-border text-primary mb-3" role="status">
                    <span class="visually-hidden">加载中...</span>
                </div>
                <div id="loading-message" class="text-dark">${message || '加载中...'}</div>
            </div>
        `;
        
        document.body.appendChild(loadingEl);
    } else {
        // 更新消息
        const messageEl = document.getElementById('loading-message');
        if (messageEl) {
            messageEl.textContent = message || '加载中...';
        }
    }
}

/**
 * 隐藏加载提示
 */
function hideLoading() {
    const loadingEl = document.getElementById('loading-overlay');
    if (loadingEl) {
        document.body.removeChild(loadingEl);
    }
}

/**
 * 显示成功消息
 */
function showSuccess(message) {
    // 如果页面上有toast容器，则显示toast消息
    const toastContainer = document.getElementById('toast-container');
    if (toastContainer) {
        const toastId = 'success-toast-' + Date.now();
        const toastHtml = `
            <div id="${toastId}" class="toast align-items-center text-white bg-success border-0" role="alert" aria-live="assertive" aria-atomic="true">
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
        const toast = new bootstrap.Toast(toastElement, { delay: 3000 });
        toast.show();
    } else {
        // 否则使用alert
        alert(message);
    }
}