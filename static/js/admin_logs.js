/**
 * 大数据智能降雨预测平台 - 管理后台系统日志功能
 */

// 全局变量
let systemLogs = [];
let currentLogLevel = null;

/**
 * 加载系统日志
 */
async function loadSystemLogs() {
    try {
        showLoading();
        
        // 构建API请求URL
        let url = '/api/admin/logs';
        const params = new URLSearchParams();
        
        if (currentLogLevel) {
            params.append('level', currentLogLevel);
        }
        
        params.append('limit', 100);
        
        if (params.toString()) {
            url += '?' + params.toString();
        }
        
        // 获取系统日志
        const response = await fetch(url);
        
        if (!response.ok) {
            throw new Error('获取系统日志失败');
        }
        
        const result = await response.json();
        systemLogs = result.logs || [];
        
        // 更新日志表格
        updateLogsTable();
        
        // 更新当前筛选显示
        updateCurrentFilterDisplay();
        
        hideLoading();
    } catch (error) {
        console.error('加载系统日志失败:', error);
        hideLoading();
        showToast('加载系统日志失败: ' + error.message, 'error');
    }
}

/**
 * 更新日志表格
 */
function updateLogsTable() {
    const logsTableBody = document.getElementById('logsTableBody');
    if (!logsTableBody) return;
    
    logsTableBody.innerHTML = '';
    
    if (systemLogs && systemLogs.length > 0) {
        systemLogs.forEach(log => {
            // 日志级别对应的样式类
            const levelClass = {
                'info': 'text-info',
                'warning': 'text-warning',
                'error': 'text-danger'
            };
            
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${log.timestamp}</td>
                <td><span class="${levelClass[log.level] || 'text-secondary'}">${log.level}</span></td>
                <td>${log.message}</td>
                <td>${log.user || '-'}</td>
            `;
            
            logsTableBody.appendChild(row);
        });
    } else {
        const row = document.createElement('tr');
        row.innerHTML = '<td colspan="4" class="text-center">暂无日志数据</td>';
        logsTableBody.appendChild(row);
    }
}

/**
 * 更新当前筛选显示
 */
function updateCurrentFilterDisplay() {
    const currentFilterDisplay = document.getElementById('currentLogFilter');
    if (!currentFilterDisplay) return;
    
    if (currentLogLevel) {
        currentFilterDisplay.textContent = `当前筛选: ${currentLogLevel}`;
        currentFilterDisplay.style.display = 'inline-block';
    } else {
        currentFilterDisplay.style.display = 'none';
    }
}

/**
 * 筛选日志
 */
function filterLogs(level) {
    // 如果点击的是当前级别，则取消筛选
    if (level === currentLogLevel) {
        currentLogLevel = null;
    } else {
        currentLogLevel = level;
    }
    
    // 重新加载日志
    loadSystemLogs();
}

/**
 * 导出日志
 */
function exportLogs() {
    if (systemLogs.length === 0) {
        showToast('没有可导出的日志', 'warning');
        return;
    }
    
    try {
        // 将日志转换为CSV格式
        let csvContent = "时间戳,级别,消息,用户\n";
        
        systemLogs.forEach(log => {
            const row = [
                log.timestamp,
                log.level,
                `"${log.message.replace(/"/g, '""')}"`, // 处理引号
                log.user || ''
            ];
            
            csvContent += row.join(',') + "\n";
        });
        
        // 创建下载链接
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        
        // 设置下载文件名（当前日期_logs.csv）
        const date = new Date().toISOString().split('T')[0];
        link.setAttribute('href', url);
        link.setAttribute('download', `${date}_logs.csv`);
        link.style.visibility = 'hidden';
        
        // 触发下载
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        showToast('日志导出成功', 'success');
    } catch (error) {
        console.error('导出日志失败:', error);
        showToast('导出日志失败: ' + error.message, 'error');
    }
}

/**
 * 初始化日志页面
 */
function initLogsPage() {
    // 创建日志筛选按钮
    createLogFilterButtons();
    
    // 创建导出按钮
    createExportLogButton();
    
    // 添加事件监听器
    addLogPageEventListeners();
}

/**
 * 创建日志筛选按钮
 */
function createLogFilterButtons() {
    const logFilterContainer = document.querySelector('#logs .card-header');
    if (!logFilterContainer || document.getElementById('logFilterBtns')) return;
    
    const filterHtml = `
        <div id="logFilterBtns" class="btn-group btn-group-sm ms-2">
            <button type="button" class="btn btn-outline-info filter-log" data-level="info">Info</button>
            <button type="button" class="btn btn-outline-warning filter-log" data-level="warning">Warning</button>
            <button type="button" class="btn btn-outline-danger filter-log" data-level="error">Error</button>
            <button type="button" class="btn btn-outline-secondary filter-log" data-level="">全部</button>
        </div>
        <span id="currentLogFilter" class="badge bg-secondary ms-2" style="display: none;"></span>
    `;
    
    logFilterContainer.insertAdjacentHTML('beforeend', filterHtml);
}

/**
 * 创建导出按钮
 */
function createExportLogButton() {
    const logActionContainer = document.querySelector('#logs .card-header');
    if (!logActionContainer || document.getElementById('exportLogBtn')) return;
    
    const exportBtnHtml = `
        <button id="exportLogBtn" class="btn btn-sm btn-outline-primary float-end">
            <i class="bi bi-download"></i> 导出日志
        </button>
    `;
    
    logActionContainer.insertAdjacentHTML('beforeend', exportBtnHtml);
}

/**
 * 添加日志页面事件监听器
 */
function addLogPageEventListeners() {
    // 筛选按钮
    document.querySelectorAll('.filter-log').forEach(btn => {
        btn.addEventListener('click', function() {
            const level = this.getAttribute('data-level');
            filterLogs(level);
        });
    });
    
    // 导出按钮
    const exportBtn = document.getElementById('exportLogBtn');
    if (exportBtn) {
        exportBtn.addEventListener('click', exportLogs);
    }
}

// 初始化日志页面
document.addEventListener('DOMContentLoaded', function() {
    initLogsPage();
}); 