/**
 * 大数据智能降雨预测平台 - 管理后台交互脚本（增强版）
 */

// 全局变量
let users = [];
let dataFiles = [];
let usageLogs = [];
let systemLogs = [];
let backups = [];
let systemSettings = {};

// DOM加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 初始化事件监听器
    initEventListeners();
    
    // 加载初始数据
    loadDashboardData();
    
    // 初始化图表
    initCharts();
});

/**
 * 初始化事件监听器
 */
function initEventListeners() {
    // 刷新按钮
    document.getElementById('refreshBtn')?.addEventListener('click', function() {
        loadDashboardData();
    });
    
    // 添加用户按钮
    document.getElementById('addUserBtn')?.addEventListener('click', function() {
        showUserModal();
    });
    
    // 保存用户按钮
    document.getElementById('saveUserBtn')?.addEventListener('click', function() {
        saveUser();
    });
    
    // 保存设置按钮
    document.getElementById('saveSettingsBtn')?.addEventListener('click', function() {
        saveSettings();
    });
    
    // 标签页切换事件
    document.querySelectorAll('.nav-link[data-bs-toggle="tab"]').forEach(tabEl => {
        tabEl.addEventListener('shown.bs.tab', function (event) {
            const target = event.target.getAttribute('href').substring(1);
            loadTabData(target);
        });
    });
    
    // 统计时间段按钮
    document.getElementById('weeklyStatBtn')?.addEventListener('click', function() {
        updateStatCharts('weekly');
    });
    
    document.getElementById('monthlyStatBtn')?.addEventListener('click', function() {
        updateStatCharts('monthly');
    });
    
    document.getElementById('yearlyStatBtn')?.addEventListener('click', function() {
        updateStatCharts('yearly');
    });
    
    // 导出统计按钮
    document.getElementById('exportStatBtn')?.addEventListener('click', function() {
        exportStatistics();
    });
    
    // 日志级别选择
    document.querySelectorAll('.dropdown-item[data-level]').forEach(item => {
        item.addEventListener('click', function(e) {
            e.preventDefault();
            const level = this.getAttribute('data-level');
            filterLogs(level);
        });
    });
}

/**
 * 加载仪表盘数据
 */
async function loadDashboardData() {
    try {
        showLoading();
        
        // 获取系统概况数据
        const response = await fetch('/api/admin/dashboard');
        
        if (!response.ok) {
            throw new Error('获取仪表盘数据失败');
        }
        
        const data = await response.json();
        
        // 更新计数统计
        document.getElementById('userCount').textContent = data.stats.user_count || 0;
        document.getElementById('dataCount').textContent = data.stats.data_count || 0;
        document.getElementById('predictionCount').textContent = data.stats.prediction_count || 0;
        document.getElementById('visitCount').textContent = data.stats.visit_count || 0;
        
        // 更新图表
        updateDashboardCharts(data.stats);
        
        // 更新最近日志
        updateRecentLogs(data.logs);
        
        hideLoading();
    } catch (error) {
        console.error('加载仪表盘数据失败:', error);
        hideLoading();
        showToast('加载仪表盘数据失败: ' + error.message, 'error');
    }
}

/**
 * 更新最近日志列表
 */
function updateRecentLogs(logs) {
    const logsList = document.getElementById('recentLogsList');
    if (!logsList) return;
    
    logsList.innerHTML = '';
    
    if (logs && logs.length > 0) {
        logs.forEach(log => {
            // 日志级别对应的样式类
            const levelClass = {
                'info': 'text-info',
                'warning': 'text-warning',
                'error': 'text-danger'
            };
            
            const logItem = document.createElement('li');
            logItem.className = 'list-group-item';
            logItem.innerHTML = `
                <div class="d-flex justify-content-between align-items-center">
                    <span class="${levelClass[log.level] || 'text-secondary'}">
                        <i class="bi ${log.level === 'error' ? 'bi-exclamation-triangle' : 
                                      log.level === 'warning' ? 'bi-exclamation-circle' : 
                                      'bi-info-circle'}"></i>
                        ${log.message}
                    </span>
                    <small class="text-muted">${log.timestamp}</small>
                </div>
                ${log.user ? `<small class="text-muted">用户: ${log.user}</small>` : ''}
            `;
            
            logsList.appendChild(logItem);
        });
    } else {
        const emptyItem = document.createElement('li');
        emptyItem.className = 'list-group-item text-center text-muted';
        emptyItem.textContent = '暂无日志数据';
        
        logsList.appendChild(emptyItem);
    }
}

/**
 * 加载标签页数据
 */
function loadTabData(tabId) {
    switch(tabId) {
        case 'users':
            loadUsers();
            break;
        case 'data':
            loadDataFiles();
            break;
        case 'statistics':
            loadStatistics('monthly');
            break;
        case 'logs':
            loadSystemLogs();
            break;
        case 'settings':
            loadSettings();
            break;
    }
}

/**
 * 初始化图表
 */
function initCharts() {
    // 初始化注册用户图表
    const userRegistrationCtx = document.getElementById('userRegistrationChart');
    if (userRegistrationCtx) {
        window.userRegistrationChart = new Chart(userRegistrationCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: '新注册用户',
                    data: [],
                    borderColor: 'rgba(75, 192, 192, 1)',
                    tension: 0.1,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: '最近30天用户注册趋势'
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
    }
    
    // 初始化系统使用情况图表
    const systemUsageCtx = document.getElementById('systemUsageChart');
    if (systemUsageCtx) {
        window.systemUsageChart = new Chart(systemUsageCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: '上传数据',
                    data: [],
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }, {
                    label: '执行预测',
                    data: [],
                    backgroundColor: 'rgba(255, 99, 132, 0.5)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: '最近7天系统使用情况'
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
    }
}

/**
 * 更新仪表盘图表
 */
function updateDashboardCharts(data) {
    // 更新用户注册趋势图
    if (window.userRegistrationChart && data.date_labels && data.registration_data) {
        window.userRegistrationChart.data.labels = data.date_labels;
        window.userRegistrationChart.data.datasets[0].data = data.registration_data;
        window.userRegistrationChart.update();
    }
    
    // 更新系统使用情况图
    if (window.systemUsageChart && data.date_labels) {
        window.systemUsageChart.data.labels = data.date_labels;
        window.systemUsageChart.data.datasets[0].data = data.upload_data;
        window.systemUsageChart.data.datasets[1].data = data.prediction_data;
        window.systemUsageChart.update();
    }
}

/**
 * 加载用户列表
 */
async function loadUsers(page = 1) {
    try {
        showLoading();
        
        const response = await fetch('/api/admin/users');
        
        if (!response.ok) {
            throw new Error('获取用户数据失败');
        }
        
        const data = await response.json();
        users = data.users || [];
        
        // 更新用户表格
        updateUserTable(users);
        
        hideLoading();
    } catch (error) {
        console.error('加载用户数据失败:', error);
        hideLoading();
        showToast('加载用户数据失败: ' + error.message, 'error');
    }
}

/**
 * 更新用户表格
 */
function updateUserTable(users) {
    const userTableBody = document.getElementById('userTableBody');
    if (!userTableBody) return;
    
    userTableBody.innerHTML = '';
    
    if (users && users.length > 0) {
        users.forEach(user => {
            const row = document.createElement('tr');
            
            // 状态样式
            const isActive = user.status === 'active';
            const statusClass = isActive ? 'bg-success' : 'bg-danger';
            const statusText = isActive ? '启用' : '禁用';
            
            row.innerHTML = `
                <td>${user.id || '未知ID'}</td>
                <td>${user.username}</td>
                <td>${user.register_time}</td>
                <td>${user.last_login || '从未登录'}</td>
                <td>${user.usage_count}</td>
                <td><span class="badge ${statusClass}">${statusText}</span></td>
                <td>
                    <div class="d-flex gap-2">
                        <button type="button" class="btn btn-outline-primary btn-sm edit-user" data-username="${user.username}">
                            <i class="bi bi-pencil"></i> 编辑
                        </button>
                        <button type="button" class="btn btn-outline-${isActive ? 'danger' : 'success'} btn-sm toggle-status" 
                            data-username="${user.username}" data-status="${user.status}">
                            <i class="bi bi-${isActive ? 'x-circle' : 'check-circle'}"></i> 
                            ${isActive ? '禁用' : '启用'}
                        </button>
                    </div>
                </td>
            `;
            
            userTableBody.appendChild(row);
        });
        
        // 添加事件监听器
        addUserTableEventListeners();
    } else {
        const row = document.createElement('tr');
        row.innerHTML = '<td colspan="7" class="text-center">暂无用户数据</td>';
        userTableBody.appendChild(row);
    }
}

/**
 * 添加用户表格事件监听器
 */
function addUserTableEventListeners() {
    // 编辑用户按钮
    document.querySelectorAll('.edit-user').forEach(btn => {
        btn.addEventListener('click', function() {
            const username = this.getAttribute('data-username');
            editUser(username);
        });
    });
    
    // 切换用户状态按钮
    document.querySelectorAll('.toggle-status').forEach(btn => {
        btn.addEventListener('click', function() {
            const username = this.getAttribute('data-username');
            const currentStatus = this.getAttribute('data-status');
            const newStatus = currentStatus === 'active' ? 'disabled' : 'active';
            
            updateUserStatus(username, newStatus);
        });
    });
}

/**
 * 编辑用户
 */
function editUser(username) {
    // 查找用户
    const user = users.find(u => u.username === username);
    if (!user) {
        showToast('未找到用户数据', 'error');
        return;
    }
    
    // 获取模态框元素
    const modal = document.getElementById('userModal') || createUserModal();
    
    // 设置模态框标题
    modal.querySelector('.modal-title').textContent = `编辑用户: ${username}`;
    
    // 填充表单
    modal.querySelector('#username').value = username;
    modal.querySelector('#username').disabled = true; // 禁用用户名修改
    modal.querySelector('#password').value = '';
    modal.querySelector('#password').placeholder = '留空表示不修改';
    
    if (modal.querySelector('#email')) {
        modal.querySelector('#email').value = user.email || '';
    }
    
    if (modal.querySelector('#status')) {
        modal.querySelector('#status').value = user.status || 'active';
    }
    
    if (modal.querySelector('#role')) {
        modal.querySelector('#role').value = user.role || 'user';
    }
    
    // 设置保存按钮动作
    const saveBtn = modal.querySelector('#saveUserBtn');
    saveBtn.onclick = function() {
        saveEditedUser(username);
    };
    
    // 显示模态框
    const modalInstance = new bootstrap.Modal(modal);
    modalInstance.show();
}

/**
 * 保存编辑后的用户
 */
async function saveEditedUser(username) {
    const modal = document.getElementById('userModal');
    if (!modal) return;
    
    // 收集表单数据
    const userData = {
        status: modal.querySelector('#status').value
    };
    
    // 只有当密码不为空时才包含密码
    const password = modal.querySelector('#password').value;
    if (password) {
        userData.password = password;
    }
    
    // 可选字段
    if (modal.querySelector('#email')) {
        userData.email = modal.querySelector('#email').value;
    }
    
    if (modal.querySelector('#role')) {
        userData.role = modal.querySelector('#role').value;
    }
    
    try {
        showLoading();
        
        // 发送请求更新用户
        const response = await fetch(`/api/admin/users/${username}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(userData)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || '更新用户失败');
        }
        
        // 关闭模态框
        const modalInstance = bootstrap.Modal.getInstance(modal);
        modalInstance.hide();
        
        // 重新加载用户列表
        await loadUsers();
        
        // 显示成功消息
        showToast('用户信息已更新', 'success');
    } catch (error) {
        console.error('更新用户失败:', error);
        showToast('更新用户失败: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

/**
 * 更新用户状态
 */
async function updateUserStatus(username, newStatus) {
    try {
        showLoading();
        
        console.log(`正在更新用户 ${username} 状态为 ${newStatus}`);
        
        // 发送请求更新状态
        const response = await fetch(`/api/admin/users/${username}/status`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                status: newStatus
            })
        });
        
        const responseData = await response.json();
        console.log('服务器响应:', responseData);
        
        if (!response.ok) {
            throw new Error(responseData.error || '更新用户状态失败');
        }
        
        // 重新加载用户列表
        await loadUsers();
        
        // 显示成功消息
        showToast(`用户 ${username} 状态已更新为 ${newStatus === 'active' ? '激活' : '禁用'}`, 'success');
    } catch (error) {
        console.error('更新用户状态失败:', error);
        showToast('更新用户状态失败: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

/**
 * 显示添加用户模态框
 */
function showUserModal() {
    // 获取模态框元素
    const modal = document.getElementById('userModal') || createUserModal();
    
    // 设置模态框标题
    modal.querySelector('.modal-title').textContent = '添加用户';
    
    // 清空表单
    modal.querySelector('#username').value = '';
    modal.querySelector('#username').disabled = false; // 允许设置用户名
    modal.querySelector('#password').value = '';
    modal.querySelector('#password').placeholder = '输入密码';
    
    if (modal.querySelector('#email')) {
        modal.querySelector('#email').value = '';
    }
    
    if (modal.querySelector('#status')) {
        modal.querySelector('#status').value = 'active';
    }
    
    if (modal.querySelector('#role')) {
        modal.querySelector('#role').value = 'user';
    }
    
    // 设置保存按钮动作
    const saveBtn = modal.querySelector('#saveUserBtn');
    saveBtn.onclick = function() {
        saveNewUser();
    };
    
    // 显示模态框
    const modalInstance = new bootstrap.Modal(modal);
    modalInstance.show();
}

/**
 * 创建用户模态框（如果不存在）
 */
function createUserModal() {
    const modalHtml = `
        <div class="modal fade" id="userModal" tabindex="-1" aria-labelledby="userModalLabel" aria-hidden="true">
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title" id="userModalLabel">添加用户</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                        <form id="userForm">
                            <div class="mb-3">
                                <label for="username" class="form-label">用户名</label>
                                <input type="text" class="form-control" id="username" required>
                            </div>
                            <div class="mb-3">
                                <label for="password" class="form-label">密码</label>
                                <input type="password" class="form-control" id="password" required>
                            </div>
                            <div class="mb-3">
                                <label for="email" class="form-label">电子邮箱</label>
                                <input type="email" class="form-control" id="email">
                            </div>
                            <div class="mb-3">
                                <label for="status" class="form-label">状态</label>
                                <select class="form-select" id="status">
                                    <option value="active">激活</option>
                                    <option value="disabled">禁用</option>
                                </select>
                            </div>
                            <div class="mb-3">
                                <label for="role" class="form-label">角色</label>
                                <select class="form-select" id="role">
                                    <option value="user">普通用户</option>
                                    <option value="admin">管理员</option>
                                </select>
                            </div>
                        </form>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">取消</button>
                        <button type="button" class="btn btn-primary" id="saveUserBtn">保存</button>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // 添加到DOM
    document.body.insertAdjacentHTML('beforeend', modalHtml);
    
    return document.getElementById('userModal');
}

/**
 * 保存新用户
 */
async function saveNewUser() {
    const modal = document.getElementById('userModal');
    if (!modal) return;
    
    // 获取表单数据
    const username = modal.querySelector('#username').value;
    const password = modal.querySelector('#password').value;
    
    if (!username || !password) {
        showToast('用户名和密码不能为空', 'error');
        return;
    }
    
    // 收集表单数据
    const userData = {
        username: username,
        password: password,
        status: modal.querySelector('#status').value
    };
    
    // 可选字段
    if (modal.querySelector('#email')) {
        userData.email = modal.querySelector('#email').value;
    }
    
    if (modal.querySelector('#role')) {
        userData.role = modal.querySelector('#role').value;
    }
    
    try {
        showLoading();
        
        // 发送请求创建用户
        const response = await fetch('/api/admin/users', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(userData)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || '创建用户失败');
        }
        
        // 关闭模态框
        const modalInstance = bootstrap.Modal.getInstance(modal);
        modalInstance.hide();
        
        // 重新加载用户列表
        await loadUsers();
        
        // 显示成功消息
        showToast('用户已创建', 'success');
    } catch (error) {
        console.error('创建用户失败:', error);
        showToast('创建用户失败: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

/**
 * 显示加载指示器
 */
function showLoading() {
    // 显示加载提示
    const loadingEl = document.getElementById('loadingIndicator') || createLoadingIndicator();
    loadingEl.style.display = 'flex';
}

/**
 * 隐藏加载指示器
 */
function hideLoading() {
    const loadingEl = document.getElementById('loadingIndicator');
    if (loadingEl) {
        loadingEl.style.display = 'none';
    }
}

/**
 * 创建加载指示器（如果不存在）
 */
function createLoadingIndicator() {
    const loadingHtml = `
        <div id="loadingIndicator" style="position: fixed; top: 0; left: 0; right: 0; bottom: 0; 
             background: rgba(0,0,0,0.5); display: flex; justify-content: center; align-items: center; z-index: 9999;">
            <div class="spinner-border text-light" role="status">
                <span class="visually-hidden">加载中...</span>
            </div>
        </div>
    `;
    
    // 添加到DOM
    document.body.insertAdjacentHTML('beforeend', loadingHtml);
    
    return document.getElementById('loadingIndicator');
}

/**
 * 显示提示消息
 */
function showToast(message, type = 'info') {
    // 消息类型样式
    const typeClass = {
        'info': 'bg-info',
        'success': 'bg-success',
        'warning': 'bg-warning',
        'error': 'bg-danger'
    };
    
    // 创建Toast元素
    const toastId = 'toast-' + Date.now();
    const toastHtml = `
        <div id="${toastId}" class="toast align-items-center ${typeClass[type] || 'bg-info'} text-white border-0" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
        </div>
    `;
    
    // 创建或获取Toast容器
    let toastContainer = document.getElementById('toastContainer');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toastContainer';
        toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        document.body.appendChild(toastContainer);
    }
    
    // 添加Toast到容器
    toastContainer.insertAdjacentHTML('beforeend', toastHtml);
    
    // 显示Toast
    const toastEl = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastEl, {
        autohide: true,
        delay: 5000
    });
    toast.show();
    
    // 自动删除
    toastEl.addEventListener('hidden.bs.toast', function() {
        this.remove();
    });
} 