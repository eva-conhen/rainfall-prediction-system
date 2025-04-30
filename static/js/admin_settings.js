/**
 * 大数据智能降雨预测平台 - 管理后台系统设置功能
 */

// 全局变量
let systemSettings = {};

/**
 * 加载系统设置
 */
async function loadSettings() {
    try {
        showLoading();
        
        // 获取系统设置
        const response = await fetch('/api/admin/settings');
        
        if (!response.ok) {
            throw new Error('获取系统设置失败');
        }
        
        systemSettings = await response.json();
        
        // 更新设置表单
        updateSettingsForm();
        
        hideLoading();
    } catch (error) {
        console.error('加载系统设置失败:', error);
        hideLoading();
        showToast('加载系统设置失败: ' + error.message, 'error');
    }
}

/**
 * 更新设置表单
 */
function updateSettingsForm() {
    // 系统名称
    const systemNameInput = document.getElementById('systemName');
    if (systemNameInput) {
        systemNameInput.value = systemSettings.system_name || '大数据智能降雨预测平台';
    }
    
    // 数据保留天数
    const dataRetentionInput = document.getElementById('dataRetention');
    if (dataRetentionInput) {
        dataRetentionInput.value = systemSettings.data_retention_days || 30;
    }
    
    // 日志级别
    const logLevelSelect = document.getElementById('logLevel');
    if (logLevelSelect) {
        logLevelSelect.value = systemSettings.log_level || 'info';
    }
    
    // 自动备份
    const backupEnabledCheck = document.getElementById('backupEnabled');
    if (backupEnabledCheck) {
        backupEnabledCheck.checked = systemSettings.backup_enabled === undefined ? true : systemSettings.backup_enabled;
        updateBackupIntervalVisibility();
    }
    
    // 备份间隔天数
    const backupIntervalInput = document.getElementById('backupInterval');
    if (backupIntervalInput) {
        backupIntervalInput.value = systemSettings.backup_interval_days || 7;
    }
    
    // 邮件通知
    const emailNotificationsCheck = document.getElementById('emailNotifications');
    if (emailNotificationsCheck) {
        emailNotificationsCheck.checked = systemSettings.email_notifications || false;
        updateAdminEmailVisibility();
    }
    
    // 管理员邮箱
    const adminEmailInput = document.getElementById('adminEmail');
    if (adminEmailInput) {
        adminEmailInput.value = systemSettings.admin_email || '';
    }
    
    // 模型自动更新
    const modelAutoUpdateCheck = document.getElementById('modelAutoUpdate');
    if (modelAutoUpdateCheck) {
        modelAutoUpdateCheck.checked = systemSettings.model_auto_update === undefined ? true : systemSettings.model_auto_update;
    }
    
    // 添加事件监听器（如果尚未添加）
    addSettingsEventListeners();
}

/**
 * 添加设置表单事件监听器
 */
function addSettingsEventListeners() {
    // 自动备份切换
    const backupEnabledCheck = document.getElementById('backupEnabled');
    if (backupEnabledCheck && !backupEnabledCheck.hasListener) {
        backupEnabledCheck.addEventListener('change', updateBackupIntervalVisibility);
        backupEnabledCheck.hasListener = true;
    }
    
    // 邮件通知切换
    const emailNotificationsCheck = document.getElementById('emailNotifications');
    if (emailNotificationsCheck && !emailNotificationsCheck.hasListener) {
        emailNotificationsCheck.addEventListener('change', updateAdminEmailVisibility);
        emailNotificationsCheck.hasListener = true;
    }
    
    // 保存设置按钮
    const saveSettingsBtn = document.getElementById('saveSettingsBtn');
    if (saveSettingsBtn && !saveSettingsBtn.hasListener) {
        saveSettingsBtn.addEventListener('click', saveSettings);
        saveSettingsBtn.hasListener = true;
    }
}

/**
 * 更新备份间隔天数输入框可见性
 */
function updateBackupIntervalVisibility() {
    const backupEnabledCheck = document.getElementById('backupEnabled');
    const backupIntervalGroup = document.getElementById('backupIntervalGroup');
    
    if (backupEnabledCheck && backupIntervalGroup) {
        backupIntervalGroup.style.display = backupEnabledCheck.checked ? 'block' : 'none';
    }
}

/**
 * 更新管理员邮箱输入框可见性
 */
function updateAdminEmailVisibility() {
    const emailNotificationsCheck = document.getElementById('emailNotifications');
    const adminEmailGroup = document.getElementById('adminEmailGroup');
    
    if (emailNotificationsCheck && adminEmailGroup) {
        adminEmailGroup.style.display = emailNotificationsCheck.checked ? 'block' : 'none';
    }
}

/**
 * 保存系统设置
 */
async function saveSettings() {
    // 收集表单数据
    const settings = {
        system_name: document.getElementById('systemName').value,
        data_retention_days: parseInt(document.getElementById('dataRetention').value),
        log_level: document.getElementById('logLevel').value,
        backup_enabled: document.getElementById('backupEnabled').checked,
        backup_interval_days: parseInt(document.getElementById('backupInterval').value),
        email_notifications: document.getElementById('emailNotifications').checked,
        admin_email: document.getElementById('adminEmail').value,
        model_auto_update: document.getElementById('modelAutoUpdate').checked
    };
    
    // 验证表单数据
    if (!settings.system_name) {
        showToast('系统名称不能为空', 'error');
        return;
    }
    
    if (isNaN(settings.data_retention_days) || settings.data_retention_days < 1) {
        showToast('数据保留天数必须是大于0的整数', 'error');
        return;
    }
    
    if (isNaN(settings.backup_interval_days) || settings.backup_interval_days < 1) {
        showToast('备份间隔天数必须是大于0的整数', 'error');
        return;
    }
    
    if (settings.email_notifications && !validateEmail(settings.admin_email)) {
        showToast('请输入有效的管理员邮箱地址', 'error');
        return;
    }
    
    try {
        showLoading();
        
        // 发送请求保存设置
        const response = await fetch('/api/admin/settings', {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(settings)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || '保存设置失败');
        }
        
        // 更新全局设置对象
        systemSettings = settings;
        
        // 显示成功消息
        showToast('系统设置已保存', 'success');
    } catch (error) {
        console.error('保存系统设置失败:', error);
        showToast('保存系统设置失败: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

/**
 * 验证邮箱地址
 */
function validateEmail(email) {
    const re = /^(([^<>()\[\]\\.,;:\s@"]+(\.[^<>()\[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/;
    return re.test(String(email).toLowerCase());
}

/**
 * 创建设置面板
 */
function createSettingsPanel() {
    const settingsPanelId = 'settingsPanel';
    
    // 检查面板是否已存在
    if (document.getElementById(settingsPanelId)) {
        return;
    }
    
    const panelHtml = `
        <div id="${settingsPanelId}" class="card">
            <div class="card-header">
                <h5 class="mb-0">系统设置</h5>
            </div>
            <div class="card-body">
                <form id="settingsForm">
                    <div class="mb-3">
                        <label for="systemName" class="form-label">系统名称</label>
                        <input type="text" class="form-control" id="systemName" required>
                    </div>
                    
                    <div class="mb-3">
                        <label for="dataRetention" class="form-label">数据保留天数</label>
                        <input type="number" class="form-control" id="dataRetention" min="1" required>
                        <div class="form-text">设置数据文件和日志的保留天数，超过此期限的数据将被自动清理</div>
                    </div>
                    
                    <div class="mb-3">
                        <label for="logLevel" class="form-label">日志级别</label>
                        <select class="form-select" id="logLevel">
                            <option value="debug">Debug</option>
                            <option value="info">Info</option>
                            <option value="warning">Warning</option>
                            <option value="error">Error</option>
                        </select>
                    </div>
                    
                    <div class="mb-3 form-check">
                        <input type="checkbox" class="form-check-input" id="backupEnabled">
                        <label class="form-check-label" for="backupEnabled">启用自动备份</label>
                    </div>
                    
                    <div class="mb-3" id="backupIntervalGroup">
                        <label for="backupInterval" class="form-label">备份间隔天数</label>
                        <input type="number" class="form-control" id="backupInterval" min="1" required>
                    </div>
                    
                    <div class="mb-3 form-check">
                        <input type="checkbox" class="form-check-input" id="emailNotifications">
                        <label class="form-check-label" for="emailNotifications">启用邮件通知</label>
                    </div>
                    
                    <div class="mb-3" id="adminEmailGroup">
                        <label for="adminEmail" class="form-label">管理员邮箱</label>
                        <input type="email" class="form-control" id="adminEmail">
                        <div class="form-text">接收系统通知和警报的邮箱地址</div>
                    </div>
                    
                    <div class="mb-3 form-check">
                        <input type="checkbox" class="form-check-input" id="modelAutoUpdate">
                        <label class="form-check-label" for="modelAutoUpdate">启用模型自动更新</label>
                        <div class="form-text">当有新数据上传时，自动重新训练模型</div>
                    </div>
                    
                    <div class="d-grid gap-2 d-md-flex justify-content-md-end">
                        <button type="button" class="btn btn-primary" id="saveSettingsBtn">
                            <i class="bi bi-save"></i> 保存设置
                        </button>
                    </div>
                </form>
            </div>
        </div>
    `;
    
    // 查找目标容器
    const targetContainer = document.querySelector('#settings');
    if (targetContainer) {
        targetContainer.innerHTML = panelHtml;
        
        // 初始化事件监听器
        addSettingsEventListeners();
        
        // 初始化表单可见性
        updateBackupIntervalVisibility();
        updateAdminEmailVisibility();
    }
}

// 初始化设置面板
document.addEventListener('DOMContentLoaded', function() {
    createSettingsPanel();
}); 