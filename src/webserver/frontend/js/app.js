/**
 * Biometric System - Main JavaScript Application
 * Handles API communication and UI interactions
 */

// API Configuration
const API_BASE_URL = window.location.origin;

// Global state
let currentUser = {
    login: null,
    privilege: null,
    token: null
};

// ============================================================================
// AUTHENTICATION
// ============================================================================

/**
 * Get current authentication token
 */
function getToken() {
    return localStorage.getItem('token');
}

/**
 * Get current user info
 */
function getCurrentUser() {
    return {
        login: localStorage.getItem('login'),
        privilege: localStorage.getItem('privilege'),
        token: localStorage.getItem('token')
    };
}

/**
 * Logout user
 */
function logout() {
    localStorage.removeItem('token');
    localStorage.removeItem('privilege');
    localStorage.removeItem('login');
    
    window.location.href = '/';
}

/**
 * Check if user is authenticated
 */
function requireAuth(requiredPrivilege = null) {
    const user = getCurrentUser();
    
    if (!user.token) {
        window.location.href = '/';
        return false;
    }
    
    if (requiredPrivilege && user.privilege !== requiredPrivilege) {
        alert('Insufficient privileges');
        logout();
        return false;
    }
    
    currentUser = user;
    return true;
}

// ============================================================================
// API CALLS
// ============================================================================

/**
 * Make authenticated API call
 */
async function apiCall(endpoint, method = 'GET', body = null) {
    const token = getToken();
    
    const options = {
        method,
        headers: {
            'Authorization': `Bearer ${token}`
        }
    };
    
    // Only set Content-Type for JSON (not for FormData)
    if (body && !(body instanceof FormData)) {
        options.headers['Content-Type'] = 'application/json';
        options.body = JSON.stringify(body);
    } else if (body instanceof FormData) {
        // Let browser set Content-Type with boundary for FormData
        options.body = body;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, options);
        
        // Handle 401 Unauthorized
        if (response.status === 401) {
            alert('Session expired. Please login again.');
            logout();
            throw new Error('Unauthorized');
        }
        
        // Try to parse JSON response
        let data;
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
            data = await response.json();
        } else {
            // Non-JSON response (probably HTML error page)
            const text = await response.text();
            console.error('Non-JSON response:', text.substring(0, 200));
            throw new Error(`Server error (HTTP ${response.status})`);
        }
        
        if (!response.ok) {
            throw new Error(data.error || data.detail || `HTTP ${response.status}`);
        }
        
        return data;
    } catch (error) {
        console.error('API call failed:', error);
        throw error;
    }
}

/**
 * Get all users
 */
async function getUsers() {
    return await apiCall('/api/users');
}

/**
 * Get user by ID
 */
async function getUser(userId) {
    return await apiCall(`/api/users/${userId}`);
}

/**
 * Add new user
 */
async function addUser(userId, nome, files) {
    // Create FormData
    const formData = new FormData();
    formData.append('user_id', userId);
    formData.append('name', nome);  // API expects 'name', not 'nome'
    
    // Append all image files
    for (let i = 0; i < files.length; i++) {
        formData.append('images', files[i]);
    }
    
    return await apiCall('/api/users', 'POST', formData);
}

/**
 * Update user
 */
async function updateUser(userId, nome = null, files = null) {
    const formData = new FormData();
    
    if (nome) {
        formData.append('name', nome);  // API expects 'name', not 'nome'
    }
    
    if (files) {
        for (let i = 0; i < files.length; i++) {
            formData.append('images', files[i]);
        }
    }
    
    return await apiCall(`/api/users/${userId}`, 'PUT', formData);
}

/**
 * Delete user
 */
async function deleteUser(userId) {
    return await apiCall(`/api/users/${userId}`, 'DELETE');
}

/**
 * Delete all users
 */
async function deleteAllUsers() {
    return await apiCall('/api/users', 'DELETE');
}

/**
 * Verify fingerprint (1:1)
 */
async function verifyFingerprint(userId, imageFile) {
    const formData = new FormData();
    formData.append('user_id', userId);
    formData.append('image', imageFile);
    
    return await apiCall('/api/verify', 'POST', formData);
}

/**
 * Identify fingerprint (1:N)
 */
async function identifyFingerprint(imageFile) {
    const formData = new FormData();
    formData.append('image', imageFile);
    
    return await apiCall('/api/identify', 'POST', formData);
}

/**
 * Load templates from folder
 */
async function loadTemplatesFromFolder(folderPath) {
    // API expects FormData, not JSON body
    const formData = new FormData();
    formData.append('folder_path', folderPath);
    
    // Start the job
    const jobResponse = await apiCall('/api/admin/load_folder', 'POST', formData);
    const jobId = jobResponse.job_id;
    
    // Poll for job completion
    while (true) {
        await new Promise(resolve => setTimeout(resolve, 500)); // Wait 500ms
        
        const jobStatus = await apiCall(`/api/admin/jobs/${jobId}`);
        
        if (jobStatus.status === 'completed') {
            // Parse result JSON
            const result = JSON.parse(jobStatus.result || '{}');
            return {
                enrolled: result.enrolled || 0,
                skipped: 0, // Not tracked separately
                errors: result.failed || 0,
                total_users: result.total || 0,
                failed_users: result.failed_users || []
            };
        } else if (jobStatus.status === 'failed') {
            throw new Error(jobStatus.error || 'Job failed');
        }
        // If status is 'processing' or 'pending', continue polling
    }
}

/**
 * Get system statistics
 */
async function getStats() {
    return await apiCall('/api/admin/stats');
}

// ============================================================================
// UI HELPERS
// ============================================================================

/**
 * Show alert message
 */
function showAlert(container, message, type = 'info', duration = 5000) {
    if (typeof container === 'string') {
        container = document.getElementById(container);
    }
    
    if (!container) {
        console.error('Alert container not found');
        return;
    }
    
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} fade-in`;
    
    const icons = {
        success: '✓',
        danger: '✗',
        warning: '⚠',
        info: 'ℹ'
    };
    
    alert.innerHTML = `
        <span style="font-size: 1.25rem;">${icons[type] || '•'}</span>
        <span>${message}</span>
    `;
    
    container.appendChild(alert);
    
    // Auto-remove
    if (duration > 0) {
        setTimeout(() => {
            alert.style.opacity = '0';
            setTimeout(() => alert.remove(), 300);
        }, duration);
    }
    
    return alert;
}

/**
 * Show loading overlay
 */
function showLoading(message = 'Loading...') {
    let overlay = document.getElementById('loading-overlay');
    
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'loading-overlay';
        overlay.className = 'loading-overlay';
        overlay.innerHTML = `
            <div style="text-align: center;">
                <div class="spinner" style="margin: 0 auto 1rem;"></div>
                <div id="loading-message" style="color: white; font-weight: 600;">${message}</div>
            </div>
        `;
        document.body.appendChild(overlay);
    }
    
    const messageEl = overlay.querySelector('#loading-message');
    if (messageEl) {
        messageEl.textContent = message;
    }
    
    overlay.classList.add('active');
}

/**
 * Hide loading overlay
 */
function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.classList.remove('active');
    }
}

/**
 * Format timestamp
 */
function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleString('pt-PT');
}

/**
 * Format quality score
 */
function formatQuality(quality) {
    const percent = (quality * 100).toFixed(1);
    
    if (quality >= 0.8) {
        return `<span class="badge badge-success">${percent}%</span>`;
    } else if (quality >= 0.6) {
        return `<span class="badge badge-warning">${percent}%</span>`;
    } else {
        return `<span class="badge badge-danger">${percent}%</span>`;
    }
}

/**
 * Read file as base64
 */
function readFileAsBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        
        reader.onload = () => {
            // Remove data URL prefix (e.g., "data:image/png;base64,")
            const base64 = reader.result.split(',')[1];
            resolve(base64);
        };
        
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

/**
 * Read multiple files as base64
 */
async function readFilesAsBase64(files) {
    const promises = Array.from(files).map(file => readFileAsBase64(file));
    return await Promise.all(promises);
}

/**
 * Create modal
 */
function createModal(title, content, buttons = []) {
    const modal = document.createElement('div');
    modal.className = 'modal active';
    
    let buttonsHtml = '';
    buttons.forEach(btn => {
        buttonsHtml += `
            <button class="btn btn-${btn.type || 'secondary'}" onclick="${btn.onclick}">
                ${btn.text}
            </button>
        `;
    });
    
    modal.innerHTML = `
        <div class="modal-content">
            <div class="modal-header">
                <h3>${title}</h3>
                <button class="modal-close" onclick="this.closest('.modal').remove()">×</button>
            </div>
            <div class="modal-body">
                ${content}
            </div>
            <div class="card-footer">
                ${buttonsHtml}
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // Close on outside click
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.remove();
        }
    });
    
    return modal;
}

/**
 * Confirm dialog
 */
function confirm(message, onConfirm, onCancel = null) {
    const modal = createModal(
        'Confirm',
        `<p>${message}</p>`,
        [
            {
                text: 'Cancel',
                type: 'secondary',
                onclick: `this.closest('.modal').remove(); ${onCancel ? onCancel : ''}`
            },
            {
                text: 'Confirm',
                type: 'danger',
                onclick: `this.closest('.modal').remove(); ${onConfirm}`
            }
        ]
    );
    
    return modal;
}

// ============================================================================
// EXPORT
// ============================================================================

// Make functions globally available
window.BiometricAPI = {
    // Auth
    getToken,
    getCurrentUser,
    logout,
    requireAuth,
    
    // API
    apiCall,
    getUsers,
    getUser,
    addUser,
    updateUser,
    deleteUser,
    deleteAllUsers,
    verifyFingerprint,
    identifyFingerprint,
    loadTemplatesFromFolder,
    getStats,
    
    // UI
    showAlert,
    showLoading,
    hideLoading,
    formatTimestamp,
    formatQuality,
    readFileAsBase64,
    readFilesAsBase64,
    createModal,
    confirm
};
