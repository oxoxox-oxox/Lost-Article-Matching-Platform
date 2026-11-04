/**
 * Lost Article Matching Platform (LAMP)
 * Global Frontend Utilities & Authentication Helper
 */

// 1. Dynamic API Base URL: Adapts to localhost, custom ports, LAN IPs, and production domains
const API_BASE_URL = (typeof window !== 'undefined' && window.location && window.location.origin)
    ? window.location.origin
    : '';

if (typeof window !== 'undefined') {
    window.API_BASE_URL = API_BASE_URL;
}

// 2. Unified Token Helpers: Harmonize dual 'token' and 'access_token' local storage keys
function getAuthToken() {
    try {
        return localStorage.getItem('token') || localStorage.getItem('access_token') || '';
    } catch (e) {
        return '';
    }
}

function setAuthToken(token) {
    try {
        if (token) {
            localStorage.setItem('token', token);
            localStorage.setItem('access_token', token);
        } else {
            clearAuthToken();
        }
    } catch (e) {
        console.error('Failed to set auth token:', e);
    }
}

function clearAuthToken() {
    try {
        localStorage.removeItem('token');
        localStorage.removeItem('access_token');
        localStorage.removeItem('current_user');
    } catch (e) {
        console.error('Failed to clear auth token:', e);
    }
}

// Expose token helpers to global window object
if (typeof window !== 'undefined') {
    window.getAuthToken = getAuthToken;
    window.setAuthToken = setAuthToken;
    window.clearAuthToken = clearAuthToken;
}

// 3. Global Toast Notification System (default 3000ms duration per UX best practices)
function msg(message, type = 'default', duration = 3000) {
    if (typeof document === 'undefined') return;

    // 1. Remove existing toast if present
    const existingToast = document.querySelector('.custom-toast');
    if (existingToast) {
        existingToast.remove();
    }

    // 2. Create toast element
    const toast = document.createElement('div');
    toast.className = `custom-toast ${type}`;
    toast.textContent = message;

    // 3. Styling with frosted glass effect and smooth spring transition
    toast.style.cssText = `
        position: fixed;
        top: 0;
        left: 50%;
        transform: translate(-50%, -100%);
        padding: 12px 24px;
        border-radius: 8px;
        color: #1f2937;
        font-size: 14px;
        font-weight: 500;
        z-index: 99999;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
        margin: 0;
        border: 1px solid rgba(0, 0, 0, 0.06);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        pointer-events: none;
    `;

    // 4. Color theme mapping
    const bgColorMap = {
        default: 'rgba(219, 234, 254, 0.95)', // Light blue
        success: 'rgba(220, 252, 231, 0.95)', // Light green
        warn: 'rgba(254, 243, 199, 0.95)',    // Light yellow
        error: 'rgba(254, 226, 226, 0.95)'    // Light red
    };
    toast.style.backgroundColor = bgColorMap[type] || bgColorMap.default;

    // 5. Mount and animate in
    document.body.appendChild(toast);
    requestAnimationFrame(() => {
        toast.style.transform = 'translate(-50%, 20px)';
    });

    // 6. Graceful slide-up and fade-out after duration
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translate(-50%, 0px)';
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 300);
    }, duration);
}

if (typeof window !== 'undefined') {
    window.msg = msg;
}

// 4. Defensive AuthManager: Only operates if auth-related elements exist on current page
class AuthManager {
    constructor() {
        this.token = getAuthToken();
        try {
            this.currentUser = JSON.parse(localStorage.getItem('current_user') || 'null');
        } catch (e) {
            this.currentUser = null;
        }
        this.init();
    }

    init() {
        // Core defensive check: silently exit if current page has no auth form elements
        if (!this.hasAuthElements()) {
            return;
        }
        this.bindEvents();
        this.checkAuthStatus();
    }

    hasAuthElements() {
        if (typeof document === 'undefined') return false;
        return !!(
            document.getElementById('show-register') ||
            document.getElementById('show-login') ||
            document.getElementById('loginForm') ||
            document.getElementById('registerForm') ||
            document.getElementById('login-form') ||
            document.getElementById('register-form') ||
            document.getElementById('dashboard')
        );
    }

    _safeAddListener(id, event, handler) {
        if (typeof document === 'undefined') return;
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener(event, handler);
        }
    }

    _safeClassListToggle(id, addClass, removeClass) {
        if (typeof document === 'undefined') return;
        const el = document.getElementById(id);
        if (el) {
            if (removeClass) el.classList.remove(removeClass);
            if (addClass) el.classList.add(addClass);
        }
    }

    _safeSetText(id, text) {
        if (typeof document === 'undefined') return;
        const el = document.getElementById(id);
        if (el) {
            el.textContent = text;
        }
    }

    bindEvents() {
        this._safeAddListener('show-register', 'click', (e) => {
            e.preventDefault();
            this.showRegisterForm();
        });

        this._safeAddListener('show-login', 'click', (e) => {
            e.preventDefault();
            this.showLoginForm();
        });

        this._safeAddListener('loginForm', 'submit', (e) => {
            e.preventDefault();
            this.handleLogin(e.target);
        });

        this._safeAddListener('registerForm', 'submit', (e) => {
            e.preventDefault();
            this.handleRegister(e.target);
        });

        this._safeAddListener('logout-btn', 'click', () => {
            this.handleLogout();
        });
    }

    showLoginForm() {
        this._safeClassListToggle('login-form', null, 'hidden');
        this._safeClassListToggle('register-form', 'hidden', null);
        this._safeClassListToggle('dashboard', 'hidden', null);
    }

    showRegisterForm() {
        this._safeClassListToggle('login-form', 'hidden', null);
        this._safeClassListToggle('register-form', null, 'hidden');
        this._safeClassListToggle('dashboard', 'hidden', null);
    }

    showDashboard() {
        this._safeClassListToggle('login-form', 'hidden', null);
        this._safeClassListToggle('register-form', 'hidden', null);
        this._safeClassListToggle('dashboard', null, 'hidden');
    }

    showNotification(message, type = 'info') {
        const notification = document.getElementById('notification');
        if (notification) {
            notification.textContent = message;
            notification.className = `notification ${type} show`;
            setTimeout(() => {
                notification.classList.remove('show');
            }, 4000);
        } else if (typeof msg === 'function') {
            msg(message, type === 'error' ? 'error' : (type === 'success' ? 'success' : 'default'));
        }
    }

    async handleLogin(form) {
        const formData = new FormData(form);
        const data = {
            username_or_email: formData.get('email') || formData.get('username_or_email'),
            password: formData.get('password')
        };

        try {
            const response = await fetch(`${API_BASE_URL}/auth/login`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (response.ok) {
                this.token = result.access_token;
                setAuthToken(this.token);

                await this.fetchUserInfo();
                this.showNotification('登录成功！', 'success');
                this.showDashboard();
            } else {
                this.showNotification(result.detail || '登录失败', 'error');
            }
        } catch (error) {
            console.error('Login error:', error);
            this.showNotification('网络错误，请稍后重试', 'error');
        }
    }

    async handleRegister(form) {
        const formData = new FormData(form);
        const password = formData.get('password');
        const confirmPassword = formData.get('confirmPassword');

        if (confirmPassword !== null && password !== confirmPassword) {
            this.showNotification('两次输入的密码不一致', 'error');
            return;
        }

        const data = {
            username: formData.get('username'),
            email: formData.get('email'),
            password: password,
            honeypot: ''
        };

        try {
            const response = await fetch(`${API_BASE_URL}/auth/register`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (response.ok) {
                this.showNotification('注册成功！请前往邮箱查收验证码', 'success');
                this.showLoginForm();
                if (form.reset) form.reset();
            } else {
                this.showNotification(result.detail || '注册失败', 'error');
            }
        } catch (error) {
            console.error('Register error:', error);
            this.showNotification('网络错误，请稍后重试', 'error');
        }
    }

    async fetchUserInfo() {
        if (!this.token) return;
        try {
            const response = await fetch(`${API_BASE_URL}/auth/me`, {
                method: 'GET',
                headers: {
                    'Authorization': `Bearer ${this.token}`,
                    'Content-Type': 'application/json'
                }
            });

            if (response.ok) {
                const user = await response.json();
                this.currentUser = user;
                localStorage.setItem('current_user', JSON.stringify(user));
                this.updateDashboard();
            }
        } catch (error) {
            console.error('Fetch user info error:', error);
        }
    }

    updateDashboard() {
        if (this.currentUser) {
            this._safeSetText('user-username', this.currentUser.username || '');
            this._safeSetText('user-email', this.currentUser.email || '');
            this._safeSetText('user-status', this.currentUser.is_active ? '活跃' : '未激活');
            this._safeSetText('token-display', this.token || '未获取到 Token');
        }
    }

    handleLogout() {
        this.token = null;
        this.currentUser = null;
        clearAuthToken();

        this.showNotification('已退出登录', 'success');
        this.showLoginForm();
        const loginForm = document.getElementById('loginForm');
        if (loginForm && loginForm.reset) {
            loginForm.reset();
        }
    }

    checkAuthStatus() {
        if (this.token) {
            this.fetchUserInfo().then(() => {
                if (this.currentUser) {
                    this.showDashboard();
                } else {
                    this.showLoginForm();
                }
            });
        } else {
            this.showLoginForm();
        }
    }

    async apiRequest(endpoint, options = {}) {
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
                ...(this.token && { 'Authorization': `Bearer ${this.token}` })
            }
        };

        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            ...defaultOptions,
            ...options,
            headers: {
                ...defaultOptions.headers,
                ...options.headers
            }
        });

        if (response.status === 401) {
            this.handleLogout();
            throw new Error('未授权，请重新登录');
        }

        return response;
    }
}

if (typeof window !== 'undefined') {
    window.AuthManager = AuthManager;
}

// Global initialization
if (typeof document !== 'undefined') {
    document.addEventListener('DOMContentLoaded', () => {
        try {
            window.authManager = new AuthManager();
        } catch (err) {
            console.error('AuthManager initialization error:', err);
        }
    });
}
