const API_BASE_URL = 'http://localhost';

function msg(message, type = 'default') {
    // 1. Remove existing toast (to achieve override effect)
    const existingToast = document.querySelector('.custom-toast');
    if (existingToast) {
        existingToast.remove();
    }

    // 2. Create toast element
    const toast = document.createElement('div');
    toast.className = `custom-toast ${type}`;
    toast.textContent = message;

    // 3. Inline core CSS styles (translucent light background + black text)
    toast.style.cssText = `
        position: fixed;
        top: 0;
        left: 50%;
        transform: translate(-50%, -100%);
        padding: 12px 24px;
        border-radius: 8px;
        color: #333333; /* Black text (dark grey is softer) */
        font-size: 14px;
        font-weight: 500;
        z-index: 9999;
        transition: all 0.3s ease;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
        margin: 0;
        border: 1px solid rgba(0, 0, 0, 0.05);
        backdrop-filter: blur(8px); /* Frosted glass effect (optional) */
    `;

    // 4. Set translucent light background according to type
    const bgColorMap = {
        default: 'rgba(219, 234, 254, 0.9)', // Light blue (translucent)
        success: 'rgba(220, 252, 231, 0.9)', // Light green (translucent)
        warn: 'rgba(254, 243, 199, 0.9)',    // Light yellow (translucent)
        error: 'rgba(254, 226, 226, 0.9)'    // Light red (translucent)
    };
    toast.style.backgroundColor = bgColorMap[type] || bgColorMap.default;

    // 5. Add to page and trigger display animation
    document.body.appendChild(toast);
    setTimeout(() => {
        toast.style.transform = 'translate(-50%, 20px)'; // Slide in from top
    }, 10);

    // 6. Fade out and remove after 1 second
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translate(-50%, 10px)'; // Move up slightly + transparent
        setTimeout(() => {
            toast.remove(); // Remove DOM completely
        }, 300); // Wait for transition animation to complete
    }, 1000); // Fade out after 1 second
}
class AuthManager {
    constructor() {
        this.token = localStorage.getItem('access_token');
        this.currentUser = JSON.parse(localStorage.getItem('current_user') || 'null');
        this.init();
    }

    init() {
        this.bindEvents();
        this.checkAuthStatus();
    }

    bindEvents() {
        document.getElementById('show-register').addEventListener('click', (e) => {
            e.preventDefault();
            this.showRegisterForm();
        });

        document.getElementById('show-login').addEventListener('click', (e) => {
            e.preventDefault();
            this.showLoginForm();
        });

        document.getElementById('loginForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleLogin(e.target);
        });

        document.getElementById('registerForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleRegister(e.target);
        });

        document.getElementById('logout-btn').addEventListener('click', () => {
            this.handleLogout();
        });
    }

    showLoginForm() {
        document.getElementById('login-form').classList.remove('hidden');
        document.getElementById('register-form').classList.add('hidden');
        document.getElementById('dashboard').classList.add('hidden');
    }

    showRegisterForm() {
        document.getElementById('login-form').classList.add('hidden');
        document.getElementById('register-form').classList.remove('hidden');
        document.getElementById('dashboard').classList.add('hidden');
    }

    showDashboard() {
        document.getElementById('login-form').classList.add('hidden');
        document.getElementById('register-form').classList.add('hidden');
        document.getElementById('dashboard').classList.remove('hidden');
    }

    showNotification(message, type = 'info') {
        const notification = document.getElementById('notification');
        notification.textContent = message;
        notification.className = `notification ${type} show`;

        setTimeout(() => {
            notification.classList.remove('show');
        }, 5000);
    }

    async handleLogin(form) {
        const formData = new FormData(form);
        const data = {
            email: formData.get('email'),
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
                localStorage.setItem('access_token', this.token);

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

        if (password !== confirmPassword) {
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
                this.showNotification('注册成功！请登录', 'success');
                this.showLoginForm();
                form.reset();
            } else {
                this.showNotification(result.detail || '注册失败', 'error');
            }
        } catch (error) {
            console.error('Register error:', error);
            this.showNotification('网络错误，请稍后重试', 'error');
        }
    }

    async fetchUserInfo() {
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
            document.getElementById('user-username').textContent = this.currentUser.username;
            document.getElementById('user-email').textContent = this.currentUser.email;
            document.getElementById('user-status').textContent = this.currentUser.is_active ? '活跃' : '未激活';
            document.getElementById('token-display').textContent = this.token || '未获取到 Token';
        }
    }

    handleLogout() {
        this.token = null;
        this.currentUser = null;
        localStorage.removeItem('access_token');
        localStorage.removeItem('current_user');

        this.showNotification('已退出登录', 'success');
        this.showLoginForm();
        document.getElementById('loginForm').reset();
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

document.addEventListener('DOMContentLoaded', () => {
    window.authManager = new AuthManager();
});