// static/js/theme-toggle.js

const themeToggle = document.getElementById('themeToggle');
const root = document.documentElement;

// Apply saved theme on load
const savedTheme = localStorage.getItem('theme');
if (savedTheme === 'dark') {
    root.setAttribute('data-bs-theme', 'dark');
    themeToggle.checked = true;
}

// Toggle handler
themeToggle?.addEventListener('change', () => {
    const newTheme = themeToggle.checked ? 'dark' : 'light';
    root.setAttribute('data-bs-theme', newTheme);
    localStorage.setItem('theme', newTheme);
});
