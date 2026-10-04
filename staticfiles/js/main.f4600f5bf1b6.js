/* SmartBiz AI — Main JavaScript */

// ── Sidebar toggle ────────────────────────────────────────────
(function () {
    const sidebar = document.getElementById('sidebar');
    const mainWrapper = document.getElementById('mainWrapper');
    const topnavToggle = document.getElementById('topnavToggle');
    const sidebarToggle = document.getElementById('sidebarToggle');

    // Create overlay for mobile
    const overlay = document.createElement('div');
    overlay.className = 'sidebar-overlay';
    document.body.appendChild(overlay);

    function openSidebar() {
        sidebar.classList.add('open');
        overlay.classList.add('visible');
    }

    function closeSidebar() {
        sidebar.classList.remove('open');
        overlay.classList.remove('visible');
    }

    if (topnavToggle) {
        topnavToggle.addEventListener('click', () => {
            if (sidebar.classList.contains('open')) {
                closeSidebar();
            } else {
                openSidebar();
            }
        });
    }

    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', closeSidebar);
    }

    overlay.addEventListener('click', closeSidebar);
})();

// ── Alert auto-dismiss ────────────────────────────────────────
document.querySelectorAll('.alert').forEach(function (alert) {
    setTimeout(function () {
        alert.style.opacity = '0';
        alert.style.transition = 'opacity 0.4s ease';
        setTimeout(function () { alert.remove(); }, 400);
    }, 5000);
});

// ── Number counter animation for stat cards ───────────────────
function animateCounter(el) {
    const target = parseFloat(el.getAttribute('data-target') || el.textContent.replace(/[^0-9.]/g, ''));
    if (isNaN(target)) return;

    const isFloat = el.getAttribute('data-decimals') === 'true';
    const prefix = el.getAttribute('data-prefix') || '';
    const suffix = el.getAttribute('data-suffix') || '';
    const duration = 1200;
    const start = performance.now();

    function update(now) {
        const elapsed = now - start;
        const progress = Math.min(elapsed / duration, 1);
        const ease = 1 - Math.pow(1 - progress, 3);
        const current = target * ease;
        el.textContent = prefix + (isFloat ? current.toFixed(1) : Math.round(current).toLocaleString()) + suffix;
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

// Run counters on page load for elements with data-target
document.querySelectorAll('[data-target]').forEach(function (el) {
    animateCounter(el);
});

// ── Chart.js global defaults ──────────────────────────────────
if (typeof Chart !== 'undefined') {
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.font.family = "'Inter', sans-serif";
    Chart.defaults.plugins.legend.labels.color = '#94a3b8';
    Chart.defaults.plugins.legend.labels.boxWidth = 12;
    Chart.defaults.plugins.legend.labels.padding = 16;
    Chart.defaults.plugins.tooltip.backgroundColor = '#161d2e';
    Chart.defaults.plugins.tooltip.borderColor = 'rgba(99,102,241,0.3)';
    Chart.defaults.plugins.tooltip.borderWidth = 1;
    Chart.defaults.plugins.tooltip.padding = 10;
    Chart.defaults.plugins.tooltip.titleColor = '#f1f5f9';
    Chart.defaults.plugins.tooltip.bodyColor = '#94a3b8';
    Chart.defaults.scale.grid.color = 'rgba(99,102,241,0.08)';
    Chart.defaults.scale.ticks.color = '#64748b';
}

// ── Utility: create a doughnut chart ─────────────────────────
function makeDoughnut(id, labels, data, colors) {
    const ctx = document.getElementById(id);
    if (!ctx || !data || data.length === 0) return null;
    return new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: colors || defaultPalette(data.length),
                borderWidth: 0,
                hoverOffset: 6,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '68%',
            plugins: {
                legend: { position: 'right' }
            }
        }
    });
}

// ── Utility: create a bar chart ───────────────────────────────
function makeBar(id, labels, data, label, color) {
    const ctx = document.getElementById(id);
    if (!ctx || !data || data.length === 0) return null;
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: label || 'Count',
                data: data,
                backgroundColor: color || 'rgba(99,102,241,0.7)',
                borderColor: color || 'rgba(99,102,241,1)',
                borderWidth: 0,
                borderRadius: 6,
                borderSkipped: false,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true, ticks: { precision: 0 } }
            }
        }
    });
}

// ── Utility: create a line chart ─────────────────────────────
function makeLine(id, labels, data, label) {
    const ctx = document.getElementById(id);
    if (!ctx || !data || data.length === 0) return null;
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: label || 'Count',
                data: data,
                borderColor: 'rgba(99,102,241,1)',
                backgroundColor: 'rgba(99,102,241,0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointBackgroundColor: 'rgba(99,102,241,1)',
                pointRadius: 4,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: { y: { beginAtZero: true, ticks: { precision: 0 } } }
        }
    });
}

// ── Default color palette ─────────────────────────────────────
function defaultPalette(n) {
    const palette = [
        'rgba(99,102,241,0.85)', 'rgba(6,182,212,0.85)', 'rgba(16,185,129,0.85)',
        'rgba(245,158,11,0.85)', 'rgba(244,63,94,0.85)', 'rgba(139,92,246,0.85)',
        'rgba(59,130,246,0.85)', 'rgba(236,72,153,0.85)',
    ];
    return palette.slice(0, n);
}
