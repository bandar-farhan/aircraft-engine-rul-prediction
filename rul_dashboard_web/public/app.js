const state = {
    payload: null,
    engines: null,
    schedule: null,
    sessionId: getSessionId(),
    charts: {}
};

const HEALTH = {
    critical: {
        key: 'immediate',
        label: 'Immediate Check',
        shortLabel: 'Needs Immediate Check',
        note: 'Predicted 20 cycles or less',
        empty: 'Pending'
    },
    high: {
        key: 'soon',
        label: 'Upcoming Service',
        shortLabel: 'Upcoming',
        note: 'Predicted 21-50 cycles',
        empty: 'Pending'
    },
    medium: {
        key: 'monitor',
        label: 'Monitor',
        shortLabel: 'Monitor',
        note: 'Predicted 51-90 cycles',
        empty: 'Pending'
    },
    low: {
        key: 'healthy',
        label: 'Healthy',
        shortLabel: 'Healthy',
        note: 'Predicted more than 90 cycles',
        empty: 'Pending'
    }
};

const HEALTH_ORDER = ['critical', 'high', 'medium', 'low'];
const $ = (id) => document.getElementById(id);

document.addEventListener('DOMContentLoaded', () => {
    bindEvents();
    bindActiveNav();
    addAssistantMessage('Ask me about engine health, maintenance planning, or which engines need attention first. If engine data is missing, I will say that clearly.');
    refreshAll();
});

function bindEvents() {
    if ($('stopServerBtn')) $('stopServerBtn').addEventListener('click', stopServer);
    if ($('prioritySubset')) $('prioritySubset').addEventListener('change', renderEngines);
    if ($('clearChatBtn')) $('clearChatBtn').addEventListener('click', clearChat);

    document.querySelectorAll('.prompt-chip').forEach((button) => {
        button.addEventListener('click', () => {
            $('chatInput').value = button.textContent.trim();
            $('chatInput').focus();
        });
    });

    $('chatForm').addEventListener('submit', sendChat);
}

function bindActiveNav() {
    const links = document.querySelectorAll('.nav-list a');
    const sections = [];

    // Click handler for instant response
    links.forEach(link => {
        link.addEventListener('click', () => {
            links.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
        });

        const id = link.getAttribute('href')?.replace('#', '');
        if (id) {
            const section = document.getElementById(id);
            if (section) sections.push({ link, section });
        }
    });

    if (!sections.length) return;

    // Scroll handler for updating as user scrolls
    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            if (entry.isIntersecting) {
                links.forEach((l) => l.classList.remove('active'));
                const match = sections.find((s) => s.section === entry.target);
                if (match) match.link.classList.add('active');
            }
        });
    }, { rootMargin: '-30% 0px -50% 0px' });

    sections.forEach((s) => observer.observe(s.section));

    // Fallback for when the user scrolls to the absolute bottom
    window.addEventListener('scroll', () => {
        if ((window.innerHeight + Math.round(window.scrollY)) >= document.body.offsetHeight - 2) {
            links.forEach(l => l.classList.remove('active'));
            if (links.length > 0) {
                links[links.length - 1].classList.add('active');
            }
        }
    });
}

function getSessionId() {
    const key = 'rul_dashboard_session';
    let value = localStorage.getItem(key);
    if (!value) {
        value = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
        localStorage.setItem(key, value);
    }
    return value;
}

function clearChat() {
    const key = 'rul_dashboard_session';
    const newValue = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    localStorage.setItem(key, newValue);
    state.sessionId = newValue;
    $('chatMessages').innerHTML = '';
    addAssistantMessage('Chat history cleared! Ask me about engine health, maintenance planning, or which engines need attention first.');
}

async function api(path, options = {}) {
    const response = await fetch(path, {
        headers: { 'Content-Type': 'application/json' },
        ...options
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `Request failed: ${response.status}`);
    return data;
}

async function refreshAll() {
    setServerStatus('Checking', 'Loading dashboard data', '');
    try {
        const [payload, engines, schedule] = await Promise.all([
            api('/api/results'),
            api('/api/engines'),
            api('/api/schedule')
        ]);

        state.payload = payload;
        state.engines = engines;
        state.schedule = schedule;
        renderAll();
        setServerStatus('Ready', 'Server online', 'ok');
        $('aiStatus').textContent = 'Ready';
    } catch (error) {
        setServerStatus('Offline', error.message, 'bad');
    }
}

function setServerStatus(title, detail, mode) {
    $('serverStatus').textContent = title;
    $('serverDetail').textContent = detail;
    $('serverDot').className = `server-dot ${mode || ''}`.trim();
}

function renderAll() {
    renderHealthOverview();
    renderEngines();
    renderSchedule();
    renderInteractiveCharts();
}

function hasEngineData() {
    return getPriorityRows().length > 0;
}

function getPriorityRows() {
    if (!state.engines) return [];
    return Object.values(state.engines).flatMap((subset) => subset.priority?.rows || []);
}

function getScheduleRows() {
    if (!state.schedule) return [];
    return Object.values(state.schedule).flatMap((table) => table.rows || []);
}

function getHealthCounts(rows = getPriorityRows()) {
    const counts = { critical: 0, high: 0, medium: 0, low: 0 };
    rows.forEach((row) => {
        if (counts[row.maintenance_priority] !== undefined) {
            counts[row.maintenance_priority]++;
        }
    });
    return counts;
}

// -- Interactive Chart.js Rendering --
function renderInteractiveCharts() {
    const rows = getPriorityRows().filter((row) => Number.isFinite(Number(row.ensemble_predicted_RUL)));
    if (!rows.length || typeof Chart === 'undefined') {
        ['fleetHealthChart', 'criticalEnginesChart', 'fleetActionChart'].forEach(id => {
            const ctx = $(id);
            if (ctx && !state.charts[id]) {
                const parent = ctx.parentElement;
                parent.innerHTML = '<div class="empty" style="border:none; background:transparent;">No engine data available yet.</div>';
            }
        });
        return;
    }

    renderFleetDonut(rows);
    renderTopCriticalEngines(rows);
    renderFleetActions(rows);
}

function renderFleetDonut(rows) {
    const counts = getHealthCounts(rows);
    const data = [counts.critical, counts.high, counts.medium, counts.low];
    const labels = [HEALTH.critical.label, HEALTH.high.label, HEALTH.medium.label, HEALTH.low.label];
    // Colors match our CSS variables roughly
    const colors = ['#d46a62', '#d9a64a', '#6f92be', '#6fae86'];

    const ctx = $('fleetHealthChart');
    if (!ctx) return;

    if (state.charts.fleet) state.charts.fleet.destroy();

    state.charts.fleet = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: colors,
                borderWidth: 0,
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'right', labels: { boxWidth: 12, font: { size: 11, family: 'Segoe UI' } } }
            },
            cutout: '65%'
        }
    });
}

function renderTopCriticalEngines(rows) {
    // Sort by lowest RUL first, grab top 6
    const topEngines = rows
        .slice()
        .sort((a, b) => Number(a.ensemble_predicted_RUL) - Number(b.ensemble_predicted_RUL))
        .slice(0, 6);

    const labels = topEngines.map(row => `${row.subset} E${row.engine_id}`);
    const data = topEngines.map(row => Number(row.ensemble_predicted_RUL));

    // Color them based on priority
    const colors = topEngines.map(row => {
        if (row.maintenance_priority === 'critical') return '#d46a62';
        if (row.maintenance_priority === 'high') return '#d9a64a';
        if (row.maintenance_priority === 'medium') return '#6f92be';
        return '#6fae86';
    });

    const ctx = $('criticalEnginesChart');
    if (!ctx) return;

    if (state.charts.critical) state.charts.critical.destroy();

    state.charts.critical = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Cycles Left',
                data: data,
                backgroundColor: colors,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { beginAtZero: true, grid: { color: '#e6eaf0' }, ticks: { font: { size: 10 } } },
                y: { grid: { display: false }, ticks: { font: { size: 10, family: 'Segoe UI' } } }
            }
        }
    });
}

function renderFleetActions(rows) {
    const subsetMap = {};
    rows.forEach(row => {
        const subset = row.subset || 'Unknown';
        if (!subsetMap[subset]) {
            subsetMap[subset] = { critical: 0, high: 0, medium: 0 };
        }
        if (row.maintenance_priority === 'critical') subsetMap[subset].critical++;
        else if (row.maintenance_priority === 'high') subsetMap[subset].high++;
        else if (row.maintenance_priority === 'medium') subsetMap[subset].medium++;
    });

    const labels = Object.keys(subsetMap).sort();
    const criticalData = labels.map(l => subsetMap[l].critical);
    const highData = labels.map(l => subsetMap[l].high);
    const monitorData = labels.map(l => subsetMap[l].medium);

    const ctx = $('fleetActionChart');
    if (!ctx) return;

    if (state.charts.action) state.charts.action.destroy();

    state.charts.action = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                { label: 'Immediate', data: criticalData, backgroundColor: '#d46a62', borderRadius: 2 },
                { label: 'Upcoming Service', data: highData, backgroundColor: '#d9a64a', borderRadius: 2 },
                { label: 'Monitor', data: monitorData, backgroundColor: '#6f92be', borderRadius: 2 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 11, family: 'Segoe UI' } } }
            },
            scales: {
                x: { stacked: true, grid: { display: false }, ticks: { font: { size: 10 } } },
                y: { stacked: true, beginAtZero: true, grid: { color: '#e6eaf0' }, ticks: { font: { size: 10 } } }
            }
        }
    });
}

function renderHealthOverview() {
    const rows = getPriorityRows()
        .filter((row) => Number.isFinite(Number(row.ensemble_predicted_RUL)))
        .sort((a, b) => Number(a.ensemble_predicted_RUL) - Number(b.ensemble_predicted_RUL));
    const ready = rows.length > 0;
    const counts = getHealthCounts(rows);

    if (ready) {
        const first = rows[0];
        const health = HEALTH[first.maintenance_priority] || HEALTH.medium;
        $('nextEngineTitle').textContent = `Check ${first.subset} Engine ${first.engine_id} first`;
        $('nextEngineText').textContent = `${health.label}: predicted ${formatNumber(first.ensemble_predicted_RUL)} cycles left. ${first.suggested_action}`;
    } else {
        $('nextEngineTitle').textContent = 'Engine health data is not ready yet';
        $('nextEngineText').textContent = 'Engine health, priority, and maintenance planning data is not ready yet.';
    }

    $('healthSummary').innerHTML = HEALTH_ORDER.map((priority) => {
        const health = HEALTH[priority];
        return `
            <article class="health-card ${health.key}">
                <span>${escapeHtml(health.shortLabel)}</span>
                <strong>${ready ? counts[priority] : health.empty}</strong>
                <small>${escapeHtml(health.note)}</small>
            </article>
        `;
    }).join('');

    renderHealthDistribution(counts, ready);
}

function renderHealthDistribution(counts, ready) {
    if (!ready) {
        $('healthDistribution').innerHTML = `
            <div class="empty">Engine health groups are not ready yet. Until then, no engine health count is shown as real.</div>
        `;
        return;
    }

    const total = Object.values(counts).reduce((sum, value) => sum + value, 0);
    $('healthDistribution').innerHTML = HEALTH_ORDER.map((priority) => {
        const health = HEALTH[priority];
        const value = counts[priority];
        const width = total ? Math.max((value / total) * 100, value > 0 ? 3 : 0) : 0;
        return `
            <div class="health-bar-row">
                <div class="health-bar-label">${escapeHtml(health.label)}</div>
                <div class="health-bar-track">
                    <div class="health-bar-fill ${health.key}" style="--width:${width}%"></div>
                </div>
                <div class="bar-value">${value}</div>
            </div>
        `;
    }).join('');
}

function renderEngines() {
    const subset = $('prioritySubset').value;
    const rows = getPriorityRows()
        .filter((row) => subset === 'all' || row.subset === subset)
        .filter((row) => Number.isFinite(Number(row.ensemble_predicted_RUL)))
        .sort((a, b) => Number(a.ensemble_predicted_RUL) - Number(b.ensemble_predicted_RUL));

    if (!rows.length) {
        $('engineEmpty').classList.remove('hidden');
        $('engineTableWrap').classList.add('hidden');
        $('engineEmpty').textContent = 'Engine health data is not ready yet. Engine ranking will appear here when real prediction data is available.';
        return;
    }

    $('engineEmpty').classList.add('hidden');
    $('engineTableWrap').classList.remove('hidden');
    $('priorityRows').innerHTML = rows.map((row) => {
        const health = HEALTH[row.maintenance_priority] || HEALTH.medium;
        return `
            <tr>
                <td>${escapeHtml(row.subset)}</td>
                <td>Engine ${escapeHtml(row.engine_id)}</td>
                <td><span class="health-tag ${health.key}">${escapeHtml(health.label)}</span></td>
                <td>${formatNumber(row.ensemble_predicted_RUL)}</td>
                <td>${formatNumber(row.actual_RUL)}</td>
                <td>${escapeHtml(row.suggested_action)}</td>
            </tr>
        `;
    }).join('');
}

function renderSchedule() {
    const rows = getScheduleRows();

    if (!rows.length) {
        $('scheduleGrid').innerHTML = `
            <div class="empty">No maintenance plan is available yet. The plan will appear here when real engine prediction data is available.</div>
        `;
        return;
    }

    const priorityOrder = { critical: 0, high: 1, medium: 2, low: 3 };
    $('scheduleGrid').innerHTML = rows
        .slice()
        .sort((a, b) => (priorityOrder[a.maintenance_priority] ?? 9) - (priorityOrder[b.maintenance_priority] ?? 9))
        .map((row) => {
            const health = HEALTH[row.maintenance_priority] || HEALTH.medium;
            return `
                <article class="plan-card">
                    <span class="health-tag ${health.key}">${escapeHtml(health.label)}</span>
                    <h3>${escapeHtml(row.subset)} - ${escapeHtml(row.suggested_schedule_window)}</h3>
                    <p><strong>${escapeHtml(row.engine_count)}</strong> engines. Average predicted cycles left: ${formatNumber(row.avg_predicted_RUL)}.</p>
                    <p class="engines">Engines: ${escapeHtml(String(row.engines || ''))}</p>
                </article>
            `;
        }).join('');
}

async function startRun(path, label) {
    if (!confirm(`${label}. This can take a long time on CPU. Continue?`)) return;

    try {
        await api(path, { method: 'POST', body: JSON.stringify({}) });
        setServerStatus('Running', label, 'ok');
    } catch (error) {
        setServerStatus('Run blocked', error.message, 'bad');
    }
}

async function stopServer() {
    if (!confirm('Stop the Engine Health dashboard server now? The website will be unavailable until you run start.cmd again.')) return;

    try {
        const result = await api('/api/shutdown', { method: 'POST', body: JSON.stringify({}) });
        setServerStatus('Stopping', result.message || 'Server is stopping', 'bad');
    } catch (error) {
        setServerStatus('Stop blocked', error.message, 'bad');
    }
}

async function sendChat(event) {
    event.preventDefault();
    const input = $('chatInput');
    const message = input.value.trim();
    if (!message) return;

    input.value = '';
    addUserMessage(message);
    const pending = addAssistantMessage('Thinking...');
    $('aiStatus').textContent = 'Working...';

    try {
        const data = await api('/api/chat', {
            method: 'POST',
            body: JSON.stringify({ message, sessionId: state.sessionId })
        });
        pending.textContent = data.reply;
        $('aiStatus').textContent = 'Ready';
    } catch (error) {
        pending.classList.add('error');
        pending.textContent = error.message;
        $('aiStatus').textContent = 'Error';
    }
}

function addUserMessage(text) {
    addMessage(text, 'user');
}

function addAssistantMessage(text) {
    return addMessage(text, 'assistant');
}

function addMessage(text, type) {
    const node = document.createElement('div');
    node.className = `message ${type}`;
    node.textContent = text;
    $('chatMessages').appendChild(node);
    $('chatMessages').scrollTop = $('chatMessages').scrollHeight;
    return node;
}

function formatNumber(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return 'Not ready';
    return number.toLocaleString(undefined, { maximumFractionDigits: 3 });
}

function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}
