/* ── AquaPredict AI — main.js ── */

let histChart    = null;
let compareChart = null;
let forecastChart = null;

// ═══════════════════════════════════════════
// NAVIGATION
// ═══════════════════════════════════════════
function showSection(name) {
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
  document.getElementById('section-' + name).classList.add('active');
  document.getElementById('pill-' + name).classList.add('active');

  if (name === 'history') loadHistory();
  if (name === 'compare') loadCompare();
}

// ═══════════════════════════════════════════
// METRICS
// ═══════════════════════════════════════════
async function loadMetrics() {
  const r = await fetch('/api/metrics');
  const d = await r.json();
  document.getElementById('m-r2').textContent   = d.r2;
  document.getElementById('m-rmse').textContent = d.rmse + ' m';
  document.getElementById('m-mae').textContent  = d.mae  + ' m';
}

// ═══════════════════════════════════════════
// PREDICTION
// ═══════════════════════════════════════════
const DISTRICT_CONFIG = {
    'Jaisalmer': { rf: 110, tmp: 46.5, pop: 600000, agri: 150000 },
    'Barmer': { rf: 140, tmp: 46.0, pop: 2600000, agri: 400000 },
    'Bikaner': { rf: 135, tmp: 45.5, pop: 2400000, agri: 420000 },
    'Jodhpur': { rf: 190, tmp: 44.0, pop: 3700000, agri: 480000 },
    'Nagaur': { rf: 195, tmp: 44.5, pop: 3300000, agri: 550000 },
    'Sri Ganganagar': { rf: 210, tmp: 43.0, pop: 1900000, agri: 700000 },
    'Hanumangarh': { rf: 200, tmp: 42.5, pop: 1800000, agri: 680000 },
    'Churu': { rf: 215, tmp: 44.0, pop: 2000000, agri: 490000 },
    'Sikar': { rf: 220, tmp: 43.5, pop: 2700000, agri: 500000 },
    'Jhunjhunu': { rf: 230, tmp: 43.0, pop: 2200000, agri: 450000 },
    'Pali': { rf: 250, tmp: 43.0, pop: 2000000, agri: 380000 },
    'Jalor': { rf: 260, tmp: 43.0, pop: 1800000, agri: 360000 },
    'Ajmer': { rf: 300, tmp: 41.5, pop: 2600000, agri: 420000 },
    'Bhilwara': { rf: 340, tmp: 40.5, pop: 2500000, agri: 460000 },
    'Sirohi': { rf: 320, tmp: 40.0, pop: 1100000, agri: 300000 },
    'Rajsamand': { rf: 360, tmp: 39.5, pop: 1200000, agri: 280000 },
    'Tonk': { rf: 360, tmp: 40.5, pop: 1500000, agri: 310000 },
    'Jaipur': { rf: 390, tmp: 40.0, pop: 6700000, agri: 530000 },
    'Alwar': { rf: 590, tmp: 38.5, pop: 3700000, agri: 490000 }
};

function updateDistrictValues() {
    const d = document.getElementById('p-district').value;
    const config = DISTRICT_CONFIG[d];
    if (config) {
        document.getElementById('p-rainfall').value = config.rf;
        document.getElementById('p-temp').value = config.tmp;
        document.getElementById('p-pop').value = config.pop;
        document.getElementById('p-agri').value = config.agri;
    }
}

async function runPredict() {
  const btn = document.querySelector('.btn-predict');
  btn.textContent = '⏳ Computing...';
  btn.disabled = true;

  const body = {
    district:         document.getElementById('p-district').value,
    year:             +document.getElementById('p-year').value,
    rainfall:         +document.getElementById('p-rainfall').value,
    temperature:      +document.getElementById('p-temp').value,
    population:       +document.getElementById('p-pop').value,
    agriculture_area: +document.getElementById('p-agri').value,
  };

  try {
    const r = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const d = await r.json();
    if (d.error) { alert('Error: ' + d.error); }
    else { renderResult(d); }
  } catch (e) {
    alert('Network error: ' + e);
  }

  btn.textContent = '⚡ Run AI Prediction';
  btn.disabled = false;
}

function renderResult(d) {
  document.getElementById('result-idle').classList.add('hidden');
  const out = document.getElementById('result-output');
  out.classList.remove('hidden');

  const depth = d.groundwater_level_m;
  const risk  = d.risk;

  // Animated ring (max display = 120m → full circle)
  const pct    = Math.min(depth / 120, 1);
  const circ   = 2 * Math.PI * 85; // ~534
  const offset = circ * (1 - pct);
  const fill   = document.getElementById('ring-fill');
  fill.style.strokeDashoffset = offset;
  fill.style.stroke = riskColor(risk);

  document.getElementById('depth-val').textContent = depth.toFixed(1);

  const badge = document.getElementById('risk-badge');
  badge.textContent = risk + ' Risk';
  badge.className   = 'risk-badge risk-' + risk;

  document.getElementById('r-district').textContent = d.district;
  document.getElementById('r-year').textContent     = d.year;
  document.getElementById('r-depth').textContent    = depth.toFixed(2) + ' m';
  document.getElementById('r-risk').textContent     = risk;

  const bar = document.getElementById('risk-bar-fill');
  bar.style.width      = (pct * 100).toFixed(0) + '%';
  bar.style.background = riskColor(risk);

  const captions = {
    Low:      '✅ Water table is at a manageable depth. Sustainable usage is feasible.',
    Moderate: '⚠️ Moderate depletion detected. Conservation measures are advised.',
    High:     '🔴 High depletion! Immediate water management intervention needed.',
    Critical: '🚨 Critical shortage! Emergency government action urgently required.',
  };
  document.getElementById('risk-caption').textContent = captions[risk] || '';
}

function riskColor(risk) {
  return { Low: '#10b981', Moderate: '#f59e0b', High: '#ef4444', Critical: '#dc2626' }[risk] || '#3b82f6';
}

// ═══════════════════════════════════════════
// HISTORICAL CHART
// ═══════════════════════════════════════════
async function loadHistory() {
  const dist = document.getElementById('h-district').value;
  const r    = await fetch('/api/history?district=' + encodeURIComponent(dist));
  const d    = await r.json();

  if (histChart) histChart.destroy();
  const ctx = document.getElementById('historyChart').getContext('2d');
  histChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: d.years,
      datasets: [{
        label: dist + ' – Groundwater Depth (m)',
        data:  d.levels,
        borderColor: '#3b82f6',
        backgroundColor: 'rgba(59,130,246,.07)',
        borderWidth: 2.5,
        pointRadius: 3,
        pointHoverRadius: 6,
        fill: true,
        tension: 0.4,
      }]
    },
    options: chartOptions('Groundwater Depth Below Surface (meters)')
  });

  const mn  = Math.min(...d.levels).toFixed(2);
  const mx  = Math.max(...d.levels).toFixed(2);
  const avg = (d.levels.reduce((a, b) => a + b, 0) / d.levels.length).toFixed(2);
  const chg = (d.levels[d.levels.length - 1] - d.levels[0]).toFixed(2);
  document.getElementById('hist-stats').innerHTML = `
    <div class="stat-pill">Shallowest <strong>${mn} m</strong></div>
    <div class="stat-pill">Deepest <strong>${mx} m</strong></div>
    <div class="stat-pill">Average <strong>${avg} m</strong></div>
    <div class="stat-pill">Change 1990→2025 <strong>+${chg} m</strong></div>
  `;
}

// ═══════════════════════════════════════════
// COMPARE CHART — all 19 districts
// ═══════════════════════════════════════════
async function loadCompare() {
  const r = await fetch('/api/all_history');
  const d = await r.json();  // [{district, level}, ...]

  const sorted    = [...d].sort((a, b) => b.level - a.level);
  const labels    = sorted.map(x => x.district);
  const values    = sorted.map(x => x.level);
  const maxLevel  = Math.max(...values);
  const colors    = values.map(v => {
    const t = v / maxLevel;
    const r = Math.round(239 * t + 59 * (1 - t));
    const g = Math.round(68 * t + 130 * (1 - t));
    const b = Math.round(68 * t + 246 * (1 - t));
    return `rgba(${r},${g},${b},0.85)`;
  });

  if (compareChart) compareChart.destroy();
  const ctx = document.getElementById('compareChart').getContext('2d');
  compareChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Groundwater Depth in 2025 (meters)',
        data: values,
        backgroundColor: colors,
        borderRadius: 6,
        borderSkipped: false,
      }]
    },
    options: {
      ...chartOptions('Groundwater Depth (m below surface)'),
      plugins: {
        ...chartOptions().plugins,
        legend: { display: false },
      },
      scales: {
        x: { ticks: { color: '#a1a1aa', font: { family: 'Inter', size: 10 }, maxRotation: 40 }, grid: { color: 'rgba(255,255,255,.05)' } },
        y: { title: { display: true, text: 'Depth (m)', color: '#a1a1aa' }, ticks: { color: '#a1a1aa' }, grid: { color: 'rgba(255,255,255,.05)' } }
      }
    }
  });

  const deepest   = sorted[0];
  const shallowest = sorted[sorted.length - 1];
  const avg       = (values.reduce((a, b) => a + b, 0) / values.length).toFixed(2);
  document.getElementById('compare-stats').innerHTML = `
    <div class="stat-pill">Most Depleted <strong>${deepest.district} (${deepest.level} m)</strong></div>
    <div class="stat-pill">Least Depleted <strong>${shallowest.district} (${shallowest.level} m)</strong></div>
    <div class="stat-pill">State Average <strong>${avg} m</strong></div>
    <div class="stat-pill">Districts Compared <strong>19</strong></div>
  `;
}

// ═══════════════════════════════════════════
// FORECAST CHART
// ═══════════════════════════════════════════
async function loadForecast() {
  const dist  = document.getElementById('f-district').value;
  const years = document.getElementById('f-years').value;
  const r     = await fetch(`/api/forecast?district=${encodeURIComponent(dist)}&years=${years}`);
  const d     = await r.json();

  if (forecastChart) forecastChart.destroy();
  const ctx = document.getElementById('forecastChart').getContext('2d');
  forecastChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: d.years,
      datasets: [{
        label: dist + ' – Projected Depth (m)',
        data: d.levels,
        borderColor: '#f59e0b',
        backgroundColor: 'rgba(245,158,11,.08)',
        borderWidth: 2.5,
        borderDash: [6, 3],
        pointRadius: 4,
        pointHoverRadius: 7,
        fill: true,
        tension: 0.4,
      }]
    },
    options: chartOptions('Projected Groundwater Depth (meters)')
  });

  const first = d.levels[0].toFixed(2);
  const last  = d.levels[d.levels.length - 1].toFixed(2);
  const diff  = (d.levels[d.levels.length - 1] - d.levels[0]).toFixed(2);
  document.getElementById('forecast-stats').innerHTML = `
    <div class="stat-pill">Start (${d.years[0]}) <strong>${first} m</strong></div>
    <div class="stat-pill">End (${d.years[d.years.length - 1]}) <strong>${last} m</strong></div>
    <div class="stat-pill">Projected Change <strong>+${diff} m</strong></div>
  `;
}

// ═══════════════════════════════════════════
// SHARED CHART OPTIONS
// ═══════════════════════════════════════════
function chartOptions(yLabel) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { labels: { color: '#fafafa', font: { family: 'Inter' } } },
      tooltip: {
        backgroundColor: '#18181b',
        borderColor: '#3f3f46',
        borderWidth: 1,
        titleColor: '#3b82f6',
        bodyColor: '#fafafa',
      }
    },
    scales: {
      x: { ticks: { color: '#a1a1aa', font: { family: 'Inter', size: 11 } }, grid: { color: 'rgba(255,255,255,.05)' } },
      y: {
        title: { display: true, text: yLabel || '', color: '#a1a1aa', font: { family: 'Inter', size: 11 } },
        ticks: { color: '#a1a1aa', font: { family: 'Inter', size: 11 } },
        grid:  { color: 'rgba(255,255,255,.05)' }
      }
    }
  };
}

// ═══════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════
window.addEventListener('DOMContentLoaded', () => {
  loadMetrics();
  updateDistrictValues();
  setTimeout(loadForecast, 300);
});
