// Theme toggle
const root = document.documentElement;
const savedTheme = localStorage.getItem('theme');
if (savedTheme === 'dark') root.classList.add('dark');
document.getElementById('themeToggle')?.addEventListener('click', ()=>{
  root.classList.toggle('dark');
  localStorage.setItem('theme', root.classList.contains('dark') ? 'dark':'light');
});

let chart;
async function renderForecast(n = 7) {
    const btn = document.querySelector('button[type="submit"]');
    const loader = document.getElementById('loading');
    btn.disabled = true; btn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Memproses...';
    loader.style.display = 'inline-block';
    const res = await fetch(`/api/forecast?n=${n}`);
    const data = await res.json();
    if (data.error) { alert(data.error); btn.disabled = false; btn.innerHTML = '<i class="bi bi-magic me-1"></i>Forecast'; loader.style.display = 'none'; return; }

    // Get history for context
    const hres = await fetch('/api/history?max_points=200');
    const hist = await hres.json();
    const labels = [...hist.dates, ...data.dates];
    const padding = new Array(hist.values.length - 1).fill(null);
    const forecastValues = [hist.values[hist.values.length - 1], ...data.predictions];

    const ctx = document.getElementById('chart');
    if (chart) chart.destroy();
    chart = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets: [
            { label: 'History', data: hist.values, borderColor: getComputedStyle(root).getPropertyValue('--accent').trim() || '#0d6efd', tension: 0.25, pointRadius: 0 },
            { label: 'Forecast', data: [...padding, ...forecastValues], borderColor: '#fb7185', borderDash: [6,4], tension: 0.25, pointRadius: 2 }
        ]},
        options: { responsive: true, maintainAspectRatio: false, scales: { x: { ticks: { maxTicksLimit: 10, color: getComputedStyle(root).getPropertyValue('--muted').trim() } }, y: { ticks: { color: getComputedStyle(root).getPropertyValue('--muted').trim() } } }, plugins: { legend: { labels: { color: getComputedStyle(root).getPropertyValue('--text').trim() } } } }
    });

    // Download CSV
    document.getElementById('btnDownloadCsv').onclick = () => {
        const rows = [['Tanggal','Prediksi']].concat(data.dates.map((d,i)=>[d, data.predictions[i]]));
        const csv = rows.map(r=>r.join(',')).join('\n');
        const blob = new Blob([csv], {type:'text/csv'});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a'); a.href = url; a.download = `forecast_${n}hari.csv`; a.click(); URL.revokeObjectURL(url);
    };

    btn.disabled = false; btn.innerHTML = '<i class="bi bi-magic me-1"></i>Forecast';
    loader.style.display = 'none';
}

// Auto render with default 7 days
renderForecast(7);


