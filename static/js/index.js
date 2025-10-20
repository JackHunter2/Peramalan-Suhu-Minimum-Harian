// Theme toggle
const root = document.documentElement;
const savedTheme = localStorage.getItem('theme');
if (savedTheme === 'dark') root.classList.add('dark');
document.getElementById('themeToggle')?.addEventListener('click', ()=>{
  root.classList.toggle('dark');
  localStorage.setItem('theme', root.classList.contains('dark') ? 'dark':'light');
});

// Drag & drop upload UX
const dz = document.getElementById('dropzone');
const fileInput = document.getElementById('file');
dz?.addEventListener('click', () => fileInput?.click());
dz?.addEventListener('dragover', (e) => { e.preventDefault(); dz.classList.add('dragover'); });
dz?.addEventListener('dragleave', () => dz.classList.remove('dragover'));
dz?.addEventListener('drop', (e) => {
    e.preventDefault(); dz.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) { fileInput.files = e.dataTransfer.files; }
});

// Load history chart
let historyChart;
async function loadHistory() {
    try {
        const res = await fetch('/api/history');
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        const ctx = document.getElementById('historyChart');
        const values = data.values;
        const labels = data.dates;
        const empty = document.getElementById('historyEmpty');
        if (empty) empty.style.display = values.length ? 'none' : 'block';
        historyChart = new Chart(ctx, {
            type: 'line',
            data: { labels, datasets: [{ label: data.target || 'Nilai', data: values, borderColor: getComputedStyle(root).getPropertyValue('--accent').trim() || '#0d6efd', tension: 0.25, pointRadius: 0 }] },
            options: { responsive: true, maintainAspectRatio: false, scales: { x: { ticks: { maxTicksLimit: 8, color: getComputedStyle(root).getPropertyValue('--muted').trim() } }, y:{ ticks:{ color: getComputedStyle(root).getPropertyValue('--muted').trim() } } }, plugins: { legend: { labels:{ color: getComputedStyle(root).getPropertyValue('--text').trim() } } } }
        });
    } catch (_) { /* ignore */ }
}
loadHistory();

// Forecast interaktif
let forecastChart;
document.getElementById('btnForecast')?.addEventListener('click', async () => {
    const n = parseInt(document.getElementById('nDays').value || '7', 10);
    const btn = document.getElementById('btnForecast');
    btn.disabled = true; btn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Memproses...';
    document.getElementById('loading').style.display = 'inline-block';
    try {
        const res = await fetch(`/api/forecast?n=${n}`);
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        document.getElementById('forecastSection').style.display = 'block';
        const ctx = document.getElementById('forecastChart');
        const histRes = await fetch('/api/history?max_points=200');
        const hist = await histRes.json();
        const combinedLabels = [...hist.dates, ...data.dates];
        const padding = new Array(hist.values.length - 1).fill(null);
        const forecastValues = [hist.values[hist.values.length - 1], ...data.predictions];
        if (forecastChart) forecastChart.destroy();
        forecastChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: combinedLabels,
                datasets: [
                    { label: 'History', data: hist.values, borderColor: getComputedStyle(root).getPropertyValue('--accent').trim() || '#0d6efd', tension: 0.25, pointRadius: 0 },
                    { label: 'Forecast', data: [...padding, ...forecastValues], borderColor: '#fb7185', borderDash: [6,4], tension: 0.25, pointRadius: 2 }
                ]
            },
            options: { responsive: true, maintainAspectRatio: false, scales: { x: { ticks: { maxTicksLimit: 10, color: getComputedStyle(root).getPropertyValue('--muted').trim() } }, y:{ ticks:{ color: getComputedStyle(root).getPropertyValue('--muted').trim() } } }, plugins: { legend: { labels:{ color: getComputedStyle(root).getPropertyValue('--text').trim() } } } }
        });

        document.getElementById('btnDownloadCsv').onclick = () => {
            const rows = [['Tanggal','Prediksi']].concat(data.dates.map((d,i)=>[d, data.predictions[i]]));
            const csv = rows.map(r=>r.join(',')).join('\n');
            const blob = new Blob([csv], {type:'text/csv'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a'); a.href = url; a.download = `forecast_${n}hari.csv`; a.click(); URL.revokeObjectURL(url);
        };
    } catch (err) {
        alert(err.message || 'Gagal melakukan forecast');
    } finally {
        btn.disabled = false; btn.innerHTML = '<i class="bi bi-magic me-1"></i>Forecast';
        document.getElementById('loading').style.display = 'none';
    }
});


