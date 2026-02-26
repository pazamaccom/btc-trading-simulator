/* ===== BTC Trading Simulator — App Logic ===== */
(function () {
  'use strict';

  // ===== STATE =====
  let currentTF = '1h';
  let currentStrategy = null;
  let equityChart = null;
  let sortState = { col: null, dir: 'asc' };

  const STRATEGY_COLORS = {
    'RSI': '#58a6ff',
    'Bollinger Bands': '#bc8cff',
    'MA Crossover': '#00d97e',
    'MACD': '#f0883e',
    'Volume Breakout': '#39d2c0'
  };

  const STRATEGY_SHORT = {
    'RSI': 'RSI',
    'Bollinger Bands': 'BB',
    'MA Crossover': 'MA',
    'MACD': 'MACD',
    'Volume Breakout': 'VOL'
  };

  // ===== UTILITY =====
  function fmt(n, decimals) {
    if (n == null || isNaN(n)) return '—';
    if (decimals === undefined) decimals = 2;
    return Number(n).toFixed(decimals);
  }

  function fmtPct(n) {
    if (n == null || isNaN(n)) return '—';
    const sign = n > 0 ? '+' : '';
    return sign + Number(n).toFixed(2) + '%';
  }

  function fmtPrice(n) {
    if (n == null) return '—';
    return '$' + Number(n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  function fmtMoney(n) {
    if (n == null) return '—';
    const sign = n >= 0 ? '+' : '-';
    const abs = Math.abs(n);
    return sign + '$' + abs.toFixed(0);
  }

  function fmtDate(iso) {
    const d = new Date(iso);
    const mo = (d.getMonth() + 1).toString().padStart(2, '0');
    const day = d.getDate().toString().padStart(2, '0');
    const h = d.getHours().toString().padStart(2, '0');
    const m = d.getMinutes().toString().padStart(2, '0');
    return `${mo}-${day} ${h}:${m}`;
  }

  function fmtDateShort(iso) {
    const d = new Date(iso);
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  }

  function valClass(n) {
    if (n > 0) return 'positive';
    if (n < 0) return 'negative';
    return 'neutral';
  }

  // ===== DATA ACCESS =====
  function tfData() { return DATA[currentTF]; }
  function strategies() { return tfData().strategies; }
  function strategyNames() { return Object.keys(strategies()); }

  function bestStrategyName() {
    let best = null, bestRet = -Infinity;
    for (const [name, s] of Object.entries(strategies())) {
      if (s.best.total_return_pct > bestRet) {
        bestRet = s.best.total_return_pct;
        best = name;
      }
    }
    return best;
  }

  // ===== HEADER =====
  function updateHeader() {
    const d = tfData();
    const start = fmtDateShort(d.date_range.start);
    const end = fmtDateShort(d.date_range.end);
    const lo = fmtPrice(d.price_range.min);
    const hi = fmtPrice(d.price_range.max);
    document.getElementById('headerSub').textContent =
      `${d.candles} candles · ${start} → ${end} · Range: ${lo} – ${hi}`;
  }

  // ===== SUMMARY BAR =====
  function updateSummary() {
    const strats = strategies();
    let totalCombinations = 0;
    let bestStrat = null, bestReturn = -Infinity, bestSharpe = -Infinity, worstDD = 0, buyHold = null;

    for (const [name, s] of Object.entries(strats)) {
      totalCombinations += s.optimization.length;
      if (s.best.total_return_pct > bestReturn) {
        bestReturn = s.best.total_return_pct;
        bestStrat = name;
      }
      if (s.best.sharpe_ratio > bestSharpe) bestSharpe = s.best.sharpe_ratio;
      if (s.best.max_drawdown_pct > worstDD) worstDD = s.best.max_drawdown_pct;
      if (buyHold === null) buyHold = s.best.buy_hold_return_pct;
    }

    document.getElementById('sumCombinations').textContent = totalCombinations;
    document.getElementById('sumBestStrategy').textContent = bestStrat;
    const retEl = document.getElementById('sumBestReturn');
    retEl.textContent = fmtPct(bestReturn);
    retEl.className = 'summary-value mono ' + valClass(bestReturn);
    document.getElementById('sumBestSharpe').textContent = fmt(bestSharpe, 3);
    const ddEl = document.getElementById('sumWorstDD');
    ddEl.textContent = '-' + fmt(worstDD) + '%';
    ddEl.className = 'summary-value mono negative';
    const bhEl = document.getElementById('sumBuyHold');
    bhEl.textContent = fmtPct(buyHold);
    bhEl.className = 'summary-value mono ' + valClass(buyHold);
  }

  // ===== STRATEGY CARDS =====
  function buildCards() {
    const container = document.getElementById('strategyCards');
    container.innerHTML = '';
    const strats = strategies();
    const best = bestStrategyName();

    for (const name of strategyNames()) {
      const s = strats[name].best;
      const card = document.createElement('div');
      card.className = 'strat-card fade-in';
      if (name === best) card.classList.add('best');
      if (name === currentStrategy) card.classList.add('active');

      const color = STRATEGY_COLORS[name];
      card.style.borderBottomColor = name === currentStrategy ? color : 'transparent';
      card.dataset.strategy = name;

      card.innerHTML = `
        <div class="strat-name">
          <span class="strat-dot" style="background:${color}"></span>
          ${name}
        </div>
        <div class="strat-metrics">
          <div class="metric">
            <span class="metric-label">Return</span>
            <span class="metric-value ${valClass(s.total_return_pct)}">${fmtPct(s.total_return_pct)}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Sharpe</span>
            <span class="metric-value ${valClass(s.sharpe_ratio)}">${fmt(s.sharpe_ratio, 3)}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Win Rate</span>
            <span class="metric-value neutral">${fmt(s.win_rate_pct, 1)}%</span>
          </div>
          <div class="metric">
            <span class="metric-label">Max DD</span>
            <span class="metric-value negative">-${fmt(s.max_drawdown_pct)}%</span>
          </div>
          <div class="metric">
            <span class="metric-label">Trades</span>
            <span class="metric-value neutral">${s.num_trades}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Avg Win</span>
            <span class="metric-value positive">${s.avg_win_pct > 0 ? '+' : ''}${fmt(s.avg_win_pct)}%</span>
          </div>
        </div>
      `;

      card.addEventListener('click', () => selectStrategy(name));
      container.appendChild(card);
    }
  }

  function selectStrategy(name) {
    currentStrategy = name;
    // Update card active states
    document.querySelectorAll('.strat-card').forEach(c => {
      const sn = c.dataset.strategy;
      c.classList.toggle('active', sn === name);
      c.style.borderBottomColor = sn === name ? STRATEGY_COLORS[sn] : 'transparent';
    });
    updateChart();
    updateTrades();
    updateOptimization();
  }

  // ===== EQUITY CHART =====
  function updateChart() {
    const s = strategies()[currentStrategy].best;
    const curve = s.equity_curve;
    const color = STRATEGY_COLORS[currentStrategy];

    // Prepare data
    const labels = curve.map(p => new Date(p.time));
    const equityData = curve.map(p => p.equity);
    // Normalize buy & hold to start at same value as equity
    const startPrice = curve[0].price;
    const startEquity = curve[0].equity;
    const bhData = curve.map(p => (p.price / startPrice) * startEquity);

    // Title
    document.getElementById('chartTitle').textContent = `${currentStrategy} — Equity Curve`;

    // Legend
    document.getElementById('chartLegend').innerHTML = `
      <div class="legend-item"><span class="legend-line" style="background:${color}"></span>${STRATEGY_SHORT[currentStrategy]}</div>
      <div class="legend-item"><span class="legend-line dashed"></span>Buy &amp; Hold</div>
    `;

    if (equityChart) equityChart.destroy();

    const ctx = document.getElementById('equityChart').getContext('2d');
    equityChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: currentStrategy,
            data: equityData,
            borderColor: color,
            backgroundColor: color + '18',
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 4,
            pointHoverBackgroundColor: color,
            fill: true,
            tension: 0.1,
            order: 1
          },
          {
            label: 'Buy & Hold',
            data: bhData,
            borderColor: '#6e768166',
            borderWidth: 1.5,
            borderDash: [5, 4],
            pointRadius: 0,
            pointHoverRadius: 3,
            fill: false,
            tension: 0.1,
            order: 2
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        animation: { duration: 500, easing: 'easeOutQuart' },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#1c2128',
            borderColor: '#30363d',
            borderWidth: 1,
            titleFont: { family: "'JetBrains Mono'", size: 11 },
            bodyFont: { family: "'JetBrains Mono'", size: 11 },
            titleColor: '#8b949e',
            bodyColor: '#e6edf3',
            padding: 10,
            displayColors: true,
            callbacks: {
              title: (items) => {
                const d = items[0].label;
                return d;
              },
              label: (ctx) => {
                return ` ${ctx.dataset.label}: $${ctx.parsed.y.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
              }
            }
          }
        },
        scales: {
          x: {
            type: 'time',
            time: {
              unit: currentTF === '1h' ? 'day' : 'month',
              displayFormats: {
                day: 'MMM d',
                month: 'MMM yyyy'
              }
            },
            ticks: {
              color: '#6e7681',
              font: { family: "'JetBrains Mono'", size: 10 },
              maxTicksLimit: 10
            },
            grid: { color: '#21262d', drawBorder: false }
          },
          y: {
            position: 'right',
            ticks: {
              color: '#6e7681',
              font: { family: "'JetBrains Mono'", size: 10 },
              callback: v => '$' + v.toLocaleString()
            },
            grid: { color: '#21262d', drawBorder: false }
          }
        }
      }
    });
  }

  // ===== TRADES TABLE =====
  function buildTradeRows(trades) {
    // Pair up buys and sells
    const rows = [];
    let idx = 0;
    for (const t of trades) {
      idx++;
      rows.push({
        idx,
        type: t.type,
        time: t.time,
        price: t.price,
        pnl: t.pnl != null ? t.pnl : null,
        pnl_pct: t.pnl_pct != null ? t.pnl_pct : null
      });
    }
    return rows;
  }

  function updateTrades() {
    const s = strategies()[currentStrategy].best;
    const trades = s.trades;
    const rows = buildTradeRows(trades);

    document.getElementById('tradesTitle').textContent = `${currentStrategy} — Trade Log`;
    document.getElementById('tradeCount').textContent = `${s.num_trades} round trips · ${trades.length} executions`;

    // Sort
    let sorted = [...rows];
    if (sortState.col) {
      sorted.sort((a, b) => {
        let va = a[sortState.col], vb = b[sortState.col];
        if (va == null) va = sortState.dir === 'asc' ? Infinity : -Infinity;
        if (vb == null) vb = sortState.dir === 'asc' ? Infinity : -Infinity;
        if (typeof va === 'string') return sortState.dir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
        return sortState.dir === 'asc' ? va - vb : vb - va;
      });
    }

    const tbody = document.getElementById('tradesBody');
    tbody.innerHTML = sorted.map(r => `
      <tr class="fade-in">
        <td class="muted">${r.idx}</td>
        <td class="${r.type === 'BUY' ? 'buy' : 'sell'}">${r.type}</td>
        <td>${fmtDate(r.time)}</td>
        <td>${fmtPrice(r.price)}</td>
        <td class="${r.pnl != null ? valClass(r.pnl) : 'muted'}">${r.pnl != null ? fmtMoney(r.pnl) : '—'}</td>
        <td class="${r.pnl_pct != null ? valClass(r.pnl_pct) : 'muted'}">${r.pnl_pct != null ? fmtPct(r.pnl_pct) : '—'}</td>
      </tr>
    `).join('');

    // Header sort indicators
    document.querySelectorAll('#tradesTable thead th').forEach(th => {
      th.classList.remove('sort-asc', 'sort-desc');
      if (th.dataset.sort === sortState.col) {
        th.classList.add(sortState.dir === 'asc' ? 'sort-asc' : 'sort-desc');
      }
    });
  }

  // ===== OPTIMIZATION TABLE =====
  function updateOptimization() {
    const s = strategies()[currentStrategy];
    const opt = s.optimization;

    document.getElementById('optTitle').textContent = `${currentStrategy} — Parameter Optimization`;
    document.getElementById('optCount').textContent = `${opt.length} combinations`;

    // Determine param keys
    const paramKeys = opt.length > 0 ? Object.keys(opt[0].params) : [];

    // Build header
    const thead = document.getElementById('optHead');
    thead.innerHTML = `<tr>
      <th>#</th>
      ${paramKeys.map(k => `<th>${k.replace(/_/g, ' ')}</th>`).join('')}
      <th>Return %</th>
      <th>Sharpe</th>
      <th>Win Rate</th>
      <th>Trades</th>
      <th>Max DD</th>
    </tr>`;

    // Find best row (highest return)
    let bestIdx = 0;
    opt.forEach((o, i) => {
      if (o.total_return_pct > opt[bestIdx].total_return_pct) bestIdx = i;
    });

    const tbody = document.getElementById('optBody');
    tbody.innerHTML = opt.map((o, i) => `
      <tr class="${i === bestIdx ? 'best-row' : ''} fade-in">
        <td class="muted">${i + 1}</td>
        ${paramKeys.map(k => `<td>${o.params[k]}</td>`).join('')}
        <td class="${valClass(o.total_return_pct)}">${fmtPct(o.total_return_pct)}</td>
        <td class="${valClass(o.sharpe_ratio)}">${fmt(o.sharpe_ratio, 3)}</td>
        <td>${fmt(o.win_rate_pct, 1)}%</td>
        <td>${o.num_trades}</td>
        <td class="negative">-${fmt(o.max_drawdown_pct)}%</td>
      </tr>
    `).join('');
  }

  // ===== EVENT HANDLERS =====
  function initTimeframeToggle() {
    document.querySelectorAll('.tf-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        if (btn.dataset.tf === currentTF) return;
        currentTF = btn.dataset.tf;
        document.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentStrategy = bestStrategyName();
        sortState = { col: null, dir: 'asc' };
        render();
      });
    });
  }

  function initTableSort() {
    document.querySelectorAll('#tradesTable thead th').forEach(th => {
      th.addEventListener('click', () => {
        const col = th.dataset.sort;
        if (!col) return;
        if (sortState.col === col) {
          sortState.dir = sortState.dir === 'asc' ? 'desc' : 'asc';
        } else {
          sortState.col = col;
          sortState.dir = 'asc';
        }
        updateTrades();
      });
    });
  }

  // ===== RENDER ALL =====
  function render() {
    updateHeader();
    updateSummary();
    buildCards();
    updateChart();
    updateTrades();
    updateOptimization();
  }

  // ===== INIT =====
  function init() {
    currentStrategy = bestStrategyName();
    initTimeframeToggle();
    initTableSort();
    render();
  }

  // Wait for DOM
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
