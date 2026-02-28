// ===== BTC Trading Simulator v7 Dashboard =====
// All data from data.js globals: BACKTEST_DATA, VERSION_COMPARISON

(function () {
  'use strict';

  // ===== UTILITIES =====
  const fmt = (n, d = 2) => n != null ? Number(n).toFixed(d) : '\u2014';
  const fmtPct = (n) => n != null ? (n >= 0 ? '+' : '') + fmt(n) + '%' : '\u2014';
  const fmtUsd = (n) => n != null ? '$' + Number(n).toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 }) : '\u2014';
  const fmtDate = (d) => {
    if (!d) return '\u2014';
    const dt = new Date(d);
    return dt.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  };
  const fmtDateShort = (d) => {
    if (!d) return '\u2014';
    const dt = new Date(d);
    return dt.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  };

  const colorVal = (n, invert = false) => {
    if (n == null) return 'val-neutral';
    const positive = invert ? n < 0 : n > 0;
    const negative = invert ? n > 0 : n < 0;
    if (positive) return 'val-pos';
    if (negative) return 'val-neg';
    return 'val-neutral';
  };

  // ===== CHART.JS DEFAULTS =====
  Chart.defaults.color = '#8888a8';
  Chart.defaults.borderColor = 'rgba(26,26,46,0.6)';
  Chart.defaults.font.family = "'Inter', sans-serif";
  Chart.defaults.font.size = 11;
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.legend.labels.pointStyleWidth = 8;
  Chart.defaults.plugins.legend.labels.padding = 16;
  Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(12,12,20,0.95)';
  Chart.defaults.plugins.tooltip.borderColor = 'rgba(0,255,163,0.2)';
  Chart.defaults.plugins.tooltip.borderWidth = 1;
  Chart.defaults.plugins.tooltip.titleFont = { family: "'JetBrains Mono', monospace", size: 12, weight: '600' };
  Chart.defaults.plugins.tooltip.bodyFont = { family: "'JetBrains Mono', monospace", size: 11 };
  Chart.defaults.plugins.tooltip.padding = 10;
  Chart.defaults.plugins.tooltip.cornerRadius = 4;
  Chart.defaults.animation = { duration: 600, easing: 'easeOutQuart' };

  // ===== DATA EXTRACTION =====
  const data = BACKTEST_DATA;
  const vc = typeof VERSION_COMPARISON !== 'undefined' ? VERSION_COMPARISON : {};
  const strategies = data.strategies;
  const stratNames = Object.keys(strategies);

  // Strategy colors — categorized
  const STRAT_COLORS = {
    // Technical (cyan family)
    'MA Crossover': '#00d4ff',
    'RSI': '#00ffa3',
    'Bollinger': '#22d3ee',
    'MACD': '#06b6d4',
    'Volume Breakout': '#67e8f9',
    'Confluence Trend': '#2dd4bf',
    'Confluence Reversal': '#34d399',
    'Adaptive': '#5eead4',
    // Alternative (purple family)
    'FNG Contrarian': '#a855f7',
    'FNG Momentum': '#c084fc',
    'On-Chain Activity': '#8b5cf6',
    'Hash Rate': '#7c3aed',
    'Mempool Pressure': '#d946ef',
    // Hybrid (yellow/orange family)
    'MA + FNG Hybrid': '#ffd000',
    'Confluence + AltData': '#ff8a00',
    // ML (rose/pink family)
    'ML RandomForest': '#f43f5e',
    'ML GradientBoost': '#fb923c',
    'ML RF Short-Horizon': '#e879f9',
    'ML RF Conservative': '#f472b6',
    // Ensemble (green/emerald family)
    'Ensemble Balanced': '#00ffa3',
    'Ensemble Aggressive': '#22d3ee',
    'Ensemble Conservative': '#fbbf24',
  };

  const CAT_COLORS = {
    'technical': '#00d4ff',
    'alternative': '#a855f7',
    'hybrid': '#ffd000',
    'ml': '#f43f5e',
    'ensemble': '#00ffa3',
  };

  const getColor = (name) => STRAT_COLORS[name] || '#8888a8';
  const getCategory = (name) => (strategies[name] && strategies[name].category) || 'technical';

  // ===== HEADER =====
  document.getElementById('meta-oos-period').textContent = fmtDate(data.date_range.oos_start) + ' \u2192 ' + fmtDate(data.date_range.end);
  document.getElementById('meta-candles').textContent = data.total_candles.toLocaleString();
  document.getElementById('meta-price-range').textContent = fmtUsd(data.price_range.min) + ' \u2014 ' + fmtUsd(data.price_range.max);
  document.getElementById('meta-method').textContent =
    data.ml_available ? 'Price + Alt Data + Ensemble ML + Futures' :
    data.alt_data_available ? 'Price + Alt Data' : 'Price Only';

  // ===== KPI CARDS =====
  let bestName = '', bestReturn = -Infinity;
  for (const name of stratNames) {
    if (strategies[name].total_return_pct > bestReturn) {
      bestReturn = strategies[name].total_return_pct;
      bestName = name;
    }
  }
  const bhReturn = strategies[stratNames[0]].buy_hold_return_pct;
  const alpha = bestReturn - bhReturn;

  document.getElementById('kpi-best-name').textContent = bestName;
  document.getElementById('kpi-best-return').textContent = fmtPct(bestReturn) + ' OOS return';
  document.getElementById('kpi-bh').textContent = fmtPct(bhReturn);
  document.getElementById('kpi-bh').className = 'kpi-value ' + colorVal(bhReturn);
  document.getElementById('kpi-alpha').textContent = fmtPct(alpha);
  document.getElementById('kpi-count').textContent = stratNames.length;

  // Count categories
  const catCounts = { technical: 0, alternative: 0, hybrid: 0, ml: 0, ensemble: 0 };
  for (const name of stratNames) {
    const cat = getCategory(name);
    if (catCounts[cat] !== undefined) catCounts[cat]++;
  }
  const catParts = [];
  if (catCounts.technical) catParts.push(`${catCounts.technical} TA`);
  if (catCounts.alternative) catParts.push(`${catCounts.alternative} Alt`);
  if (catCounts.hybrid) catParts.push(`${catCounts.hybrid} Hybrid`);
  if (catCounts.ml) catParts.push(`${catCounts.ml} ML`);
  if (catCounts.ensemble) catParts.push(`${catCounts.ensemble} Ensemble`);
  document.getElementById('kpi-categories-sub').textContent = catParts.join('  \u00b7  ');

  // ===== CATEGORY FILTER =====
  let activeFilter = 'all';

  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      activeFilter = btn.dataset.cat;
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderTable();
    });
  });

  // ===== STRATEGY TABLE =====
  let tableData = stratNames.map(name => {
    const s = strategies[name];
    return {
      name,
      category: getCategory(name),
      return: s.total_return_pct,
      bh: s.buy_hold_return_pct,
      alpha: s.total_return_pct - s.buy_hold_return_pct,
      sharpe: s.sharpe_ratio,
      sortino: s.sortino_ratio,
      winrate: s.win_rate_pct,
      trades: s.num_trades,
      longs: s.long_trades != null ? s.long_trades : s.num_trades,
      shorts: s.short_trades != null ? s.short_trades : 0,
      dd: s.max_drawdown_pct,
      pf: s.profit_factor,
    };
  });

  let sortCol = 'return';
  let sortAsc = false;

  function renderTable() {
    let filtered = tableData;
    if (activeFilter !== 'all') {
      filtered = tableData.filter(r => r.category === activeFilter);
    }

    const sorted = [...filtered].sort((a, b) => {
      let va = a[sortCol], vb = b[sortCol];
      if (va === Infinity) va = 999999;
      if (vb === Infinity) vb = 999999;
      if (typeof va === 'string') {
        return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
      }
      return sortAsc ? va - vb : vb - va;
    });

    const tbody = document.getElementById('strategy-tbody');
    tbody.innerHTML = '';

    for (const row of sorted) {
      const tr = document.createElement('tr');
      tr.dataset.strategy = row.name;

      const catBadge = `<span class="cat-badge cat-${row.category}">${row.category}</span>`;
      const pfDisplay = row.pf === Infinity ? '\u221e' : fmt(row.pf, 3);
      const pfClass = row.pf === Infinity ? 'val-pos' : colorVal(row.pf - 1);

      const cells = [
        { val: row.name, cls: '', html: false },
        { val: catBadge, cls: '', html: true },
        { val: fmtPct(row.return), cls: colorVal(row.return) },
        { val: fmtPct(row.bh), cls: colorVal(row.bh) },
        { val: fmtPct(row.alpha), cls: colorVal(row.alpha) },
        { val: fmt(row.sharpe, 3), cls: colorVal(row.sharpe) },
        { val: fmt(row.sortino, 3), cls: colorVal(row.sortino) },
        { val: fmt(row.winrate, 1) + '%', cls: colorVal(row.winrate - 50) },
        { val: row.trades, cls: '' },
        { val: row.longs, cls: '' },
        { val: row.shorts > 0 ? row.shorts : '\u2014', cls: row.shorts > 0 ? 'val-short' : '' },
        { val: '-' + fmt(row.dd, 2) + '%', cls: 'val-neg' },
        { val: pfDisplay, cls: pfClass },
      ];

      for (const c of cells) {
        const td = document.createElement('td');
        if (c.html) {
          td.innerHTML = c.val;
        } else {
          td.textContent = c.val;
        }
        if (c.cls) td.className = c.cls;
        tr.appendChild(td);
      }

      tr.addEventListener('click', () => showDetail(row.name));
      tbody.appendChild(tr);
    }

    document.querySelectorAll('#strategy-table th').forEach(th => {
      th.classList.remove('sorted-asc', 'sorted-desc');
    });
    const sortedTh = document.querySelector(`#strategy-table th[data-col="${sortCol}"]`);
    if (sortedTh) sortedTh.classList.add(sortAsc ? 'sorted-asc' : 'sorted-desc');
  }

  document.querySelectorAll('#strategy-table th').forEach(th => {
    th.addEventListener('click', () => {
      const col = th.dataset.col;
      if (sortCol === col) {
        sortAsc = !sortAsc;
      } else {
        sortCol = col;
        sortAsc = col === 'dd';
      }
      renderTable();
    });
  });

  renderTable();

  // ===== STRATEGY DETAIL =====
  let exitChartInstance = null;

  function showDetail(name) {
    const s = strategies[name];
    const section = document.getElementById('detail-section');
    section.classList.remove('hidden');

    document.querySelectorAll('#strategy-table tbody tr').forEach(tr => {
      tr.classList.toggle('active-row', tr.dataset.strategy === name);
    });

    const cat = getCategory(name);
    document.getElementById('detail-title').innerHTML =
      `${name} <span class="cat-badge cat-${cat}">${cat}</span>`;

    // Refit log table
    const refitHead = document.getElementById('refit-thead');
    const refitBody = document.getElementById('refit-tbody');

    if (s.refit_log && s.refit_log.length > 0) {
      const paramKeys = Object.keys(s.refit_log[0].params);
      refitHead.innerHTML = '<th>Date</th><th>Day</th>' +
        paramKeys.map(k => `<th>${k}</th>`).join('') +
        '<th>Train Score</th>';

      refitBody.innerHTML = '';
      for (const r of s.refit_log) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${fmtDateShort(r.date)}</td><td style="text-align:right">${r.day}</td>` +
          paramKeys.map(k => `<td style="text-align:right">${r.params[k]}</td>`).join('') +
          `<td style="text-align:right">${fmt(r.train_score, 2)}</td>`;
        refitBody.appendChild(tr);
      }
    } else {
      refitHead.innerHTML = '<th>No refit data</th>';
      refitBody.innerHTML = '';
    }

    // Exit breakdown doughnut — updated colors for v6 exit types
    if (exitChartInstance) exitChartInstance.destroy();
    const exitCtx = document.getElementById('exit-chart').getContext('2d');
    const eb = s.exit_breakdown || {};
    const exitLabels = Object.keys(eb).filter(k => eb[k] > 0);
    const exitValues = exitLabels.map(k => eb[k]);
    
    const EXIT_COLORS = {
      'stop_loss': '#ff3860',
      'take_profit': '#00ffa3',
      'signal': '#00d4ff',
      'trailing_stop': '#ffd000',
      'time_exit': '#a855f7',
      'close': '#8888a8',
    };
    const exitColors = exitLabels.map(l => EXIT_COLORS[l] || '#8888a8');

    exitChartInstance = new Chart(exitCtx, {
      type: 'doughnut',
      data: {
        labels: exitLabels.map(l => l.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())),
        datasets: [{
          data: exitValues,
          backgroundColor: exitColors,
          borderColor: '#06060b',
          borderWidth: 2,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '55%',
        plugins: {
          legend: { position: 'bottom', labels: { padding: 12 } },
          tooltip: {
            callbacks: {
              label: ctx => `${ctx.label}: ${ctx.raw} trade${ctx.raw !== 1 ? 's' : ''}`
            }
          }
        }
      }
    });

    // Trade log
    const tradeBody = document.getElementById('trade-tbody');
    tradeBody.innerHTML = '';
    if (s.trades && s.trades.length > 0) {
      s.trades.forEach((t, i) => {
        const tr = document.createElement('tr');
        const pnl = t.pnl != null ? t.pnl : '';
        const pnlPct = t.pnl_pct != null ? t.pnl_pct : '';
        const pnlClass = pnl !== '' ? colorVal(pnl) : '';
        const side = t.side || 'long';
        const sideClass = side === 'short' ? 'val-short' : 'val-long';
        tr.innerHTML = `
          <td>${i + 1}</td>
          <td>${t.type}</td>
          <td class="${sideClass}">${side.toUpperCase()}</td>
          <td>${fmtDateShort(t.time)}</td>
          <td>${fmtUsd(Math.round(t.price))}</td>
          <td>${t.amount != null ? fmt(t.amount, 6) : '\u2014'}</td>
          <td class="${pnlClass}">${pnl !== '' ? (pnl >= 0 ? '+' : '') + fmtUsd(Math.round(pnl)) : '\u2014'}</td>
          <td class="${pnlClass}">${pnlPct !== '' ? fmtPct(pnlPct) : '\u2014'}</td>
        `;
        tradeBody.appendChild(tr);
      });
    }

    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  document.getElementById('detail-close').addEventListener('click', () => {
    document.getElementById('detail-section').classList.add('hidden');
    document.querySelectorAll('#strategy-table tbody tr').forEach(tr => tr.classList.remove('active-row'));
  });

  // ===== CATEGORY BREAKDOWN CHART =====
  (function () {
    const categories = ['technical', 'alternative', 'hybrid', 'ml', 'ensemble'];
    const catData = {};

    for (const cat of categories) {
      catData[cat] = { names: [], returns: [], alphas: [] };
    }

    for (const name of stratNames) {
      const s = strategies[name];
      const cat = getCategory(name);
      if (catData[cat]) {
        catData[cat].names.push(name);
        catData[cat].returns.push(s.total_return_pct);
        catData[cat].alphas.push(s.total_return_pct - s.buy_hold_return_pct);
      }
    }

    const allNames = [];
    const allReturns = [];
    const allColors = [];
    const allBorderColors = [];

    for (const cat of categories) {
      const cd = catData[cat];
      if (cd.names.length === 0) continue;
      const indices = cd.returns.map((_, i) => i).sort((a, b) => cd.returns[b] - cd.returns[a]);
      for (const i of indices) {
        allNames.push(cd.names[i]);
        allReturns.push(cd.returns[i]);
        allColors.push(CAT_COLORS[cat] + '99');
        allBorderColors.push(CAT_COLORS[cat]);
      }
    }

    const ctx = document.getElementById('category-chart').getContext('2d');
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: allNames,
        datasets: [{
          label: 'OOS Return %',
          data: allReturns,
          backgroundColor: allColors,
          borderColor: allBorderColors,
          borderWidth: 1.5,
          borderRadius: 3,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: ctx => {
                const name = allNames[ctx.dataIndex];
                const cat = getCategory(name);
                return `${name} [${cat}]: ${fmtPct(ctx.raw)}`;
              }
            }
          },
          annotation: {
            annotations: {
              zeroLine: {
                type: 'line',
                yMin: 0, yMax: 0,
                borderColor: 'rgba(255,255,255,0.15)',
                borderWidth: 1,
                borderDash: [4, 4],
              },
              bhLine: {
                type: 'line',
                yMin: bhReturn, yMax: bhReturn,
                borderColor: 'rgba(255,56,96,0.4)',
                borderWidth: 1.5,
                borderDash: [6, 4],
                label: {
                  display: true,
                  content: 'Buy & Hold: ' + fmtPct(bhReturn),
                  position: 'start',
                  font: { family: "'JetBrains Mono', monospace", size: 9 },
                  color: 'rgba(255,56,96,0.8)',
                  backgroundColor: 'rgba(12,12,20,0.8)',
                  padding: 4,
                }
              }
            }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: {
              font: { family: "'JetBrains Mono', monospace", size: 9 },
              maxRotation: 45,
            }
          },
          y: {
            grid: { color: 'rgba(26,26,46,0.4)' },
            ticks: {
              callback: v => v + '%',
              font: { family: "'JetBrains Mono', monospace", size: 10 }
            },
            title: {
              display: true,
              text: 'OOS Return %',
              font: { family: "'JetBrains Mono', monospace", size: 11 },
              color: '#6a6a88'
            }
          }
        }
      }
    });
  })();

  // ===== VERSION COMPARISON CHART =====
  (function () {
    const allStrats = [...new Set([
      ...Object.keys(vc.v1_static || {}),
      ...Object.keys(vc.v2_static || {}),
      ...Object.keys(vc.v3_oos || {}),
      ...Object.keys(vc.v4_oos || {}),
      ...Object.keys(vc.v5_oos || {}),
      ...Object.keys(vc.v6_oos || {}),
      ...Object.keys(vc.v7_oos || {}),
    ])];

    allStrats.sort((a, b) => {
      const av = vc.v7_oos?.[a] ?? vc.v6_oos?.[a] ?? vc.v5_oos?.[a] ?? vc.v4_oos?.[a] ?? vc.v3_oos?.[a] ?? -999;
      const bv = vc.v7_oos?.[b] ?? vc.v6_oos?.[b] ?? vc.v5_oos?.[b] ?? vc.v4_oos?.[b] ?? vc.v3_oos?.[b] ?? -999;
      return bv - av;
    });

    if (allStrats.length === 0) return;

    const ctx = document.getElementById('version-chart').getContext('2d');
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: allStrats,
        datasets: [
          {
            label: 'v1 (Static, Overfit)',
            data: allStrats.map(s => vc.v1_static?.[s] ?? null),
            backgroundColor: 'rgba(120,120,140,0.5)',
            borderColor: 'rgba(120,120,140,0.7)',
            borderWidth: 1, borderRadius: 2,
          },
          {
            label: 'v2 (Risk-Managed)',
            data: allStrats.map(s => vc.v2_static?.[s] ?? null),
            backgroundColor: 'rgba(255,208,0,0.5)',
            borderColor: 'rgba(255,208,0,0.7)',
            borderWidth: 1, borderRadius: 2,
          },
          {
            label: 'v3 (Rolling OOS)',
            data: allStrats.map(s => vc.v3_oos?.[s] ?? null),
            backgroundColor: 'rgba(0,255,163,0.6)',
            borderColor: 'rgba(0,255,163,0.8)',
            borderWidth: 1, borderRadius: 2,
          },
          {
            label: 'v4 (+ Alt Data)',
            data: allStrats.map(s => vc.v4_oos?.[s] ?? null),
            backgroundColor: 'rgba(168,85,247,0.6)',
            borderColor: 'rgba(168,85,247,0.8)',
            borderWidth: 1, borderRadius: 2,
          },
          {
            label: 'v5 (+ ML Signals)',
            data: allStrats.map(s => vc.v5_oos?.[s] ?? null),
            backgroundColor: 'rgba(244,63,94,0.6)',
            borderColor: 'rgba(244,63,94,0.8)',
            borderWidth: 1, borderRadius: 2,
          },
          {
            label: 'v6 (Ensemble)',
            data: allStrats.map(s => vc.v6_oos?.[s] ?? null),
            backgroundColor: 'rgba(0,255,163,0.5)',
            borderColor: 'rgba(0,255,163,0.7)',
            borderWidth: 1, borderRadius: 2,
          },
          {
            label: 'v7 (Shorts+DynExit)',
            data: allStrats.map(s => vc.v7_oos?.[s] ?? null),
            backgroundColor: 'rgba(34,211,238,0.8)',
            borderColor: 'rgba(34,211,238,1)',
            borderWidth: 1.5, borderRadius: 2,
          },
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'top' },
          tooltip: {
            callbacks: {
              label: ctx => `${ctx.dataset.label}: ${ctx.raw != null ? fmtPct(ctx.raw) : 'N/A'}`
            }
          },
          annotation: {
            annotations: {
              zeroLine: {
                type: 'line',
                yMin: 0, yMax: 0,
                borderColor: 'rgba(255,255,255,0.15)',
                borderWidth: 1,
                borderDash: [4, 4],
              }
            }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: {
              font: { family: "'JetBrains Mono', monospace", size: 9 },
              maxRotation: 45,
            }
          },
          y: {
            grid: { color: 'rgba(26,26,46,0.4)' },
            ticks: {
              callback: v => v + '%',
              font: { family: "'JetBrains Mono', monospace", size: 10 }
            },
            title: {
              display: true,
              text: 'Return %',
              font: { family: "'JetBrains Mono', monospace", size: 11 },
              color: '#6a6a88'
            }
          }
        }
      }
    });
  })();

  // ===== EQUITY CURVES =====
  (function () {
    const priceData = data.price_data;
    if (!priceData || priceData.length === 0) return;

    const startPrice = priceData[0].close;
    const bhEquity = priceData.map(p => ({
      time: p.time,
      equity: 10000 * (p.close / startPrice)
    }));

    const labels = priceData.map(p => {
      const d = new Date(p.time);
      return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    });

    const datasets = [];

    for (const name of stratNames) {
      const s = strategies[name];
      if (!s.equity_curve || s.equity_curve.length === 0) continue;

      let eqData = s.equity_curve.map(e => e.equity);
      while (eqData.length < labels.length) {
        eqData.unshift(10000);
      }
      if (eqData.length > labels.length) {
        eqData = eqData.slice(eqData.length - labels.length);
      }

      datasets.push({
        label: name,
        data: eqData,
        borderColor: getColor(name),
        backgroundColor: 'transparent',
        borderWidth: 1.5,
        pointRadius: 0,
        pointHitRadius: 4,
        tension: 0.1,
      });
    }

    datasets.push({
      label: 'Buy & Hold',
      data: bhEquity.map(e => e.equity),
      borderColor: 'rgba(120,120,140,0.7)',
      backgroundColor: 'transparent',
      borderWidth: 2,
      borderDash: [6, 3],
      pointRadius: 0,
      pointHitRadius: 4,
      tension: 0.1,
    });

    const ctx = document.getElementById('equity-chart').getContext('2d');
    new Chart(ctx, {
      type: 'line',
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: {
            position: 'top',
            labels: { font: { family: "'JetBrains Mono', monospace", size: 9 } }
          },
          tooltip: {
            callbacks: {
              label: ctx => `${ctx.dataset.label}: ${fmtUsd(Math.round(ctx.raw))}`
            }
          },
          annotation: {
            annotations: {
              startLine: {
                type: 'line',
                yMin: 10000, yMax: 10000,
                borderColor: 'rgba(255,255,255,0.08)',
                borderWidth: 1,
                borderDash: [4, 4],
                label: {
                  display: true,
                  content: '$10K Start',
                  position: 'end',
                  font: { family: "'JetBrains Mono', monospace", size: 9 },
                  color: 'rgba(255,255,255,0.3)',
                  backgroundColor: 'transparent',
                }
              }
            }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: {
              maxTicksLimit: 12,
              font: { family: "'JetBrains Mono', monospace", size: 10 }
            }
          },
          y: {
            grid: { color: 'rgba(26,26,46,0.4)' },
            ticks: {
              callback: v => fmtUsd(v),
              font: { family: "'JetBrains Mono', monospace", size: 10 }
            },
            title: {
              display: true,
              text: 'Portfolio Value',
              font: { family: "'JetBrains Mono', monospace", size: 11 },
              color: '#6a6a88'
            }
          }
        }
      }
    });
  })();

  // ===== RISK-RETURN SCATTER =====
  (function () {
    const scatterData = stratNames.map(name => {
      const s = strategies[name];
      return {
        x: s.max_drawdown_pct,
        y: s.total_return_pct,
        label: name,
        category: getCategory(name),
      };
    });

    const priceData = data.price_data;
    if (!priceData || priceData.length === 0) return;

    const startPrice = priceData[0].close;
    let peak = startPrice;
    let maxDD = 0;
    for (const p of priceData) {
      if (p.close > peak) peak = p.close;
      const dd = (peak - p.close) / peak * 100;
      if (dd > maxDD) maxDD = dd;
    }

    scatterData.push({
      x: maxDD,
      y: bhReturn,
      label: 'Buy & Hold',
      category: 'benchmark',
    });

    const ctx = document.getElementById('scatter-chart').getContext('2d');
    new Chart(ctx, {
      type: 'scatter',
      data: {
        datasets: [{
          data: scatterData.map(d => ({ x: d.x, y: d.y })),
          backgroundColor: scatterData.map(d => {
            if (d.label === 'Buy & Hold') return 'rgba(120,120,140,0.8)';
            return CAT_COLORS[d.category] || '#8888a8';
          }),
          borderColor: scatterData.map(d => {
            if (d.label === 'Buy & Hold') return 'rgba(120,120,140,1)';
            return CAT_COLORS[d.category] || '#8888a8';
          }),
          borderWidth: 2,
          pointRadius: scatterData.map(d => d.label === 'Buy & Hold' ? 8 : 7),
          pointStyle: scatterData.map(d => {
            if (d.label === 'Buy & Hold') return 'rectRot';
            if (d.category === 'ensemble') return 'star';
            if (d.category === 'ml') return 'star';
            if (d.category === 'alternative') return 'triangle';
            if (d.category === 'hybrid') return 'rect';
            return 'circle';
          }),
          pointHoverRadius: 10,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: ctx => {
                const d = scatterData[ctx.dataIndex];
                return `${d.label} [${d.category}]: Return ${fmtPct(d.y)}, Max DD -${fmt(d.x)}%`;
              }
            }
          },
          annotation: {
            annotations: {
              zeroLine: {
                type: 'line',
                yMin: 0, yMax: 0,
                borderColor: 'rgba(255,255,255,0.12)',
                borderWidth: 1,
                borderDash: [4, 4],
              }
            }
          },
        },
        scales: {
          x: {
            reverse: false,
            grid: { color: 'rgba(26,26,46,0.4)' },
            ticks: {
              callback: v => '-' + v + '%',
              font: { family: "'JetBrains Mono', monospace", size: 10 }
            },
            title: {
              display: true,
              text: 'Max Drawdown %',
              font: { family: "'JetBrains Mono', monospace", size: 11 },
              color: '#6a6a88'
            }
          },
          y: {
            grid: { color: 'rgba(26,26,46,0.4)' },
            ticks: {
              callback: v => v + '%',
              font: { family: "'JetBrains Mono', monospace", size: 10 }
            },
            title: {
              display: true,
              text: 'OOS Return %',
              font: { family: "'JetBrains Mono', monospace", size: 11 },
              color: '#6a6a88'
            }
          }
        }
      },
      plugins: [{
        id: 'scatterLabels',
        afterDraw(chart) {
          const ctx = chart.ctx;
          ctx.save();
          ctx.font = "500 9px 'JetBrains Mono', monospace";
          ctx.textAlign = 'center';

          const meta = chart.getDatasetMeta(0);
          const positions = [];

          meta.data.forEach((point, i) => {
            const d = scatterData[i];
            let labelX = point.x;
            let labelY = point.y - 14;

            for (const pos of positions) {
              const dx = Math.abs(labelX - pos.x);
              const dy = Math.abs(labelY - pos.y);
              if (dx < 70 && dy < 12) {
                labelY = pos.y - 13;
              }
            }
            positions.push({ x: labelX, y: labelY });

            if (d.label === 'Buy & Hold') {
              ctx.fillStyle = 'rgba(120,120,140,0.9)';
            } else {
              ctx.fillStyle = CAT_COLORS[d.category] || '#8888a8';
            }
            ctx.fillText(d.label, labelX, labelY);
          });
          ctx.restore();
        }
      }]
    });
  })();

  // ===== ENSEMBLE FEATURE IMPORTANCE CHART =====
  (function () {
    const featureCanvas = document.getElementById('feature-chart');
    if (!featureCanvas) return;

    // Aggregate feature importance across ensemble strategies
    const featureScores = {};
    let sampleCount = 0;

    for (const name of stratNames) {
      const s = strategies[name];
      // Look at both 'ensemble' and 'ml' categories
      if (s.category !== 'ensemble' && s.category !== 'ml') continue;
      if (!s.feature_importance || s.feature_importance.length === 0) continue;

      for (const entry of s.feature_importance) {
        if (!entry.top_features) continue;
        sampleCount++;
        for (const [feat, score] of Object.entries(entry.top_features)) {
          featureScores[feat] = (featureScores[feat] || 0) + score;
        }
      }
    }

    if (sampleCount === 0 || Object.keys(featureScores).length === 0) {
      const section = featureCanvas.closest('.chart-card') || featureCanvas.closest('section');
      if (section) section.style.display = 'none';
      return;
    }

    const sorted = Object.entries(featureScores)
      .map(([feat, score]) => ({ feat, score: score / sampleCount }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 15);

    const labels = sorted.map(d => d.feat).reverse();
    const values = sorted.map(d => d.score).reverse();

    // Green gradient for ensemble (instead of rose for ML)
    const barColors = values.map((v, i) => {
      const t = i / (values.length - 1 || 1);
      const r = Math.round(0 + t * 0);
      const g = Math.round(160 + t * 95);
      const b = Math.round(100 + t * 63);
      return `rgba(${r},${g},${b},0.8)`;
    });
    const borderColors = barColors.map(c => c.replace('0.8)', '1)'));

    const ctx = featureCanvas.getContext('2d');
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'Avg Importance',
          data: values,
          backgroundColor: barColors,
          borderColor: borderColors,
          borderWidth: 1,
          borderRadius: 3,
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: ctx => `Importance: ${ctx.raw.toFixed(4)}`
            }
          }
        },
        scales: {
          x: {
            grid: { color: 'rgba(26,26,46,0.4)' },
            ticks: {
              callback: v => v.toFixed(2),
              font: { family: "'JetBrains Mono', monospace", size: 10 }
            },
            title: {
              display: true,
              text: 'Avg Feature Importance',
              font: { family: "'JetBrains Mono', monospace", size: 11 },
              color: '#6a6a88'
            }
          },
          y: {
            grid: { display: false },
            ticks: {
              font: { family: "'JetBrains Mono', monospace", size: 10 },
              color: '#b0b0cc',
            }
          }
        }
      }
    });
  })();

})();
