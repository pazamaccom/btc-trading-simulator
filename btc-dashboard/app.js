// ===== BTC Trading Simulator v3 Dashboard =====
// All data from data.js globals: BACKTEST_DATA, VERSION_COMPARISON

(function () {
  'use strict';

  // ===== UTILITIES =====
  const fmt = (n, d = 2) => n != null ? Number(n).toFixed(d) : '—';
  const fmtPct = (n) => n != null ? (n >= 0 ? '+' : '') + fmt(n) + '%' : '—';
  const fmtUsd = (n) => n != null ? '$' + Number(n).toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 }) : '—';
  const fmtDate = (d) => {
    if (!d) return '—';
    const dt = new Date(d);
    return dt.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  };
  const fmtDateShort = (d) => {
    if (!d) return '—';
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
  const vc = VERSION_COMPARISON;
  const strategies = data.strategies;
  const stratNames = Object.keys(strategies);

  // Strategy colors
  const STRAT_COLORS = {
    'RSI': '#00ffa3',
    'Bollinger': '#00d4ff',
    'MA Crossover': '#ffd000',
    'MACD': '#ff8a00',
    'Volume Breakout': '#a855f7',
    'Confluence Trend': '#ff3860',
    'Confluence Reversal': '#22d3ee',
    'Adaptive': '#f472b6',
  };

  const getColor = (name) => STRAT_COLORS[name] || '#8888a8';

  // ===== HEADER =====
  document.getElementById('meta-oos-period').textContent = fmtDate(data.date_range.oos_start) + ' → ' + fmtDate(data.date_range.end);
  document.getElementById('meta-candles').textContent = data.total_candles.toLocaleString();
  document.getElementById('meta-price-range').textContent = fmtUsd(data.price_range.min) + ' — ' + fmtUsd(data.price_range.max);
  document.getElementById('meta-method').textContent = 'Rolling Walk-Forward';

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

  // ===== STRATEGY TABLE =====
  let tableData = stratNames.map(name => {
    const s = strategies[name];
    return {
      name,
      return: s.total_return_pct,
      bh: s.buy_hold_return_pct,
      alpha: s.total_return_pct - s.buy_hold_return_pct,
      sharpe: s.sharpe_ratio,
      sortino: s.sortino_ratio,
      winrate: s.win_rate_pct,
      trades: s.num_trades,
      dd: s.max_drawdown_pct,
      pf: s.profit_factor,
      refits: s.num_refits,
    };
  });

  let sortCol = 'return';
  let sortAsc = false;

  function renderTable() {
    const sorted = [...tableData].sort((a, b) => {
      let va = a[sortCol], vb = b[sortCol];
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

      const cells = [
        { val: row.name, cls: '' },
        { val: fmtPct(row.return), cls: colorVal(row.return) },
        { val: fmtPct(row.bh), cls: colorVal(row.bh) },
        { val: fmtPct(row.alpha), cls: colorVal(row.alpha) },
        { val: fmt(row.sharpe, 3), cls: colorVal(row.sharpe) },
        { val: fmt(row.sortino, 3), cls: colorVal(row.sortino) },
        { val: fmt(row.winrate, 1) + '%', cls: colorVal(row.winrate - 50) },
        { val: row.trades, cls: '' },
        { val: '-' + fmt(row.dd, 2) + '%', cls: 'val-neg' },
        { val: fmt(row.pf, 3), cls: colorVal(row.pf - 1) },
        { val: row.refits, cls: '' },
      ];

      for (const c of cells) {
        const td = document.createElement('td');
        td.textContent = c.val;
        if (c.cls) td.className = c.cls;
        tr.appendChild(td);
      }

      tr.addEventListener('click', () => showDetail(row.name));
      tbody.appendChild(tr);
    }

    // Highlight sorted column
    document.querySelectorAll('#strategy-table th').forEach(th => {
      th.classList.remove('sorted-asc', 'sorted-desc');
    });
    const sortedTh = document.querySelector(`#strategy-table th[data-col="${sortCol}"]`);
    if (sortedTh) sortedTh.classList.add(sortAsc ? 'sorted-asc' : 'sorted-desc');
  }

  // Table sorting
  document.querySelectorAll('#strategy-table th').forEach(th => {
    th.addEventListener('click', () => {
      const col = th.dataset.col;
      if (sortCol === col) {
        sortAsc = !sortAsc;
      } else {
        sortCol = col;
        sortAsc = col === 'dd'; // default ascending for drawdown
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

    // Highlight row
    document.querySelectorAll('#strategy-table tbody tr').forEach(tr => {
      tr.classList.toggle('active-row', tr.dataset.strategy === name);
    });

    document.getElementById('detail-title').textContent = name + ' — Strategy Details';

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

    // Exit breakdown doughnut
    if (exitChartInstance) exitChartInstance.destroy();
    const exitCtx = document.getElementById('exit-chart').getContext('2d');
    const eb = s.exit_breakdown || {};
    const exitLabels = Object.keys(eb);
    const exitValues = Object.values(eb);
    const exitColors = ['#ff3860', '#00ffa3', '#00d4ff', '#ffd000'];

    exitChartInstance = new Chart(exitCtx, {
      type: 'doughnut',
      data: {
        labels: exitLabels.map(l => l.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())),
        datasets: [{
          data: exitValues,
          backgroundColor: exitColors.slice(0, exitLabels.length),
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
        tr.innerHTML = `
          <td>${i + 1}</td>
          <td>${t.type}</td>
          <td>${fmtDateShort(t.time)}</td>
          <td>${fmtUsd(Math.round(t.price))}</td>
          <td>${t.amount != null ? fmt(t.amount, 6) : '—'}</td>
          <td class="${pnlClass}">${pnl !== '' ? (pnl >= 0 ? '+' : '') + fmtUsd(Math.round(pnl)) : '—'}</td>
          <td class="${pnlClass}">${pnlPct !== '' ? fmtPct(pnlPct) : '—'}</td>
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

  // ===== VERSION COMPARISON CHART =====
  (function () {
    const allStrats = [...new Set([
      ...Object.keys(vc.v1_static || {}),
      ...Object.keys(vc.v2_static || {}),
      ...Object.keys(vc.v3_oos || {})
    ])];

    // Sort by v3 return descending
    allStrats.sort((a, b) => (vc.v3_oos[b] || 0) - (vc.v3_oos[a] || 0));

    const ctx = document.getElementById('version-chart').getContext('2d');
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: allStrats,
        datasets: [
          {
            label: 'v1 (Static, Overfit)',
            data: allStrats.map(s => vc.v1_static?.[s] ?? null),
            backgroundColor: 'rgba(120,120,140,0.6)',
            borderColor: 'rgba(120,120,140,0.8)',
            borderWidth: 1,
            borderRadius: 2,
          },
          {
            label: 'v2 (Risk-Managed Static)',
            data: allStrats.map(s => vc.v2_static?.[s] ?? null),
            backgroundColor: 'rgba(255,208,0,0.6)',
            borderColor: 'rgba(255,208,0,0.8)',
            borderWidth: 1,
            borderRadius: 2,
          },
          {
            label: 'v3 (Rolling OOS)',
            data: allStrats.map(s => vc.v3_oos?.[s] ?? null),
            backgroundColor: 'rgba(0,255,163,0.7)',
            borderColor: 'rgba(0,255,163,0.9)',
            borderWidth: 1,
            borderRadius: 2,
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
                yMin: 0,
                yMax: 0,
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
            ticks: { font: { family: "'JetBrains Mono', monospace", size: 10 } }
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
    // Build Buy & Hold equity from price_data starting at $10,000
    const priceData = data.price_data;
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

    // Add strategy equity curves
    for (const name of stratNames) {
      const s = strategies[name];
      datasets.push({
        label: name,
        data: s.equity_curve.map(e => e.equity),
        borderColor: getColor(name),
        backgroundColor: 'transparent',
        borderWidth: 1.5,
        pointRadius: 0,
        pointHitRadius: 4,
        tension: 0.1,
      });
    }

    // Add Buy & Hold
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
        interaction: {
          mode: 'index',
          intersect: false,
        },
        plugins: {
          legend: {
            position: 'top',
            labels: {
              font: { family: "'JetBrains Mono', monospace", size: 10 },
            }
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
                yMin: 10000,
                yMax: 10000,
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
      };
    });

    // Buy & Hold reference point
    const priceData = data.price_data;
    const startPrice = priceData[0].close;
    // Calculate B&H max drawdown
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
    });

    const ctx = document.getElementById('scatter-chart').getContext('2d');
    new Chart(ctx, {
      type: 'scatter',
      data: {
        datasets: [{
          data: scatterData.map(d => ({ x: d.x, y: d.y })),
          backgroundColor: scatterData.map(d =>
            d.label === 'Buy & Hold' ? 'rgba(120,120,140,0.8)' : getColor(d.label)
          ),
          borderColor: scatterData.map(d =>
            d.label === 'Buy & Hold' ? 'rgba(120,120,140,1)' : getColor(d.label)
          ),
          borderWidth: 2,
          pointRadius: scatterData.map(d => d.label === 'Buy & Hold' ? 8 : 7),
          pointStyle: scatterData.map(d => d.label === 'Buy & Hold' ? 'rectRot' : 'circle'),
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
                return `${d.label}: Return ${fmtPct(d.y)}, Max DD -${fmt(d.x)}%`;
              }
            }
          },
          // Labels on points
          annotation: {
            annotations: {
              zeroLine: {
                type: 'line',
                yMin: 0,
                yMax: 0,
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
          ctx.font = "500 10px 'JetBrains Mono', monospace";
          ctx.textAlign = 'center';

          const meta = chart.getDatasetMeta(0);
          const positions = [];
          
          meta.data.forEach((point, i) => {
            const d = scatterData[i];
            let labelX = point.x;
            let labelY = point.y - 16;
            
            // Shift labels to avoid overlap
            for (const pos of positions) {
              const dx = Math.abs(labelX - pos.x);
              const dy = Math.abs(labelY - pos.y);
              if (dx < 80 && dy < 14) {
                labelY = pos.y - 15;
              }
            }
            positions.push({ x: labelX, y: labelY });
            
            ctx.fillStyle = d.label === 'Buy & Hold' ? 'rgba(120,120,140,0.9)' : getColor(d.label);
            ctx.fillText(d.label, labelX, labelY);
          });
          ctx.restore();
        }
      }]
    });
  })();

})();
