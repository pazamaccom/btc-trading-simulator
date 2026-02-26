/* ═══════════════════════════════════════════════════════════════
   BTC Trading Simulator v2 — Dashboard Logic
   ═══════════════════════════════════════════════════════════════ */

(function () {
    'use strict';

    // ── State ──
    let currentTimeframe = '1h';
    let currentSort = { key: 'total_return_pct', dir: 'desc' };
    let selectedStrategy = null;
    let charts = {};

    // ── Color Palette ──
    const STRATEGY_COLORS = {
        'RSI Enhanced':             '#00e5a0',
        'Bollinger Enhanced':       '#00c8ff',
        'MA Crossover Enhanced':    '#8866ff',
        'MACD Enhanced':            '#ffaa00',
        'Volume Breakout Enhanced': '#ff4466',
        'Confluence Trend':         '#ff66cc',
        'Confluence Reversal':      '#44ddff',
        'Adaptive':                 '#ccff00',
        'Buy & Hold':               '#555570'
    };

    const V1_V2_MAP = {
        'RSI':              'RSI Enhanced',
        'Bollinger Bands':  'Bollinger Enhanced',
        'MA Crossover':     'MA Crossover Enhanced',
        'MACD':             'MACD Enhanced',
        'Volume Breakout':  'Volume Breakout Enhanced'
    };

    // ── Helpers ──
    function fmt(v, decimals = 2) {
        if (v == null || isNaN(v)) return '—';
        return Number(v).toFixed(decimals);
    }

    function fmtPct(v) {
        if (v == null || isNaN(v)) return '—';
        return (v >= 0 ? '+' : '') + Number(v).toFixed(2) + '%';
    }

    function fmtPrice(v) {
        if (v == null) return '—';
        return '$' + Number(v).toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 });
    }

    function fmtDate(iso) {
        if (!iso) return '—';
        const d = new Date(iso);
        return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
    }

    function fmtDateTime(iso) {
        if (!iso) return '—';
        const d = new Date(iso);
        return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) + ' ' +
               d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false });
    }

    function valClass(v, invert) {
        if (v == null || isNaN(v)) return 'val-neutral';
        if (invert) return v > 0 ? 'val-neg' : v < 0 ? 'val-pos' : 'val-neutral';
        return v > 0 ? 'val-pos' : v < 0 ? 'val-neg' : 'val-neutral';
    }

    function getData() {
        return BACKTEST_DATA[currentTimeframe];
    }

    function getStrategies() {
        return getData().strategies;
    }

    // ── Scatter Label Plugin ──
    const scatterLabelsPlugin = {
        id: 'scatterLabels',
        afterDatasetsDraw(chart) {
            if (chart.config.type !== 'scatter') return;
            const ctx = chart.ctx;
            ctx.save();
            ctx.font = "10px 'JetBrains Mono', monospace";
            ctx.textBaseline = 'middle';

            const area = chart.chartArea;

            chart.data.datasets.forEach((dataset, i) => {
                const meta = chart.getDatasetMeta(i);
                if (meta.hidden) return;
                ctx.fillStyle = dataset.borderColor || '#8888a8';
                meta.data.forEach(point => {
                    let x = point.x + 12;
                    let textWidth = ctx.measureText(dataset.label).width;
                    // If label would go off right edge, place it to the left
                    if (x + textWidth > area.right) {
                        x = point.x - textWidth - 12;
                        ctx.textAlign = 'left';
                    } else {
                        ctx.textAlign = 'left';
                    }
                    ctx.fillText(dataset.label, x, point.y - 2);
                });
            });
            ctx.restore();
        }
    };
    Chart.register(scatterLabelsPlugin);

    // ── Chart.js Defaults ──
    Chart.defaults.color = '#8888a8';
    Chart.defaults.borderColor = '#1a1a2e';
    Chart.defaults.font.family = "'JetBrains Mono', 'Source Code Pro', monospace";
    Chart.defaults.font.size = 11;
    Chart.defaults.plugins.legend.labels.usePointStyle = true;
    Chart.defaults.plugins.legend.labels.pointStyleWidth = 10;
    Chart.defaults.plugins.legend.labels.padding = 16;
    Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(12,12,20,0.95)';
    Chart.defaults.plugins.tooltip.borderColor = '#2a2a44';
    Chart.defaults.plugins.tooltip.borderWidth = 1;
    Chart.defaults.plugins.tooltip.titleFont = { family: "'JetBrains Mono', monospace", size: 11 };
    Chart.defaults.plugins.tooltip.bodyFont = { family: "'JetBrains Mono', monospace", size: 11 };
    Chart.defaults.plugins.tooltip.padding = 10;
    Chart.defaults.plugins.tooltip.cornerRadius = 4;

    // ── Initialize ──
    function init() {
        updateHeader();
        renderKPIStrip();
        renderPerformanceTable();
        renderComparisonChart();
        renderEquityChart();
        renderWalkForwardTable();
        renderRiskChart();
        bindEvents();
    }

    // ── Header ──
    function updateHeader() {
        const d = getData();
        document.getElementById('date-range').textContent =
            fmtDate(d.date_range.start) + ' → ' + fmtDate(d.date_range.end);
        document.getElementById('candle-count').textContent =
            d.candles.toLocaleString();
        document.getElementById('price-range').textContent =
            fmtPrice(d.price_range.min) + ' – ' + fmtPrice(d.price_range.max);
    }

    // ── KPI Strip ──
    function renderKPIStrip() {
        const strats = getStrategies();
        const names = Object.keys(strats);
        const d = getData();

        // Best strategy
        let bestName = names[0], bestReturn = -Infinity;
        let worstName = names[0], worstReturn = Infinity;
        let totalTrades = 0;
        let positiveCount = 0;

        names.forEach(name => {
            const r = strats[name].best.total_return_pct;
            totalTrades += strats[name].best.num_trades;
            if (r > bestReturn) { bestReturn = r; bestName = name; }
            if (r < worstReturn) { worstReturn = r; worstName = name; }
            if (r > 0) positiveCount++;
        });

        const bhReturn = strats[names[0]].best.buy_hold_return_pct;

        const strip = document.getElementById('kpi-strip');
        strip.innerHTML = `
            <div class="kpi-card kpi-green">
                <div class="kpi-label">Best Strategy</div>
                <div class="kpi-value val-pos">${fmtPct(bestReturn)}</div>
                <div class="kpi-sub">${bestName}</div>
            </div>
            <div class="kpi-card kpi-red">
                <div class="kpi-label">Buy & Hold</div>
                <div class="kpi-value ${valClass(bhReturn)}">${fmtPct(bhReturn)}</div>
                <div class="kpi-sub">Benchmark</div>
            </div>
            <div class="kpi-card kpi-cyan">
                <div class="kpi-label">Strategies Tested</div>
                <div class="kpi-value" style="color: var(--cyan)">${names.length}</div>
                <div class="kpi-sub">${positiveCount} profitable</div>
            </div>
            <div class="kpi-card kpi-amber">
                <div class="kpi-label">Total Trades</div>
                <div class="kpi-value" style="color: var(--amber)">${totalTrades}</div>
                <div class="kpi-sub">Across all strategies</div>
            </div>
            <div class="kpi-card kpi-purple">
                <div class="kpi-label">Alpha vs B&H</div>
                <div class="kpi-value ${valClass(bestReturn - bhReturn)}">${fmtPct(bestReturn - bhReturn)}</div>
                <div class="kpi-sub">Best strat - Buy&Hold</div>
            </div>
        `;
    }

    // ── Section 1: Performance Table ──
    function renderPerformanceTable() {
        const strats = getStrategies();
        const names = Object.keys(strats);
        const rows = names.map(name => {
            const b = strats[name].best;
            return {
                name,
                total_return_pct: b.total_return_pct,
                sharpe_ratio: b.sharpe_ratio,
                sortino_ratio: b.sortino_ratio,
                win_rate_pct: b.win_rate_pct,
                num_trades: b.num_trades,
                max_drawdown_pct: b.max_drawdown_pct,
                profit_factor: b.profit_factor,
                calmar_ratio: b.calmar_ratio
            };
        });

        // Sort
        rows.sort((a, b) => {
            let va = a[currentSort.key];
            let vb = b[currentSort.key];
            if (currentSort.key === 'name') {
                va = va || '';
                vb = vb || '';
                return currentSort.dir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
            }
            va = va == null ? -Infinity : va;
            vb = vb == null ? -Infinity : vb;
            return currentSort.dir === 'asc' ? va - vb : vb - va;
        });

        const tbody = document.getElementById('performance-tbody');
        tbody.innerHTML = rows.map(r => {
            const isNew = ['Confluence Trend', 'Confluence Reversal', 'Adaptive'].includes(r.name);
            const badge = isNew
                ? '<span class="strat-badge badge-new">NEW</span>'
                : '<span class="strat-badge badge-enhanced">ENH</span>';
            const sel = selectedStrategy === r.name ? ' selected' : '';
            return `<tr class="clickable${sel}" data-strategy="${r.name}">
                <td><span class="strat-name">${r.name}${badge}</span></td>
                <td class="num ${valClass(r.total_return_pct)}">${fmtPct(r.total_return_pct)}</td>
                <td class="num ${valClass(r.sharpe_ratio)}">${fmt(r.sharpe_ratio, 3)}</td>
                <td class="num ${valClass(r.sortino_ratio)}">${fmt(r.sortino_ratio, 3)}</td>
                <td class="num">${fmt(r.win_rate_pct, 1)}%</td>
                <td class="num">${r.num_trades}</td>
                <td class="num ${valClass(r.max_drawdown_pct, true)}">${fmt(r.max_drawdown_pct, 2)}%</td>
                <td class="num ${valClass(r.profit_factor)}">${r.profit_factor != null ? fmt(r.profit_factor, 3) : '—'}</td>
                <td class="num ${valClass(r.calmar_ratio)}">${fmt(r.calmar_ratio, 3)}</td>
            </tr>`;
        }).join('');

        // Update sort icons
        document.querySelectorAll('#performance-table thead th').forEach(th => {
            th.classList.remove('sort-asc', 'sort-desc');
            if (th.dataset.sort === currentSort.key) {
                th.classList.add(currentSort.dir === 'asc' ? 'sort-asc' : 'sort-desc');
            }
        });
    }

    // ── Section 2: v1 vs v2 Comparison Chart ──
    function renderComparisonChart() {
        const v1Data = V1_COMPARISON[currentTimeframe];
        const strats = getStrategies();

        const v1Names = Object.keys(V1_V2_MAP);
        const labels = v1Names.map(n => n);
        const v1Returns = v1Names.map(n => v1Data[n] ? v1Data[n].return : 0);
        const v2Returns = v1Names.map(n => {
            const v2Name = V1_V2_MAP[n];
            return strats[v2Name] ? strats[v2Name].best.total_return_pct : 0;
        });

        if (charts.comparison) charts.comparison.destroy();

        const ctx = document.getElementById('comparison-chart').getContext('2d');
        charts.comparison = new Chart(ctx, {
            type: 'bar',
            data: {
                labels,
                datasets: [
                    {
                        label: 'v1 Return %',
                        data: v1Returns,
                        backgroundColor: v1Returns.map(v => v >= 0 ? 'rgba(85, 85, 112, 0.7)' : 'rgba(85, 85, 112, 0.4)'),
                        borderColor: '#555570',
                        borderWidth: 1,
                        borderRadius: 3,
                        barPercentage: 0.7,
                        categoryPercentage: 0.65
                    },
                    {
                        label: 'v2 Return %',
                        data: v2Returns,
                        backgroundColor: v2Returns.map(v => v >= 0 ? 'rgba(0, 229, 160, 0.7)' : 'rgba(255, 68, 102, 0.5)'),
                        borderColor: v2Returns.map(v => v >= 0 ? '#00e5a0' : '#ff4466'),
                        borderWidth: 1,
                        borderRadius: 3,
                        barPercentage: 0.7,
                        categoryPercentage: 0.65
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { position: 'top' },
                    tooltip: {
                        callbacks: {
                            label: ctx => ctx.dataset.label + ': ' + fmtPct(ctx.parsed.y)
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { font: { size: 10 } }
                    },
                    y: {
                        grid: { color: '#1a1a2e' },
                        ticks: {
                            callback: v => v + '%',
                            font: { size: 10 }
                        },
                        title: {
                            display: true,
                            text: 'Total Return %',
                            font: { size: 11 }
                        }
                    }
                }
            }
        });
    }

    // ── Section 3: Equity Curves ──
    function renderEquityChart() {
        const strats = getStrategies();
        const names = Object.keys(strats);

        // Use the first strategy's equity curve for time labels
        const firstEc = strats[names[0]].best.equity_curve;
        const labels = firstEc.map(p => p.time);
        const basePrice = firstEc[0].price;

        const datasets = [];

        // Buy & Hold baseline
        datasets.push({
            label: 'Buy & Hold',
            data: firstEc.map(p => (p.price / basePrice) * 10000),
            borderColor: '#555570',
            backgroundColor: 'transparent',
            borderWidth: 2,
            borderDash: [6, 3],
            pointRadius: 0,
            tension: 0.1,
            order: 10
        });

        // Strategy equity curves
        names.forEach((name, i) => {
            const ec = strats[name].best.equity_curve;
            datasets.push({
                label: name,
                data: ec.map(p => p.equity),
                borderColor: STRATEGY_COLORS[name] || '#888',
                backgroundColor: 'transparent',
                borderWidth: 1.8,
                pointRadius: 0,
                tension: 0.1,
                order: i
            });
        });

        if (charts.equity) charts.equity.destroy();

        const ctx = document.getElementById('equity-chart').getContext('2d');
        charts.equity = new Chart(ctx, {
            type: 'line',
            data: { labels, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            font: { size: 10 },
                            boxWidth: 14,
                            padding: 12
                        },
                        onClick: function (e, legendItem, legend) {
                            const index = legendItem.datasetIndex;
                            const ci = legend.chart;
                            const meta = ci.getDatasetMeta(index);
                            meta.hidden = meta.hidden === null ? !ci.data.datasets[index].hidden : null;
                            ci.update();
                        }
                    },
                    tooltip: {
                        callbacks: {
                            title: ctx => {
                                if (ctx[0]) return fmtDateTime(labels[ctx[0].dataIndex]);
                                return '';
                            },
                            label: ctx => ctx.dataset.label + ': $' + fmt(ctx.parsed.y, 2)
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: {
                            maxTicksLimit: 10,
                            callback: function(value, index) {
                                return fmtDate(labels[index]);
                            },
                            font: { size: 9 }
                        }
                    },
                    y: {
                        grid: { color: '#1a1a2e' },
                        ticks: {
                            callback: v => '$' + v.toLocaleString(),
                            font: { size: 10 }
                        },
                        title: {
                            display: true,
                            text: 'Portfolio Value ($)',
                            font: { size: 11 }
                        }
                    }
                }
            }
        });
    }

    // ── Section 4: Strategy Details ──
    function renderStrategyDetails(name) {
        selectedStrategy = name;
        const strat = getStrategies()[name];
        if (!strat) return;

        document.getElementById('detail-placeholder').style.display = 'none';
        document.getElementById('detail-content').style.display = 'block';
        document.getElementById('detail-strategy-name').textContent = name;

        // Params
        const params = strat.best.params || {};
        const paramsEl = document.getElementById('detail-params');
        paramsEl.innerHTML = Object.entries(params).map(([k, v]) =>
            `<div class="param-item">
                <span class="param-key">${k}</span>
                <span class="param-val">${v}</span>
            </div>`
        ).join('');

        // Exit Pie Chart
        const eb = strat.best.exit_breakdown || {};
        const pieLabels = [];
        const pieData = [];
        const pieColors = [];
        const colorMap = {
            stop_loss: '#ff4466',
            take_profit: '#00e5a0',
            signal: '#00c8ff',
            close: '#ffaa00'
        };
        const labelMap = {
            stop_loss: 'Stop Loss',
            take_profit: 'Take Profit',
            signal: 'Signal Exit',
            close: 'Close'
        };

        for (const [key, val] of Object.entries(eb)) {
            if (val > 0) {
                pieLabels.push(labelMap[key] || key);
                pieData.push(val);
                pieColors.push(colorMap[key] || '#888');
            }
        }

        if (charts.exitPie) charts.exitPie.destroy();

        if (pieData.length > 0) {
            const ctx = document.getElementById('exit-pie-chart').getContext('2d');
            charts.exitPie = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: pieLabels,
                    datasets: [{
                        data: pieData,
                        backgroundColor: pieColors,
                        borderColor: '#0f0f1a',
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    cutout: '55%',
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: { font: { size: 10 }, padding: 12 }
                        },
                        tooltip: {
                            callbacks: {
                                label: ctx => {
                                    const total = ctx.dataset.data.reduce((a, b) => a + b, 0);
                                    const pct = ((ctx.parsed / total) * 100).toFixed(1);
                                    return ctx.label + ': ' + ctx.parsed + ' (' + pct + '%)';
                                }
                            }
                        }
                    }
                }
            });
        }

        // Trades table
        const trades = strat.best.trades || [];
        const tradesTbody = document.getElementById('trades-tbody');
        tradesTbody.innerHTML = trades.map((t, i) => {
            const isSell = t.type.startsWith('SELL');
            const pnlClass = t.pnl != null ? valClass(t.pnl) : '';
            return `<tr>
                <td class="num">${i + 1}</td>
                <td style="color: ${isSell ? '#ff4466' : '#00e5a0'}">${t.type}</td>
                <td>${fmtDateTime(t.time)}</td>
                <td class="num">${fmtPrice(t.price)}</td>
                <td class="num">${t.amount != null ? fmt(t.amount, 6) : '—'}</td>
                <td class="num ${pnlClass}">${t.pnl != null ? (t.pnl >= 0 ? '+$' : '-$') + Math.abs(t.pnl).toFixed(2) : '—'}</td>
                <td class="num ${pnlClass}">${t.pnl_pct != null ? fmtPct(t.pnl_pct) : '—'}</td>
            </tr>`;
        }).join('');

        // Walk-forward folds
        const wfContainer = document.getElementById('detail-walkforward-container');
        const wf = strat.walk_forward;
        if (wf && wf.folds && wf.folds.length > 0) {
            wfContainer.style.display = 'block';
            const wfTbody = document.getElementById('wf-fold-tbody');
            wfTbody.innerHTML = wf.folds.map(f => {
                const paramStr = Object.entries(f.params || {}).map(([k,v]) => k + '=' + v).join(', ');
                return `<tr>
                    <td class="num">${f.fold}</td>
                    <td style="font-size:10px; max-width: 300px; white-space: normal; word-break: break-all;">${paramStr}</td>
                    <td class="num ${valClass(f.train_return)}">${fmtPct(f.train_return)}</td>
                    <td class="num ${valClass(f.test_return)}">${fmtPct(f.test_return)}</td>
                    <td class="num ${valClass(f.train_sharpe)}">${fmt(f.train_sharpe, 3)}</td>
                    <td class="num ${valClass(f.test_sharpe)}">${fmt(f.test_sharpe, 3)}</td>
                </tr>`;
            }).join('');
        } else {
            wfContainer.style.display = 'none';
        }

        // Highlight row
        document.querySelectorAll('#performance-tbody tr').forEach(tr => {
            tr.classList.toggle('selected', tr.dataset.strategy === name);
        });

        // Scroll into view
        document.getElementById('details').scrollIntoView({ behavior: 'smooth' });
    }

    // ── Section 5: Walk-Forward Analysis ──
    function renderWalkForwardTable() {
        const strats = getStrategies();
        const tbody = document.getElementById('walkforward-tbody');
        const rows = [];

        for (const [name, strat] of Object.entries(strats)) {
            const wf = strat.walk_forward;
            if (!wf) continue;
            rows.push({
                name,
                avg_oos_return: wf.avg_oos_return_pct,
                avg_oos_sharpe: wf.avg_oos_sharpe,
                avg_train_return: wf.avg_train_return_pct,
                overfit_ratio: wf.overfit_ratio
            });
        }

        tbody.innerHTML = rows.map(r => {
            let assessment, assessClass;
            const or = Math.abs(r.overfit_ratio);
            if (or >= 0.5 && or <= 1.5 && r.overfit_ratio > 0) {
                assessment = 'ROBUST';
                assessClass = 'assessment-good';
            } else if ((or >= 0.3 && or < 0.5) || (or > 1.5 && or <= 3.0) || r.overfit_ratio < 0) {
                assessment = 'CAUTION';
                assessClass = 'assessment-caution';
            } else {
                assessment = 'OVERFIT';
                assessClass = 'assessment-poor';
            }
            // Special: if OOS return is 0 and overfit is 0, it means no trades OOS
            if (r.avg_oos_return === 0 && r.overfit_ratio === 0) {
                assessment = 'NO OOS DATA';
                assessClass = 'assessment-caution';
            }
            // Negative overfit with negative meanings
            if (r.overfit_ratio < 0) {
                assessment = 'UNSTABLE';
                assessClass = 'assessment-poor';
            }

            return `<tr>
                <td><span class="strat-name">${r.name}</span></td>
                <td class="num ${valClass(r.avg_oos_return)}">${fmtPct(r.avg_oos_return)}</td>
                <td class="num ${valClass(r.avg_oos_sharpe)}">${fmt(r.avg_oos_sharpe, 3)}</td>
                <td class="num ${valClass(r.avg_train_return)}">${fmtPct(r.avg_train_return)}</td>
                <td class="num">${fmt(r.overfit_ratio, 3)}</td>
                <td class="num ${assessClass}">${assessment}</td>
            </tr>`;
        }).join('');
    }

    // ── Section 6: Risk-Return Scatter ──
    function renderRiskChart() {
        const strats = getStrategies();
        const points = [];

        for (const [name, strat] of Object.entries(strats)) {
            const b = strat.best;
            points.push({
                x: b.max_drawdown_pct,
                y: b.total_return_pct,
                name
            });
        }

        // Add Buy & Hold
        const firstStrat = Object.values(strats)[0].best;
        const bhReturn = firstStrat.buy_hold_return_pct;
        // Approximate buy & hold drawdown from price range
        const d = getData();
        const bhDD = ((d.price_range.max - d.price_range.min) / d.price_range.max * 100);
        points.push({ x: bhDD, y: bhReturn, name: 'Buy & Hold' });

        if (charts.risk) charts.risk.destroy();

        const ctx = document.getElementById('risk-chart').getContext('2d');
        charts.risk = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: points.map(p => ({
                    label: p.name,
                    data: [{ x: p.x, y: p.y }],
                    backgroundColor: STRATEGY_COLORS[p.name] || '#888',
                    borderColor: STRATEGY_COLORS[p.name] || '#888',
                    pointRadius: 7,
                    pointHoverRadius: 10,
                    pointStyle: p.name === 'Buy & Hold' ? 'rectRot' : 'circle'
                }))
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { font: { size: 10 }, padding: 10 }
                    },
                    tooltip: {
                        callbacks: {
                            title: ctx => ctx[0] ? ctx[0].dataset.label : '',
                            label: ctx => [
                                'Return: ' + fmtPct(ctx.parsed.y),
                                'Max DD: ' + fmt(ctx.parsed.x, 2) + '%'
                            ]
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Max Drawdown %',
                            font: { size: 12, weight: '600' }
                        },
                        grid: { color: '#1a1a2e' },
                        ticks: {
                            callback: v => v + '%',
                            font: { size: 10 }
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Total Return %',
                            font: { size: 12, weight: '600' }
                        },
                        grid: { color: '#1a1a2e' },
                        ticks: {
                            callback: v => v + '%',
                            font: { size: 10 }
                        }
                    }
                }
            }
        });

        // Done - labels handled by the plugin below
    }

    // ── Event Bindings ──
    function bindEvents() {
        // Timeframe tabs
        document.querySelectorAll('.tf-tab').forEach(tab => {
            tab.addEventListener('click', () => {
                if (tab.dataset.tf === currentTimeframe) return;
                currentTimeframe = tab.dataset.tf;
                document.querySelectorAll('.tf-tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                selectedStrategy = null;
                document.getElementById('detail-placeholder').style.display = 'block';
                document.getElementById('detail-content').style.display = 'none';
                updateHeader();
                renderKPIStrip();
                renderPerformanceTable();
                renderComparisonChart();
                renderEquityChart();
                renderWalkForwardTable();
                renderRiskChart();
            });
        });

        // Sort columns
        document.querySelectorAll('#performance-table thead th.sortable').forEach(th => {
            th.addEventListener('click', () => {
                const key = th.dataset.sort;
                if (currentSort.key === key) {
                    currentSort.dir = currentSort.dir === 'desc' ? 'asc' : 'desc';
                } else {
                    currentSort.key = key;
                    currentSort.dir = 'desc';
                }
                renderPerformanceTable();
            });
        });

        // Row click for details
        document.getElementById('performance-tbody').addEventListener('click', (e) => {
            const row = e.target.closest('tr[data-strategy]');
            if (row) {
                renderStrategyDetails(row.dataset.strategy);
            }
        });

        // Scroll-based nav highlighting
        const sections = ['performance', 'comparison', 'equity', 'details', 'walkforward', 'risk'];
        const navLinks = document.querySelectorAll('.nav-link');

        const observer = new IntersectionObserver(entries => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    navLinks.forEach(link => {
                        link.classList.toggle('active',
                            link.getAttribute('href') === '#' + entry.target.id);
                    });
                }
            });
        }, { rootMargin: '-30% 0px -60% 0px' });

        sections.forEach(id => {
            const el = document.getElementById(id);
            if (el) observer.observe(el);
        });
    }

    // ── Boot ──
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
