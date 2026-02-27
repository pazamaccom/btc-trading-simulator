// BACKTEST_DATA assembled from split files (data_a.js through data_f.js)
const BACKTEST_DATA = {
  "1h": Object.assign({}, BD_1H_A, {
    strategies: Object.assign({}, BD_1H_A.strategies, BD_1H_B.strategies, BD_1H_C.strategies)
  }),
  "1d": Object.assign({}, BD_1D_A, {
    strategies: Object.assign({}, BD_1D_A.strategies, BD_1D_B.strategies, BD_1D_C.strategies)
  })
};