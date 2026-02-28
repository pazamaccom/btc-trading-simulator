// BTC Trading Simulator v6 — Compressed Data (v6 ensemble)
// Requires pako.js loaded before this script
(function() {
  var b64 = "PLACEHOLDER";
  var raw = atob(b64); var ua = new Uint8Array(raw.length);
  for (var i = 0; i < raw.length; i++) ua[i] = raw.charCodeAt(i);
  var src = pako.inflate(ua, { to: 'string' });
  // Define as window globals instead of const (for cross-script access)
  src = src.replace(/^const /gm, 'window.');
  (new Function(src))();
})();
