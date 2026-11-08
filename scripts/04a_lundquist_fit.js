// Lundquist flux-rope fits to hourly OMNI GSE field inside magnetic clouds (run in a browser on the cdaweb.gsfc.nasa.gov origin
// after the OMNI2_H0_MRG1HR download that fills window._vb with "yyMMddHHmm:bx by bz V|..." strings).
// Free parameters: axis latitude th and longitude ph (GSE), impact parameter p (units of R), axial field B0, centre shift s0;
// handedness H = +-1 is tried separately. Output: start, chi (rms residual / mean |B|), th, ph, p, B0, H, s0, mean V, n hours.
const X0 = 2.405;
const J0 = x => { let s = 0, t = 1; for (let k = 0; k < 25; k++) { if (k > 0) t *= -(x * x / 4) / (k * k); s += t; } return s; };
const J1 = x => { let s = 0, t = x / 2; for (let k = 0; k < 25; k++) { if (k > 0) t *= -(x * x / 4) / (k * (k + 1)); s += t; } return s; };
function model(q, h, n) {
  const [th, ph, p, b0, s0] = q;
  const ax = [Math.cos(th) * Math.cos(ph), Math.cos(th) * Math.sin(ph), Math.sin(th)];
  const xs = [-1, 0, 0]; const d = -ax[0];
  let xc = [xs[0] - d * ax[0], xs[1] - d * ax[1], xs[2] - d * ax[2]]; const nx = Math.hypot(...xc) || 1e-9; xc = xc.map(v => v / nx);
  const yc = [ax[1] * xc[2] - ax[2] * xc[1], ax[2] * xc[0] - ax[0] * xc[2], ax[0] * xc[1] - ax[1] * xc[0]];
  const half = Math.sqrt(Math.max(1 - p * p, 1e-3)); const out = [];
  for (let i = 0; i < n; i++) {
    const u = -half + 2 * half * (n > 1 ? i / (n - 1) : 0.5) + s0 * half; const r = Math.min(Math.hypot(u, p), 1.2), phi = Math.atan2(p, u);
    const ba = b0 * J0(X0 * r), bt = b0 * h * J1(X0 * r);
    out.push([0, 1, 2].map(k => ba * ax[k] + bt * (-Math.sin(phi) * xc[k] + Math.cos(phi) * yc[k])));
  }
  return out;
}
function nelder(f, x0, step, it = 1500) {
  const n = x0.length; let S = [x0.slice()]; for (let i = 0; i < n; i++) { const x = x0.slice(); x[i] += step[i]; S.push(x); }
  let F = S.map(f);
  for (let k = 0; k < it; k++) {
    const o = F.map((v, i) => i).sort((a, b) => F[a] - F[b]); S = o.map(i => S[i]); F = o.map(i => F[i]);
    if (Math.abs(F[n] - F[0]) < 1e-9) break;
    const c = Array(n).fill(0); for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) c[j] += S[i][j] / n;
    const pt = a => c.map((v, j) => v + a * (S[n][j] - v));
    const xr = pt(-1), fr = f(xr);
    if (fr < F[0]) { const xe = pt(-2), fe = f(xe); if (fe < fr) { S[n] = xe; F[n] = fe; } else { S[n] = xr; F[n] = fr; } }
    else if (fr < F[n - 1]) { S[n] = xr; F[n] = fr; }
    else { const xk = pt(0.5), fk = f(xk); if (fk < F[n]) { S[n] = xk; F[n] = fk; } else { for (let i = 1; i <= n; i++) { S[i] = S[i].map((v, j) => S[0][j] + 0.5 * (v - S[0][j])); F[i] = f(S[i]); } } }
  }
  return [S[0], F[0]];
}
function fitEvent(rows) {
  const n = rows.length; const idx = [], obs = [];
  rows.forEach((r, i) => { if (r.slice(0, 3).every(v => isFinite(v))) { idx.push(i); obs.push(r.slice(0, 3)); } });
  if (idx.length < 6) return null;
  const bm = obs.reduce((s, b) => s + Math.hypot(...b), 0) / obs.length; const bmax = Math.max(...obs.map(b => Math.hypot(...b)));
  let best = null;
  for (const h of [1, -1]) for (const th0 of [-0.8, 0, 0.8]) for (let k = 0; k < 6; k++) for (const p0 of [0.1, 0.5]) {
    const f = q => {
      const [th, ph, p, b0, s0] = q; let pen = 0;
      if (Math.abs(th) > Math.PI / 2) pen += 10 * (Math.abs(th) - Math.PI / 2); if (p < 0 || p > 0.95) pen += 10 * Math.abs(p < 0 ? p : p - 0.95);
      if (b0 < 0) pen += 10 * -b0 / bm; if (Math.abs(s0) > 0.3) pen += 10 * (Math.abs(s0) - 0.3);
      const m = model([Math.max(-1.57, Math.min(1.57, th)), ph, Math.max(0, Math.min(0.95, p)), Math.abs(b0), Math.max(-0.3, Math.min(0.3, s0))], h, n);
      let s = 0; idx.forEach((i, j) => { for (let c = 0; c < 3; c++) s += (m[i][c] - obs[j][c]) ** 2; });
      return Math.sqrt(s / (3 * idx.length)) / bm + pen;
    };
    const [x, v] = nelder(f, [th0, k * Math.PI / 3, p0, bmax, 0], [0.3, 0.5, 0.2, 0.3 * bmax, 0.1]);
    if (!best || v < best.v) best = { v, x, h };
  }
  const V = rows.map(r => r[3]).filter(v => isFinite(v)); const Vm = V.reduce((a, b) => a + b, 0) / V.length;
  return [best.v, ...best.x.map(v => +v.toFixed(4)), best.h, Math.round(Vm), n];
}
window._lq = []; window._stl = "start";
(async () => {
  for (const blk of window._vb) {
    const [k, v] = blk.split(":"); if (v === "ERR") continue;
    const rows = v.split("|").map(r => r.split(" ").map(x => x === "" ? NaN : +x));
    const f = fitEvent(rows); window._lq.push(k + "," + (f ? f.map(z => typeof z === "number" ? +z.toFixed(4) : z).join(",") : "NA"));
    window._stl = "n " + window._lq.length; await new Promise(r => setTimeout(r, 0));
  }
  window._stl += " DONE";
})();
"launched";
