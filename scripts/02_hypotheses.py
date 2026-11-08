#!/usr/bin/env python3
"""Tests of the 'interlocked merger / partial flux annihilation / early energy release' picture with observed quantities.
T1: expansion and compression of merged ICMEs against the launch separation of the CMEs (early merging = small separation).
T2: magnetic flux proxy against the number N of associated CMEs: Phi ~ N^alpha; annihilated fraction for N = 2.
T3: temperature excess (relative to single ICMEs at the same expansion and speed) against the launch separation."""
import json, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
B = Path(__file__).resolve().parent; ROOT = B.parents[2]
X = pd.read_csv(B / "icme_plasma.csv", dtype={"dist": str}); X = X[(X.dur_h >= 6) & X.V.notna()].copy()
AU = 1.496e8
X["zeta"] = (X.Vfront - X.Vrear) / (X.dur_h * 3600) * AU / X.V ** 2; X["size_au"] = X.V * X.dur_h * 3600 / AU; X["flux"] = X.B * X.size_au
R = pd.read_csv(B.parent / "data/rc_icme_2011_2026.csv", dtype=str); R["dist"] = R.disturbance.str.replace("S", "").str[2:]
R["cmes"] = R.cme_times.str.split("/").apply(lambda v: sorted(pd.Timestamp(x[:8] + "T" + x[8:], tz="UTC") for x in v))
R.loc[R.disturbance.str.startswith("202203131047"), "cme_times"] = "202203092348/202203100200/202203101848"
R["cmes"] = R.cme_times.str.split("/").apply(lambda v: sorted(pd.Timestamp(x[:8] + "T" + x[8:], tz="UTC") for x in v))
R["dt_launch_h"] = R.cmes.apply(lambda v: (v[-1] - v[0]).total_seconds() / 3600)
# CDAW speeds of the associated CMEs and the earliest catch-up distance (constant speeds)
rows = []
for l in open(ROOT / "data/CME_all.txt", errors="ignore"):
    m = re.match(r"(\d{4}/\d\d/\d\d)\s+(\d\d:\d\d:\d\d)\s+(\S+)\s+(\d+)\s+(\d+)", l)
    if m: rows.append((pd.Timestamp(f"{m[1]} {m[2]}", tz="UTC"), int(m[5])))
C = pd.DataFrame(rows, columns=["t", "v"])
def speeds(v):
    out = []
    for t in v:
        c = C[(C.t - t).abs() <= pd.Timedelta(minutes=30)]; out.append(int(c.iloc[(c.t - t).abs().argmin()].v) if len(c) else np.nan)
    return out
def catch(v, s):
    best = np.nan
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            if np.isfinite(s[i]) and np.isfinite(s[j]) and s[j] > s[i]:
                r = s[i] * s[j] * (v[j] - v[i]).total_seconds() / (s[j] - s[i]) / AU
                best = r if not np.isfinite(best) else min(best, r)
    return best
R["speeds"] = R.cmes.apply(speeds); R["r_catch"] = [catch(a, b) for a, b in zip(R.cmes, R.speeds)]
X = X.merge(R[["dist", "dt_launch_h", "r_catch"]], on="dist", how="left")
S, M = X[X.ncme == 1], X[X.ncme >= 2].copy(); out = {}
# T1
for c in ("zeta", "N", "B"):
    r = spearmanr(M.dt_launch_h, M[c], nan_policy="omit"); out[f"T1_{c}"] = dict(rho=float(r[0]), p=float(r[1]), n=int(M[[c, "dt_launch_h"]].dropna().shape[0]))
    print(f"T1 merged: rho(launch separation, {c}) = {r[0]:.2f}, p = {r[1]:.2f}")
m = M.r_catch.notna(); print("merged with a computable catch-up distance:", int(m.sum()), "median r_catch", round(M.r_catch.median(), 2), "au")
if m.sum() >= 6:
    r = spearmanr(M.r_catch[m], M.zeta[m]); out["T1_rcatch_zeta"] = dict(rho=float(r[0]), p=float(r[1]), n=int(m.sum())); print(" rho(r_catch, zeta)", round(r[0], 2), "p", round(r[1], 2))
# T2: flux against N
X["Ncl"] = np.where(X.ncme >= 3, 3, X.ncme); Y = X[X.ncme >= 1]
for n, z in Y.groupby("Ncl"): print(f"T2 N={n}{'+' if n == 3 else ''}: n={len(z)}, median flux {z.flux.median():.2f} nT au, B {z.B.median():.1f}, size {z.size_au.median():.2f} au")
rng = np.random.default_rng(3)
def alpha(d): return np.polyfit(np.log10(d.ncme), np.log10(d.flux), 1)[0]
a0 = alpha(Y); bs = [alpha(Y.sample(len(Y), replace=True, random_state=int(s))) for s in rng.integers(0, 1e9, 3000)]
out["T2_alpha"] = dict(val=float(a0), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), n=int(len(Y))); print(f"T2: Phi ~ N^alpha, alpha = {a0:.2f} [{np.percentile(bs,2.5):.2f}, {np.percentile(bs,97.5):.2f}]")
two = X[X.ncme == 2]; f = 1 - two.flux.median() / (2 * S.flux.median())
bf = [1 - two.flux.sample(len(two), replace=True, random_state=int(s)).median() / (2 * S.flux.sample(len(S), replace=True, random_state=int(s) + 1).median()) for s in rng.integers(0, 1e9, 3000)]
allm = 1 - M.flux.median() / (M.ncme.median() * S.flux.median())
out["T2_fann_N2"] = dict(val=float(f), lo=float(np.percentile(bf, 2.5)), hi=float(np.percentile(bf, 97.5)), n=int(len(two))); print(f"T2: annihilated fraction for N=2: {f:.2f} [{np.percentile(bf,2.5):.2f}, {np.percentile(bf,97.5):.2f}] (n={len(two)}); all merged with median N={M.ncme.median():.0f}: {allm:.2f}")
# T3: temperature excess against launch separation
k = np.linalg.lstsq(np.column_stack([np.ones(len(S)), S.zeta, np.log10(S.V)]), np.log10(S.Tratio), rcond=None)[0]
M["Texc"] = np.log10(M.Tratio) - (k[0] + k[1] * M.zeta + k[2] * np.log10(M.V))
r = spearmanr(M.dt_launch_h, M.Texc, nan_policy="omit"); out["T3"] = dict(rho=float(r[0]), p=float(r[1])); print(f"T3 merged: rho(launch separation, T excess) = {r[0]:.2f}, p = {r[1]:.2f}")
lo, hi = M[M.dt_launch_h <= M.dt_launch_h.median()], M[M.dt_launch_h > M.dt_launch_h.median()]
print(f"   T excess (dex): close launches {lo.Texc.median():+.2f} (n={len(lo)}), distant launches {hi.Texc.median():+.2f} (n={len(hi)}), MWU p {mannwhitneyu(lo.Texc, hi.Texc).pvalue:.2f}")
out["T3_split"] = dict(close=float(lo.Texc.median()), distant=float(hi.Texc.median()), p=float(mannwhitneyu(lo.Texc, hi.Texc).pvalue), split_h=float(M.dt_launch_h.median()))
M[["dist", "ncme", "dt_launch_h", "r_catch", "zeta", "Tratio", "Texc", "flux", "N", "B", "sigv"]].to_csv(B / "merged_details.csv", index=False)
json.dump(out, open(B / "hypotheses.json", "w"), indent=1)
