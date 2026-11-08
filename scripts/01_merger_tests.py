#!/usr/bin/env python3
"""Merged (two or more associated CMEs) versus single-CME ICMEs, Richardson-Cane list 2011-2025 with hourly OMNI plasma inside the ICME.
Tests of the 'merged structure' hypothesis: weaker expansion, heating beyond the adiabatic expectation, magnetic flux addition."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
B = Path(__file__).resolve().parent
X = pd.read_csv(B / "icme_plasma.csv", dtype={"dist": str, "start": str, "end": str})
X = X[(X.dur_h >= 6) & X.V.notna()].copy()
AU = 1.496e8
X["zeta"] = (X.Vfront - X.Vrear) / (X.dur_h * 3600) * AU / X.V ** 2          # dimensionless expansion rate (Demoulin & Dasso 2009)
X["size_au"] = X.V * X.dur_h * 3600 / AU                                       # radial size
X["flux"] = X.B * X.size_au                                                    # flux per unit length proxy, nT au
X["grp"] = np.where(X.ncme >= 2, "merged", np.where(X.ncme == 1, "single", "none"))
S, M = X[X.grp == "single"], X[X.grp == "merged"]
rng = np.random.default_rng(1); out = {"n_single": len(S), "n_merged": len(M)}
def cmp(c, log=False):
    a, b = S[c].dropna(), M[c].dropna()
    f = (lambda v: np.log10(v[v > 0])) if log else (lambda v: v)
    d0 = np.median(f(b)) - np.median(f(a)); bs = [np.median(f(b.sample(len(b), replace=True, random_state=int(s)))) - np.median(f(a.sample(len(a), replace=True, random_state=int(s) + 7))) for s in rng.integers(0, 1e9, 3000)]
    p = mannwhitneyu(a, b).pvalue
    r = dict(single=float(np.median(a)), merged=float(np.median(b)), diff=float(d0), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), p=float(p), log=log)
    print(f"{c:7s} single {r['single']:.3g} merged {r['merged']:.3g}  diff{' (dex)' if log else ''} {d0:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]  MWU p {p:.3g}"); return r
for c, lg in (("zeta", False), ("Tratio", True), ("beta", True), ("B", True), ("size_au", True), ("flux", True), ("V", False), ("N", True), ("shB", True), ("dV", False)):
    out[c] = cmp(c, lg)
out["mc_frac"] = dict(single=float((S.mc == 2).mean()), merged=float((M.mc == 2).mean())); print("magnetic-cloud fraction", out["mc_frac"])
# heating beyond the reduced expansion: regress log Tratio on zeta and log V in single ICMEs, apply to merged
from numpy.linalg import lstsq
def fit(D):
    A = np.column_stack([np.ones(len(D)), D.zeta, np.log10(D.V)]); return lstsq(A, np.log10(D.Tratio), rcond=None)[0]
D = S.dropna(subset=["zeta", "Tratio", "V"]); D = D[D.Tratio > 0]; k = fit(D)
Mm = M.dropna(subset=["zeta", "Tratio", "V"]); pred = k[0] + k[1] * Mm.zeta + k[2] * np.log10(Mm.V); exc = np.log10(Mm.Tratio) - pred
bs = []
for s in range(2000):
    d = D.sample(len(D), replace=True, random_state=s); kk = fit(d); m = Mm.sample(len(Mm), replace=True, random_state=s + 99)
    bs.append(np.median(np.log10(m.Tratio) - (kk[0] + kk[1] * m.zeta + kk[2] * np.log10(m.V))))
out["heat_excess"] = dict(coef=list(map(float, k)), excess=float(np.median(exc)), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
print("single fit log Tratio = a + b zeta + c log V:", np.round(k, 3), "; merged excess (dex):", round(np.median(exc), 3), np.round(np.percentile(bs, [2.5, 97.5]), 3))
r = spearmanr(S.zeta, np.log10(S.Tratio), nan_policy="omit"); out["rho_zeta_T_single"] = float(r[0]); print("single: rho(zeta, log Tratio)", round(r[0], 2), "p", round(r[1], 3))
# adiabatic expectation: T ~ (volume)^-(gamma-1); a merged ICME expanding with zeta_m instead of zeta_s keeps a factor (r/r0)^{-2(gamma-1)(zeta_m - zeta_s)}... evaluated from 0.3 to 1 au
g = 5 / 3; zs, zm = out["zeta"]["single"], out["zeta"]["merged"]
for r0 in (0.3, 0.5):
    fac = (1 / r0) ** (2 * (g - 1) * (zs - zm)); print(f"adiabatic temperature gain from reduced expansion since {r0} au: x{fac:.2f}")
    out[f"adiabatic_gain_{r0}"] = float(fac)
json.dump(out, open(B / "merger_tests.json", "w"), indent=1)
