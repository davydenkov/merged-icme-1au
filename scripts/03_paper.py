#!/usr/bin/env python3
"""All numbers and figures of the merged-ICME paper. Groups: single-CME ICMEs (one associated CME) and merged ICMEs (two or more) in the
Richardson-Cane list 2011-2025; ICMEs without an associated CME are excluded. Medians with 95% bootstrap intervals; Mann-Whitney tests."""
import json, re
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr
B = Path(__file__).resolve().parent; F = B / "paper/figures"
C1, C2, C3, INK, MUTE = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6, "axes.edgecolor": MUTE,
                     "xtick.color": MUTE, "ytick.color": MUTE, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "figure.constrained_layout.use": True})
AU = 1.496e8
X = pd.read_csv(B / "icme_plasma.csv", dtype={"dist": str}); X = X[(X.dur_h >= 6) & X.V.notna()].copy()
X["year"] = 2000 + X.dist.str[:2].astype(int)
X["zeta"] = (X.Vfront - X.Vrear) / (X.dur_h * 3600) * AU / X.V ** 2; X["size_au"] = X.V * X.dur_h * 3600 / AU; X["flux"] = X.B * X.size_au
X["f75"] = 1e3 * X.d75 / X.nmin; X["r5"] = np.log10(X.in5 / X.up5); X["r1"] = np.log10(X.in1 / X.up1)
# energetic protons at the shock (3-9 MeV GOES ratio) from the ESP sample
E = pd.read_csv(B.parent / "data/esp_sample.csv", parse_dates=["t_dist"]); E["dist"] = E.t_dist.dt.strftime("%y%m%d%H%M")
X = X.merge(E[["dist", "Glow_ratio"]], on="dist", how="left")
S, M = X[X.ncme == 1].copy(), X[X.ncme >= 2].copy(); rng = np.random.default_rng(11); N = {}
N["nAll"], N["nSingle"], N["nMerged"], N["nNone"] = len(X), len(S), len(M), int((X.ncme == 0).sum())
N["nMergedRecent"] = int((M.year >= 2021).sum())
def med_ci(v, log=False):
    v = v.dropna(); v = np.log10(v[v > 0]) if log else v
    bs = [np.median(rng.choice(v.values, len(v))) for _ in range(3000)]; return float(np.median(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
def diff_ci(c, log=False):
    a, b = S[c].dropna(), M[c].dropna()
    if log: a, b = np.log10(a[a > 0]), np.log10(b[b > 0])
    d = [np.median(rng.choice(b.values, len(b))) - np.median(rng.choice(a.values, len(a))) for _ in range(3000)]
    return float(np.median(b) - np.median(a)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(mannwhitneyu(a, b).pvalue)
for c, lg, k in (("zeta", False, "Zeta"), ("N", True, "Dens"), ("B", True, "Bfield"), ("shB", True, "ShB"), ("flux", True, "Flux"), ("Tratio", True, "Trat"), ("beta", True, "Beta"),
                 ("sigv", True, "Sig"), ("siga", True, "Siga"), ("size_au", True, "Size"), ("V", False, "Vel"), ("rho_bn", False, "Rbn"), ("f75", False, "Dips"), ("r5", False, "Ion"), ("r1", False, "Ionl"), ("Glow_ratio", True, "Esp")):
    for g, D in (("S", S), ("M", M)):
        m, lo, hi = med_ci(D[c], lg); N[f"{k}{g}"], N[f"{k}{g}lo"], N[f"{k}{g}hi"] = (10 ** m, 10 ** lo, 10 ** hi) if lg else (m, lo, hi)
    d, lo, hi, p = diff_ci(c, lg); N[f"{k}D"], N[f"{k}Dlo"], N[f"{k}Dhi"], N[f"{k}P"] = d, lo, hi, p
    if lg: N[f"{k}R"], N[f"{k}Rlo"], N[f"{k}Rhi"] = 10 ** d, 10 ** lo, 10 ** hi
N["mcS"], N["mcM"] = 100 * float((S.mc == 2).mean()), 100 * float((M.mc == 2).mean())
# flux scaling with N
Y = X[X.ncme >= 1]; a0 = np.polyfit(np.log10(Y.ncme), np.log10(Y.flux), 1)[0]
bs = [np.polyfit(np.log10(z.ncme), np.log10(z.flux), 1)[0] for z in (Y.sample(len(Y), replace=True, random_state=int(s)) for s in rng.integers(0, 1e9, 3000))]
N["alpha"], N["alphalo"], N["alphahi"] = float(a0), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
two = X[X.ncme == 2]; N["nTwo"], N["nThree"] = len(two), int((X.ncme >= 3).sum())
f = lambda t, s: 1 - np.median(t) / (2 * np.median(s)); fb = [f(rng.choice(two.flux.values, len(two)), rng.choice(S.flux.values, len(S))) for _ in range(3000)]
N["fann"], N["fannlo"], N["fannhi"] = 100 * f(two.flux.values, S.flux.values), 100 * float(np.percentile(fb, 2.5)), 100 * float(np.percentile(fb, 97.5))
# temperature excess at given expansion and speed
D = S.dropna(subset=["zeta", "Tratio", "V"]); D = D[D.Tratio > 0]
fit = lambda d: np.linalg.lstsq(np.column_stack([np.ones(len(d)), d.zeta, np.log10(d.V)]), np.log10(d.Tratio), rcond=None)[0]
k = fit(D); Mm = M.dropna(subset=["zeta", "Tratio", "V"]); exc = np.log10(Mm.Tratio) - (k[0] + k[1] * Mm.zeta + k[2] * np.log10(Mm.V))
bx = []
for s in range(2000):
    d = D.sample(len(D), replace=True, random_state=s); kk = fit(d); m = Mm.sample(len(Mm), replace=True, random_state=s + 1)
    bx.append(np.median(np.log10(m.Tratio) - (kk[0] + kk[1] * m.zeta + kk[2] * np.log10(m.V))))
N["Texc"], N["Texclo"], N["Texchi"] = float(np.median(exc)), float(np.percentile(bx, 2.5)), float(np.percentile(bx, 97.5))
N["rhoZT"] = float(spearmanr(S.zeta, np.log10(S.Tratio), nan_policy="omit")[0])
g = 5 / 3; N["adiaA"], N["adiaB"] = (1 / 0.5) ** (2 * (g - 1) * (N["ZetaS"] - N["ZetaM"])), (1 / 0.3) ** (2 * (g - 1) * (N["ZetaS"] - N["ZetaM"]))
# robustness 2021-2025
Sr, Mr = S[S.year >= 2021], M[M.year >= 2021]
for c, k2 in (("zeta", "Zeta"), ("N", "Dens"), ("flux", "Flux"), ("B", "Bfield"), ("Tratio", "Trat"), ("sigv", "Sig"), ("shB", "ShB")):
    N[f"{k2}Prec"] = float(mannwhitneyu(Sr[c].dropna(), Mr[c].dropna()).pvalue)
N["nSrec"], N["nMrec"] = len(Sr), len(Mr)
# launch separation tests
H = json.load(open(B / "hypotheses.json")); N["rhoLaunchZeta"], N["pLaunchZeta"] = H["T1_zeta"]["rho"], H["T1_zeta"]["p"]; N["rhoLaunchT"], N["pLaunchT"] = H["T3"]["rho"], H["T3"]["p"]
N["rbnAll"], N["rbnNeg"] = float(X.rho_bn.median()), 100 * float((X.rho_bn.dropna() < 0).mean())
z = X.dropna(subset=["r5", "f75"]); N["rhoDipIon"], N["pDipIon"] = map(float, spearmanr(z.f75, z.r5)); N["rhoDipSig"] = float(spearmanr(X.f75, X.siga, nan_policy="omit")[0])
json.dump(N, open(B / "paper/numbers.json", "w"), indent=1)
def fmt(k, v):
    if isinstance(v, (int, np.integer)): return str(int(v))
    v = float(v)
    if k.endswith(("P", "Prec")) or k.startswith("p"): return f"{v:.2g}" if v >= 0.001 else r"{<}0.001"
    if k.startswith("rho") or k.startswith(("Zeta", "Texc", "alpha", "Rbn")): return f"{v:.2f}"
    if k.startswith("fann"): return f"{v:.0f}"
    if abs(v) >= 100: return f"{v:.0f}"
    if abs(v) >= 10: return f"{v:.0f}" if k.startswith(("fann", "mc", "rbnNeg")) else f"{v:.1f}"
    return f"{v:.2f}" if abs(v) < 1 else f"{v:.2f}"
with open(B / "paper/macros.tex", "w") as fh:
    fh.write("% generated by 03_paper.py\n")
    for k, v in N.items():
        name = re.sub(r'[0-9]', '', k).replace("alpha", "alphaN")
        fh.write(f"\\newcommand{{\\{name}}}{{{fmt(k, v)}}}\n")
# ---------------- figures ----------------
def save(f, n): f.savefig(F / f"{n}.pdf", bbox_inches="tight"); f.savefig(F / f"{n}.eps", bbox_inches="tight"); plt.close(f)
def two_groups(ax, c, log, lab):
    ns = []
    for i, (D, col, m) in enumerate(((S, C1, "o"), (M, C2, "s"))):
        v = D[c].dropna(); v = v[v > 0] if log else v; x = i + rng.uniform(-0.18, 0.18, len(v)); ns.append(len(v))
        ax.scatter(x, v, s=9, color=col, alpha=.55, marker=m, edgecolor="none")
        q1, q2, q3 = np.percentile(v, [25, 50, 75]); ax.plot([i - .28, i + .28], [q2, q2], color=INK, lw=1.6); ax.plot([i, i], [q1, q3], color=INK, lw=.8)
    if log: ax.set_yscale("log")
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"single\n(N={ns[0]})", f"merged\n(N={ns[1]})"]); ax.set_ylabel(lab); ax.set_xlim(-.6, 1.6)
f, A = plt.subplots(1, 4, figsize=(7.2, 2.7))
for ax, (c, lg, lab, t) in zip(A, (("zeta", False, r"Expansion rate $\zeta$", "(a)"), ("N", True, r"Density (cm$^{-3}$)", "(b)"), ("B", True, "|B| in ICME (nT)", "(c)"), ("shB", True, "|B| in sheath (nT)", "(d)"))):
    two_groups(ax, c, lg, lab); ax.set_title(t, loc="left", fontsize=8.5)
A[0].set_ylim(-1.5, 2.5); save(f, "fig1")
f, A = plt.subplots(1, 2, figsize=(7.0, 3.0))
Yp = X[X.ncme >= 1]; xs = Yp.ncme.clip(upper=8) + rng.uniform(-0.12, 0.12, len(Yp))
A[0].scatter(xs, Yp.flux, s=10, color=C1, alpha=.5, edgecolor="none"); nn = np.arange(1, 9)
med = [Yp[Yp.ncme.clip(upper=8) == n].flux.median() if (Yp.ncme.clip(upper=8) == n).sum() > 0 else np.nan for n in nn]
A[0].plot(nn, med, "s", color=INK, ms=5, label="median"); s1 = S.flux.median()
A[0].plot(nn, s1 * nn ** N["alpha"], color=C2, lw=1.2, label=rf"$\Phi\propto N^{{{N['alpha']:.2f}}}$"); A[0].plot(nn, s1 * nn, color=MUTE, ls="--", lw=.9, label=r"$\Phi\propto N$ (additive)")
A[0].set_yscale("log"); A[0].set_xlabel("Number of associated CMEs, N"); A[0].set_ylabel("Flux proxy B·size (nT au)"); A[0].legend(fontsize=7); A[0].set_title("(a)", loc="left", fontsize=8.5)
Dd = S.dropna(subset=["zeta", "Tratio"]); A[1].scatter(Dd.zeta, Dd.Tratio, s=10, color=C1, alpha=.5, label="single", edgecolor="none")
A[1].scatter(Mm.zeta, Mm.Tratio, s=18, color=C2, marker="s", label="merged", edgecolor="none"); A[1].set_yscale("log"); A[1].set_xlim(-1.5, 2.5)
A[1].set_xlabel(r"Expansion rate $\zeta$"); A[1].set_ylabel(r"$T_p/T_{\rm exp}$"); A[1].legend(fontsize=7); A[1].set_title("(b)", loc="left", fontsize=8.5)
save(f, "fig2")
f, A = plt.subplots(1, 4, figsize=(7.2, 2.7))
for ax, (c, lg, lab, t) in zip(A, (("sigv", False, r"$\sigma_B/B$ (hourly)", "(a)"), ("rho_bn", False, r"corr($\delta B/B$, $\delta n/n$)", "(b)"), ("r5", False, r"log$_{10}$ ions in/upstream", "(c)"), ("Glow_ratio", True, "3–9 MeV protons at shock", "(d)"))):
    two_groups(ax, c, lg, lab); ax.set_title(t, loc="left", fontsize=8.5)
A[1].axhline(0, color=MUTE, lw=.6, ls="--"); A[2].axhline(0, color=MUTE, lw=.6, ls="--"); save(f, "fig3")
print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in N.items()}, indent=0))
