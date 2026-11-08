#!/usr/bin/env python3
"""Example of a merged ICME: 3-5 November 2021 (four associated CMEs launched on 1-2 November). OMNI 10-min averages from gs_raw,
GOES-16 SGPS protons from the local cache (timelag_ASR/code/io_data.py)."""
import sys, types
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.dates as mdates
B = Path(__file__).resolve().parent
sys.modules.setdefault("cdflib", types.ModuleType("cdflib"))                     # io_data imports cdflib only for EPHIN
sys.path.insert(0, str(B.parents[1] / "timelag_ASR/code")); from io_data import goes
C1, C2, C3, INK, MUTE = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6,
                     "axes.edgecolor": MUTE, "xtick.color": MUTE, "ytick.color": MUTE, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42})
head, *rows = open(B / "gs_raw/211103.txt").read().strip().split("|")
D = pd.DataFrame([[float(x) if x else np.nan for x in (r.split(" ") + [""] * 9)[:9]] for r in rows], columns=["F", "bx", "by", "bz", "vx", "vy", "vz", "n", "T"])
D.index = pd.Timestamp(head, tz="UTC") + pd.to_timedelta(10 * np.arange(len(D)), unit="min")
D["V"] = np.sqrt(D.vx ** 2 + D.vy ** 2 + D.vz ** 2); D["T"] *= 1e3
D["Texp"] = np.where(D.V < 500, (0.031 * D.V - 5.1) ** 2 * 1e3, (0.51 * D.V - 142) * 1e3)
g, band = goes(D.index[0], D.index[-1]); G = g.resample("10min").mean()
tsh, t0, t1 = (pd.Timestamp(s, tz="UTC") for s in ("2021-11-03 19:42", "2021-11-04 07:00", "2021-11-05 05:00"))
fig, ax = plt.subplots(5, 1, figsize=(6.6, 7.4), sharex=True, gridspec_kw=dict(hspace=0.08))
ax[0].plot(D.index, D.F, color=INK, lw=1, label=r"$|B|$")
for c, col, lab in (("bx", C1, r"$B_x$"), ("by", C3, r"$B_y$"), ("bz", C2, r"$B_z$")): ax[0].plot(D.index, D[c], color=col, lw=.8, label=lab)
ax[0].axhline(0, color=MUTE, lw=.4); ax[0].set_ylabel("B (nT, GSE)"); ax[0].legend(ncol=4, fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2)
ax[1].plot(D.index, D.V, color=INK, lw=1); ax[1].set_ylabel(r"$V$ (km s$^{-1}$)")
ax[2].semilogy(D.index, D.n, color=INK, lw=1); ax[2].set_ylabel(r"$n$ (cm$^{-3}$)")
ax[3].semilogy(D.index, D["T"], color=INK, lw=1, label=r"$T_p$"); ax[3].semilogy(D.index, D.Texp, color=C2, lw=1, ls="--", label=r"$T_{\rm exp}(V)$")
ax[3].set_ylabel(r"$T$ (K)"); ax[3].legend(fontsize=7, loc="upper right")
for ch, col in (("G1", C1), ("G4", C2), ("G6", C3)):
    lo, hi = band[ch]; ax[4].semilogy(G.index, G[ch], color=col, lw=1, label=f"{lo:g}–{hi:g} MeV")
ax[4].set_ylabel(r"$J_p$ (cm$^{-2}$s$^{-1}$sr$^{-1}$MeV$^{-1}$)", fontsize=7.5); ax[4].legend(fontsize=7, loc="upper right")
for a in ax:
    a.axvline(tsh, color=MUTE, lw=.8, ls=":"); a.axvspan(t0, t1, color=C1, alpha=.08, lw=0)
ax[1].text(tsh, ax[1].get_ylim()[1], " shock", va="top", fontsize=7, color=MUTE); ax[1].text(t1, ax[1].get_ylim()[1], "merged ejecta ", va="top", ha="right", fontsize=7, color=C1)
ax[4].xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M")); ax[4].set_xlim(D.index[0], D.index[-1])
for i, a in enumerate(ax): a.text(-0.17, 1.0, "abcde"[i], transform=a.transAxes, fontweight="bold", va="top")
fig.align_ylabels(ax)
for ext in ("pdf", "png"): fig.savefig(B / f"paper/figures/fig1_example.{ext}", dpi=150, bbox_inches="tight")
print("ok", D.loc[t0:t1, ["F", "V", "n", "T"]].mean().round(1).to_dict())
