# Single and merged interplanetary coronal mass ejections at 1 au, 2011–2025: derived parameters and analysis scripts

Author: I. I. Davydenkov, Space Research Institute of the Russian Academy of Sciences (IKI RAS), Moscow  
Contact: davydenkov.ii@phystech.edu

This archive contains the derived data and the scripts used in the manuscript
"Merged Coronal Mass Ejections at 1 au: Compression, Sublinear Magnetic Flux Addition, and No Excess Heating"
(submitted to Journal of Geophysical Research: Space Physics).

## Data (`data/`)

| File | Content |
|---|---|
| `icme_parameters_2011_2025.csv` | One row per ICME (286 ICMEs lasting at least 6 h): times from the Richardson–Cane list, number of associated CMEs, magnetic-cloud flag, mean field, speed, density and proton temperature inside the ICME, T_p/T_exp, plasma beta, front and rear speeds, expansion rate ζ, flux proxy, sheath field and density, hourly field fluctuation levels, field–density rank correlation, ACE/EPAM ion intensities inside the ICME and upstream. |
| `icme_plasma.csv` | The same quantities with the internal column names used by the scripts. |
| `rc_icme_2011_2026.csv` | Compact copy of the Richardson–Cane ICME list used (disturbance time, associated LASCO CME times, speeds). |
| `esp_sample.csv` | Proton intensity ratios at the arrival of each disturbance (GOES, SOHO/EPHIN). |
| `lundquist_raw.txt` | Lundquist flux-rope fit parameters for 122 ICMEs with magnetic-cloud flags: start time, residual, axis latitude and longitude (rad, GSE), impact parameter, B0 (nT), centre shift, handedness, mean speed, number of hours. |
| `gs_raw/` | 10-min averages of 5-min OMNI data for the three example events (|B|, Bx, By, Bz GSE in nT; Vx, Vy, Vz in km/s; n in cm^-3; Tp in kK). |

Column units: B in nT, V in km/s, n in cm^-3, T in K, durations in hours, EPAM intensities in (cm^2 s sr MeV)^-1.

## Scripts (`scripts/`)

| Script | Purpose |
|---|---|
| `01_merger_tests.py`, `02_hypotheses.py` | Group comparisons and launch-separation tests |
| `03_paper.py` | All numbers of the comparisons, flux scaling, temperature excess; Figures 2–4 |
| `04a_lundquist_fit.js` | Lundquist fits (run in a browser on hourly OMNI GSE field) |
| `04_helicity.py` | Axial flux, helicity and fit quality of single and merged clouds |
| `05_energy.py` | Energy budget of the merger |
| `07_example.py` | Figure 1 (3–5 November 2021); needs GOES-16 SGPS data, not included |
| `08_physics.py` | Alfvén crossing times, total pressure, Dst statistics |

The scripts expect `icme_plasma.csv`, `lundquist_raw.txt` and `gs_raw/` in the same folder as the scripts, and
`esp_sample.csv` in `../data/`. Python 3 with numpy, pandas, scipy and matplotlib.

## Sources of the original data

- Richardson & Cane ICME list: https://izw1.caltech.edu/ACE/ASC/DATA/level3/icmetable2.htm
- OMNI hourly, 5-min and 1-min data (NASA SPDF): doi:10.48322/1shr-ht18, doi:10.48322/gbpg-5r77, doi:10.48322/45bb-8792
- ACE/EPAM (NASA SPDF/CDAWeb, datasets AC_H2_EPM and AC_H1_EPM): doi:10.48322/xab5-ea54
- GOES-13/15 EPEAD and GOES-16 SGPS proton data: NOAA NCEI

License: CC BY 4.0.
