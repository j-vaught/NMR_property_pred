"""
Generate publication-quality NMR feature extraction pipeline diagram.
Three-panel figure: (A) synthetic spectrum with region bands,
(B) 30-bin bar chart, (C) 120-dim feature vector heatmap.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Brand colors
# ---------------------------------------------------------------------------
GARNET = "#73000A"
BLACK = "#000000"
WHITE = "#FFFFFF"
B90 = "#363636"
B70 = "#5C5C5C"
B50 = "#A2A2A2"
B30 = "#C7C7C7"
B10 = "#ECECEC"
WARM_GREY = "#676156"
SANDSTORM = "#FFF2E3"
ROSE = "#CC2E40"
ATLANTIC = "#466A9F"
CONGAREE = "#1F414D"
HORSESHOE = "#65780B"
GRASS = "#CED318"
HONEYCOMB = "#A49137"

# ---------------------------------------------------------------------------
# NMR chemical shift regions and their display colors
# ---------------------------------------------------------------------------
REGIONS = {
    "Alkyl (high-field)":    (0.0,  1.0),
    "Alkyl (mid)":           (1.0,  2.0),
    r"Alkyl ($\alpha$-hetero)": (2.0,  3.0),
    "Oxygenated":            (3.0,  4.5),
    "Olefinic (low)":        (4.5,  5.5),
    "Olefinic (high)":       (5.5,  6.5),
    "Aromatic (low)":        (6.5,  7.5),
    "Aromatic (high)":       (7.5,  8.5),
    "Aldehyde/downfield":    (8.5, 12.0),
}

REGION_COLORS = [
    "#466A9F",  # Atlantic
    "#1F414D",  # Congaree
    "#65780B",  # Horseshoe
    "#A49137",  # Honeycomb
    "#CED318",  # Grass
    "#CC2E40",  # Rose
    "#73000A",  # Garnet
    "#5C5C5C",  # B70
    "#363636",  # B90
]

# ---------------------------------------------------------------------------
# Lorentzian lineshape
# ---------------------------------------------------------------------------
def lorentzian(x, center, gamma, intensity=1.0):
    hg = gamma / 2.0
    return intensity * hg**2 / ((x - center)**2 + hg**2)


# ---------------------------------------------------------------------------
# Synthetic 1H NMR spectrum (typical alkyl-aromatic compound)
# ---------------------------------------------------------------------------
ppm = np.linspace(0, 12, 1200)

# Peaks: (center_ppm, integral, gamma)
synthetic_peaks = [
    (0.88, 9.0, 0.015),   # t-Bu or large CH3
    (0.92, 3.0, 0.012),
    (1.28, 6.0, 0.015),   # CH2 chain
    (1.32, 4.0, 0.012),
    (1.55, 2.0, 0.015),   # CH2 beta
    (2.10, 2.0, 0.015),   # CH2 alpha to C=O
    (2.15, 1.0, 0.012),
    (7.15, 1.0, 0.012),   # aromatic
    (7.22, 2.0, 0.015),
    (7.28, 1.0, 0.012),
]

spectrum = np.zeros_like(ppm)
for c, inten, g in synthetic_peaks:
    spectrum += lorentzian(ppm, c, g, inten)

# Normalize to [0, 1]
spectrum /= spectrum.max()

# ---------------------------------------------------------------------------
# Compute 30-bin features (mean signal per bin)
# ---------------------------------------------------------------------------
n_bins = 30
bin_edges = np.linspace(0, 12, n_bins + 1)
bin_means = np.zeros(n_bins)
bin_maxes = np.zeros(n_bins)
for i in range(n_bins):
    mask = (ppm >= bin_edges[i]) & (ppm < bin_edges[i + 1])
    if mask.any():
        bin_means[i] = spectrum[mask].mean()
        bin_maxes[i] = spectrum[mask].max()

# ---------------------------------------------------------------------------
# Simulate the 120-dim feature vector layout
# ---------------------------------------------------------------------------
# 60 bin features (30 mean + 30 max interleaved)
bin_features = np.zeros(60)
for i in range(30):
    bin_features[2*i] = bin_means[i]
    bin_features[2*i+1] = bin_maxes[i]

# 27 region features (integral, frac, peaks) x 9
region_keys = list(REGIONS.values())
region_features = np.zeros(27)
total_signal = spectrum.sum()
for idx, (lo, hi) in enumerate(region_keys):
    mask = (ppm >= lo) & (ppm < hi)
    region = spectrum[mask]
    region_features[3*idx] = region.sum() / total_signal  # normalized integral
    region_features[3*idx+1] = region.sum() / (total_signal + 1e-10)  # frac
    # fake peak count
    above = region > 0.05
    if len(region) > 1:
        crossings = np.diff(above.astype(int))
        region_features[3*idx+2] = max(0, (crossings == 1).sum()) / 10.0  # scaled

# 8 global stats (fake reasonable values)
global_stats = np.array([0.35, 0.55, 0.8, 0.3, 0.6, 0.45, 0.25, 1.0])

# 25 peaklist features (simulated)
np.random.seed(42)
peaklist = np.array([
    0.7, 0.4, 0.9,          # total_h, n_peaks, max_integral (scaled)
    0.6, 0.5, 0.1, 0.0, 0.0, 0.0, 0.15, 0.05, 0.0,  # h_count per region
    0.3, 0.25, 0.15, 0.1, 0.2, 0.0, 0.0,  # multiplicity fracs
    0.4, 0.6, 0.5,          # J-coupling stats
    0.7, 0.15, 0.45,        # derived ratios
])
assert len(peaklist) == 25

# Full 120-dim vector
full_vector = np.concatenate([bin_features, region_features, global_stats, peaklist])
assert len(full_vector) == 120, f"Got {len(full_vector)}"

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 5.4), facecolor=WHITE)
gs = GridSpec(1, 3, figure=fig, width_ratios=[1.3, 0.8, 1.1],
              left=0.05, right=0.97, bottom=0.13, top=0.78, wspace=0.28)

# ===== Panel A: Synthetic 1H NMR spectrum with region bands =====
ax_a = fig.add_subplot(gs[0])

# Draw region background bands
region_names = list(REGIONS.keys())
for idx, (name, (lo, hi)) in enumerate(REGIONS.items()):
    color = REGION_COLORS[idx]
    ax_a.axvspan(lo, hi, alpha=0.15, color=color, linewidth=0)

# Plot spectrum
ax_a.plot(ppm, spectrum, color=GARNET, linewidth=0.9, zorder=5)
ax_a.fill_between(ppm, 0, spectrum, color=GARNET, alpha=0.08, zorder=4)

# Reverse x-axis (NMR convention)
ax_a.set_xlim(12, 0)
ax_a.set_ylim(-0.02, 1.12)
ax_a.set_xlabel("Chemical shift (ppm)", fontsize=9, color=B90)
ax_a.set_ylabel("Normalized intensity", fontsize=9, color=B90)
ax_a.set_title("A.  Synthetic $^{1}$H NMR spectrum", fontsize=10,
               fontweight="bold", color=BLACK, loc="left", pad=55)

# Region labels along top of panel A, staggered to avoid overlap
region_label_y = [
    1.03, 1.11, 1.03, 1.11, 1.03, 1.11, 1.03, 1.11, 1.03
]
# Short names for compact display
region_short = [
    "Alkyl\n(high-field)", "Alkyl\n(mid)", r"Alkyl"+"\n"+r"($\alpha$-het.)",
    "Oxy-\ngenated", "Olefinic\n(low)", "Olefinic\n(high)",
    "Aromatic\n(low)", "Aromatic\n(high)", "Aldehyde/\ndownfield",
]
for idx, ((name, (lo, hi)), short) in enumerate(zip(REGIONS.items(), region_short)):
    mid = (lo + hi) / 2.0
    ax_a.text(mid, region_label_y[idx], short, fontsize=4.8,
              ha="center", va="bottom", linespacing=0.85,
              color=REGION_COLORS[idx], fontweight="bold",
              transform=ax_a.get_xaxis_transform())

# Clean up spines
for spine in ["top", "right"]:
    ax_a.spines[spine].set_visible(False)
ax_a.spines["bottom"].set_color(B50)
ax_a.spines["left"].set_color(B50)
ax_a.tick_params(colors=B70, labelsize=7)

# ===== Panel B: 30 spectral bins =====
ax_b = fig.add_subplot(gs[1])

bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
bar_width = 0.38

# Color each bar by which region it falls in
bar_colors = []
for bc in bin_centers:
    assigned = False
    for idx, (lo, hi) in enumerate(REGIONS.values()):
        if lo <= bc < hi:
            bar_colors.append(REGION_COLORS[idx])
            assigned = True
            break
    if not assigned:
        bar_colors.append(B50)

ax_b.bar(bin_centers, bin_means, width=bar_width, color=bar_colors,
         edgecolor=WHITE, linewidth=0.3, alpha=0.85)

ax_b.set_xlim(-0.2, 12.2)
ax_b.set_ylim(0, bin_means.max() * 1.15)
ax_b.set_xlabel("Chemical shift (ppm)", fontsize=9, color=B90)
ax_b.set_ylabel("Mean signal", fontsize=9, color=B90)
ax_b.set_title("B.  30 spectral bins (0.4 ppm)", fontsize=10,
               fontweight="bold", color=BLACK, loc="left", pad=8)

# Annotate bin count
ax_b.text(6.0, bin_means.max() * 1.05, "30 bins $\\times$ 2 = 60 features\n(mean + max per bin)",
          fontsize=6.5, ha="center", va="bottom", color=B70, style="italic")

for spine in ["top", "right"]:
    ax_b.spines[spine].set_visible(False)
ax_b.spines["bottom"].set_color(B50)
ax_b.spines["left"].set_color(B50)
ax_b.tick_params(colors=B70, labelsize=7)

# ===== Panel C: 120-dim feature vector =====
ax_c = fig.add_subplot(gs[2])

# Create stacked bar segments colored by feature group
x = np.arange(120)
colors_120 = np.empty(120, dtype=object)

# Color by group
colors_120[:60] = ATLANTIC      # bin features
colors_120[60:87] = HORSESHOE   # region features
colors_120[87:95] = HONEYCOMB   # global stats
colors_120[95:120] = GARNET     # peaklist features

ax_c.bar(x, full_vector, width=1.0, color=colors_120, edgecolor="none",
         linewidth=0)

# Group separators
for sep in [60, 87, 95]:
    ax_c.axvline(sep - 0.5, color=B30, linewidth=0.7, linestyle="--", zorder=6)

# Group labels
group_info = [
    (30,  "Spectral bins\n(60)", ATLANTIC),
    (73.5, "Region\n(27)", HORSESHOE),
    (91,  "Global\n(8)", HONEYCOMB),
    (107.5, "Peak-list\n(25)", GARNET),
]
for xpos, label, col in group_info:
    ax_c.text(xpos, full_vector.max() * 1.12, label, fontsize=6.5,
              ha="center", va="bottom", color=col, fontweight="bold")

ax_c.set_xlim(-1, 121)
ax_c.set_ylim(0, full_vector.max() * 1.35)
ax_c.set_xlabel("Feature index", fontsize=9, color=B90)
ax_c.set_ylabel("Feature value", fontsize=9, color=B90)
ax_c.set_title("C.  120-dimensional feature vector", fontsize=10,
               fontweight="bold", color=BLACK, loc="left", pad=8)

for spine in ["top", "right"]:
    ax_c.spines[spine].set_visible(False)
ax_c.spines["bottom"].set_color(B50)
ax_c.spines["left"].set_color(B50)
ax_c.tick_params(colors=B70, labelsize=7)

# ===== Arrows between panels =====
# Use fig-level annotation arrows
from matplotlib.patches import FancyArrowPatch

arrow_kw = dict(
    arrowstyle="->,head_width=0.15,head_length=0.12",
    color=B50, linewidth=1.5, connectionstyle="arc3,rad=0.0",
)

# Arrow A -> B
fig.patches.append(FancyArrowPatch(
    posA=(0.38, 0.52), posB=(0.405, 0.52),
    transform=fig.transFigure, **arrow_kw
))

# Arrow B -> C
fig.patches.append(FancyArrowPatch(
    posA=(0.635, 0.52), posB=(0.66, 0.52),
    transform=fig.transFigure, **arrow_kw
))

# ===== Save =====
out_path = "/Users/user/Downloads/untitled folder/report/figures/nmr_feature_pipeline.png"
fig.savefig(out_path, dpi=300, facecolor=WHITE, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
