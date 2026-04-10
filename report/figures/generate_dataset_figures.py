"""
Generate publication-quality dataset scale and overlap figures.
Author: J.C. Vaught
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Brand colours
# ---------------------------------------------------------------------------
GARNET = "#73000A"
BLACK = "#000000"
ATLANTIC = "#466A9F"
CONGAREE = "#1F414D"
HORSESHOE = "#65780B"
WARM_GREY = "#676156"
SANDSTORM = "#FFF2E3"
BLACK_90 = "#363636"
BLACK_70 = "#5C5C5C"
BLACK_50 = "#A2A2A2"
BLACK_30 = "#C7C7C7"
BLACK_10 = "#ECECEC"

# Global matplotlib settings
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Georgia", "serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =========================================================================
# FIGURE 1: Dataset scale horizontal bar chart
# =========================================================================
def make_dataset_scale():
    databases = {
        "NMRexp":       1_480_000,
        "NP-MRD":         281_000,
        "HMDB":           217_000,
        "NMRBank":        149_000,
        "NMRShiftDB2":     45_000,
        "SDBS":            34_600,
        "2DNMRGym":        22_348,
        "PROSPRE":         10_000,
        "BMRB/GISSMO":      3_400,
        "IR-NMR":           1_255,
        "logD":               754,
    }

    # Sort ascending so largest bar is at top
    names = list(databases.keys())[::-1]
    counts = [databases[n] for n in names]

    # Colour assignment: primary datasets in garnet tones, smaller in atlantic
    bar_colours = []
    for n in names:
        if n == "NMRexp":
            bar_colours.append(GARNET)
        elif n in ("HMDB", "NMRBank", "NP-MRD"):
            bar_colours.append(ATLANTIC)
        elif n in ("NMRShiftDB2", "SDBS"):
            bar_colours.append(CONGAREE)
        else:
            bar_colours.append(HORSESHOE)

    fig, ax = plt.subplots(figsize=(8, 4.8))

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, counts, height=0.68, color=bar_colours, edgecolor="none",
                   linewidth=0)

    # Remove rounded edges explicitly
    for bar in bars:
        bar.set_joinstyle("miter")
        bar.set_capstyle("butt")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10, color=BLACK_90)
    ax.set_xscale("log")
    ax.set_xlim(400, 4_000_000)

    # Format x-axis with short labels
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}" if x < 1000 else
                     f"{x/1000:.0f}K" if x < 1_000_000 else
                     f"{x/1_000_000:.1f}M"
    ))
    ax.set_xlabel("Number of compounds (log scale)", fontsize=11, color=BLACK_90)

    # Bar labels
    for i, (cnt, bar) in enumerate(zip(counts, bars)):
        if cnt >= 1_000_000:
            label = f"{cnt/1_000_000:.2f}M"
        elif cnt >= 1_000:
            label = f"{cnt/1_000:.1f}K" if cnt % 1000 else f"{cnt//1000}K"
        else:
            label = f"{cnt:,}"
        # Clean up .0K
        label = label.replace(".0K", "K")
        ax.text(cnt * 1.15, i, label, va="center", ha="left",
                fontsize=9, color=BLACK_70, fontweight="medium")

    # Spine styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BLACK_30)
    ax.spines["bottom"].set_color(BLACK_30)
    ax.tick_params(axis="x", colors=BLACK_70, labelsize=9)
    ax.tick_params(axis="y", length=0)

    # Light grid
    ax.xaxis.grid(True, which="major", linestyle="-", linewidth=0.4,
                  color=BLACK_30, alpha=0.6)
    ax.set_axisbelow(True)

    ax.set_title("NMR Database Scale Comparison", fontsize=13,
                 fontweight="bold", color=BLACK, pad=12)

    fig.tight_layout()
    fig.savefig("/Users/user/Downloads/untitled folder/report/figures/dataset_scale.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved dataset_scale.png")


# =========================================================================
# FIGURE 2: NMR-viscosity overlap diagram
# =========================================================================
def make_overlap_diagram():
    nmr_total = 1_800_000
    visc_total = 2_238
    overlap = 1_330

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Use a proportional-width stacked rectangle approach.
    # Full width represents NMR data. A narrow strip represents viscosity.
    # The overlap is shown as a highlighted segment within the viscosity strip.

    bar_y = 0.50
    bar_h = 0.22
    full_w = 0.82
    x0 = 0.09

    # NMR bar (full width)
    nmr_rect = mpatches.FancyBboxPatch(
        (x0, bar_y), full_w, bar_h,
        boxstyle="square,pad=0", facecolor=ATLANTIC, edgecolor=BLACK_90,
        linewidth=1.0, alpha=0.85
    )
    ax.add_patch(nmr_rect)

    # Viscosity bar: proportional width = visc_total / nmr_total * full_w
    visc_w = (visc_total / nmr_total) * full_w  # ~0.001 -- too tiny to see
    # Use a minimum visible width but annotate the true proportion
    visc_display_w = 0.065  # visible width for readability
    visc_x = x0 + full_w - visc_display_w

    visc_rect = mpatches.FancyBboxPatch(
        (visc_x, bar_y), visc_display_w, bar_h,
        boxstyle="square,pad=0", facecolor=GARNET, edgecolor=BLACK_90,
        linewidth=1.0, alpha=0.95
    )
    ax.add_patch(visc_rect)

    # Overlap region inside the viscosity bar
    overlap_frac = overlap / visc_total
    overlap_display_w = visc_display_w * overlap_frac
    overlap_x = visc_x

    overlap_rect = mpatches.FancyBboxPatch(
        (overlap_x, bar_y), overlap_display_w, bar_h,
        boxstyle="square,pad=0", facecolor=HORSESHOE, edgecolor=BLACK_90,
        linewidth=1.0, alpha=0.95
    )
    ax.add_patch(overlap_rect)

    # ---- Annotations ----
    # NMR label (centred on the NMR-only region)
    nmr_only_cx = x0 + (full_w - visc_display_w) / 2
    ax.text(nmr_only_cx, bar_y + bar_h / 2, "NMR Data\n1.80M compounds",
            ha="center", va="center", fontsize=12, color="white",
            fontweight="bold")

    # Viscosity callout with bracket
    callout_x = visc_x + visc_display_w / 2
    callout_y_top = bar_y + bar_h + 0.04

    # Brace line above viscosity bar
    ax.annotate("",
                xy=(visc_x, callout_y_top), xytext=(visc_x + visc_display_w, callout_y_top),
                arrowprops=dict(arrowstyle="-", color=BLACK_70, lw=1.0))
    ax.plot([visc_x, visc_x], [bar_y + bar_h, callout_y_top],
            color=BLACK_70, lw=1.0)
    ax.plot([visc_x + visc_display_w, visc_x + visc_display_w],
            [bar_y + bar_h, callout_y_top], color=BLACK_70, lw=1.0)

    ax.text(callout_x, callout_y_top + 0.06,
            "Viscosity Data\n2,238 compounds",
            ha="center", va="bottom", fontsize=10, color=GARNET,
            fontweight="bold")

    # Overlap callout below
    callout_y_bot = bar_y - 0.04
    ol_cx = overlap_x + overlap_display_w / 2

    ax.plot([overlap_x, overlap_x], [bar_y, callout_y_bot],
            color=BLACK_70, lw=1.0)
    ax.plot([overlap_x + overlap_display_w, overlap_x + overlap_display_w],
            [bar_y, callout_y_bot], color=BLACK_70, lw=1.0)
    ax.annotate("",
                xy=(overlap_x, callout_y_bot),
                xytext=(overlap_x + overlap_display_w, callout_y_bot),
                arrowprops=dict(arrowstyle="-", color=BLACK_70, lw=1.0))

    ax.text(ol_cx, callout_y_bot - 0.06,
            "Overlap: 1,330 compounds",
            ha="center", va="top", fontsize=10, color=HORSESHOE,
            fontweight="bold")

    # Proportion annotation
    ax.text(0.5, 0.08,
            "The overlap represents 0.07% of NMR data and 59.4% of viscosity data.",
            ha="center", va="center", fontsize=9, color=BLACK_70,
            style="italic")

    # Scale disclaimer
    ax.text(0.5, 0.02,
            "(Viscosity bar width exaggerated for visibility; true width is < 1 pixel at this scale)",
            ha="center", va="center", fontsize=7.5, color=BLACK_50)

    # Legend
    legend_patches = [
        mpatches.Patch(facecolor=ATLANTIC, edgecolor=BLACK_90, linewidth=0.6,
                       label="NMR compounds (1.80M)"),
        mpatches.Patch(facecolor=GARNET, edgecolor=BLACK_90, linewidth=0.6,
                       label="Viscosity compounds (2,238)"),
        mpatches.Patch(facecolor=HORSESHOE, edgecolor=BLACK_90, linewidth=0.6,
                       label="NMR + Viscosity overlap (1,330)"),
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=8.5,
              frameon=True, edgecolor=BLACK_30, fancybox=False,
              borderpad=0.6)

    ax.set_title("NMR and Viscosity Dataset Overlap", fontsize=13,
                 fontweight="bold", color=BLACK, pad=10)

    fig.tight_layout()
    fig.savefig("/Users/user/Downloads/untitled folder/report/figures/nmr_visc_overlap.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved nmr_visc_overlap.png")


if __name__ == "__main__":
    make_dataset_scale()
    make_overlap_diagram()
