"""
Extract thermophysical properties from thermo/chemicals libraries.
Generates T-dependent thermal conductivity and heat capacity data,
plus flash point and boiling point data for all available compounds.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict

warnings.filterwarnings("ignore")

OUT_DIR = "/Users/user/Downloads/untitled folder/data/thermo/additional_properties"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Step 1: Gather a master list of CAS numbers from multiple sources
# ---------------------------------------------------------------------------
import chemicals

chem_dir = os.path.dirname(chemicals.__file__)

# Collect CAS numbers from thermal conductivity tables
tc_liquid = pd.read_csv(
    os.path.join(chem_dir, "Thermal Conductivity",
                 "Table 2-315 Thermal Conductivity of Inorganic and Organic Liquids.tsv"),
    sep="\t"
)
tc_vdi = pd.read_csv(
    os.path.join(chem_dir, "Thermal Conductivity",
                 "VDI PPDS Thermal conductivity of saturated liquids.tsv"),
    sep="\t"
)

# Heat capacity tables
hc_perry_100 = pd.read_csv(
    os.path.join(chem_dir, "Heat Capacity", "Perry_Table_2-153_DIPPR_100.tsv"),
    sep="\t"
)
hc_perry_114 = pd.read_csv(
    os.path.join(chem_dir, "Heat Capacity", "Perry_Table_2-153_DIPPR_114.tsv"),
    sep="\t"
)
hc_poling = pd.read_csv(
    os.path.join(chem_dir, "Heat Capacity", "PolingDatabank.tsv"),
    sep="\t"
)

# Boiling points
bp_yaws = pd.read_csv(
    os.path.join(chem_dir, "Phase Change", "Yaws Boiling Points.tsv"),
    sep="\t"
)

# Flash points
fp_dippr = pd.read_csv(
    os.path.join(chem_dir, "Safety", "DIPPR T_flash Serat.csv"),
    sep="\t"
)

# Merge all CAS numbers
all_cas = set()
for df in [tc_liquid, tc_vdi, hc_perry_100, hc_perry_114, hc_poling, bp_yaws, fp_dippr]:
    if "CAS" in df.columns:
        all_cas.update(df["CAS"].dropna().astype(str).tolist())
    elif "CASRN" in df.columns:
        all_cas.update(df["CASRN"].dropna().astype(str).tolist())

# Also add from identifiers database
try:
    ident = pd.read_csv(
        os.path.join(chem_dir, "Identifiers", "dippr_2014.csv"), low_memory=False
    )
    if "CASN" in ident.columns:
        all_cas.update(ident["CASN"].dropna().astype(str).tolist())
    elif "CAS" in ident.columns:
        all_cas.update(ident["CAS"].dropna().astype(str).tolist())
except Exception:
    pass

# Filter valid CAS format
import re
cas_pattern = re.compile(r"^\d{1,7}-\d{2}-\d$")
all_cas = sorted([c for c in all_cas if cas_pattern.match(c)])
print(f"Total unique CAS numbers collected: {len(all_cas)}")

# ---------------------------------------------------------------------------
# Step 2: Extract properties using thermo.Chemical
# ---------------------------------------------------------------------------
from thermo import Chemical

kl_rows = []       # thermal conductivity
cpl_rows = []      # heat capacity
point_rows = []    # flash + boiling points

success_kl = 0
success_cpl = 0
success_point = 0
errors = 0

for idx, cas in enumerate(all_cas):
    if idx % 500 == 0:
        print(f"Processing {idx}/{len(all_cas)}... (kl={success_kl}, cpl={success_cpl}, pts={success_point})")

    try:
        c = Chemical(cas)
    except Exception:
        errors += 1
        continue

    name = c.name or ""
    smiles = c.smiles or ""
    cas_str = c.CAS or cas

    # --- Boiling point and flash point (scalar properties) ---
    tb = None
    tflash = None
    try:
        tb = c.Tb
    except Exception:
        pass
    try:
        tflash = c.Tflash
    except Exception:
        pass

    if tb is not None or tflash is not None:
        point_rows.append({
            "CAS": cas_str,
            "name": name,
            "SMILES": smiles,
            "Tb_K": tb if tb is not None else "",
            "Tflash_K": tflash if tflash is not None else "",
        })
        success_point += 1

    # --- T-dependent: liquid thermal conductivity ---
    try:
        Tm = c.Tm if c.Tm else None
        Tb = c.Tb if c.Tb else None
        if Tm and Tb and Tb > Tm:
            # Evaluate in liquid range with margin
            T_low = max(Tm, 200.0)
            T_high = min(Tb, 600.0)
            if T_high > T_low + 5:
                temps = np.linspace(T_low, T_high, 10)
                got_any = False
                for T in temps:
                    try:
                        c_t = Chemical(cas, T=float(T))
                        kl = c_t.kl
                        if kl is not None and np.isfinite(kl) and kl > 0:
                            kl_rows.append({
                                "CAS": cas_str,
                                "name": name,
                                "SMILES": smiles,
                                "T_K": round(float(T), 2),
                                "kl_WmK": round(float(kl), 6),
                            })
                            got_any = True
                    except Exception:
                        pass
                if got_any:
                    success_kl += 1
    except Exception:
        pass

    # --- T-dependent: liquid heat capacity ---
    try:
        Tm = c.Tm if c.Tm else None
        Tb = c.Tb if c.Tb else None
        if Tm and Tb and Tb > Tm:
            T_low = max(Tm, 200.0)
            T_high = min(Tb, 600.0)
            if T_high > T_low + 5:
                temps = np.linspace(T_low, T_high, 10)
                got_any = False
                for T in temps:
                    try:
                        c_t = Chemical(cas, T=float(T))
                        cplm = c_t.Cplm
                        if cplm is not None and np.isfinite(cplm) and cplm > 0:
                            cpl_rows.append({
                                "CAS": cas_str,
                                "name": name,
                                "SMILES": smiles,
                                "T_K": round(float(T), 2),
                                "Cpl_JmolK": round(float(cplm), 4),
                            })
                            got_any = True
                    except Exception:
                        pass
                if got_any:
                    success_cpl += 1
    except Exception:
        pass

print(f"\nDone processing {len(all_cas)} CAS numbers.")
print(f"  Thermal conductivity: {success_kl} compounds, {len(kl_rows)} data points")
print(f"  Heat capacity: {success_cpl} compounds, {len(cpl_rows)} data points")
print(f"  Flash/boiling points: {success_point} compounds")
print(f"  Errors (could not load Chemical): {errors}")

# ---------------------------------------------------------------------------
# Step 3: Save outputs
# ---------------------------------------------------------------------------
df_kl = pd.DataFrame(kl_rows)
df_kl.to_csv(os.path.join(OUT_DIR, "thermal_conductivity.csv"), index=False)
print(f"\nSaved thermal_conductivity.csv: {len(df_kl)} rows")

df_cpl = pd.DataFrame(cpl_rows)
df_cpl.to_csv(os.path.join(OUT_DIR, "heat_capacity.csv"), index=False)
print(f"Saved heat_capacity.csv: {len(df_cpl)} rows")

df_pts = pd.DataFrame(point_rows)
df_pts.to_csv(os.path.join(OUT_DIR, "flash_boiling_points.csv"), index=False)
print(f"Saved flash_boiling_points.csv: {len(df_pts)} rows")

# Summary stats
print("\n=== Summary Statistics ===")
if len(df_kl) > 0:
    print(f"Thermal conductivity: {df_kl['CAS'].nunique()} compounds, {len(df_kl)} points")
    print(f"  T range: {df_kl['T_K'].min():.1f} - {df_kl['T_K'].max():.1f} K")
    print(f"  kl range: {df_kl['kl_WmK'].min():.6f} - {df_kl['kl_WmK'].max():.6f} W/m/K")
    print(f"  SMILES coverage: {(df_kl['SMILES'] != '').sum() / len(df_kl) * 100:.1f}%")

if len(df_cpl) > 0:
    print(f"Heat capacity: {df_cpl['CAS'].nunique()} compounds, {len(df_cpl)} points")
    print(f"  T range: {df_cpl['T_K'].min():.1f} - {df_cpl['T_K'].max():.1f} K")
    print(f"  Cpl range: {df_cpl['Cpl_JmolK'].min():.4f} - {df_cpl['Cpl_JmolK'].max():.4f} J/mol/K")
    print(f"  SMILES coverage: {(df_cpl['SMILES'] != '').sum() / len(df_cpl) * 100:.1f}%")

if len(df_pts) > 0:
    has_tb = (df_pts["Tb_K"] != "").sum()
    has_tf = (df_pts["Tflash_K"] != "").sum()
    has_both = ((df_pts["Tb_K"] != "") & (df_pts["Tflash_K"] != "")).sum()
    print(f"Flash/boiling points: {len(df_pts)} compounds")
    print(f"  With Tb: {has_tb}")
    print(f"  With Tflash: {has_tf}")
    print(f"  With both: {has_both}")
    print(f"  SMILES coverage: {(df_pts['SMILES'] != '').sum() / len(df_pts) * 100:.1f}%")
