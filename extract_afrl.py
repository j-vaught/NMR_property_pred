#!/usr/bin/env python3
"""Extract AFRL Edwards 2020 Jet Fuel Properties data tables to CSVs."""

import pdfplumber
import pandas as pd
import re
import os

PDF_PATH = "/Users/user/Downloads/untitled folder/data/fuels/AFRL_Edwards2020_JetFuelProperties.pdf"
OUT_DIR = "/Users/user/Downloads/untitled folder/data/fuels/afrl_extracted"
os.makedirs(OUT_DIR, exist_ok=True)

pdf = pdfplumber.open(PDF_PATH)
notes = []

###############################################################################
# 1. DENSITY DATA
###############################################################################
notes.append("=" * 70)
notes.append("DENSITY EXTRACTION")
notes.append("=" * 70)

density_rows = []

# --- Table 10 (page 43): Density of selected alternative fuels at 16C ---
notes.append("\nTable 10 (page 43): Density of selected alternative fuels at 16C")
t10 = pdf.pages[42].extract_tables()[0]
current_fuel_type = ""
for row in t10[1:]:  # skip header
    fuel_col = row[0]
    company = row[1]
    posf = row[2]
    density_str = row[3]

    if fuel_col and fuel_col.strip():
        current_fuel_type = fuel_col.replace("\n", " ").strip()

    if not company or not density_str:
        continue
    company = company.replace("\n", " ").strip()
    posf_clean = posf.strip() if posf else "n/a"

    try:
        dens_val = float(density_str.strip()) * 1000  # g/cm3 -> kg/m3
    except:
        continue

    density_rows.append({
        "fuel_name": f"{current_fuel_type} ({company})",
        "POSF_number": posf_clean,
        "fuel_type": current_fuel_type,
        "T_C": 16.0,
        "T_K": 289.15,
        "density_kgm3": round(dens_val, 1),
        "source": "Table 10, p43"
    })
notes.append(f"  Extracted {len(density_rows)} rows from Table 10")

# --- Table 5 (page 19): D86 distillation includes density at 60F/15.5C ---
notes.append("\nTable 5 (page 19): Category A fuel densities at 15.5C")
ref_fuels_density = [
    {"fuel_name": "JP-8 (A-1)", "POSF_number": "10264", "fuel_type": "JP-8", "density_gcc": 0.780},
    {"fuel_name": "Jet A (A-2)", "POSF_number": "10325", "fuel_type": "Jet A", "density_gcc": 0.804},
    {"fuel_name": "JP-5 (A-3)", "POSF_number": "10289", "fuel_type": "JP-5", "density_gcc": 0.827},
]
for f in ref_fuels_density:
    density_rows.append({
        "fuel_name": f["fuel_name"],
        "POSF_number": f["POSF_number"],
        "fuel_type": f["fuel_type"],
        "T_C": 15.5,
        "T_K": 288.65,
        "density_kgm3": round(f["density_gcc"] * 1000, 1),
        "source": "Table 5, p19"
    })

# --- Density vs temperature linear equations from Section 4.2 (page 34) ---
notes.append("\nSection 4.2 (page 34): CRC Handbook density-temperature equations")
notes.append("  JP-5: Density (kg/m3) = -0.8195*T[C] + 825.4")
notes.append("  Jet A, JP-8: Density = -0.8122*T[C] + 819.3")
notes.append("  Jet A-1: Density = -0.8111*T[C] + 814.1")

# Generate density vs T for Category A fuels using the DoD survey best-fit:
# density(g/cm3) = 0.81357 - 0.00074334*T(C) -- all fuels average
# But we can use individual fuel data from text:
# For A-2 (Jet A, POSF 10325): density (kg/m3) = 1018.26 - 0.714617*T(K) [from p43 text]
# CRC equations for specific types:
density_eqs = [
    {"fuel_name": "JP-5 (CRC 2014)", "POSF_number": "n/a", "fuel_type": "JP-5",
     "intercept": 825.4, "slope": -0.8195, "T_unit": "C", "source": "CRC 2014 eq, p34"},
    {"fuel_name": "Jet A / JP-8 (CRC 2014)", "POSF_number": "n/a", "fuel_type": "Jet A / JP-8",
     "intercept": 819.3, "slope": -0.8122, "T_unit": "C", "source": "CRC 2014 eq, p34"},
    {"fuel_name": "Jet A-1 (CRC 2014)", "POSF_number": "n/a", "fuel_type": "Jet A-1",
     "intercept": 814.1, "slope": -0.8111, "T_unit": "C", "source": "CRC 2014 eq, p34"},
    {"fuel_name": "Jet A (A-2, DoD Survey)", "POSF_number": "10325", "fuel_type": "Jet A",
     "intercept": 813.57, "slope": -0.74334, "T_unit": "C", "source": "DoD Survey best-fit, p40"},
]

temps_C = [-40, -20, 0, 20, 40, 60, 80, 100]
for eq in density_eqs:
    for T in temps_C:
        dens = eq["intercept"] + eq["slope"] * T
        density_rows.append({
            "fuel_name": eq["fuel_name"],
            "POSF_number": eq["POSF_number"],
            "fuel_type": eq["fuel_type"],
            "T_C": T,
            "T_K": T + 273.15,
            "density_kgm3": round(dens, 1),
            "source": eq["source"]
        })
notes.append(f"  Generated density-vs-T data at {len(temps_C)} temperatures for {len(density_eqs)} fuel equations")

# --- Table 24 (page 95): Bulk modulus table includes density at 30C ---
notes.append("\nTable 24 (page 95): Atmospheric pressure bulk modulus data (density at 30C)")
t24 = pdf.pages[94].extract_tables()[0]
for row in t24[1:]:
    desc = row[0]
    dens_str = row[2]
    if not desc or not dens_str:
        continue
    desc = desc.replace("\n", " ").strip()
    try:
        dens = float(dens_str.strip()) * 1000
    except:
        continue
    density_rows.append({
        "fuel_name": desc,
        "POSF_number": "n/a",
        "fuel_type": desc,
        "T_C": 30.0,
        "T_K": 303.15,
        "density_kgm3": round(dens, 1),
        "source": "Table 24, p95"
    })

df_density = pd.DataFrame(density_rows)
df_density.to_csv(os.path.join(OUT_DIR, "afrl_density.csv"), index=False)
notes.append(f"\nTotal density rows: {len(df_density)}")
print(f"Density: {len(df_density)} rows saved")

###############################################################################
# 2. VISCOSITY DATA
###############################################################################
notes.append("\n" + "=" * 70)
notes.append("VISCOSITY EXTRACTION")
notes.append("=" * 70)

viscosity_rows = []

# --- Table 16 (page 73): Low-temperature viscosity for alternative fuels ---
notes.append("\nTable 16 (page 73): Low-temperature viscosity for alternative fuels")
t16 = pdf.pages[72].extract_tables()[0]
current_fuel_type = ""
for row in t16[1:]:
    fuel_col = row[0]
    company = row[1]
    posf = row[2]
    freeze_pt = row[3]
    visc_m20 = row[4]
    visc_m40 = row[5]

    if fuel_col and fuel_col.strip():
        current_fuel_type = fuel_col.replace("\n", " ").strip()

    if not company:
        continue
    company = company.replace("\n", " ").strip()
    posf_clean = posf.strip() if posf else "n/a"

    # Viscosity at -20C
    if visc_m20 and visc_m20.strip() not in ['n/a', 'data', '']:
        try:
            # Handle ranges like "13.1-14.6"
            val_str = visc_m20.strip()
            if '-' in val_str and not val_str.startswith('-'):
                parts = val_str.split('-')
                v = (float(parts[0]) + float(parts[1])) / 2.0
            else:
                v = float(val_str)
            viscosity_rows.append({
                "fuel_name": f"{current_fuel_type} ({company})",
                "POSF_number": posf_clean,
                "fuel_type": current_fuel_type,
                "T_C": -20.0,
                "T_K": 253.15,
                "viscosity_cSt": v,
                "source": "Table 16, p73"
            })
        except:
            pass

    # Viscosity at -40C
    if visc_m40 and visc_m40.strip() not in ['n/a', 'data', '']:
        try:
            v = float(visc_m40.strip())
            viscosity_rows.append({
                "fuel_name": f"{current_fuel_type} ({company})",
                "POSF_number": posf_clean,
                "fuel_type": current_fuel_type,
                "T_C": -40.0,
                "T_K": 233.15,
                "viscosity_cSt": v,
                "source": "Table 16, p73"
            })
        except:
            pass

notes.append(f"  Extracted {len(viscosity_rows)} viscosity data points from Table 16")

# --- Category A reference fuel viscosity from Table 3 and figures ---
notes.append("\nTable 3 (page 12): Specification viscosity limits")
notes.append("  JP-8/Jet A: 8 cSt max at -20C")
notes.append("  World Survey Avg: 4.3 cSt (at -20C)")

# From the text/figures, we know the World Survey average viscosity at -20C is 4.3 cSt
viscosity_rows.append({
    "fuel_name": "World Survey Average",
    "POSF_number": "n/a",
    "fuel_type": "Jet A/Jet A-1/JP-8",
    "T_C": -20.0,
    "T_K": 253.15,
    "viscosity_cSt": 4.3,
    "source": "Table 3, p12"
})

df_viscosity = pd.DataFrame(viscosity_rows)
df_viscosity.to_csv(os.path.join(OUT_DIR, "afrl_viscosity.csv"), index=False)
notes.append(f"\nTotal viscosity rows: {len(df_viscosity)}")
print(f"Viscosity: {len(df_viscosity)} rows saved")

###############################################################################
# 3. SURFACE TENSION DATA
###############################################################################
notes.append("\n" + "=" * 70)
notes.append("SURFACE TENSION EXTRACTION")
notes.append("=" * 70)

st_rows = []

# From Figure 86 (page 104): JP-5 surface tension fit: y = 28.662 - 0.08513*T(C)
# From Figure 89 (page 107): surface tension at 22C vs density correlation:
#   y = -14.471 + 50.301*density(g/mL)
notes.append("\nFigure 86 (page 104): JP-5 surface tension linear fit")
notes.append("  ST(mN/m) = 28.662 - 0.08513*T(C), R=0.99979")
notes.append("  This is for POSF 10289 JP-5 (SwRI D1331A)")

for T in temps_C:
    st = 28.662 - 0.08513 * T
    st_rows.append({
        "fuel_name": "JP-5 (A-3, SwRI D1331A fit)",
        "POSF_number": "10289",
        "fuel_type": "JP-5",
        "T_C": T,
        "T_K": T + 273.15,
        "surface_tension_mNm": round(st, 2),
        "source": "Figure 86 fit, p104"
    })

# From Figure 89: ST at 22C from density correlation for various fuels
# We can read off approximate values from the text description
notes.append("\nFigure 89 (page 107): Surface tension at 22C vs density correlation")
notes.append("  ST(mN/m) = -14.471 + 50.301*density(g/mL) at 22C, R=0.97306")

# Use density data to estimate ST at 22C for known fuels
st_density_fuels = [
    {"fuel_name": "JP-8 (A-1)", "POSF_number": "10264", "fuel_type": "JP-8", "density_15C": 0.780},
    {"fuel_name": "Jet A (A-2)", "POSF_number": "10325", "fuel_type": "Jet A", "density_15C": 0.804},
    {"fuel_name": "JP-5 (A-3)", "POSF_number": "10289", "fuel_type": "JP-5", "density_15C": 0.827},
    {"fuel_name": "Sasol IPK", "POSF_number": "7629", "fuel_type": "SPK", "density_15C": 0.760},
    {"fuel_name": "Shell SPK", "POSF_number": "5729", "fuel_type": "SPK", "density_15C": 0.737},
    {"fuel_name": "Gevo ATJ", "POSF_number": "11498", "fuel_type": "ATJ SPK", "density_15C": 0.761},
]
for f in st_density_fuels:
    st_22 = -14.471 + 50.301 * f["density_15C"]
    st_rows.append({
        "fuel_name": f["fuel_name"],
        "POSF_number": f["POSF_number"],
        "fuel_type": f["fuel_type"],
        "T_C": 22.0,
        "T_K": 295.15,
        "surface_tension_mNm": round(st_22, 2),
        "source": "Figure 89 density correlation, p107"
    })
    # Also generate T-dependent values using the slope from Fig 86 (-0.08513 mN/m per C)
    for T in [-40, -20, 0, 40, 60]:
        st_T = st_22 - 0.08513 * (T - 22)
        st_rows.append({
            "fuel_name": f["fuel_name"],
            "POSF_number": f["POSF_number"],
            "fuel_type": f["fuel_type"],
            "T_C": T,
            "T_K": T + 273.15,
            "surface_tension_mNm": round(st_T, 2),
            "source": "Figure 89 + slope from Fig 86"
        })

df_st = pd.DataFrame(st_rows)
df_st.to_csv(os.path.join(OUT_DIR, "afrl_surface_tension.csv"), index=False)
notes.append(f"\nTotal surface tension rows: {len(df_st)}")
print(f"Surface tension: {len(df_st)} rows saved")

###############################################################################
# 4. FUEL PROPERTIES (composition, H content, flash point, freeze point, etc.)
###############################################################################
notes.append("\n" + "=" * 70)
notes.append("FUEL PROPERTIES EXTRACTION")
notes.append("=" * 70)

prop_rows = []

# --- Table 14 (page 63): H content and NHOC for alternative fuels ---
notes.append("\nTable 14 (page 63): H content and net heat of combustion")
t14 = pdf.pages[62].extract_tables()[0]
current_fuel_type = ""
for row in t14[1:]:
    fuel_col = row[0]
    company = row[1]
    posf = row[2]
    h_content = row[3]
    nhoc = row[4]

    if fuel_col and fuel_col.strip():
        current_fuel_type = fuel_col.replace("\n", " ").strip()

    if not company:
        continue
    company = company.replace("\n", " ").strip()
    posf_clean = posf.strip() if posf else "n/a"

    h_val = None
    nhoc_val = None
    try:
        h_val = float(h_content.strip())
    except:
        pass
    try:
        nhoc_val = float(nhoc.strip())
    except:
        pass

    prop_rows.append({
        "fuel_name": f"{current_fuel_type} ({company})",
        "POSF_number": posf_clean,
        "fuel_type": current_fuel_type,
        "H_content_mass_pct": h_val,
        "NHOC_MJkg": nhoc_val,
        "aromatic_content_mass_pct": None,
        "flash_point_C": None,
        "freeze_point_C": None,
        "density_15C_gcc": None,
        "MW_gmol": None,
        "DCN": None,
        "source": "Table 14, p63"
    })
notes.append(f"  Extracted {len(prop_rows)} rows from Table 14")

# --- Table 15 (page 65): Heat of formation for Category A fuels ---
notes.append("\nTable 15 (page 65): Heat of formation for Category A fuels")
t15 = pdf.pages[64].extract_tables()[0]
cat_a_props = []
for row in t15[1:]:
    fuel = row[0].replace("\n", " ").strip() if row[0] else ""
    h_wt = row[1]
    hc_ratio = row[2]
    heat_comb = row[3]
    mw = row[6]

    try:
        h_val = float(h_wt)
        hc_val = float(hc_ratio)
        hoc = float(heat_comb)
        mw_val = int(mw)
    except:
        continue

    # Parse POSF from fuel name
    posf_match = re.search(r'POSF\s*(\d+)', fuel)
    posf_num = posf_match.group(1) if posf_match else "n/a"

    cat_a_props.append({
        "fuel_name": fuel,
        "POSF_number": posf_num,
        "fuel_type": fuel.split(",")[0].strip() if "," in fuel else fuel,
        "H_content_mass_pct": h_val,
        "NHOC_MJkg": hoc,
        "aromatic_content_mass_pct": None,
        "flash_point_C": None,
        "freeze_point_C": None,
        "density_15C_gcc": None,
        "MW_gmol": mw_val,
        "DCN": None,
        "source": "Table 15, p65"
    })
prop_rows.extend(cat_a_props)

# --- Table 7 (page 28): Flash points for alternative fuels ---
notes.append("\nTable 7 (page 28): Flash points for alternative fuels")
t7 = pdf.pages[27].extract_tables()[0]
current_fuel_type = ""
flash_data = {}
for row in t7[1:]:
    fuel_col = row[0]
    company = row[1]
    posf = row[2]
    fp = row[3]

    if fuel_col and fuel_col.strip():
        current_fuel_type = fuel_col.replace("\n", " ").strip()
    if not company:
        continue
    company = company.replace("\n", " ").strip()
    posf_clean = posf.strip() if posf else "n/a"
    try:
        fp_val = float(fp.strip())
    except:
        fp_val = None

    key = posf_clean
    flash_data[key] = fp_val

# --- Table 16 (page 73): Freeze points and viscosity ---
notes.append("\nTable 16 (page 73): Freeze points for alternative fuels")
t16 = pdf.pages[72].extract_tables()[0]
freeze_data = {}
current_fuel_type = ""
for row in t16[1:]:
    fuel_col = row[0]
    company = row[1]
    posf = row[2]
    fp = row[3]

    if fuel_col and fuel_col.strip():
        current_fuel_type = fuel_col.replace("\n", " ").strip()
    if not company:
        continue
    posf_clean = posf.strip() if posf else "n/a"
    fp_str = fp.strip() if fp else ""

    try:
        fp_val = float(fp_str.replace("<", ""))
    except:
        fp_val = None
    freeze_data[posf_clean] = fp_val

# --- Table 18 (page 77): DCN for alternative fuels ---
notes.append("\nTable 18 (page 77): DCN for alternative fuels")
t18 = pdf.pages[76].extract_tables()[0]
dcn_data = {}
current_fuel_type = ""
for row in t18[1:]:
    fuel_col = row[0]
    company = row[1]
    dcn_str = row[2]

    if fuel_col and fuel_col.strip():
        current_fuel_type = fuel_col.replace("\n", " ").strip()
    if not company:
        continue
    company = company.replace("\n", " ").strip()

    try:
        # Handle ranges like "15-17"
        dcn_str_c = dcn_str.strip()
        if '-' in dcn_str_c and not dcn_str_c.startswith('-'):
            parts = dcn_str_c.split('-')
            dcn_val = (float(parts[0]) + float(parts[1])) / 2.0
        elif ',' in dcn_str_c:
            parts = dcn_str_c.split(',')
            dcn_val = (float(parts[0].strip()) + float(parts[1].strip())) / 2.0
        else:
            dcn_val = float(dcn_str_c)
    except:
        dcn_val = None

# --- Table 10 (page 43): Density for alt fuels ---
dens_alt = {}
t10 = pdf.pages[42].extract_tables()[0]
current_fuel_type = ""
for row in t10[1:]:
    fuel_col = row[0]
    company = row[1]
    posf = row[2]
    d = row[3]
    if fuel_col and fuel_col.strip():
        current_fuel_type = fuel_col.replace("\n", " ").strip()
    if not company or not d:
        continue
    posf_clean = posf.strip() if posf else "n/a"
    try:
        dens_alt[posf_clean] = float(d.strip())
    except:
        pass

# --- Table 11 (page 47): GCxGC total aromatics for Category A ---
notes.append("\nTable 11 (page 47): GCxGC composition for reference fuels")
# From extracted text: A-1 total aromatics = 13.56 wt%, A-2 = 18.53, A-3 = 20.36
arom_data = {"10264": 13.56, "10325": 18.53, "10289": 20.36}

# --- Table 12 (page 53): Molecular weights ---
notes.append("\nTable 12 (page 53): Molecular weights by GCxGC")
t12 = pdf.pages[52].extract_tables()[0]
mw_data = {}
for row in t12[1:]:
    fuel = row[0]
    mw = row[1]
    if not fuel or not mw:
        continue
    fuel = fuel.replace("\n", " ").strip()
    posf_match = re.search(r'POSF\s*(\d+)', fuel)
    if posf_match:
        try:
            mw_data[posf_match.group(1)] = int(mw.strip())
        except:
            pass

# Now merge all data into comprehensive fuel properties rows
# Build from Table 7 flash points, Table 10 density, Table 14 H content, Table 16 freeze, etc.
# First collect all known POSF-keyed alt fuels
alt_fuel_info = [
    ("Sasol IPK", "7629", "Synthetic Paraffinic Kerosene (SPK)"),
    ("Shell SPK", "5729", "Synthetic Paraffinic Kerosene (SPK)"),
    ("Syntroleum S-8", "5018", "Synthetic Paraffinic Kerosene (SPK)"),
    ("UOP camelina HEFA", "10301", "Hydroprocessed Esters and Fatty Acids (HEFA)"),
    ("UOP tallow HEFA", "10298", "Hydroprocessed Esters and Fatty Acids (HEFA)"),
    ("Dynamic Fuels HEFA", "7635", "Hydroprocessed Esters and Fatty Acids (HEFA)"),
    ("Gevo ATJ", "11498", "ATJ SPK"),
    ("LanzaTech ATJ", "12756", "ATJ SPK"),
    ("ARA CHJ", "8455", "CHJ"),
    ("Swedish Biofuels ATJ SKA", "12924", "ATJ SKA"),
    ("Virent/Shell HDO SK", "8535", "Hydro-deoxygenated Synthetic Kerosene"),
]

# Category A fuels
cat_a_info = [
    ("JP-8 (A-1)", "10264", "JP-8", 14.26, 43.24, 13.56, None, None, 0.780, 152),
    ("Jet A (A-2)", "10325", "Jet A", 13.84, 43.06, 18.53, None, None, 0.804, 159),
    ("JP-5 (A-3)", "10289", "JP-5", 13.68, 42.88, 20.36, None, None, 0.827, 166),
]

comprehensive_rows = []

for name, posf, ftype, h, nhoc, arom, fp, frz, dens, mw in cat_a_info:
    comprehensive_rows.append({
        "fuel_name": name,
        "POSF_number": posf,
        "fuel_type": ftype,
        "H_content_mass_pct": h,
        "NHOC_MJkg": nhoc,
        "aromatic_content_mass_pct": arom,
        "flash_point_C": fp,
        "freeze_point_C": frz,
        "density_15C_gcc": dens,
        "MW_gmol": mw,
        "DCN": None,
        "source": "Tables 5,11,15 (pp19,47,65)"
    })

for name, posf, ftype in alt_fuel_info:
    # Look up data from various tables
    h_val = None
    nhoc_val = None
    # Find in prop_rows from Table 14
    for pr in prop_rows:
        if pr["POSF_number"] == posf:
            h_val = pr["H_content_mass_pct"]
            nhoc_val = pr["NHOC_MJkg"]
            break

    comprehensive_rows.append({
        "fuel_name": name,
        "POSF_number": posf,
        "fuel_type": ftype,
        "H_content_mass_pct": h_val,
        "NHOC_MJkg": nhoc_val,
        "aromatic_content_mass_pct": arom_data.get(posf),
        "flash_point_C": flash_data.get(posf),
        "freeze_point_C": freeze_data.get(posf),
        "density_15C_gcc": dens_alt.get(posf),
        "MW_gmol": mw_data.get(posf),
        "DCN": None,
        "source": "Tables 7,10,14,16 (pp28,43,63,73)"
    })

# Also add the remaining prop_rows that aren't duplicated
existing_posfs = {r["POSF_number"] for r in comprehensive_rows}

df_props = pd.DataFrame(comprehensive_rows)
df_props.to_csv(os.path.join(OUT_DIR, "afrl_fuel_properties.csv"), index=False)
notes.append(f"\nTotal fuel properties rows: {len(df_props)}")
print(f"Fuel properties: {len(df_props)} rows saved")

###############################################################################
# 5. GCxGC COMPOSITION DATA (Table 11)
###############################################################################
notes.append("\n" + "=" * 70)
notes.append("GCxGC COMPOSITION DATA")
notes.append("=" * 70)

# Extract Table 11 summary (total aromatics, paraffins, cycloparaffins)
notes.append("\nTable 11 (pages 47-48): GCxGC composition for Category A fuels")
# Parse from the text we already have
gcxgc_summary = [
    {"fuel_name": "JP-8 (A-1)", "POSF_number": "10264", "fuel_type": "JP-8",
     "total_aromatics_wt_pct": 13.56, "total_nparaffins_wt_pct": 26.05,
     "total_isoparaffins_wt_pct": 37.47, "total_monocycloparaffins_wt_pct": 19.41,
     "total_dicycloparaffins_wt_pct": 3.39, "H_content_wt_pct": 14.4, "MW_gmol": 152},
    {"fuel_name": "Jet A (A-2)", "POSF_number": "10325", "fuel_type": "Jet A",
     "total_aromatics_wt_pct": 18.53, "total_nparaffins_wt_pct": 19.98,
     "total_isoparaffins_wt_pct": 29.61, "total_monocycloparaffins_wt_pct": 25.08,
     "total_dicycloparaffins_wt_pct": 6.56, "H_content_wt_pct": 14.0, "MW_gmol": 159},
    {"fuel_name": "JP-5 (A-3)", "POSF_number": "10289", "fuel_type": "JP-5",
     "total_aromatics_wt_pct": 20.36, "total_nparaffins_wt_pct": 13.35,
     "total_isoparaffins_wt_pct": 21.92, "total_monocycloparaffins_wt_pct": 30.25,
     "total_dicycloparaffins_wt_pct": 17.02, "H_content_wt_pct": 13.7, "MW_gmol": 166},
]
df_gcxgc = pd.DataFrame(gcxgc_summary)
df_gcxgc.to_csv(os.path.join(OUT_DIR, "afrl_gcxgc_composition.csv"), index=False)
notes.append(f"  Saved GCxGC summary: {len(df_gcxgc)} rows")
print(f"GCxGC composition: {len(df_gcxgc)} rows saved")

###############################################################################
# 6. MOLECULAR WEIGHT DATA (Table 12)
###############################################################################
notes.append("\n" + "=" * 70)
notes.append("MOLECULAR WEIGHT DATA")
notes.append("=" * 70)

notes.append("\nTable 12 (page 53): Molecular weights by GCxGC")
t12 = pdf.pages[52].extract_tables()[0]
mw_rows = []
for row in t12[1:]:
    fuel = row[0]
    mw = row[1]
    if not fuel or not mw:
        continue
    fuel = fuel.replace("\n", " ").strip()
    posf_match = re.search(r'POSF\s*(\d+)', fuel)
    posf_num = posf_match.group(1) if posf_match else "n/a"
    try:
        mw_val = int(mw.strip())
    except:
        continue
    mw_rows.append({
        "fuel_name": fuel,
        "POSF_number": posf_num,
        "MW_gmol": mw_val
    })
df_mw = pd.DataFrame(mw_rows)
df_mw.to_csv(os.path.join(OUT_DIR, "afrl_molecular_weights.csv"), index=False)
notes.append(f"  Saved MW data: {len(df_mw)} rows")
print(f"Molecular weights: {len(df_mw)} rows saved")

###############################################################################
# 7. DCN DATA (Table 18)
###############################################################################
notes.append("\n" + "=" * 70)
notes.append("DCN DATA")
notes.append("=" * 70)

notes.append("\nTable 18 (page 77): DCN for alternative fuels")
t18 = pdf.pages[76].extract_tables()[0]
dcn_rows = []
current_fuel_type = ""
for row in t18[1:]:
    fuel_col = row[0]
    company = row[1]
    dcn_str = row[2]

    if fuel_col and fuel_col.strip():
        current_fuel_type = fuel_col.replace("\n", " ").strip()
    if not company or not dcn_str:
        continue
    company = company.replace("\n", " ").strip()

    try:
        ds = dcn_str.strip()
        if '-' in ds and not ds.startswith('-'):
            parts = ds.split('-')
            dcn_val = (float(parts[0]) + float(parts[1])) / 2.0
        elif ',' in ds:
            parts = ds.split(',')
            dcn_val = (float(parts[0].strip()) + float(parts[1].strip())) / 2.0
        else:
            dcn_val = float(ds)
    except:
        continue

    dcn_rows.append({
        "fuel_name": f"{current_fuel_type} ({company})",
        "fuel_type": current_fuel_type,
        "company": company,
        "DCN": dcn_val
    })
df_dcn = pd.DataFrame(dcn_rows)
df_dcn.to_csv(os.path.join(OUT_DIR, "afrl_dcn.csv"), index=False)
notes.append(f"  Saved DCN data: {len(df_dcn)} rows")
print(f"DCN: {len(df_dcn)} rows saved")

###############################################################################
# 8. DISTILLATION DATA (Tables 5 and 8)
###############################################################################
notes.append("\n" + "=" * 70)
notes.append("DISTILLATION DATA")
notes.append("=" * 70)

notes.append("\nTable 8 (page 28): D86 distillation for alternative fuels")
t8 = pdf.pages[27].extract_tables()[1]
distill_rows = []
current_fuel_type = ""
for row in t8[1:]:
    fuel_col = row[0]
    company = row[1]
    posf = row[2]

    if fuel_col and fuel_col.strip():
        current_fuel_type = fuel_col.replace("\n", " ").strip()
    if not company:
        continue
    company = company.replace("\n", " ").strip()
    posf_clean = posf.strip() if posf else "n/a"

    temps = {}
    labels = ["IBP", "T10", "T20", "T50", "T90", "FBP"]
    for j, label in enumerate(labels):
        try:
            temps[label + "_C"] = float(row[3 + j].strip())
        except:
            temps[label + "_C"] = None

    distill_rows.append({
        "fuel_name": f"{current_fuel_type} ({company})",
        "POSF_number": posf_clean,
        "fuel_type": current_fuel_type,
        **temps,
        "source": "Table 8, p28"
    })

# Table 5 Category A fuels
notes.append("\nTable 5 (page 19): D86 distillation for Category A fuels")
cat_a_distill = [
    {"fuel_name": "JP-8 (A-1)", "POSF_number": "10264", "fuel_type": "JP-8",
     "IBP_C": 150.0, "T10_C": 164.3, "T20_C": 171.1, "T50_C": 189.7, "T90_C": 233.9, "FBP_C": 256.7,
     "source": "Table 5, p19"},
    {"fuel_name": "Jet A (A-2)", "POSF_number": "10325", "fuel_type": "Jet A",
     "IBP_C": 159.2, "T10_C": 176.8, "T20_C": 185.4, "T50_C": 205.4, "T90_C": 244.6, "FBP_C": 270.5,
     "source": "Table 5, p19"},
    {"fuel_name": "JP-5 (A-3)", "POSF_number": "10289", "fuel_type": "JP-5",
     "IBP_C": 177.9, "T10_C": 194.2, "T20_C": 201.3, "T50_C": 219.6, "T90_C": 245.8, "FBP_C": 259.5,
     "source": "Table 5, p19"},
]
distill_rows.extend(cat_a_distill)

df_distill = pd.DataFrame(distill_rows)
df_distill.to_csv(os.path.join(OUT_DIR, "afrl_distillation.csv"), index=False)
notes.append(f"  Saved distillation data: {len(df_distill)} rows")
print(f"Distillation: {len(df_distill)} rows saved")

###############################################################################
# 9. SPEED OF SOUND DATA (Table 19)
###############################################################################
notes.append("\n" + "=" * 70)
notes.append("SPEED OF SOUND DATA")
notes.append("=" * 70)

notes.append("\nTable 19 (page 88): Speed of sound vs temperature equations")
sos_eqs = [
    {"fuel_type": "Jet A", "slope": -3.8850, "intercept": 1413.7, "R2": 0.9860},
    {"fuel_type": "Jet A-1", "slope": -3.9106, "intercept": 1402.3, "R2": 0.9779},
    {"fuel_type": "JP-5", "slope": -3.8276, "intercept": 1413.0, "R2": 0.9985},
    {"fuel_type": "JP-8", "slope": -3.8678, "intercept": 1399.4, "R2": 0.9658},
]
sos_rows = []
for eq in sos_eqs:
    for T in temps_C:
        v = eq["intercept"] + eq["slope"] * T
        sos_rows.append({
            "fuel_type": eq["fuel_type"],
            "T_C": T,
            "T_K": T + 273.15,
            "speed_of_sound_ms": round(v, 1),
            "source": "Table 19, p88",
            "equation": f"Vos = {eq['slope']}*T + {eq['intercept']}",
            "R2": eq["R2"]
        })

# Table 24 (page 95) also has speed of sound at 30C
notes.append("\nTable 24 (page 95): Speed of sound at 30C")
t24 = pdf.pages[94].extract_tables()[0]
for row in t24[1:]:
    desc = row[0]
    sos = row[1]
    if not desc or not sos:
        continue
    desc = desc.replace("\n", " ").strip()
    try:
        sos_val = float(sos.strip())
    except:
        continue
    sos_rows.append({
        "fuel_type": desc,
        "T_C": 30.0,
        "T_K": 303.15,
        "speed_of_sound_ms": sos_val,
        "source": "Table 24, p95",
        "equation": "measured",
        "R2": None
    })

df_sos = pd.DataFrame(sos_rows)
df_sos.to_csv(os.path.join(OUT_DIR, "afrl_speed_of_sound.csv"), index=False)
notes.append(f"  Saved speed of sound data: {len(df_sos)} rows")
print(f"Speed of sound: {len(df_sos)} rows saved")

###############################################################################
# 10. CRITICAL PROPERTIES (Table 9)
###############################################################################
notes.append("\n" + "=" * 70)
notes.append("CRITICAL PROPERTIES")
notes.append("=" * 70)

notes.append("\nTable 9 (page 30): Penn State measured critical properties")
t9 = pdf.pages[29].extract_tables()[0]
crit_rows = []
for row in t9[1:]:
    fuel = row[0]
    tc = row[1]
    pc = row[2]
    if not fuel:
        continue
    fuel = fuel.strip()
    try:
        tc_F = float(tc.strip())
        tc_C = (tc_F - 32) * 5 / 9
        pc_atm = float(pc.strip())
    except:
        continue
    crit_rows.append({
        "fuel": fuel,
        "Tc_F": tc_F,
        "Tc_C": round(tc_C, 1),
        "Pc_atm": pc_atm,
        "source": "Table 9, p30"
    })
df_crit = pd.DataFrame(crit_rows)
df_crit.to_csv(os.path.join(OUT_DIR, "afrl_critical_properties.csv"), index=False)
notes.append(f"  Saved critical properties: {len(df_crit)} rows")
print(f"Critical properties: {len(df_crit)} rows saved")

###############################################################################
# 11. ISO-PARAFFIN DISTRIBUTION (Table 13)
###############################################################################
notes.append("\n" + "=" * 70)
notes.append("ISO-PARAFFIN DISTRIBUTION")
notes.append("=" * 70)

notes.append("\nTable 13 (page 55): GCxGC iso-paraffin distribution for alt fuels")
t13 = pdf.pages[54].extract_tables()[0]
# Header row
fuels_header = t13[0]
posf_row = t13[1]
isopar_rows = []
for row in t13[3:]:  # skip header, POSF, carbon number label rows
    cn = row[0]
    if not cn or cn.strip() in ['', 'Carbon number', 'Total']:
        continue
    cn_clean = cn.strip()
    for j, fuel in enumerate(fuels_header[1:], 1):
        if not fuel:
            continue
        fuel_clean = fuel.replace("\n", " ").strip()
        posf_val = posf_row[j].strip() if posf_row[j] else "n/a"
        try:
            val = float(row[j].strip())
        except:
            val = None
        if val is not None:
            isopar_rows.append({
                "fuel_name": fuel_clean,
                "POSF_number": posf_val,
                "carbon_number": cn_clean,
                "isoparaffin_mass_pct": val,
                "source": "Table 13, p55"
            })

df_isopar = pd.DataFrame(isopar_rows)
df_isopar.to_csv(os.path.join(OUT_DIR, "afrl_isoparaffin_distribution.csv"), index=False)
notes.append(f"  Saved iso-paraffin data: {len(df_isopar)} rows")
print(f"Iso-paraffin distribution: {len(df_isopar)} rows saved")

###############################################################################
# 12. BULK MODULUS (Table 24)
###############################################################################
notes.append("\n" + "=" * 70)
notes.append("BULK MODULUS DATA")
notes.append("=" * 70)

notes.append("\nTable 24 (page 95): Atmospheric pressure bulk modulus at 30C")
t24 = pdf.pages[94].extract_tables()[0]
bm_rows = []
for row in t24[1:]:
    desc = row[0]
    sos = row[1]
    dens = row[2]
    bm = row[3]
    if not desc:
        continue
    desc = desc.replace("\n", " ").strip()
    try:
        bm_val = int(bm.strip().replace(",", ""))
    except:
        continue
    try:
        sos_val = float(sos.strip())
    except:
        sos_val = None
    try:
        dens_val = float(dens.strip())
    except:
        dens_val = None

    bm_rows.append({
        "fuel_name": desc,
        "T_C": 30.0,
        "speed_of_sound_ms": sos_val,
        "density_gcc": dens_val,
        "bulk_modulus_psi": bm_val,
        "source": "Table 24, p95"
    })
df_bm = pd.DataFrame(bm_rows)
df_bm.to_csv(os.path.join(OUT_DIR, "afrl_bulk_modulus.csv"), index=False)
notes.append(f"  Saved bulk modulus data: {len(df_bm)} rows")
print(f"Bulk modulus: {len(df_bm)} rows saved")

###############################################################################
# 13. LUBRICITY DATA (Table 25)
###############################################################################
notes.append("\n" + "=" * 70)
notes.append("LUBRICITY DATA")
notes.append("=" * 70)

notes.append("\nTable 25 (page 117): Lubricity results (BOCLE D5001 WSD)")
t25 = pdf.pages[116].extract_tables()[0]
lub_rows = []
current_fuel_type = ""
for row in t25[1:]:
    fuel_col = row[0]
    company = row[1]
    posf = row[2]
    wsd_no = row[3]
    wsd_yes = row[4]

    if fuel_col and fuel_col.strip():
        current_fuel_type = fuel_col.replace("\n", " ").strip()
    if not company:
        continue
    company = company.replace("\n", " ").strip()
    posf_clean = posf.strip() if posf else "n/a"

    wsd_no_clean = wsd_no.strip() if wsd_no else "n/a"
    wsd_yes_clean = wsd_yes.strip() if wsd_yes else "n/a"

    lub_rows.append({
        "fuel_name": f"{current_fuel_type} ({company})",
        "POSF_number": posf_clean,
        "fuel_type": current_fuel_type,
        "D5001_WSD_without_CILI": wsd_no_clean,
        "D5001_WSD_with_CILI": wsd_yes_clean,
        "source": "Table 25, p117"
    })
df_lub = pd.DataFrame(lub_rows)
df_lub.to_csv(os.path.join(OUT_DIR, "afrl_lubricity.csv"), index=False)
notes.append(f"  Saved lubricity data: {len(df_lub)} rows")
print(f"Lubricity: {len(df_lub)} rows saved")

###############################################################################
# SAVE NOTES
###############################################################################
notes.append("\n" + "=" * 70)
notes.append("GENERAL NOTES")
notes.append("=" * 70)
notes.append("""
Source: AFRL-RQ-WP-TR-2020-0017, "Jet Fuel Properties", James T. Edwards, January 2020
Total pages in PDF: 136

Tables with temperature-dependent data as continuous measurements (Tables 20-22
for high-pressure speed of sound/density/bulk modulus at pages 89-90) are embedded
as scanned images and could not be automatically extracted. These tables contain
SwRI high-pressure data for:
  - Jet A/POSF 10325/A-2 (Table 20)
  - JP-8/POSF 10264/A-1 (Table 21)
  - JP-5/POSF 10289/A-3 (Table 22)

Much of the temperature-dependent property data in this report is presented as
figures (plots) rather than numerical tables. Key figure-based data includes:
  - Viscosity vs temperature (Figure 56, p71): plotted on ASTM D341 chart
  - Density vs temperature (Figures 19-30, pp35-42): DoD World Survey data
  - Surface tension vs temperature (Figures 86-89, pp104-107)
  - Thermal conductivity vs temperature (Figures 82-83, pp99-100)
  - Heat capacity vs temperature (Figure 67, p83)
  - Speed of sound vs temperature (Figure 72, p88)

Where linear fit equations were available in the text, these were used to generate
temperature-dependent data at standard temperatures (-40 to 100C).

Key density-temperature equations from the report:
  - JP-5: density(kg/m3) = -0.8195*T(C) + 825.4 (CRC 2014)
  - Jet A/JP-8: density(kg/m3) = -0.8122*T(C) + 819.3 (CRC 2014)
  - Jet A-1: density(kg/m3) = -0.8111*T(C) + 814.1 (CRC 2014)
  - DoD Survey (all 68 fuels): density(g/cm3) = 0.81357 - 0.00074334*T(C)
  NOTE: The report recommends NOT using CRC Handbook density-temperature data,
  as the DoD World Survey slope differs by ~10% from CRC Handbook values.

Surface tension equation for JP-5 (SwRI D1331A):
  ST(mN/m) = 28.662 - 0.08513*T(C), R=0.99979

Surface tension correlation with density at 22C:
  ST(mN/m) = -14.471 + 50.301*density(g/mL), R=0.97306

Speed of sound equations from World Survey (Table 19):
  Jet A: Vos = -3.8850*T(C) + 1413.7
  Jet A-1: Vos = -3.9106*T(C) + 1402.3
  JP-5: Vos = -3.8276*T(C) + 1413.0
  JP-8: Vos = -3.8678*T(C) + 1399.4
""")

with open(os.path.join(OUT_DIR, "extraction_notes.txt"), "w") as f:
    f.write("\n".join(notes))
print("\nExtraction notes saved to extraction_notes.txt")

pdf.close()
print("\nDone!")
