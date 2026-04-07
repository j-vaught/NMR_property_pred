"""Task 2: Generate density vs temperature data using thermo library for all compounds with SMILES."""
import csv
import warnings
import numpy as np

warnings.filterwarnings('ignore')

OUT_FILE = "/Users/user/Downloads/untitled folder/data/thermo/density/thermo_liquid_density.csv"

# Get density data sources from chemicals
from chemicals.identifiers import search_chemical
from chemicals.volume import rho_data_COSTALD, rho_data_VDI_PPDS_2, rho_data_Perry_8E_105_l

# Collect all CAS numbers that have density correlation data
cas_sets = set()

# Perry's correlations (DIPPR 105 equation)
for cas in rho_data_Perry_8E_105_l.index:
    cas_sets.add(cas)

# VDI PPDS
for cas in rho_data_VDI_PPDS_2.index:
    cas_sets.add(cas)

# COSTALD
for cas in rho_data_COSTALD.index:
    cas_sets.add(cas)

print(f"Total unique CAS numbers with density data: {len(cas_sets)}")

rows = []
compounds_ok = 0
compounds_fail = 0

for i, cas in enumerate(sorted(cas_sets)):
    if i % 100 == 0:
        print(f"  Processing {i}/{len(cas_sets)}...")

    try:
        chem = search_chemical(cas)
        name = chem.common_name if hasattr(chem, 'common_name') else chem.name
        smiles = chem.smiles if hasattr(chem, 'smiles') else None
        if not smiles:
            continue
    except Exception:
        compounds_fail += 1
        continue

    # Get Tmin/Tmax from Perry data if available, else VDI, else use Tm/Tc
    Tmin = None
    Tmax = None

    if cas in rho_data_Perry_8E_105_l.index:
        row_data = rho_data_Perry_8E_105_l.loc[cas]
        Tmin = row_data.get('Tmin', None) if hasattr(row_data, 'get') else getattr(row_data, 'Tmin', None)
        Tmax = row_data.get('Tmax', None) if hasattr(row_data, 'get') else getattr(row_data, 'Tmax', None)

    if cas in rho_data_VDI_PPDS_2.index and (Tmin is None or Tmax is None):
        row_data = rho_data_VDI_PPDS_2.loc[cas]
        if Tmin is None:
            Tmin = row_data.get('Tm', None) if hasattr(row_data, 'get') else getattr(row_data, 'Tm', None)
        if Tmax is None:
            Tmax = row_data.get('Tc', None) if hasattr(row_data, 'get') else getattr(row_data, 'Tc', None)

    # Fallback: use melting point and critical temperature from the chemical
    if Tmin is None:
        Tmin = getattr(chem, 'Tm', None) or 200.0
    if Tmax is None:
        Tmax = getattr(chem, 'Tc', None) or 600.0

    try:
        Tmin = float(Tmin)
        Tmax = float(Tmax)
    except (TypeError, ValueError):
        Tmin = 200.0
        Tmax = 600.0

    if Tmin >= Tmax or Tmin < 50 or Tmax > 3000:
        compounds_fail += 1
        continue

    from chemicals.dippr import EQ105

    temps = np.linspace(Tmin, Tmax, 20)

    # Get MW for unit conversion (EQ105 returns mol/m3)
    MW = getattr(chem, 'MW', None)
    if MW is None:
        compounds_fail += 1
        continue
    MW = float(MW)

    # Try Perry 105 equation first
    got_data = False
    if cas in rho_data_Perry_8E_105_l.index:
        try:
            p = rho_data_Perry_8E_105_l.loc[cas]
            A, B, C, D = float(p['C1']), float(p['C2']), float(p['C3']), float(p['C4'])
            for T in temps:
                # EQ105 returns mol/m3, convert to kg/m3
                rho_molm3 = EQ105(T, A, B, C, D)
                rho_kgm3 = rho_molm3 * MW / 1000.0
                if rho_kgm3 > 0 and rho_kgm3 < 30000:
                    rows.append({
                        'CAS': cas,
                        'name': name,
                        'SMILES': smiles,
                        'T_K': round(T, 2),
                        'density_kgm3': round(rho_kgm3, 4)
                    })
            got_data = True
        except Exception:
            pass

    if not got_data and cas in rho_data_VDI_PPDS_2.index:
        try:
            p = rho_data_VDI_PPDS_2.loc[cas]
            Tc_vdi = float(p['Tc'])
            rhoc = float(p['rhoc'])
            Av = float(p['A'])
            Bv = float(p['B'])
            Cv = float(p['C'])
            Dv = float(p['D'])
            for T in temps:
                tau = 1.0 - T / Tc_vdi
                if tau <= 0:
                    continue
                # VDI PPDS: rhoc and coefficients give mol/m3
                rho_molm3 = rhoc + Av*tau**0.35 + Bv*tau**(2.0/3.0) + Cv*tau + Dv*tau**(4.0/3.0)
                rho_kgm3 = rho_molm3 * MW / 1000.0
                if rho_kgm3 > 0 and rho_kgm3 < 30000:
                    rows.append({
                        'CAS': cas,
                        'name': name,
                        'SMILES': smiles,
                        'T_K': round(T, 2),
                        'density_kgm3': round(rho_kgm3, 4)
                    })
            got_data = True
        except Exception:
            pass

    if got_data:
        compounds_ok += 1
    else:
        compounds_fail += 1

with open(OUT_FILE, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['CAS', 'name', 'SMILES', 'T_K', 'density_kgm3'])
    writer.writeheader()
    writer.writerows(rows)

print(f"\nThermo density generation complete:")
print(f"  Compounds with data: {compounds_ok}")
print(f"  Compounds failed/skipped: {compounds_fail}")
print(f"  Total data points: {len(rows)}")
print(f"  Output: {OUT_FILE}")
