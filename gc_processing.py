"""
Group contribution property estimation for NMRexp compounds
and expanded property evaluation for chemicals TSV compounds.
"""

import pandas as pd
import numpy as np
import warnings
import os
import traceback

warnings.filterwarnings('ignore')

BASE = "/Users/user/Downloads/untitled folder"
OUT = os.path.join(BASE, "data/thermo/group_contribution")
os.makedirs(OUT, exist_ok=True)

# ============================================================
# PART 1: NMRexp compounds - group contribution estimation
# ============================================================
print("=" * 60)
print("PART 1: Loading NMRexp data...")
df_nmr = pd.read_parquet(
    os.path.join(BASE, "data/nmr/nmrexp/NMRexp_1H_13C.parquet"),
    columns=['SMILES', 'NMR_type']
)
unique_smiles = df_nmr[df_nmr['NMR_type'] == '1H NMR']['SMILES'].unique()
print(f"Total unique SMILES with 1H NMR: {len(unique_smiles)}")

# Random sample of 5000
rng = np.random.RandomState(42)
if len(unique_smiles) > 5000:
    sample_smiles = rng.choice(unique_smiles, size=5000, replace=False)
else:
    sample_smiles = unique_smiles
print(f"Sampled SMILES: {len(sample_smiles)}")

temperatures = [250, 300, 350, 400, 450]

from thermo import Chemical
from thermo.group_contribution.joback import Joback
from chemicals.viscosity import Letsou_Stiel
from chemicals.interface import Brock_Bird
from rdkit import Chem
from rdkit.Chem import Descriptors

visc_rows = []
st_rows = []
count_db = 0
count_joback = 0
count_fail = 0

for i, smi in enumerate(sample_smiles):
    if (i + 1) % 500 == 0:
        print(f"  Processing {i+1}/{len(sample_smiles)} "
              f"(db={count_db}, joback={count_joback}, fail={count_fail})")

    # Try database lookup first
    method = None
    Tc = Pc = Tb = omega = MW = None

    try:
        c = Chemical(smi)
        if c.Tc and c.Pc and c.Tb:
            Tc = c.Tc
            Pc = c.Pc
            Tb = c.Tb
            omega = c.omega if c.omega else 0.0
            MW = c.MW
            method = 'database'
            count_db += 1
        else:
            raise ValueError("Missing critical properties")
    except Exception:
        # Try Joback
        try:
            j = Joback(smi)
            est = j.estimate()
            if est.get('Tc') and est.get('Pc') and est.get('Tb'):
                Tc = est['Tc']
                Pc = est['Pc']
                Tb = est['Tb']
                # Estimate omega from Tc, Pc, Tb using Edmister correlation
                try:
                    omega = (3.0 / 7.0) * (np.log10(Pc / 101325.0)) / (Tc / Tb - 1.0) - 1.0
                except:
                    omega = 0.0
                # Get MW from rdkit
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    MW = Descriptors.MolWt(mol)
                else:
                    MW = j.MW if j.MW else None
                method = 'joback'
                count_joback += 1
            else:
                raise ValueError("Joback missing properties")
        except Exception:
            count_fail += 1
            continue

    if MW is None:
        count_fail += 1
        if method == 'database':
            count_db -= 1
        else:
            count_joback -= 1
        continue

    # Compute properties at each temperature
    for T in temperatures:
        # Skip if T > Tc (supercritical)
        if T >= Tc:
            continue

        # Viscosity via Letsou-Stiel
        try:
            visc = Letsou_Stiel(T, MW, Tc, Pc, omega)
            if visc is not None and visc > 0:
                visc_rows.append({
                    'SMILES': smi, 'method': method, 'T_K': T,
                    'viscosity_Pas': visc, 'Tc': Tc, 'Pc': Pc, 'Tb': Tb
                })
        except Exception:
            pass

        # Surface tension via Brock-Bird
        try:
            st = Brock_Bird(T, Tb, Tc, Pc)
            if st is not None and st > 0:
                st_rows.append({
                    'SMILES': smi, 'method': method, 'T_K': T,
                    'surface_tension_Nm': st, 'Tc': Tc, 'Pc': Pc, 'Tb': Tb
                })
        except Exception:
            pass

print(f"\nPart 1 Results:")
print(f"  Database lookups: {count_db}")
print(f"  Joback estimates: {count_joback}")
print(f"  Failures: {count_fail}")
print(f"  Viscosity rows: {len(visc_rows)}")
print(f"  Surface tension rows: {len(st_rows)}")

df_visc = pd.DataFrame(visc_rows)
df_st = pd.DataFrame(st_rows)

df_visc.to_csv(os.path.join(OUT, "gc_viscosity.csv"), index=False)
df_st.to_csv(os.path.join(OUT, "gc_surface_tension.csv"), index=False)
print("  Saved gc_viscosity.csv and gc_surface_tension.csv")


# ============================================================
# PART 2: Chemicals TSV compounds - expanded evaluation
# ============================================================
print("\n" + "=" * 60)
print("PART 2: Expanding chemicals TSV compounds...")

# Collect unique CAS from viscosity files
visc_files = [
    "Table 2-313 Viscosity of Inorganic and Organic Liquids.tsv",
    "VDI PPDS Dynamic viscosity of saturated liquids polynomials.tsv",
    "VDI PPDS Dynamic viscosity of gases polynomials.tsv",
    "Table 2-312 Vapor Viscosity of Inorganic and Organic Substances.tsv",
    "Viswanath Natarajan Dynamic 2 term.tsv",
    "Viswanath Natarajan Dynamic 2 term Exponential.tsv",
    "Viswanath Natarajan Dynamic 3 term.tsv",
]

st_files = [
    "MuleroCachadinaParameters.tsv",
    "VDI PPDS surface tensions.tsv",
    "Jasper-Lange.tsv",
    "Somayajulu.tsv",
    "SomayajuluRevised.tsv",
]

tsv_dir = os.path.join(BASE, "data/thermo/chemicals_tsv")

def get_unique_cas(file_list):
    cas_set = set()
    for fname in file_list:
        path = os.path.join(tsv_dir, fname)
        try:
            d = pd.read_csv(path, sep='\t', usecols=['CAS'])
            cas_set.update(d['CAS'].dropna().unique())
        except Exception as e:
            print(f"  Warning reading {fname}: {e}")
    return sorted(cas_set)

visc_cas_list = get_unique_cas(visc_files)
st_cas_list = get_unique_cas(st_files)
print(f"Unique CAS in viscosity files: {len(visc_cas_list)}")
print(f"Unique CAS in surface tension files: {len(st_cas_list)}")

# Generate 10 temperatures spanning typical liquid range
temps_10 = np.linspace(250, 500, 10).tolist()

# Process viscosity compounds
print("\nProcessing viscosity compounds...")
visc_exp_rows = []
visc_success = 0
visc_fail = 0

for i, cas in enumerate(visc_cas_list):
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(visc_cas_list)}")
    try:
        c = Chemical(cas)
        smi = c.smiles if hasattr(c, 'smiles') and c.smiles else ''
        for T in temps_10:
            try:
                c2 = Chemical(cas, T=T)
                mu = c2.mu  # dynamic viscosity in Pa*s
                if mu is not None and mu > 0:
                    visc_exp_rows.append({
                        'CAS': cas, 'SMILES': smi, 'T_K': round(T, 2),
                        'viscosity_Pas': mu
                    })
            except:
                pass
        visc_success += 1
    except:
        visc_fail += 1

print(f"  Viscosity: {visc_success} succeeded, {visc_fail} failed")
print(f"  Total viscosity rows: {len(visc_exp_rows)}")

df_visc_exp = pd.DataFrame(visc_exp_rows)
df_visc_exp.to_csv(os.path.join(OUT, "chemicals_viscosity_expanded.csv"), index=False)
print("  Saved chemicals_viscosity_expanded.csv")

# Process surface tension compounds
print("\nProcessing surface tension compounds...")
st_exp_rows = []
st_success = 0
st_fail = 0

for i, cas in enumerate(st_cas_list):
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(st_cas_list)}")
    try:
        c = Chemical(cas)
        smi = c.smiles if hasattr(c, 'smiles') and c.smiles else ''
        for T in temps_10:
            try:
                c2 = Chemical(cas, T=T)
                sigma = c2.sigma  # surface tension in N/m
                if sigma is not None and sigma > 0:
                    st_exp_rows.append({
                        'CAS': cas, 'SMILES': smi, 'T_K': round(T, 2),
                        'surface_tension_Nm': sigma
                    })
            except:
                pass
        st_success += 1
    except:
        st_fail += 1

print(f"  Surface tension: {st_success} succeeded, {st_fail} failed")
print(f"  Total surface tension rows: {len(st_exp_rows)}")

df_st_exp = pd.DataFrame(st_exp_rows)
df_st_exp.to_csv(os.path.join(OUT, "chemicals_surface_tension_expanded.csv"), index=False)
print("  Saved chemicals_surface_tension_expanded.csv")

# ============================================================
# Summary
# ============================================================
summary = f"""Group Contribution Property Estimation Summary
{'=' * 50}
Date: 2026-04-06

PART 1: NMRexp Compounds (sample of {len(sample_smiles)})
  Database lookups succeeded: {count_db}
  Joback estimates succeeded: {count_joback}
  Failures (unparseable/missing data): {count_fail}
  Total compounds with properties: {count_db + count_joback}

  Viscosity (Letsou-Stiel) rows: {len(visc_rows)}
  Surface tension (Brock-Bird) rows: {len(st_rows)}
  Temperatures evaluated: {temperatures}

  Output files:
    gc_viscosity.csv
    gc_surface_tension.csv

PART 2: Chemicals TSV Expanded
  Viscosity compounds (unique CAS): {len(visc_cas_list)}
    Succeeded: {visc_success}
    Failed: {visc_fail}
    Total rows: {len(visc_exp_rows)}

  Surface tension compounds (unique CAS): {len(st_cas_list)}
    Succeeded: {st_success}
    Failed: {st_fail}
    Total rows: {len(st_exp_rows)}

  Temperatures evaluated: {[round(t,1) for t in temps_10]}

  Output files:
    chemicals_viscosity_expanded.csv
    chemicals_surface_tension_expanded.csv
"""

with open(os.path.join(OUT, "gc_summary.txt"), 'w') as f:
    f.write(summary)

print("\n" + summary)
print("All done!")
