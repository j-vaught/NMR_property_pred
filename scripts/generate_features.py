"""
Generate molecular fingerprints and descriptors for compounds
with viscosity or surface tension data from the chemicals library TSV files.
"""

import os
import sys
import csv
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from rdkit import RDLogger

# Suppress RDKit warnings for cleaner output
RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore")

# Paths
TSV_DIR = Path("/Users/user/Downloads/untitled folder/data/thermo/chemicals_tsv")
OUT_DIR = Path("/Users/user/Downloads/untitled folder/data/features")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# All TSV files to load (viscosity + surface tension)
TSV_FILES = [
    "Table 2-313 Viscosity of Inorganic and Organic Liquids.tsv",
    "Table 2-312 Vapor Viscosity of Inorganic and Organic Substances.tsv",
    "VDI PPDS Dynamic viscosity of saturated liquids polynomials.tsv",
    "VDI PPDS Dynamic viscosity of gases polynomials.tsv",
    "Viswanath Natarajan Dynamic 2 term.tsv",
    "Viswanath Natarajan Dynamic 2 term Exponential.tsv",
    "Viswanath Natarajan Dynamic 3 term.tsv",
    "Dutt Prasad 3 term.tsv",
    "Jasper-Lange.tsv",
    "MuleroCachadinaParameters.tsv",
    "Somayajulu.tsv",
    "SomayajuluRevised.tsv",
    "VDI PPDS surface tensions.tsv",
    "PolingLJ.tsv",
    "MagalhaesLJ.tsv",
]


def load_all_cas_numbers():
    """Load all unique CAS numbers and associated names from TSV files."""
    cas_name_map = {}  # CAS -> name (first encountered name)

    for fname in TSV_FILES:
        fpath = TSV_DIR / fname
        if not fpath.exists():
            print(f"  WARNING: {fname} not found, skipping.")
            continue

        df = pd.read_csv(fpath, sep="\t", dtype=str)

        # Find the CAS column (always first column, but name varies)
        cas_col = df.columns[0]
        # Find a name column
        name_col = None
        for col in df.columns:
            if col.lower() in ("chemical", "name", "fluid"):
                name_col = col
                break

        for _, row in df.iterrows():
            cas = str(row[cas_col]).strip()
            if cas and cas != "nan":
                if cas not in cas_name_map and name_col is not None:
                    name = str(row[name_col]).strip() if pd.notna(row.get(name_col)) else ""
                    cas_name_map[cas] = name
                elif cas not in cas_name_map:
                    cas_name_map[cas] = ""

        print(f"  Loaded {fname}: {len(df)} rows")

    return cas_name_map


def lookup_smiles(cas_name_map):
    """Look up SMILES for each CAS number using chemicals library."""
    from chemicals.identifiers import search_chemical

    results = []  # list of dicts with CAS, name, SMILES, formula
    failed = []

    total = len(cas_name_map)
    for i, (cas, name) in enumerate(cas_name_map.items()):
        if (i + 1) % 100 == 0:
            print(f"  Looking up SMILES: {i+1}/{total}...")

        try:
            result = search_chemical(cas)
            smiles = result.smiles if result else None
            formula = result.formula if result else None
            chem_name = result.common_name if result and result.common_name else name
        except Exception:
            smiles = None
            formula = None
            chem_name = name

        if smiles:
            results.append({
                "CAS": cas,
                "Name": chem_name if chem_name else name,
                "SMILES": smiles,
                "Formula": formula or "",
            })
        else:
            failed.append(cas)

    print(f"  SMILES lookup: {len(results)} succeeded, {len(failed)} failed out of {total}")
    return results, failed


def validate_smiles(compound_list):
    """Validate SMILES strings with RDKit, return only valid ones."""
    valid = []
    invalid = []
    for comp in compound_list:
        mol = Chem.MolFromSmiles(comp["SMILES"])
        if mol is not None:
            comp["mol"] = mol
            valid.append(comp)
        else:
            invalid.append(comp["CAS"])

    print(f"  RDKit validation: {len(valid)} valid, {len(invalid)} invalid SMILES")
    return valid


def generate_morgan_fp(mol, radius, nbits=2048):
    """Generate Morgan fingerprint as bit vector."""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return list(fp.ToBitString())


def generate_maccs(mol):
    """Generate MACCS keys fingerprint."""
    fp = MACCSkeys.GenMACCSKeys(mol)
    bits = list(fp.ToBitString())
    # MACCS keys are 167 bits (0-166), we take bits 1-166 (166 bits)
    return bits[1:]  # skip bit 0 which is always 0


def generate_rdkit_descriptors(mol):
    """Calculate all RDKit molecular descriptors."""
    desc_values = []
    for name, func in Descriptors.descList:
        try:
            val = func(mol)
            if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                desc_values.append(np.nan)
            else:
                desc_values.append(val)
        except Exception:
            desc_values.append(np.nan)
    return desc_values


def main():
    print("=" * 60)
    print("Molecular Feature Generation Pipeline")
    print("=" * 60)

    # Step 1: Load CAS numbers
    print("\n[1/5] Loading CAS numbers from TSV files...")
    cas_name_map = load_all_cas_numbers()
    print(f"  Total unique CAS numbers: {len(cas_name_map)}")

    # Step 2: Look up SMILES
    print("\n[2/5] Looking up SMILES via chemicals library...")
    compound_list, failed_cas = lookup_smiles(cas_name_map)

    # Step 3: Validate SMILES
    print("\n[3/5] Validating SMILES with RDKit...")
    valid_compounds = validate_smiles(compound_list)

    # Step 4: Save compound index
    print("\n[4/5] Saving compound index...")
    index_df = pd.DataFrame([
        {"CAS": c["CAS"], "Name": c["Name"], "SMILES": c["SMILES"], "Formula": c["Formula"]}
        for c in valid_compounds
    ])
    index_df.to_csv(OUT_DIR / "compound_smiles_index.csv", index=False)
    print(f"  Saved compound_smiles_index.csv ({len(index_df)} compounds)")

    # Step 5: Generate features
    print("\n[5/5] Generating molecular features...")

    n = len(valid_compounds)
    cas_list = [c["CAS"] for c in valid_compounds]
    smiles_list = [c["SMILES"] for c in valid_compounds]
    mols = [c["mol"] for c in valid_compounds]

    # Morgan r=2
    print("  Generating Morgan fingerprints (r=2, 2048 bits)...")
    morgan_r2_data = [generate_morgan_fp(mol, radius=2, nbits=2048) for mol in mols]
    morgan_r2_cols = [f"morgan_r2_bit{i}" for i in range(2048)]
    morgan_r2_df = pd.DataFrame(morgan_r2_data, columns=morgan_r2_cols)
    morgan_r2_df.insert(0, "SMILES", smiles_list)
    morgan_r2_df.insert(0, "CAS", cas_list)
    morgan_r2_df.to_csv(OUT_DIR / "morgan_r2_2048.csv", index=False)
    print(f"  Saved morgan_r2_2048.csv ({n} x {2048} bits)")

    # Morgan r=3
    print("  Generating Morgan fingerprints (r=3, 2048 bits)...")
    morgan_r3_data = [generate_morgan_fp(mol, radius=3, nbits=2048) for mol in mols]
    morgan_r3_cols = [f"morgan_r3_bit{i}" for i in range(2048)]
    morgan_r3_df = pd.DataFrame(morgan_r3_data, columns=morgan_r3_cols)
    morgan_r3_df.insert(0, "SMILES", smiles_list)
    morgan_r3_df.insert(0, "CAS", cas_list)
    morgan_r3_df.to_csv(OUT_DIR / "morgan_r3_2048.csv", index=False)
    print(f"  Saved morgan_r3_2048.csv ({n} x {2048} bits)")

    # MACCS keys
    print("  Generating MACCS keys (166 bits)...")
    maccs_data = [generate_maccs(mol) for mol in mols]
    maccs_cols = [f"maccs_bit{i}" for i in range(1, 167)]
    maccs_df = pd.DataFrame(maccs_data, columns=maccs_cols)
    maccs_df.insert(0, "SMILES", smiles_list)
    maccs_df.insert(0, "CAS", cas_list)
    maccs_df.to_csv(OUT_DIR / "maccs_166.csv", index=False)
    print(f"  Saved maccs_166.csv ({n} x {166} bits)")

    # RDKit descriptors
    desc_names = [name for name, _ in Descriptors.descList]
    print(f"  Generating RDKit descriptors ({len(desc_names)} descriptors)...")
    rdkit_data = []
    for i, mol in enumerate(mols):
        if (i + 1) % 100 == 0:
            print(f"    Processing {i+1}/{n}...")
        rdkit_data.append(generate_rdkit_descriptors(mol))

    rdkit_df = pd.DataFrame(rdkit_data, columns=desc_names)
    rdkit_df.insert(0, "SMILES", smiles_list)
    rdkit_df.insert(0, "CAS", cas_list)
    rdkit_df.to_csv(OUT_DIR / "rdkit_descriptors.csv", index=False)
    print(f"  Saved rdkit_descriptors.csv ({n} x {len(desc_names)} descriptors)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total unique CAS numbers found:     {len(cas_name_map)}")
    print(f"  SMILES resolved:                    {len(compound_list)}")
    print(f"  Valid RDKit molecules:               {n}")
    print(f"  Failed SMILES lookup:                {len(failed_cas)}")
    print(f"  RDKit descriptors computed:           {len(desc_names)}")
    print(f"\nOutput files in {OUT_DIR}/:")
    print(f"  - compound_smiles_index.csv")
    print(f"  - morgan_r2_2048.csv")
    print(f"  - morgan_r3_2048.csv")
    print(f"  - maccs_166.csv")
    print(f"  - rdkit_descriptors.csv")
    print(f"\nNote: mordred descriptors skipped (incompatible with Python 3.14)")


if __name__ == "__main__":
    main()
