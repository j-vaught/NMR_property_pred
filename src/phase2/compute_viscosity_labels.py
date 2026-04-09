"""
Compute viscosity labels for ALL NMRexp compounds from their SMILES.

Uses the thermo/chemicals library to estimate dynamic viscosity (Pa·s)
at multiple temperatures via group contribution methods. This generates
training labels for 1.48M compounds — 2400x more than the 611 with
experimental ThermoML data.

Output: a CSV with columns [SMILES, T_K, viscosity_Pas] saved to
the Phase 2 cache directory.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.config import Paths

paths = Paths()
CACHE_DIR = Path(__file__).resolve().parent / "cache"

# Temperatures to compute viscosity at (K)
TEMPERATURES = [253.15, 263.15, 273.15, 283.15, 293.15, 303.15, 313.15, 333.15, 353.15]


def compute_viscosity_batch(smiles_list: list[str], temperatures: list[float] = None,
                             batch_size: int = 10000) -> pd.DataFrame:
    """Compute viscosity for a list of SMILES at multiple temperatures.

    Parameters
    ----------
    smiles_list : list of SMILES strings
    temperatures : list of temperatures in K
    batch_size : print progress every N compounds

    Returns
    -------
    DataFrame with columns [SMILES, T_K, viscosity_Pas]
    """
    from thermo import Chemical

    if temperatures is None:
        temperatures = TEMPERATURES

    rows = []
    n_ok = 0
    n_fail = 0
    t0 = time.time()

    for i, smi in enumerate(smiles_list):
        try:
            for T in temperatures:
                c = Chemical(smi, T=T)
                mu = c.mu
                if mu is not None and mu > 0 and np.isfinite(mu):
                    rows.append({"SMILES": smi, "T_K": T, "viscosity_Pas": mu})
                    n_ok += 1
                else:
                    n_fail += 1
        except Exception:
            n_fail += len(temperatures)

        if (i + 1) % batch_size == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(smiles_list) - i - 1) / rate / 60
            print(f"  {i+1}/{len(smiles_list)} compounds, "
                  f"{n_ok} viscosity values, {n_fail} failures, "
                  f"{rate:.0f} cpds/sec, ETA {eta:.1f} min",
                  flush=True)

    elapsed = time.time() - t0
    print(f"\nDone: {n_ok} viscosity values from {len(smiles_list)} compounds "
          f"in {elapsed/60:.1f} min ({n_fail} failures)")

    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("  COMPUTE VISCOSITY LABELS FOR NMRexp COMPOUNDS")
    print("=" * 70)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load NMR SMILES (the 611 compounds we already matched)
    # BUT also load ALL unique NMR SMILES for maximum training data
    print("\nLoading NMR SMILES...")

    # First try the full NMRexp parquet for maximum coverage
    nmr = pd.read_parquet(paths.nmrexp, columns=["SMILES", "NMR_type"])
    nmr = nmr[nmr["NMR_type"] == "1H NMR"]
    all_smiles = nmr["SMILES"].unique().tolist()
    del nmr
    print(f"  {len(all_smiles)} unique SMILES with 1H NMR")

    # Compute viscosity at multiple temperatures
    print(f"\nComputing viscosity at {len(TEMPERATURES)} temperatures...")
    print(f"  Temperatures: {TEMPERATURES}")

    df = compute_viscosity_batch(all_smiles, TEMPERATURES, batch_size=50000)

    # Filter reasonable values
    df = df[(df["viscosity_Pas"] > 1e-6) & (df["viscosity_Pas"] < 10)]

    # Save
    out_path = CACHE_DIR / "computed_viscosity_all.csv"
    df.to_csv(out_path, index=False)

    n_compounds = df["SMILES"].nunique()
    print(f"\nSaved to {out_path}")
    print(f"  {len(df)} rows, {n_compounds} compounds")
    print(f"  Viscosity range: {df['viscosity_Pas'].min():.6f} - {df['viscosity_Pas'].max():.4f} Pa·s")

    # Also compute for just the 611 matched compounds (quick sanity check)
    matched_path = CACHE_DIR / "nmr_smiles_list.txt"
    if matched_path.exists():
        with open(matched_path) as f:
            matched_smiles = [line.strip() for line in f]
        matched_in_computed = df[df["SMILES"].isin(set(matched_smiles))]
        print(f"\n  Of 611 matched compounds: {matched_in_computed['SMILES'].nunique()} have computed viscosity")


if __name__ == "__main__":
    main()
