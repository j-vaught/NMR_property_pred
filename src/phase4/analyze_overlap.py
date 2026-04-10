"""
Compute the overlap between all NMR sources and all viscosity labels.

This tells us how much ACTUAL training data we have for supervised
NMR → viscosity prediction.

Sources:
  NMR (with SMILES):
    - NMRexp (Parquet, 1.48M)
    - NMRBank (CSV, 149K)
    - NMRShiftDB2 (SDF, 58K)  [via InChI]
    - SDBS (CSV, 30K)
    - NP-MRD (JSON bundles, 281K)
    - HMDB metabolites XML (217K)
    - BMRB relational table Chem_comp_SMILES.csv (3,315)
    - 2DNMRGym (HSQC, 22K)
    - IR-NMR multimodal (1,255)
    - logd_predictor (754)

  Viscosity (with SMILES):
    - ThermoML pure_viscosity.csv
    - J Cheminf 2024 viscosity dataset (957 compounds)
    - chemicals_viscosity_expanded.csv (group contribution, if exists)

Output: counts and overlap of canonical SMILES.
"""

import csv
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.data_utils import canonical_smiles, inchi_to_canonical_smiles
from shared.config import Paths

paths = Paths()

DATA_NMR = paths.data / "nmr"


def canonicalize(smi):
    """Canonicalize one SMILES, return None on failure."""
    if not smi or not isinstance(smi, str):
        return None
    try:
        return canonical_smiles(smi)
    except Exception:
        return None


def load_nmr_smiles() -> dict[str, set]:
    """Return dict of source_name -> set of canonical SMILES."""
    sources: dict[str, set] = {}

    # 1. NMRexp
    print("Loading NMRexp...")
    nmr = pd.read_parquet(
        DATA_NMR / "nmrexp" / "NMRexp_1H_13C.parquet",
        columns=["SMILES", "NMR_type"],
    )
    nmr = nmr[nmr["NMR_type"] == "1H NMR"]
    raw_smiles = nmr["SMILES"].dropna().unique()
    del nmr
    print(f"  {len(raw_smiles)} raw SMILES → canonicalizing...")
    nmrexp_canon = set()
    for i, smi in enumerate(raw_smiles):
        c = canonicalize(smi)
        if c:
            nmrexp_canon.add(c)
        if (i + 1) % 200000 == 0:
            print(f"    {i+1}/{len(raw_smiles)}")
    sources["NMRexp"] = nmrexp_canon
    print(f"  → {len(nmrexp_canon)} canonical")

    # 2. NMRBank
    print("Loading NMRBank...")
    nb_path = DATA_NMR / "nmrbank" / "NMRBank_full" / "NMRBank_data_with_unique_SMILES_149135_in_156621.csv"
    if nb_path.exists():
        nb = pd.read_csv(nb_path, usecols=["SMILES", "1H NMR chemical shifts"])
        nb = nb[nb["1H NMR chemical shifts"].notna()]
        nb_canon = set()
        for smi in nb["SMILES"].dropna().unique():
            c = canonicalize(smi)
            if c:
                nb_canon.add(c)
        sources["NMRBank"] = nb_canon
        del nb
        print(f"  → {len(nb_canon)} canonical")

    # 3. NMRShiftDB2 (SD file)
    print("Loading NMRShiftDB2...")
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")

        sd_path = DATA_NMR / "nmrshiftdb2" / "nmrshiftdb2withsignals.sd"
        nmrshiftdb_canon = set()
        if sd_path.exists():
            suppl = Chem.SDMolSupplier(str(sd_path))
            for mol in suppl:
                if mol is None:
                    continue
                props = mol.GetPropsAsDict()
                has_1h = any("1H" in k or "Spectrum 1H" in k for k in props)
                if has_1h:
                    smi = Chem.MolToSmiles(mol)
                    if smi:
                        nmrshiftdb_canon.add(smi)
        sources["NMRShiftDB2"] = nmrshiftdb_canon
        print(f"  → {len(nmrshiftdb_canon)} canonical")
    except Exception as e:
        print(f"  ERROR: {e}")

    # 4. SDBS (from scraped CSVs)
    print("Loading SDBS...")
    sdbs_canon = set()
    for csv_file in (DATA_NMR / "sdbs").glob("sdbs_fast_*.csv"):
        try:
            df = pd.read_csv(csv_file)
            if "inchi" in df.columns:
                df = df[df["inchi"].notna()]
                for inchi in df["inchi"]:
                    try:
                        c = inchi_to_canonical_smiles(inchi)
                        if c:
                            sdbs_canon.add(c)
                    except Exception:
                        pass
        except Exception:
            pass
    # Also the retry/part files
    for csv_file in list((DATA_NMR / "sdbs").glob("sdbs_part*.csv")) + list((DATA_NMR / "sdbs").glob("sdbs_retry*.csv")):
        try:
            df = pd.read_csv(csv_file)
            if "inchi" in df.columns:
                df = df[df["inchi"].notna()]
                for inchi in df["inchi"]:
                    try:
                        c = inchi_to_canonical_smiles(inchi)
                        if c:
                            sdbs_canon.add(c)
                    except Exception:
                        pass
        except Exception:
            pass
    sources["SDBS"] = sdbs_canon
    print(f"  → {len(sdbs_canon)} canonical")

    # 5. BMRB relational table (SMILES column direct)
    print("Loading BMRB relational tables...")
    bmrb_path = DATA_NMR / "bmrb_gissmo" / "relational_tables" / "Chem_comp_SMILES.csv"
    if bmrb_path.exists():
        bmrb = pd.read_csv(bmrb_path)
        bmrb_canon = set()
        # Column name may vary
        smi_cols = [c for c in bmrb.columns if "SMILES" in c or "smiles" in c]
        if smi_cols:
            for smi in bmrb[smi_cols[0]].dropna():
                c = canonicalize(smi)
                if c:
                    bmrb_canon.add(c)
        sources["BMRB"] = bmrb_canon
        print(f"  → {len(bmrb_canon)} canonical")

    # 6. NP-MRD (JSON bundles inside zips)
    print("Loading NP-MRD (sampling from JSON zips)...")
    np_mrd_canon = set()
    json_zips = list((DATA_NMR / "np_mrd").glob("npmrd_natural_products_*_json.zip"))
    for zf_path in json_zips:
        try:
            with zipfile.ZipFile(zf_path) as zf:
                for name in zf.namelist():
                    if name.endswith(".json"):
                        try:
                            with zf.open(name) as f:
                                data = json.loads(f.read())
                            # Try to find SMILES
                            smi = data.get("smiles") or data.get("canonical_smiles") or data.get("SMILES")
                            if not smi and "structures" in data:
                                smi = data["structures"].get("smiles")
                            if smi:
                                c = canonicalize(smi)
                                if c:
                                    np_mrd_canon.add(c)
                        except Exception:
                            continue
        except Exception as e:
            print(f"  {zf_path.name}: error {e}")
    sources["NP-MRD"] = np_mrd_canon
    print(f"  → {len(np_mrd_canon)} canonical")

    # 7. HMDB metabolites XML
    print("Loading HMDB metabolites XML...")
    hmdb_canon = set()
    hmdb_zip = DATA_NMR / "hmdb" / "hmdb_metabolites.zip"
    if hmdb_zip.exists():
        try:
            with zipfile.ZipFile(hmdb_zip) as zf:
                import xml.etree.ElementTree as ET
                for name in zf.namelist():
                    if not name.endswith(".xml"):
                        continue
                    try:
                        with zf.open(name) as f:
                            # Stream parse to save memory
                            ns = ""
                            for event, elem in ET.iterparse(f, events=("end",)):
                                tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                                if tag == "smiles" and elem.text:
                                    c = canonicalize(elem.text.strip())
                                    if c:
                                        hmdb_canon.add(c)
                                    elem.clear()
                                elif tag == "metabolite":
                                    elem.clear()
                    except Exception:
                        continue
                    break  # Usually one big XML per zip
        except Exception as e:
            print(f"  error: {e}")
    sources["HMDB"] = hmdb_canon
    print(f"  → {len(hmdb_canon)} canonical")

    # 8. logD predictor
    print("Loading logD predictor...")
    logd_canon = set()
    for csv_file in (DATA_NMR / "logd_predictor").rglob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            smi_cols = [c for c in df.columns if c.upper() == "SMILES"]
            if smi_cols:
                for smi in df[smi_cols[0]].dropna():
                    c = canonicalize(smi)
                    if c:
                        logd_canon.add(c)
        except Exception:
            pass
    sources["logD"] = logd_canon
    print(f"  → {len(logd_canon)} canonical")

    # 9. IR-NMR multimodal
    print("Loading IR-NMR multimodal...")
    ir_nmr_path = DATA_NMR / "ir_nmr_multimodal" / "NMR_data.parquet"
    if ir_nmr_path.exists():
        ir_nmr = pd.read_parquet(ir_nmr_path, columns=["smiles"])
        ir_nmr_canon = set()
        for smi in ir_nmr["smiles"].dropna().unique():
            c = canonicalize(smi)
            if c:
                ir_nmr_canon.add(c)
        sources["IR-NMR"] = ir_nmr_canon
        del ir_nmr
        print(f"  → {len(ir_nmr_canon)} canonical")

    return sources


def load_viscosity_smiles() -> dict[str, set]:
    """Return dict of source_name -> set of canonical SMILES with viscosity."""
    sources: dict[str, set] = {}

    # 1. ThermoML pure_viscosity
    print("Loading ThermoML viscosity...")
    tml_path = paths.thermoml_viscosity
    if tml_path.exists():
        df = pd.read_csv(tml_path)
        df = df.dropna(subset=["InChI"])
        canon = set()
        for inchi in df["InChI"]:
            try:
                c = inchi_to_canonical_smiles(inchi)
                if c:
                    canon.add(c)
            except Exception:
                pass
        sources["ThermoML"] = canon
        print(f"  → {len(canon)} canonical")

    # 2. J Cheminf 2024 viscosity dataset
    print("Loading J Cheminf 2024 viscosity...")
    jcheminf_path = paths.data / "viscosity_dataset" / "viscosity_smiles_3582.csv"
    if jcheminf_path.exists():
        df = pd.read_csv(jcheminf_path)
        canon = set()
        for smi in df["CANON_SMILES"].dropna().unique():
            c = canonicalize(smi)
            if c:
                canon.add(c)
        sources["J_Cheminf_2024"] = canon
        print(f"  → {len(canon)} canonical")

    # 3. chemicals_viscosity_expanded (group contribution)
    print("Loading chemicals_viscosity_expanded...")
    gc_path = paths.thermo / "group_contribution" / "chemicals_viscosity_expanded.csv"
    if gc_path.exists():
        df = pd.read_csv(gc_path)
        canon = set()
        for smi in df["SMILES"].dropna().unique():
            c = canonicalize(smi)
            if c:
                canon.add(c)
        sources["chemicals_GC"] = canon
        print(f"  → {len(canon)} canonical")

    return sources


def main():
    print("=" * 70)
    print("  Overlap Analysis: NMR sources ∩ Viscosity labels")
    print("=" * 70)
    print()

    print("--- Loading NMR sources ---")
    nmr_sources = load_nmr_smiles()
    print()

    print("--- Loading viscosity sources ---")
    visc_sources = load_viscosity_smiles()
    print()

    # Union of all NMR
    all_nmr = set()
    for s in nmr_sources.values():
        all_nmr |= s
    print(f"\nUnion of all NMR sources: {len(all_nmr):,} unique canonical SMILES")

    # Union of all viscosity
    all_visc = set()
    for s in visc_sources.values():
        all_visc |= s
    print(f"Union of all viscosity labels: {len(all_visc):,} unique canonical SMILES")

    # The crucial number
    overlap = all_nmr & all_visc
    print(f"\n*** OVERLAP (NMR ∩ Viscosity): {len(overlap):,} unique compounds ***\n")

    # Per-source breakdown
    print("Per-NMR-source overlap with any viscosity source:")
    print(f"  {'Source':<20s} {'NMR size':>10s} {'∩ visc':>10s} {'%':>6s}")
    print(f"  {'-'*50}")
    for name, s in sorted(nmr_sources.items(), key=lambda x: -len(x[1])):
        inter = len(s & all_visc)
        pct = inter / max(len(s), 1) * 100
        print(f"  {name:<20s} {len(s):>10,} {inter:>10,} {pct:>5.1f}%")

    print("\nPer-viscosity-source overlap with any NMR source:")
    print(f"  {'Source':<20s} {'Visc size':>10s} {'∩ NMR':>10s} {'%':>6s}")
    print(f"  {'-'*50}")
    for name, s in sorted(visc_sources.items(), key=lambda x: -len(x[1])):
        inter = len(s & all_nmr)
        pct = inter / max(len(s), 1) * 100
        print(f"  {name:<20s} {len(s):>10,} {inter:>10,} {pct:>5.1f}%")

    # Save the actual overlapping SMILES to a file
    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "nmr_visc_overlap_smiles.txt", "w") as f:
        for s in sorted(overlap):
            f.write(s + "\n")
    print(f"\nOverlapping SMILES written to: {out_dir / 'nmr_visc_overlap_smiles.txt'}")

    # Also save individual set sizes for reference
    with open(out_dir / "source_sizes.json", "w") as f:
        import json
        json.dump({
            "nmr_sources": {k: len(v) for k, v in nmr_sources.items()},
            "visc_sources": {k: len(v) for k, v in visc_sources.items()},
            "union_nmr": len(all_nmr),
            "union_visc": len(all_visc),
            "overlap": len(overlap),
        }, f, indent=2)


if __name__ == "__main__":
    main()
