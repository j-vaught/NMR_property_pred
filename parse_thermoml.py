#!/usr/bin/env python3
"""Parse ThermoML archive for viscosity, surface tension, and density data."""

import tarfile
import json
import csv
import os
import sys
import time
from collections import defaultdict

ARCHIVE = "data/thermo/thermoml/ThermoML.v2020-09-30.tgz"
OUTDIR = "data/thermo/thermoml_parsed"

# Property matching keywords
PROP_MATCHERS = {
    "viscosity": lambda name: "viscosity" in name.lower(),
    "surface_tension": lambda name: "surface tension" in name.lower(),
    "density": lambda name: "density" in name.lower() or "specific volume" in name.lower(),
}

# Unit info embedded in ePropName strings (used for conversion notes)
# Viscosity: Pa*s, density: kg/m3, surface tension: N/m are the standard ThermoML units


def get_variable_type(var):
    """Extract variable type info from a Variable dict.
    Returns (type_key, type_value, regnum_org) where type_key is e.g. 'eTemperature'
    """
    # Variable type can be in VariableID or Variable-MethodID
    vid = var.get("VariableID") or var.get("Variable-MethodID") or {}
    vt = vid.get("VariableType", {})
    regnum = None
    # RegNum can be at var level or inside VariableID
    rn = var.get("RegNum") or vid.get("RegNum")
    if rn:
        regnum = rn.get("nOrgNum")

    for k, v in vt.items():
        if k == "tml_elements":
            continue
        return k, v, regnum
    return None, None, regnum


def get_constraint_type(con):
    """Extract constraint type info. Returns (type_key, type_value, regnum_org, value)."""
    cid = con.get("ConstraintID", {})
    ct = cid.get("ConstraintType", {})
    value = con.get("nConstraintValue")
    regnum = None
    rn = con.get("RegNum") or cid.get("RegNum")
    if rn:
        regnum = rn.get("nOrgNum")

    for k, v in ct.items():
        if k == "tml_elements":
            continue
        return k, v, regnum, value
    return None, None, regnum, value


def classify_property(prop):
    """Return property category or None."""
    pmid = prop.get("Property-MethodID", {})
    pg = pmid.get("PropertyGroup", {})
    for gn, gv in pg.items():
        if gn == "tml_elements":
            continue
        pname = gv.get("ePropName", "")
        for cat, matcher in PROP_MATCHERS.items():
            if matcher(pname):
                return cat, pname, prop.get("nPropNumber")
    return None, None, None


def build_compound_map(compounds):
    """Build a map from nOrgNum -> compound info."""
    cmap = {}
    for c in compounds:
        orgnum = c.get("RegNum", {}).get("nOrgNum")
        if orgnum is None:
            continue
        names = c.get("sCommonName", [])
        name = names[0] if names else ""
        cas = c.get("sCASRegistryNumber", "")
        inchi = c.get("sStandardInChI", "")
        inchikey = c.get("sStandardInChIKey", "")
        formula = c.get("sFormulaMolec", "")
        cmap[orgnum] = {
            "name": name,
            "CAS": cas,
            "InChI": inchi,
            "InChIKey": inchikey,
            "formula": formula,
        }
    return cmap


def parse_file(data, results):
    """Parse one JSON file and append extracted data points to results."""
    compounds = data.get("Compound", [])
    if not compounds:
        return
    cmap = build_compound_map(compounds)

    for pmd in data.get("PureOrMixtureData", []):
        # Identify components in this PMD block
        pmd_components = pmd.get("Component", [])
        comp_orgnums = []
        for comp in pmd_components:
            rn = comp.get("RegNum", {}).get("nOrgNum")
            if rn is not None:
                comp_orgnums.append(rn)

        n_components = len(comp_orgnums)
        is_pure = n_components <= 1

        # Identify properties of interest
        prop_info = {}  # nPropNumber -> (category, ePropName)
        for prop in pmd.get("Property", []):
            cat, pname, pnum = classify_property(prop)
            if cat is not None:
                prop_info[pnum] = (cat, pname)

        if not prop_info:
            continue

        # Parse variable definitions
        var_defs = {}  # nVarNumber -> (type_key, type_value, regnum)
        for var in pmd.get("Variable", []):
            vnum = var.get("nVarNumber")
            tk, tv, rn = get_variable_type(var)
            var_defs[vnum] = (tk, tv, rn)

        # Parse constraint values (fixed T, P, or composition)
        constraint_T = None
        constraint_P = None
        constraint_compositions = {}  # regnum -> (comp_type, value)
        for con in pmd.get("Constraint", []):
            tk, tv, rn, val = get_constraint_type(con)
            if tk == "eTemperature" and val is not None:
                constraint_T = val
            elif tk == "ePressure" and val is not None:
                constraint_P = val
            elif tk == "eComponentComposition" and val is not None and rn is not None:
                constraint_compositions[rn] = (tv, val)

        # Process each data point
        for nv in pmd.get("NumValues", []):
            # Extract variable values
            T = constraint_T
            P = constraint_P
            compositions = dict(constraint_compositions)  # copy constraints

            for vv in nv.get("VariableValue", []):
                vnum = vv.get("nVarNumber")
                val = vv.get("nVarValue")
                if vnum not in var_defs:
                    continue
                tk, tv, rn = var_defs[vnum]
                if tk == "eTemperature":
                    T = val
                elif tk == "ePressure":
                    P = val
                elif tk == "eComponentComposition" and rn is not None:
                    compositions[rn] = (tv, val)

            # Extract property values
            for pv in nv.get("PropertyValue", []):
                pnum = pv.get("nPropNumber")
                if pnum not in prop_info:
                    continue
                cat, pname = prop_info[pnum]
                prop_val = pv.get("nPropValue")
                if prop_val is None:
                    continue

                # Handle specific volume -> density conversion
                if "specific volume" in pname.lower():
                    # Specific volume in m3/kg, convert to density kg/m3
                    if prop_val != 0:
                        prop_val = 1.0 / prop_val
                    else:
                        continue

                if is_pure:
                    # Pure compound
                    orgnum = comp_orgnums[0] if comp_orgnums else None
                    info = cmap.get(orgnum, {})
                    row = {
                        "CAS": info.get("CAS", ""),
                        "name": info.get("name", ""),
                        "InChI": info.get("InChI", ""),
                        "T_K": T,
                        "P_kPa": P,
                        "value": prop_val,
                    }
                    results[f"pure_{cat}"].append(row)
                else:
                    # Mixture
                    comp_names = []
                    comp_cas = []
                    comp_inchi = []
                    comp_fractions = []
                    comp_frac_types = []
                    for orgnum in comp_orgnums:
                        info = cmap.get(orgnum, {})
                        comp_names.append(info.get("name", ""))
                        comp_cas.append(info.get("CAS", ""))
                        comp_inchi.append(info.get("InChI", ""))
                        if orgnum in compositions:
                            ftype, fval = compositions[orgnum]
                            comp_fractions.append(str(fval) if fval is not None else "")
                            comp_frac_types.append(ftype)
                        else:
                            comp_fractions.append("")
                            comp_frac_types.append("")

                    row = {
                        "n_components": n_components,
                        "component_names": "; ".join(comp_names),
                        "component_CAS": "; ".join(comp_cas),
                        "component_InChI": "; ".join(comp_inchi),
                        "composition_type": "; ".join(comp_frac_types),
                        "composition_values": "; ".join(comp_fractions),
                        "T_K": T,
                        "P_kPa": P,
                        "value": prop_val,
                    }
                    results[f"mixture_{cat}"].append(row)


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    results = defaultdict(list)

    tf = tarfile.open(ARCHIVE, "r:gz")
    members = tf.getnames()
    total = len(members)
    print(f"Total files in archive: {total}")

    t0 = time.time()
    processed = 0
    errors = 0

    for i, m in enumerate(members):
        if not m.endswith(".json"):
            continue
        try:
            f = tf.extractfile(m)
            if f is None:
                continue
            data = json.load(f)
            parse_file(data, results)
            processed += 1
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"Error in {m}: {e}", file=sys.stderr)

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            print(f"  Processed {i+1}/{total} files ({elapsed:.1f}s) ...", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone. Processed {processed} JSON files in {elapsed:.1f}s. Errors: {errors}")

    # Print counts
    print("\nData point counts:")
    for key in sorted(results.keys()):
        print(f"  {key}: {len(results[key])}")

    # Write pure CSVs
    pure_fields = ["CAS", "name", "InChI", "T_K", "P_kPa"]
    value_col_map = {
        "pure_viscosity": "viscosity_Pas",
        "pure_surface_tension": "surface_tension_Nm",
        "pure_density": "density_kgm3",
    }
    for key in ["pure_viscosity", "pure_surface_tension", "pure_density"]:
        rows = results.get(key, [])
        if not rows:
            continue
        vcol = value_col_map[key]
        outpath = os.path.join(OUTDIR, f"{key}.csv")
        with open(outpath, "w", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=pure_fields + [vcol])
            writer.writeheader()
            for row in rows:
                out = {k: row[k] for k in pure_fields}
                out[vcol] = row["value"]
                writer.writerow(out)
        print(f"Wrote {outpath} ({len(rows)} rows)")

    # Write mixture CSVs
    mix_fields = [
        "n_components", "component_names", "component_CAS", "component_InChI",
        "composition_type", "composition_values", "T_K", "P_kPa",
    ]
    mix_value_col_map = {
        "mixture_viscosity": "viscosity_Pas",
        "mixture_surface_tension": "surface_tension_Nm",
        "mixture_density": "density_kgm3",
    }
    for key in ["mixture_viscosity", "mixture_surface_tension", "mixture_density"]:
        rows = results.get(key, [])
        if not rows:
            continue
        vcol = mix_value_col_map[key]
        outpath = os.path.join(OUTDIR, f"{key}.csv")
        with open(outpath, "w", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=mix_fields + [vcol])
            writer.writeheader()
            for row in rows:
                out = {k: row[k] for k in mix_fields}
                out[vcol] = row["value"]
                writer.writerow(out)
        print(f"Wrote {outpath} ({len(rows)} rows)")

    # Write summary
    summary_path = os.path.join(OUTDIR, "summary.txt")
    with open(summary_path, "w") as fout:
        fout.write("ThermoML Parsing Summary\n")
        fout.write("=" * 40 + "\n")
        fout.write(f"Archive: {ARCHIVE}\n")
        fout.write(f"Total files in archive: {total}\n")
        fout.write(f"JSON files processed: {processed}\n")
        fout.write(f"Parse errors: {errors}\n")
        fout.write(f"Processing time: {elapsed:.1f}s\n\n")
        fout.write("Data point counts:\n")
        for key in sorted(results.keys()):
            fout.write(f"  {key}: {len(results[key])}\n")
        fout.write(f"\nTotal data points: {sum(len(v) for v in results.values())}\n")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
