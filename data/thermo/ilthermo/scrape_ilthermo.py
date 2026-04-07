"""
Scrape viscosity and surface tension data from NIST ILThermo database.
"""

import requests
import json
import csv
import os
import re
import time
import html
import sys
from collections import defaultdict

BASE_URL = "https://ilthermo.boulder.nist.gov/ILT2"
OUTPUT_DIR = "/Users/user/Downloads/untitled folder/data/thermo/ilthermo"
DELAY = 0.3
MAX_RETRIES = 3

PROPERTIES = {
    "PusA": {"name": "Viscosity", "col_nice": "viscosity_mPas", "subdir": "viscosity"},
    "ETUw": {"name": "Surface tension liquid-gas", "col_nice": "surface_tension_mNm", "subdir": "surface_tension"},
}

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://ilthermo.boulder.nist.gov/",
})


def log(msg):
    print(msg, flush=True)


def clean_html(s):
    if s is None:
        return ""
    s = html.unescape(s)
    s = re.sub(r'<[^>]+>', '', s)
    return s.strip()


def safe_filename(name):
    name = clean_html(name)
    name = re.sub(r'[^\w\s\-\(\)\[\],]', '_', name)
    name = re.sub(r'\s+', '_', name)
    return name[:120]


def fetch_with_retry(url, params=None, timeout=15):
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, params=params, timeout=timeout)
            if r.status_code == 403:
                log(f"  403 Forbidden (Cloudflare?). Waiting 5s...")
                time.sleep(5)
                continue
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            log(f"  Timeout (attempt {attempt+1}/{MAX_RETRIES})")
            time.sleep(2)
        except requests.exceptions.ConnectionError:
            log(f"  Connection error (attempt {attempt+1}/{MAX_RETRIES})")
            time.sleep(5)
        except Exception as e:
            log(f"  Error: {e} (attempt {attempt+1}/{MAX_RETRIES})")
            time.sleep(2)
    return None


def fetch_search_results(prp_key):
    url = f"{BASE_URL}/ilsearch"
    params = {'cmp': '', 'ncmp': '1', 'year': '', 'auth': '', 'keyw': '', 'prp': prp_key}
    return fetch_with_retry(url, params, timeout=120)


def fetch_dataset(setid):
    url = f"{BASE_URL}/ilset"
    return fetch_with_retry(url, {'set': setid}, timeout=15)


def parse_dataset(ds, prp_key):
    rows = []
    dhead = ds.get('dhead', [])
    data = ds.get('data', [])
    components = ds.get('components', [])

    compound_name = ""
    formula = ""
    if components:
        compound_name = clean_html(components[0].get('name', ''))
        formula = clean_html(components[0].get('formula', ''))

    temp_idx = None
    prop_idx = None
    pressure_idx = None

    for i, col in enumerate(dhead):
        col_name = clean_html(col[0]).lower() if col[0] else ""
        if 'temperature' in col_name:
            temp_idx = i
        elif 'viscosity' in col_name:
            prop_idx = i
        elif 'surface tension' in col_name:
            prop_idx = i
        elif 'pressure' in col_name:
            pressure_idx = i

    if temp_idx is None or prop_idx is None:
        return compound_name, formula, rows

    prp_info = PROPERTIES[prp_key]

    for row in data:
        try:
            temp_val = float(row[temp_idx][0])
            prop_val = float(row[prop_idx][0])
            # Convert Pa*s -> mPa*s or N/m -> mN/m
            prop_val_converted = prop_val * 1000.0

            pressure_kPa = None
            if pressure_idx is not None and pressure_idx < len(row):
                try:
                    pressure_kPa = float(row[pressure_idx][0])
                except (ValueError, IndexError):
                    pass

            rows.append({
                'temperature_K': temp_val,
                'pressure_kPa': pressure_kPa,
                prp_info['col_nice']: prop_val_converted,
            })
        except (ValueError, IndexError, TypeError):
            continue

    return compound_name, formula, rows


def main():
    sys.stdout.reconfigure(line_buffering=True)

    os.makedirs(os.path.join(OUTPUT_DIR, "viscosity"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "surface_tension"), exist_ok=True)

    total_datapoints = 0
    total_datasets = 0
    compounds_seen = set()
    compound_data = defaultdict(lambda: {"formula": "", "viscosity_rows": [], "surface_tension_rows": [], "setids": []})

    for prp_key, prp_info in PROPERTIES.items():
        log(f"\n{'='*60}")
        log(f"Searching for: {prp_info['name']} (key={prp_key})")
        log(f"{'='*60}")

        search_data = fetch_search_results(prp_key)
        if not search_data:
            log("Failed to fetch search results!")
            continue

        results = search_data.get('res', [])
        header = search_data.get('header', [])
        count = search_data.get('cnt', 0)
        log(f"Found {count} datasets ({len(results)} returned)")

        setid_idx = header.index('setid') if 'setid' in header else 0
        cmp1_idx = header.index('cmp1') if 'cmp1' in header else 4
        nm1_idx = header.index('nm1') if 'nm1' in header else 8
        np_idx = header.index('np') if 'np' in header else 7

        failed = 0
        succeeded = 0
        for i, row in enumerate(results):
            setid = row[setid_idx]
            compound_name_search = row[nm1_idx] if nm1_idx < len(row) and row[nm1_idx] else "unknown"

            if (i + 1) % 50 == 0:
                log(f"  [{i+1}/{len(results)}] ok={succeeded} fail={failed} | {compound_name_search[:50]}")

            time.sleep(DELAY)

            try:
                ds = fetch_dataset(setid)
                if ds is None:
                    failed += 1
                    continue

                compound_name, formula, data_rows = parse_dataset(ds, prp_key)

                if not compound_name:
                    compound_name = clean_html(compound_name_search)
                if not data_rows:
                    failed += 1
                    continue

                succeeded += 1
                total_datasets += 1
                total_datapoints += len(data_rows)
                compounds_seen.add(compound_name)

                cd = compound_data[compound_name]
                cd['formula'] = formula if formula else cd['formula']
                cd['setids'].append(setid)

                if prp_key == "PusA":
                    cd['viscosity_rows'].extend(data_rows)
                else:
                    cd['surface_tension_rows'].extend(data_rows)

            except Exception as e:
                failed += 1
                if failed <= 10:
                    log(f"  Error on set {setid}: {e}")
                continue

        log(f"  Done: {succeeded} succeeded, {failed} failed out of {len(results)}")

    # Write per-compound CSVs
    log(f"\nWriting CSV files...")

    for compound_name, cd in compound_data.items():
        safe_name = safe_filename(compound_name)

        if cd['viscosity_rows']:
            fpath = os.path.join(OUTPUT_DIR, "viscosity", f"{safe_name}.csv")
            with open(fpath, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=[
                    'compound_name', 'formula', 'temperature_K', 'pressure_kPa', 'viscosity_mPas'
                ])
                w.writeheader()
                for row in cd['viscosity_rows']:
                    w.writerow({
                        'compound_name': compound_name,
                        'formula': cd['formula'],
                        'temperature_K': row['temperature_K'],
                        'pressure_kPa': row.get('pressure_kPa', ''),
                        'viscosity_mPas': row['viscosity_mPas'],
                    })

        if cd['surface_tension_rows']:
            fpath = os.path.join(OUTPUT_DIR, "surface_tension", f"{safe_name}.csv")
            with open(fpath, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=[
                    'compound_name', 'formula', 'temperature_K', 'pressure_kPa', 'surface_tension_mNm'
                ])
                w.writeheader()
                for row in cd['surface_tension_rows']:
                    w.writerow({
                        'compound_name': compound_name,
                        'formula': cd['formula'],
                        'temperature_K': row['temperature_K'],
                        'pressure_kPa': row.get('pressure_kPa', ''),
                        'surface_tension_mNm': row['surface_tension_mNm'],
                    })

    # Write master index
    index_path = os.path.join(OUTPUT_DIR, "master_index.csv")
    with open(index_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'compound_name', 'formula', 'has_viscosity', 'viscosity_points',
            'has_surface_tension', 'surface_tension_points', 'dataset_ids'
        ])
        w.writeheader()
        for compound_name in sorted(compound_data.keys()):
            cd = compound_data[compound_name]
            w.writerow({
                'compound_name': compound_name,
                'formula': cd['formula'],
                'has_viscosity': bool(cd['viscosity_rows']),
                'viscosity_points': len(cd['viscosity_rows']),
                'has_surface_tension': bool(cd['surface_tension_rows']),
                'surface_tension_points': len(cd['surface_tension_rows']),
                'dataset_ids': ';'.join(cd['setids']),
            })

    vis_compounds = sum(1 for cd in compound_data.values() if cd['viscosity_rows'])
    st_compounds = sum(1 for cd in compound_data.values() if cd['surface_tension_rows'])
    vis_points = sum(len(cd['viscosity_rows']) for cd in compound_data.values())
    st_points = sum(len(cd['surface_tension_rows']) for cd in compound_data.values())

    log(f"\n{'='*60}")
    log(f"SCRAPING COMPLETE")
    log(f"{'='*60}")
    log(f"Total unique compounds: {len(compounds_seen)}")
    log(f"  With viscosity data: {vis_compounds} ({vis_points} data points)")
    log(f"  With surface tension data: {st_compounds} ({st_points} data points)")
    log(f"Total datasets fetched: {total_datasets}")
    log(f"Total data points: {total_datapoints}")
    log(f"Master index: {index_path}")
    log(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
