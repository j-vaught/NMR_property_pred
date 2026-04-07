"""Task 3: Compile NIST WebBook density data from individual CSVs into one master CSV."""
import os
import csv
import re

NIST_DIR = "/Users/user/Downloads/untitled folder/data/thermo/nist_webbook/"
OUT_FILE = "/Users/user/Downloads/untitled folder/data/thermo/density/nist_webbook_density.csv"

rows = []
files_with_density = 0

for fname in sorted(os.listdir(NIST_DIR)):
    if not fname.endswith('.csv'):
        continue
    cas_raw = fname.replace('.csv', '')
    # Convert C7732185 -> 7732-18-5 format
    digits = cas_raw.lstrip('C')
    # CAS format: last digit is check, preceding 2 are middle, rest are first group
    if len(digits) >= 3:
        cas = digits[:-3] + '-' + digits[-3:-1] + '-' + digits[-1]
    else:
        cas = digits

    fpath = os.path.join(NIST_DIR, fname)
    with open(fpath, 'r') as f:
        reader = csv.DictReader(f)
        if 'Density_kgm3' not in (reader.fieldnames or []):
            continue
        files_with_density += 1
        for row in reader:
            density_val = row.get('Density_kgm3', '').strip()
            temp_val = row.get('Temperature_K', '').strip()
            if density_val and temp_val:
                try:
                    rows.append({
                        'CAS': cas,
                        'T_K': float(temp_val),
                        'density_kgm3': float(density_val)
                    })
                except ValueError:
                    pass

with open(OUT_FILE, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['CAS', 'T_K', 'density_kgm3'])
    writer.writeheader()
    writer.writerows(rows)

print(f"NIST WebBook: {files_with_density} compounds, {len(rows)} data points -> {OUT_FILE}")
