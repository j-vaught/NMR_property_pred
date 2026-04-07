#!/usr/bin/env python3
"""
Scrape viscosity and surface tension data from NIST WebBook for all available fluids.
Outputs individual CSV files per fluid and a master summary CSV.
"""

import os
import re
import time
import csv
import urllib.request
import urllib.error
from html.parser import HTMLParser

OUTPUT_DIR = "/Users/user/Downloads/untitled folder/data/thermo/nist_webbook"
DELAY = 0.5  # seconds between requests
MIN_FILE_SIZE = 10 * 1024  # 10KB threshold for valid HTML


# ── Step 1: Get list of all fluids from NIST WebBook ──────────────────────────

class FluidDropdownParser(HTMLParser):
    """Parse the fluid selection page to extract fluid IDs and names from <option> tags."""
    def __init__(self):
        super().__init__()
        self.in_select = False
        self.select_name = None
        self.fluids = []  # list of (cas_id, name)
        self.current_value = None
        self.current_text = ""

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "select":
            name = attrs_dict.get("name", "")
            if name == "ID":
                self.in_select = True
                self.select_name = name
        elif tag == "option" and self.in_select:
            self.current_value = attrs_dict.get("value", "")
            self.current_text = ""

    def handle_data(self, data):
        if self.in_select and self.current_value is not None:
            self.current_text += data

    def handle_endtag(self, tag):
        if tag == "option" and self.in_select and self.current_value is not None:
            name = self.current_text.strip()
            cas_id = self.current_value.strip()
            if cas_id and name and cas_id.startswith("C"):
                self.fluids.append((cas_id, name))
            self.current_value = None
            self.current_text = ""
        elif tag == "select" and self.in_select:
            self.in_select = False


def fetch_url(url, retries=3):
    """Fetch URL content with retries."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            print(f"  Attempt {attempt+1} failed for {url}: {e}")
            if attempt < retries - 1:
                time.sleep(2)
    return None


def get_fluid_list():
    """Fetch the NIST WebBook fluid page and extract all fluid CAS IDs and names."""
    url = "https://webbook.nist.gov/chemistry/fluid/"
    html = fetch_url(url)
    if html is None:
        raise RuntimeError("Failed to fetch NIST WebBook fluid list page")

    parser = FluidDropdownParser()
    parser.feed(html)
    print(f"Found {len(parser.fluids)} fluids on NIST WebBook")
    return parser.fluids


# ── Step 2: Download saturation data HTML ─────────────────────────────────────

def build_sat_url(cas_id):
    """Build the URL for saturation-curve data with wide T range."""
    return (
        f"https://webbook.nist.gov/cgi/fluid.cgi?"
        f"Action=Load&ID={cas_id}&Type=SatT&Digits=5"
        f"&TUnit=K&PUnit=MPa&DUnit=kg/m3&HUnit=kJ/mol"
        f"&WUnit=m/s&VisUnit=uPa*s&STUnit=N/m"
        f"&TLow=50&THigh=1500&TInc=1"
    )


def download_fluid_html(cas_id, name):
    """Download saturation data HTML for a fluid, skip if already cached."""
    html_path = os.path.join(OUTPUT_DIR, f"{cas_id}.html")

    # Skip if valid file exists
    if os.path.exists(html_path) and os.path.getsize(html_path) > MIN_FILE_SIZE:
        print(f"  SKIP {cas_id} ({name}) - already have valid HTML")
        return html_path

    url = build_sat_url(cas_id)
    print(f"  Downloading {cas_id} ({name})...")
    html = fetch_url(url)

    if html is None:
        print(f"  FAIL {cas_id} ({name}) - could not download")
        return None

    # Check if we got actual data (not an error page)
    if "<table" not in html.lower():
        print(f"  WARN {cas_id} ({name}) - no table data in response ({len(html)} bytes)")
        # Still save it for inspection but mark as small
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        return None

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = len(html) / 1024
    print(f"  OK   {cas_id} ({name}) - {size_kb:.0f} KB")
    return html_path


# ── Step 3: Parse HTML tables into CSV ────────────────────────────────────────

class TableParser(HTMLParser):
    """Parse NIST HTML tables to extract rows of data."""
    def __init__(self):
        super().__init__()
        self.tables = []       # list of tables, each is list of rows
        self.current_table = None
        self.current_row = None
        self.current_cell = ""
        self.in_cell = False
        self.in_table = False
        self.cell_tag = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            attrs_dict = dict(attrs)
            cls = attrs_dict.get("class", "")
            if "small" in cls:
                self.in_table = True
                self.current_table = []
        elif tag == "tr" and self.in_table:
            self.current_row = []
        elif tag in ("td", "th") and self.current_row is not None:
            self.in_cell = True
            self.current_cell = ""
            self.cell_tag = tag

    def handle_data(self, data):
        if self.in_cell:
            self.current_cell += data

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.in_cell:
            self.current_row.append(self.current_cell.strip())
            self.in_cell = False
        elif tag == "tr" and self.current_row is not None and self.in_table:
            if self.current_row:
                self.current_table.append(self.current_row)
            self.current_row = None
        elif tag == "table" and self.in_table:
            if self.current_table:
                self.tables.append(self.current_table)
            self.current_table = None
            self.in_table = False


def parse_html_to_csv(cas_id, name):
    """Parse an HTML file and extract temperature, viscosity, surface tension, density."""
    html_path = os.path.join(OUTPUT_DIR, f"{cas_id}.html")
    csv_path = os.path.join(OUTPUT_DIR, f"{cas_id}.csv")

    if not os.path.exists(html_path) or os.path.getsize(html_path) < MIN_FILE_SIZE:
        return None

    with open(html_path, "r", encoding="utf-8", errors="replace") as f:
        html = f.read()

    parser = TableParser()
    parser.feed(html)

    if not parser.tables:
        print(f"  WARN {cas_id} ({name}) - no data tables found")
        return None

    # First table = liquid phase, second = vapor phase
    # We want liquid phase data for viscosity, surface tension, density
    liquid_table = parser.tables[0]
    if len(liquid_table) < 2:
        return None

    headers = liquid_table[0]

    # Find column indices
    temp_idx = None
    visc_idx = None
    st_idx = None
    dens_idx = None

    for i, h in enumerate(headers):
        hl = h.lower()
        if "temperature" in hl:
            temp_idx = i
        elif "viscosity" in hl:
            visc_idx = i
        elif "surf" in hl and "tension" in hl:
            st_idx = i
        elif "density" in hl:
            dens_idx = i

    if temp_idx is None:
        print(f"  WARN {cas_id} ({name}) - no Temperature column found")
        return None

    rows_out = []
    for row in liquid_table[1:]:
        if len(row) <= max(x for x in [temp_idx, visc_idx, st_idx, dens_idx] if x is not None):
            continue

        # Skip rows that aren't liquid phase data
        phase_col = None
        for i, h in enumerate(headers):
            if "phase" in h.lower():
                phase_col = i
                break
        if phase_col is not None and phase_col < len(row):
            if "liquid" not in row[phase_col].lower():
                continue

        temp = row[temp_idx] if temp_idx is not None else ""
        visc = row[visc_idx] if visc_idx is not None and visc_idx < len(row) else ""
        st = row[st_idx] if st_idx is not None and st_idx < len(row) else ""
        dens = row[dens_idx] if dens_idx is not None and dens_idx < len(row) else ""

        # Clean values - remove non-numeric except dots and minus signs
        def clean_val(v):
            v = v.strip()
            if v == "" or v == "undefined" or v == "Not Available":
                return ""
            # Remove any whitespace
            v = v.replace(" ", "")
            # Check if it's a valid number
            try:
                float(v)
                return v
            except ValueError:
                return ""

        temp = clean_val(temp)
        visc = clean_val(visc)
        st = clean_val(st)
        dens = clean_val(dens)

        if temp:
            rows_out.append([temp, visc, st, dens])

    if not rows_out:
        print(f"  WARN {cas_id} ({name}) - no valid data rows extracted")
        return None

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Temperature_K", "Viscosity_uPas", "SurfaceTension_Nm", "Density_kgm3"])
        writer.writerows(rows_out)

    has_visc = any(r[1] != "" for r in rows_out)
    has_st = any(r[2] != "" for r in rows_out)
    temps = [float(r[0]) for r in rows_out if r[0]]

    info = {
        "cas": cas_id,
        "name": name,
        "t_min": min(temps) if temps else None,
        "t_max": max(temps) if temps else None,
        "n_points": len(rows_out),
        "has_viscosity": has_visc,
        "has_surface_tension": has_st,
    }

    print(f"  CSV  {cas_id} ({name}): {len(rows_out)} pts, "
          f"T=[{info['t_min']:.0f}, {info['t_max']:.0f}] K, "
          f"visc={'Y' if has_visc else 'N'}, ST={'Y' if has_st else 'N'}")
    return info


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Get fluid list
    print("=" * 70)
    print("Step 1: Fetching fluid list from NIST WebBook")
    print("=" * 70)
    fluids = get_fluid_list()

    if not fluids:
        print("ERROR: No fluids found. Exiting.")
        return

    for cas_id, name in fluids:
        print(f"  {cas_id:20s}  {name}")

    # Step 2: Download HTML for each fluid
    print("\n" + "=" * 70)
    print(f"Step 2: Downloading saturation data for {len(fluids)} fluids")
    print("=" * 70)

    downloaded = 0
    skipped = 0
    failed = 0

    for i, (cas_id, name) in enumerate(fluids):
        result = download_fluid_html(cas_id, name)
        if result:
            html_path = os.path.join(OUTPUT_DIR, f"{cas_id}.html")
            if os.path.getsize(html_path) > MIN_FILE_SIZE:
                downloaded += 1
            else:
                failed += 1
        else:
            failed += 1

        # Delay between new downloads (not skips)
        if not (os.path.exists(os.path.join(OUTPUT_DIR, f"{cas_id}.html"))
                and os.path.getsize(os.path.join(OUTPUT_DIR, f"{cas_id}.html")) > MIN_FILE_SIZE):
            time.sleep(DELAY)
        elif i < len(fluids) - 1:
            # Small delay even for skips to be polite
            pass

    print(f"\nDownload summary: {downloaded} OK, {failed} failed")

    # Step 3: Parse all HTML to CSV
    print("\n" + "=" * 70)
    print("Step 3: Parsing HTML files to CSV")
    print("=" * 70)

    all_info = []
    for cas_id, name in fluids:
        info = parse_html_to_csv(cas_id, name)
        if info:
            all_info.append(info)

    # Step 4: Write master CSV
    print("\n" + "=" * 70)
    print("Step 4: Writing master CSV")
    print("=" * 70)

    master_path = os.path.join(OUTPUT_DIR, "nist_webbook_all.csv")
    with open(master_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["CAS", "Name", "T_min_K", "T_max_K", "n_points",
                         "has_viscosity", "has_surface_tension"])
        for info in sorted(all_info, key=lambda x: x["name"]):
            writer.writerow([
                info["cas"],
                info["name"],
                f"{info['t_min']:.1f}" if info["t_min"] else "",
                f"{info['t_max']:.1f}" if info["t_max"] else "",
                info["n_points"],
                info["has_viscosity"],
                info["has_surface_tension"],
            ])

    print(f"\nMaster CSV written: {master_path}")
    print(f"Total fluids with data: {len(all_info)} / {len(fluids)}")

    # Summary
    n_visc = sum(1 for i in all_info if i["has_viscosity"])
    n_st = sum(1 for i in all_info if i["has_surface_tension"])
    print(f"Fluids with viscosity data: {n_visc}")
    print(f"Fluids with surface tension data: {n_st}")


if __name__ == "__main__":
    main()
