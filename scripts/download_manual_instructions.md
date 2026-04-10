# Manual NMR Dataset Download Instructions

Four Tier-1 datasets cannot be fully automated (bot detection, offline server, or registration required). Download these yourself and drop the files into the indicated directories on **comech-2422**.

## 1. CH-NMR-NP (JEOL Japanese natural products NMR database)

**Why we want it**: ~35,500 natural product NMR entries, heavy coverage of Japanese and pre-2010 literature that are likely NOT in NMRexp (which only scrapes 2010-2024 English-language paper SIs). Potentially the highest-novelty single source.

### Steps

1. Visit **https://ch-nmr-np.jeol.co.jp/en/nmrdb/**
2. Click **Sign Up** (free, one-time)
   - Provide: name, affiliation (University of South Carolina), country, email (jvaught@sc.edu)
   - Confirm email when they send verification
3. After login, look for a **Download** / **Export** option
4. If no bulk download button exists:
   - Email JEOL R&D: **nmrdb@jeol.co.jp** (or the contact on their help page)
   - Subject: "Academic research data access request — bulk download of CH-NMR-NP for ML training"
   - Body: brief 3-4 sentence justification mentioning you're a PhD student working on NMR→property ML at University of South Carolina, the specific project scope (fuel property prediction from 1H NMR), and that you'll cite CH-NMR-NP if used in any publication
5. Place any downloaded files (zip, SDF, or individual compound files) in:
   ```
   ~/NMR_property_pred/data/nmr/ch_nmr_np/
   ```
6. After dropping files in, run:
   ```
   ssh comech-2422 "source ~/NMR_property_pred/.venv/bin/activate && python3 ~/NMR_property_pred/scripts/verify_nmr_datasets.py"
   ```

**Expected size**: ~2 GB if full download, much less if partial/metadata-only

---

## 2. HMDB (Human Metabolome Database)

**Why we want it**: ~1-2K experimental 1H NMR metabolite standards + ~313K predicted spectra. Pristine reference data.

**Why manual**: HMDB's Cloudflare blocks all automated HTTP requests from server IPs (returns 403 Forbidden even with browser User-Agent). Only a real browser works.

### Steps

1. Visit **https://hmdb.ca/downloads** in a regular browser
2. Find and download these zip files (they're listed on the Downloads page):
   - `hmdb_metabolites.zip` (~200 MB) — main metabolite XML with SMILES
   - `hmdb_nmr_peak_lists.zip` (~150 MB) — experimental peak lists
   - `hmdb_experimental_nmr_metabolites.zip` (~500 MB) — full experimental spectra
   - `hmdb_predicted_nmr_spectra.zip` (~3 GB, optional) — 313K predicted spectra
   - `structures.zip` (~50 MB) — SDF structures
3. scp the files to the server:
   ```
   scp hmdb_*.zip structures.zip comech-2422:~/NMR_property_pred/data/nmr/hmdb/
   ```
4. Run verify script

**Expected total size**: ~1-4 GB depending on which subsets you grab

---

## 3. BML-NMR (Birmingham Metabolite Library)

**Why we want it**: 208 metabolite standards × 16 spectra each = 3,328 raw experimental 1H NMR spectra with FIDs. Very high quality.

**Why manual**: `http://www.bml-nmr.org/` is currently unreachable from our server (connection timeout, possibly permanently offline or blocked).

### Steps

1. Try visiting **http://www.bml-nmr.org/** in a regular browser (if offline, skip and move on — this dataset is unavailable)
2. If accessible, look for bulk download or FID archive link
3. Alternative: BML-NMR data is mirrored in the BMRB metabolomics FTP we already download automatically — you may already have these compounds!
4. Place any manual download in `~/NMR_property_pred/data/nmr/bml_nmr/`

**Note**: This is a minor dataset. If you can't get it, it's fine — BMRB covers similar compounds.

---

## 4. DLQMA Hydrocarbon NMR Mixtures (Anal. Chem. 2026)

**Why we want it**: Directly relevant to Phase 3 — it has 1H NMR of hydrocarbon mixtures (close to fuel chemistry). Small dataset but high domain match.

### Steps

1. Access the paper via institutional subscription:
   - **Liu et al. "DLQMA: Deep Learning Framework for Qualitative & Quantitative NMR of Complex Hydrocarbon Mixtures"**
   - **Anal. Chem. 2026**, DOI: `10.1021/acs.analchem.5c04983`
   - Full URL: https://pubs.acs.org/doi/10.1021/acs.analchem.5c04983
   - If University of South Carolina library doesn't have ACS access, try:
     - Google Scholar for a preprint
     - ChemRxiv
     - Direct email to the corresponding author (usually on first page)
2. Open the **Supporting Information** section of the paper
3. Download the supplementary ZIP (usually linked as "Supporting Info" or "SI")
4. Place the ZIP (unzipped if possible) in:
   ```
   ~/NMR_property_pred/data/nmr/dlqma/
   ```
5. Run verify script to confirm it was picked up

**Expected size**: <100 MB

---

## After both manual downloads

1. Re-run verification:
   ```bash
   ssh comech-2422 "source ~/NMR_property_pred/.venv/bin/activate && python3 ~/NMR_property_pred/scripts/verify_nmr_datasets.py"
   ```
2. Check that the inventory markdown now shows both:
   ```bash
   ssh comech-2422 "cat ~/NMR_property_pred/data/nmr/DATASETS_INVENTORY.md"
   ```

---

## Not strictly needed, but optional follow-ups

### ACD/Labs Spectral Database (Commercial)
If we ever want more scale, ACD/Labs has the largest commercial NMR database (~millions of 1H spectra). Contact `sciencesolutions.wiley.com` for academic research licensing terms. Expect significant cost unless they have a research-use waiver.

### SpectraBase (Wiley, Commercial)
Similar to ACD/Labs. Free account gives limited access; bulk requires institutional license.

### Reaxys / SciFinder
Commercial. Most universities have subscriptions already — ask the chemistry librarian at USC if there's bulk NMR data access included.
