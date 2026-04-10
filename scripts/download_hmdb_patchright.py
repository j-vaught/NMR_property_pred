"""
Download HMDB NMR bulk files via patchright (Cloudflare bypass).

Runs on the Mac (has Chrome installed), then scp the results to the server.

Files targeted (from HMDB agent research 2026-04-09):
  1. hmdb_metabolites.zip (909.7 MB) — also on Wayback as backup
  2. spectral_data/peak_lists_txt/hmdb_nmr_peak_lists.zip (0.9 MB)
  3. spectral_data/spectra_xml/hmdb_nmr_spectra.zip (872.8 MB)

Optional (off by default due to size):
  - hmdb_all_spectra.zip (43.8 GB) — all spectra types
  - hmdb_fid_files.zip (1.95 GB) — raw FIDs

The trick: patchright uses real Chrome via channel="chrome" which bypasses
Cloudflare's "Just a moment..." JS challenge. Headed mode works reliably.
"""

import asyncio
import sys
import time
from pathlib import Path

OUTPUT_DIR = Path("/Users/user/Downloads/untitled folder/data/nmr/hmdb_download")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [
    # (relative path on hmdb.ca, local filename, expected size MB)
    ("hmdb_metabolites.zip", "hmdb_metabolites.zip", 910),
    ("spectral_data/peak_lists_txt/hmdb_nmr_peak_lists.zip", "hmdb_nmr_peak_lists.zip", 1),
    ("spectral_data/spectra_xml/hmdb_nmr_spectra.zip", "hmdb_nmr_spectra.zip", 873),
]


async def main():
    from patchright.async_api import async_playwright

    print("=" * 70)
    print("HMDB Patchright Downloader")
    print("=" * 70)
    print(f"Output dir: {OUTPUT_DIR}")

    async with async_playwright() as p:
        print("\nLaunching Chrome (channel=chrome)...")
        browser = await p.chromium.launch(
            headless=False,
            channel="chrome",
        )
        ctx = await browser.new_context(
            accept_downloads=True,
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = await ctx.new_page()

        print("Navigating to https://hmdb.ca/downloads and waiting for Cloudflare...")
        await page.goto("https://hmdb.ca/downloads", wait_until="domcontentloaded", timeout=60000)

        # Wait for Cloudflare challenge to clear (typically ~4 seconds)
        for i in range(20):
            title = await page.title()
            if "Just a moment" not in title and "hmdb" in title.lower():
                print(f"  ✓ Challenge cleared after {i} seconds (title: {title[:60]})")
                break
            print(f"  waiting... title='{title[:60]}'")
            await asyncio.sleep(1)
        else:
            print("  ⚠ Challenge didn't clear, continuing anyway")

        # Small extra wait
        await asyncio.sleep(2)

        # Download each target
        for rel, fname, expected_mb in TARGETS:
            url = f"https://hmdb.ca/system/downloads/current/{rel}"
            dest = OUTPUT_DIR / fname

            # Skip if complete
            if dest.exists() and dest.stat().st_size > expected_mb * 1e6 * 0.9:
                print(f"\n✓ SKIP {fname}: already exists ({dest.stat().st_size/1e6:.0f} MB)")
                continue

            print(f"\nDownloading {fname} from {url}")
            print(f"  expected ~{expected_mb} MB")
            t0 = time.time()

            try:
                r = await ctx.request.get(url, timeout=0)
                if r.status != 200:
                    print(f"  ✗ HTTP {r.status}")
                    continue

                body = await r.body()
                size_mb = len(body) / 1e6
                dest.write_bytes(body)
                elapsed = time.time() - t0
                print(f"  ✓ saved {size_mb:.1f} MB in {elapsed:.1f}s")

            except Exception as e:
                print(f"  ✗ error: {e}")

        print("\n" + "=" * 70)
        print("Done. Closing browser in 3 seconds...")
        await asyncio.sleep(3)
        await browser.close()

    print(f"\nFinal contents of {OUTPUT_DIR}:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        if f.is_file():
            print(f"  {f.name}: {f.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    asyncio.run(main())
