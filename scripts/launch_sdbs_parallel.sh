#!/bin/bash
# Launch 30 parallel SDBS scrapers, each covering a segment of 1-35000
# Each writes to its own CSV, merged later

cd ~/NMR_property_pred
source .venv/bin/activate

TOTAL=35000
WORKERS=10
CHUNK=$((TOTAL / WORKERS))

echo "Launching $WORKERS parallel scrapers, $CHUNK compounds each..."

for i in $(seq 0 $((WORKERS - 1))); do
    START=$((i * CHUNK + 1))
    END=$(((i + 1) * CHUNK))
    if [ $i -eq $((WORKERS - 1)) ]; then
        END=$TOTAL
    fi

    echo "  Worker $i: SDBS #$START to #$END"
    nohup python3 scripts/scrape_sdbs.py $START $END > /tmp/sdbs_worker_${i}.log 2>&1 &
done

echo "All $WORKERS workers launched. Check /tmp/sdbs_worker_*.log for progress."
echo "Merge results later: cat data/nmr/sdbs/sdbs_compounds.csv"
