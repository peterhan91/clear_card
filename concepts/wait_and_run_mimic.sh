#!/bin/bash
# Wait for H5 preprocessing to finish, then launch full MIMIC pipeline.
# Usage: nohup bash concepts/wait_and_run_mimic.sh > concepts/results/mimic_watcher.log 2>&1 &
set -e
cd /cbica/home/hanti/codes/clear_card

H5_PID=1875959

echo "Waiting for H5 preprocessing (PID $H5_PID) to finish..."
while kill -0 $H5_PID 2>/dev/null; do
    sleep 60
done
echo "H5 preprocessing done: $(date)"

# Quick sanity check
for f in data/h5_1024/mimic_train.h5 data/h5_1024/mimic_validate.h5 data/h5_1024/mimic_test.h5; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found. Aborting."
        exit 1
    fi
done
echo "All H5 files present. Launching experiments..."

bash concepts/run_mimic_all.sh

echo ""
echo "=========================================="
echo "MIMIC done. Launching CheXchoNet pipeline..."
echo "=========================================="

bash concepts/run_chexchonet_all.sh

echo ""
echo "=========================================="
echo "ALL PIPELINES COMPLETE: $(date)"
echo "=========================================="
