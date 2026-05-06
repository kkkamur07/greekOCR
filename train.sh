#!/bin/bash
while [ $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) -gt 0 ]; do 
    echo "GPU busy, waiting... $(date)"
    sleep 60
done
echo "GPU free! Starting job at $(date)"
python3 -m ocr.main