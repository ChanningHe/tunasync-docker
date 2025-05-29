#!/bin/bash

LOG_FILE="/log/tunasync.log"

echo "$(date): 开始执行同步" | tee -a "$LOG_FILE"

cd /usr/local/lib/tunasync
python3 /sync.py

echo "$(date): 同步完成" | tee -a "$LOG_FILE"