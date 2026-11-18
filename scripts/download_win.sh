#!/usr/bin/env bash
# Dev helper (Git Bash on Windows): download the 50 FineVideo files to D:\finevideo.
# Files are fetched one at a time (so finished files are usable early) with retries,
# because the connection drops with WinError 10060 now and then.
export HF_TOKEN_PATH='D:\hf_cache\token' HF_HOME='D:\hf_cache' HF_HUB_DOWNLOAD_TIMEOUT=60
for i in $(seq -f "%05g" 0 49); do
  f="train-$i-of-01357.parquet"
  for try in 1 2 3 4 5 6 7 8 9 10; do
    [ -f "/d/finevideo/data/$f" ] && break
    hf download HuggingFaceFV/finevideo --repo-type dataset --include "data/$f" --local-dir 'D:\finevideo' || sleep 10
  done
  echo "$(date +%T) $f $([ -f /d/finevideo/data/$f ] && echo ok || echo FAILED)"
done
ls /d/finevideo/data | wc -l
