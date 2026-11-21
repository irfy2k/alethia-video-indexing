# Hardware for the full run

The full 50-file run (results.jsonl.gz) ran on a Kaggle notebook (kaggle/full_run.ipynb, committed run, repo commit 8e7e3a3). Raw output is in kaggle/output/: hardware.txt holds nvidia-smi, lscpu, free and df, and the two run logs are there too.

| | |
|---|---|
| GPU | 2x NVIDIA Tesla T4, 15 GB each (driver 580.178.04, CUDA 13.0 driver) |
| CPU | Intel Xeon @ 2.00 GHz, 4 vCPUs (2 cores x 2 threads), KVM guest |
| RAM | 31 GB, no swap |
| Disk | Kaggle overlay filesystem; input Parquet in /tmp on that overlay (the device type is not exposed) |
| OS | Kaggle Docker image (Linux x86_64), Python 3.13.16 |
| Software | torch 2.14.1+cu126, ultralytics 8.4.215, pip install -r requirements.txt |

## Total time

**8,920 s = 2.48 h wall** for all 50 files, with both GPUs running at the same time. Each GPU had its own run.py process with 2 decode workers and 25 files (`CUDA_VISIBLE_DEVICES` split):

| GPU | videos | frames decoded | frames inferred | wall |
|---|---|---|---|---|
| 0 | 825 | 6,503,326 | 932,410 | 8,563 s |
| 1 | 825 | 6,478,289 | 912,099 | 8,919 s |

The parquet download (untimed, like the grader's prep) and model download are not included. Output: 461,757 lines, 1,650 videos, 0 error lines.

On the grader's machine (one RTX 4090, 32 threads) I expect this to be much faster: the 4090 has several times the T4's FP16 throughput, and 32 threads give room for about 30 decode workers instead of 2.
