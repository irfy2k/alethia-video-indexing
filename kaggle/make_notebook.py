"""Generates kaggle/full_run.ipynb. Run: python kaggle/make_notebook.py"""
import json
import os

CELLS = [
("md", "# Alethia take-home: full 50-file run on Kaggle\n"
       "Settings: Accelerator **GPU T4 x2**, Internet **on**, Secret `HF_TOKEN` attached. "
       "Then *Save Version -> Save & Run All*. Outputs land in `/kaggle/working`."),
("md", "### 1. Hardware info"),
("code", '''import os, subprocess
HW = "/kaggle/working/hardware.txt"
with open(HW, "w") as f:
    for c in ["nvidia-smi", "lscpu", "free -g", "df -h / /tmp /kaggle/working", "python --version"]:
        out = subprocess.run(c, shell=True, capture_output=True, text=True).stdout
        print(f"$ {c}\\n{out}"); f.write(f"$ {c}\\n{out}\\n")'''),
("md", "### 2. Clone repo and install"),
("code", '''REPO = "/kaggle/working/repo"
!rm -rf {REPO} && git clone -q https://github.com/irfy2k/alethia-video-indexing {REPO} && git -C {REPO} log -1 --oneline
!pip install -q -r {REPO}/requirements.txt
import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'''),
("md", "### 3. Preparation command from the README (downloads yolo11l.pt)"),
("code", '''!cd {REPO} && python -c "from ultralytics import YOLO; YOLO('yolo11l.pt')" && ls -la {REPO}/yolo11l.pt'''),
("md", "### 4. Hugging Face token from Kaggle Secrets (never printed)"),
("code", '''from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
print("token loaded:", bool(os.environ["HF_TOKEN"]))'''),
("md", "### 5. Helpers: download a set of files, run run.py on both GPUs (half the files each)"),
("code", '''import glob, shutil, time
from huggingface_hub import hf_hub_download
DATA = "/tmp/finevideo"
FILES = [f"data/train-{i:05d}-of-01357.parquet" for i in range(50)]
NGPU = max(1, torch.cuda.device_count())
WORKERS = max(1, (os.cpu_count() or 2) // NGPU)

def download(files):
    for f in files:
        for attempt in range(5):
            try:
                hf_hub_download("HuggingFaceFV/finevideo", f, repo_type="dataset", local_dir=DATA); break
            except Exception as e:
                print("retry", f, e); time.sleep(10)

def run(files, out_prefix):
    """Run run.py on `files`, split across GPUs. Returns (seconds, output paths)."""
    procs, outs = [], []
    for g in range(NGPU):
        part = files[g::NGPU]
        if not part: continue
        d = f"/tmp/in_gpu{g}"; shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
        for f in part: os.symlink(os.path.join(DATA, f), os.path.join(d, os.path.basename(f)))
        out = f"{out_prefix}_gpu{g}.jsonl"; outs.append(out)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        log = open(f"{out_prefix}_gpu{g}.log", "w")
        procs.append((subprocess.Popen(["python", "run.py", "--input", d, "--output", out,
                                        "--workers", str(WORKERS)], cwd=REPO, env=env,
                                       stdout=log, stderr=subprocess.STDOUT), log))
    t = time.time()
    for p, log in procs: p.wait(); log.close()
    return time.time() - t, outs

def count(paths):
    n = e = 0; vids = set()
    for p in paths:
        for line in open(p):
            r = json.loads(line); n += 1; vids.add(r["video"]); e += "error" in r
    return n, e, len(vids)
import json
print("GPUs:", NGPU, "workers per GPU:", WORKERS, "free /tmp GB:", shutil.disk_usage("/tmp").free / 1e9)'''),
("md", "### 6. Smoke test on 1 file (stops here if anything is wrong)"),
("code", '''download(FILES[:1])
secs, outs = run(FILES[:1], "/kaggle/working/smoke")
n, e, v = count(outs)
print(f"smoke: {secs:.0f}s, {n} lines, {v} videos, {e} error lines")
!tail -3 /kaggle/working/smoke_gpu0.log
assert n > 1000 and v >= 20 and e == 0, "smoke test failed - check smoke_gpu0.log"'''),
("md", "### 7. Full run on all 50 files\n"
       "If disk allows, all 50 are downloaded first and timed as one run; otherwise chunks of 10 "
       "(download untimed, run times summed)."),
("code", '''free = shutil.disk_usage("/tmp").free / 1e9
CHUNK = 50 if free > 40 else 10
print("chunk size", CHUNK)
total, all_outs = 0.0, []
for c in range(0, 50, CHUNK):
    files = FILES[c:c + CHUNK]
    t = time.time(); download(files); print(f"downloaded {len(files)} files in {time.time()-t:.0f}s")
    secs, outs = run(files, f"/kaggle/working/run_{c:02d}")
    total += secs; all_outs += outs
    print(f"files {c}-{c+len(files)-1}: run.py {secs:.0f}s (total {total/60:.1f} min)")
    if CHUNK < 50:
        for f in files: os.remove(os.path.join(DATA, f))
msg = f"FULL RUN: 50 files, run.py wall time {total:.0f}s = {total/3600:.2f} h on {NGPU} GPU(s), chunk={CHUNK}"
print(msg)
open(HW, "a").write(msg + "\\n")'''),
("md", "### 8. Merge, check, gzip"),
("code", '''import gzip
with gzip.open("/kaggle/working/results.jsonl.gz", "wt", compresslevel=9) as gz:
    for p in all_outs: gz.write(open(p).read())
n, e, v = count(all_outs)
print(f"results: {n} lines, {v} videos, {e} error lines")
!cat /kaggle/working/run_*_gpu*.log | grep -E "videos|wall" | tail -10
!ls -la /kaggle/working'''),
("md", "Download **results.jsonl.gz**, **hardware.txt** and the **run_*.log** files from the Output tab."),
]


def cell(kind, src):
    c = {"cell_type": "markdown" if kind == "md" else "code", "metadata": {}, "source": src}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


nb = {"nbformat": 4, "nbformat_minor": 4, "cells": [cell(k, s) for k, s in CELLS],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}}}
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_run.ipynb")
json.dump(nb, open(path, "w"), indent=1)
print("wrote", path)
