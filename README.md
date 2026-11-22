# Alethia: YOLO11l per-second object sampler

`run.py` reads every `.parquet` file in a folder. Each row holds an MP4 video in the `mp4` column. For each video it decodes frames with PyAV, samples K=4 frames per second, letterboxes them to 640 px, and runs YOLO11l (`yolo11l.pt`) on the GPU in FP16 with batched inference. It keeps five COCO classes (person, car, dog, laptop, cell phone), aggregates detections per second, and writes the results as JSONL.

## System packages

OpenCV (pulled in by ultralytics) needs `libGL.so.1`. Minimal Ubuntu images may not have it:

```
sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0
```

If `python -c "import cv2"` already works, you can skip this.

## Install

Inside the Python 3.11 venv you created, from the repo root:

```
pip install -r requirements.txt
```

## Preparation (run once, with internet, untimed)

From the repo root:

```
python -c "from ultralytics import YOLO; YOLO('yolo11l.pt')"
```

This downloads `yolo11l.pt` next to `run.py`.

## Run (works offline)

```
python run.py --input <folder of .parquet files> --output results.jsonl
```

Optional flags:

| Flag | Default | Meaning |
|---|---|---|
| `--input` | required | Folder of `.parquet` files |
| `--output` | required | Output JSONL path |
| `--k` | `3` | Frames sampled per second per video |
| `--workers` | CPU count - 2 (min 1) | Decode worker processes |
| `--batch` | `64` | Frames per GPU batch |
| `--threads` | `1` | Codec threads per worker |
| `--videos` | all videos | Text file of whitespace-separated ids `<parquet basename>:<row>` |
| `--model` | `yolo11l.pt` next to `run.py` | Weights path |

## Repo layout

- `run.py`: entry point
- `tools/`: `common.py` and `decode.py` are shared by `run.py`. `ref.py`, `score.py`, `probe.py`, `subset.py`, and `test_common.py` are dev-only.
- `kaggle/full_run.ipynb`: Kaggle notebook version of the full run
- `docs/`: plan and data notes
- `HARDWARE.md`, `WRITEUP.md`: hardware and write-up
- `results.jsonl.gz`: reference results
- `ai_chats/`: development chat logs

## Correctness

Reference output is produced by `tools/ref.py`, and `tools/score.py` compares a results file against it.
