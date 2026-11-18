# Implementation plan

Goal: `python run.py --input <dir of .parquet> --output results.jsonl`, as fast as possible on
RTX 4090 / 32 threads / Ubuntu 22.04 / Python 3.11, while passing the grader thresholds
(object F1 >= 0.95, people exact >= 85%, people within 1 >= 95%) against an every-frame reference.

Dev box: WSL2 Ubuntu 22.04, venv `~/alethia-venv` (Python 3.11), data at `/mnt/d/finevideo/data`.
Repo: `/mnt/c/Programming/Alethia`.

## Fixed semantics (both ref.py and run.py)

- Classes (COCO ids): person 0, car 2, dog 16, laptop 63, cell phone 67. Output names exactly
  `person`, `car`, `dog`, `laptop`, `cell phone`.
- Model: `yolo11l.pt`, imgsz 640, conf 0.4, iou 0.7, class-aware NMS, BGR input.
- Video id: `<parquet basename>:<row index from 0>`.
- Rotation: apply the stream's display-matrix rotation to frames before inference.
- Timestamps: `t = frame.time - t0`, `t0` = time of first decoded frame. `second = floor(t)`.
  Frames with missing pts: fall back to `index / avg_fps`.
- Emit one line for every second `0..floor(t_last)`, including empty ones
  (`"objects": [], "people": 0`).
- Per-second aggregation: object present if detected in >= half of that second's (used) frames;
  people = median of per-frame person counts, rounded down at .5.
- Errors: video that cannot be opened/decoded -> one line
  `{"video": ..., "error": "<short msg>"}`. Failure mid-way -> keep the seconds already
  completed, plus one error line. Never crash the whole run on one video.

## Phase A — accuracy baseline (Sonnet)

1. `tools/ref.py`: plain, obviously-correct reference. PyAV decode every frame, rotate,
   `YOLO("yolo11l.pt").predict(frames, imgsz=640, conf=0.4, iou=0.7, classes=[...])`
   in batches of same-shape frames, aggregate as above. Same output format.
2. `tools/score.py ref.jsonl cand.jsonl`: grader metrics. Object F1 over (video, second, object)
   triples, people exact %, people within-1 %, plus count of missing/extra seconds and error lines.
3. `tools/subset.py`: build a dev subset (~40 videos chosen for diversity: resolutions, fps,
   rotation, long/short, codecs) into a small parquet folder. Probe stats over all 50 files
   (codec, resolution, fps, duration, rotation, row-group layout) → `docs/data_stats.md`.

## Phase B — fast run.py (Sonnet), each step validated with score.py vs ref

Order = expected win size. Keep a flag/constant for each so we can A/B.

1. Frame sampling: K frames per second, spread evenly across the second by timestamp. Only sampled
   frames are converted (`to_ndarray`) and sent to the GPU. Majority/median over sampled frames.
   Measure K in {1,2,3,5}; choose the smallest K with a clear margin over thresholds.
2. Cheap decode: PyAV threaded decode (`thread_type="AUTO"`), convert only sampled frames, let
   swscale resize directly to letterbox size (no full-res BGR array). Try `skip_frame="NONREF"`
   (skips B-frame decode) and keep it only if accuracy holds.
3. Parallelism: N decode worker processes (default ~ cpu_count-2), one video per task, feeding a
   bounded queue; main process batches frames across videos for the GPU and aggregates per video.
   Parquet: read only the needed row (`ParquetFile.read_row_group` / `iter_batches(columns=["mp4"])`),
   never load a whole 0.5 GB file per worker if avoidable.
4. Inference: FP16 TensorRT engine (dynamic batch, built in the README prep step by
   `prepare.py` on the grader's 4090), own GPU preprocessing (uint8 -> half, letterbox pad 114,
   /255), ultralytics `ops.non_max_suppression` on GPU. Fallback to ultralytics PyTorch FP16 if the
   engine is missing or fails to load. Check rect (e.g. 640x384) vs square 640x640 letterbox on
   accuracy; TRT engine shape choice follows from that.
5. Output: write lines as each video finishes; flush; order irrelevant.

## Streamlined scope (decided 2026-11-18, user wants minimum time/compute)

- TensorRT dropped: PyTorch FP16 + frame sampling + parallel decode only. No `tensorrt` in requirements.
- Reference: only the train-00000 videos finished by tonight; no top-up runs.
- One Opus review pass on the final run.py (not per commit).
- Kaggle notebook does a 1-file smoke check and the 50-file run in ONE session (one click for the user).
- Docs drafted by Haiku in parallel with the review.

## Phase C — packaging (Haiku)

`requirements.txt` with exact versions (torch from the cu126 index via `--extra-index-url`; grader
driver only guarantees CUDA 12), `README.md` (install, apt-get if any, prep command, run command),
`HARDWARE.md`, `results.jsonl.gz`, `WRITEUP.md` draft from measured numbers, `ai_chats/` copied from
`~/.claude/projects/` with tokens scrubbed to `[REMOVED]`.

## QA gate (Opus reviewer) before each commit of run.py

- score.py vs ref on the dev subset passes with margin; damaged-file test (truncated mp4,
  garbage bytes, zero-length row) yields error lines, no crash.
- Fresh-venv install from requirements.txt works; run.py works offline after prep.
- Code review: second boundaries, last second included, rotation, t0 handling, error paths.
