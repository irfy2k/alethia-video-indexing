# Write-up

All numbers come from docs/NOTES.md and the Kaggle run.

## 1. Your timings

My first working version was tools/ref.py: every frame, FP32 ultralytics predict. I did not run it on all 50 files. At 57 frames/s on my laptop, the 50 files' 13.5 million frames would take about 66 hours (an extrapolation).

The final version ran all 50 files on Kaggle (2x T4) in 8,920 s (2.48 h). See HARDWARE.md.

The change that saved the most time was frame sampling: K=4 frames per second. On my 14 test videos it decoded 108,538 frames and ran the model on 15,362, about 1 in 7.

## 2. Your pipeline

A reader thread reads the mp4 column of each Parquet file, one row group at a time, and hands each video to spawned worker processes. Each worker decodes with PyAV. Timestamps are taken relative to the first decoded frame. I apply the stream rotation, then keep the first frame in each of K sub-intervals of every second. Each kept frame is letterboxed like ultralytics rect mode (for example 640x384).

The main process groups frames of the same shape into batches. An inference thread runs YOLO11l in FP16, then one batched NMS on person boxes only. For each second, an object counts if it appears in at least half of the sampled frames, and the people count is the median. The main process writes one JSON line per second. A damaged video gives an error line, and the seconds already finished are kept.

## 3. Where the time goes

I measured with per-stage timers in a copy of run.py, nvidia-smi sampling, and stage micro-benchmarks.

The profile, taken before the NMS change, showed the inference thread spending 89 s in the forward pass, 49 s in NMS and 3 s pinning memory. The main thread waited 84 s on a full inference queue. Mean GPU utilisation was 38%. Decode was not the limit: 1,780 frames/s per core, and the forward pass runs at 230 img/s at batch 64.

I replaced ultralytics' per-image NMS loop with one batched NMS on person boxes. The forward pass is now the main cost.

| Version (14 videos, K=4) | Wall |
|---|---|
| Serial main loop | 193 s |
| Inference in its own thread, pinned upload | 158 s |
| Batched person-only NMS | 106 s |

## 4. Checking correctness

I built an every-frame reference (tools/ref.py) and tools/score.py, which computes the grader's metrics: object F1, people exact, people within 1. Running my pipeline with K=1000 reproduces the reference exactly (F1 1.0, exact 100%). That test caught a letterbox bug.

I re-scored every speed change on 14 videos. I checked that the batched NMS gives the same result as the ultralytics NMS on 1,693 frames. I also ran damaged-file tests (truncated mp4, garbage bytes, empty row), which produce error lines and no crash.

## 5. Working with AI

The AI suggested CUDA graphs to speed up the forward pass. The first attempt crashed, because the Detect head caches anchors per input shape and the cache was freed. After a fix, I measured it. The outputs were identical, but the speed was the same: 226 against 224 img/s, since the forward pass is already compute-bound at batch 64. The 14-video run got slower, 113 s against 106 s, because capture costs 1.3 s per input shape. I rejected it and reverted. I did not accept the suggestion on trust; the measurement decided.

## 6. Next steps

First, adaptive sampling: take 2 frames per second and add frames only where they disagree. I estimate 35-40% less GPU work.

Second, a TensorRT FP16 engine built in the prep step on the grader's GPU, with the PyTorch path as a fallback. I expect 1.5-2.5x on inference. I dropped it because of deadline risk: it needs dynamic shapes and a fallback.

Third, cut copies: pass frames between processes through shared memory instead of pickling, and move YUV frames to the GPU and convert there.
