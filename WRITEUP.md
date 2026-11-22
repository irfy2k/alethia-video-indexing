# Write-up

## 1. Your timings

My first working version was tools/ref.py, which runs plain ultralytics predict on every frame in FP32. I never ran it on all 50 files. It managed 57 frames/s on my laptop, and the files hold about 13.5M frames, so that works out to roughly 66 hours (estimated, not measured).

The final version did all 50 files on Kaggle (2x T4) in 8,920 s, about 2.5 hours. Details are in HARDWARE.md.

The biggest win was sampling 4 frames per second instead of using every frame. On my 14 test videos that meant 15,362 frames through the model instead of 108,538.

## 2. Your pipeline

A reader thread goes through each Parquet file one row group at a time and passes the mp4 bytes to a pool of worker processes. Each worker decodes with PyAV and applies the rotation. It keeps the first frame in each quarter-second and letterboxes it the same way ultralytics does (640x360 becomes 640x384).

Back in the main process, frames with the same shape are grouped into batches of 64. A separate thread runs YOLO11l in FP16 on the GPU, then does one batched NMS over the person boxes only. That's all I need, since the other classes only need a yes or no.

For each second, an object counts if it shows up in at least half the sampled frames, and the people count is the median. If a video breaks partway through, I keep the seconds that finished and write an error line.

## 3. Where the time goes

I timed each stage inside a copy of run.py, watched GPU usage with nvidia-smi, and benchmarked each stage on its own.

The first profile surprised me. The GPU was busy only 38% of the time. The inference thread spent 89 s in the forward pass, but also 49 s in NMS, because ultralytics runs NMS in a Python loop, one image at a time. Decoding wasn't the problem: one core decodes about 1,780 frames/s.

I replaced that loop with a single batched NMS on person boxes, and the 14-video run dropped from 158 s to 106 s. Now the forward pass is what's left.

| Version (14 videos, K=4) | Wall time |
|---|---|
| Everything in one loop | 193 s |
| Inference on its own thread, pinned memory | 158 s |
| Batched person-only NMS | 106 s |

## 4. Checking correctness

Before speeding anything up I built a slow reference that runs every frame (tools/ref.py), plus tools/score.py, which computes the grader's three metrics. Running my pipeline on every frame (K=1000) should match the reference exactly. At first it didn't: I was letterboxing to 640x640 instead of 640x384. After fixing that it matched 100%.

After that I re-scored every speed change on 14 videos. I checked that my batched NMS gives the same result as ultralytics' on 1,693 frames, and I fed in broken files (truncated mp4, garbage bytes, an empty row) to make sure they give error lines and don't crash.

## 5. Working with AI

The AI suggested CUDA graphs to speed up the forward pass. The first try crashed with an illegal memory access. It turned out the Detect head caches its anchors per input shape, and that cache was getting freed. Once that was fixed the outputs matched, but it wasn't any faster: 226 vs 224 img/s, because at batch 64 the GPU is already busy doing the actual compute. The full 14-video run got slower (113 s vs 106 s), since capturing a graph costs about 1.3 s for every new input shape. I reverted it and went with the measurement.

## 6. Next steps

1. **Adaptive sampling.** Start with 2 frames per second and only add more where they disagree. I'd guess that cuts GPU work by 35-40%.
2. **TensorRT.** Build a TensorRT FP16 engine during the prep step on the grader's GPU, with PyTorch as a fallback. It's probably 1.5-2.5x faster at inference, but it needs dynamic shapes, so I didn't risk it before the deadline.
3. **Fewer copies.** Pass frames between processes through shared memory instead of pickling them, and do the colour conversion on the GPU.
