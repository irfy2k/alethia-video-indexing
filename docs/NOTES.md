# Measurement log (source for WRITEUP.md)

All on the dev laptop (RTX 3080 Laptop 16 GB @95 W, i9-10980HK 8C/16T, WSL2 Ubuntu 22.04), 14 reference
videos from train-00000 (3,846 s of video, 108,538 frames).

## Reference (tools/ref.py, every frame, FP32 ultralytics predict)
- 57 frames/s alone. Extrapolated to all 50 files (~13.5 M frames): ~66 h. This is the "first working version".

## K sweep (run.py, frames sampled per second), scored vs reference
| K | F1 | people exact | within 1 | wall |
|---|---|---|---|---|
| 1 (3 videos) | 0.892 | 79.0% | 94.6% | - |
| 2 (3 videos) | 0.929 | 85.1% | 97.5% | - |
| 3 | 0.979 | 93.8% | 99.3% | 175 s |
| 4 | 0.982 | 94.3% | 99.5% | 193 s |
| 5 | 0.987 | 95.8% | 99.7% | 233 s |
| 6 | 0.986 | 95.8% | 99.7% | 317 s |
K=3 on the first 3 videos only cleared F1 by 0.004 (crowd scene), so K=4 was chosen.

At K=4 the 14-video run decodes 108,538 frames and runs the model on 15,362 (1 in 7.07).
Batched person-only NMS was checked against ultralytics non_max_suppression on 1,693 real frames: 0 differ.

## Bugs caught by the reference comparison
- 640x640 square letterbox vs the reference's rect letterbox (640x384): people exact 89.8% -> fixed, K=1000
  then matched the reference exactly (F1 1.0, exact 100%).

## Speed steps (14 videos, K=4, wall incl. ~12 s startup)
| version | wall |
|---|---|
| v1 serial main loop | 193 s |
| inference in own thread + pinned upload | 158 s |
| batched person-only NMS instead of ultralytics per-image NMS loop | 106 s |
| + CUDA graphs (tried, reverted) | 113 s |

## Profiling (instrumented run, before the NMS change)
inference thread: forward 89 s, NMS 49 s, pin 3 s; main thread blocked on full inference queue 84 s;
GPU utilisation 38% mean. Stage micro-benchmarks: decode 1,780 frames/s/core; PyAV to_ndarray(bgr24)
2.2-4.3 ms/frame; letterbox 0.28 ms; pickle round trip 0.32 ms/frame; forward FP16 b64 230 img/s;
H2D pinned 10.8 GB/s vs pageable 3.9 GB/s.

## Rejected ideas (and why)
- CUDA graphs: identical outputs but forward is compute-bound at batch 64 (226 vs 224 img/s); capture costs
  1.3 s per input shape. Net slower. Needed a fix for the Detect head's per-shape anchor cache first.
- cudnn.benchmark: no change (230 vs 229 img/s).
- skip_frame=NONREF: halves decoded frames but decode is not the bottleneck.
- TensorRT: likely 1.5-2.5x on inference but the engine must be built on the grader GPU in the prep step,
  needs dynamic shapes + a fallback; dropped for deadline risk.
- Adaptive sampling (2 frames/s, add frames only where they disagree): est. 35-40% less GPU work; next step.
