"""Dev-only: time each pipeline stage in isolation to find the bottleneck.
python tools/profile_stages.py"""
import io
import pickle
import time

import av
import numpy as np
import pyarrow.parquet as pq
import torch
import torchvision
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.nms import non_max_suppression

PQ = "/mnt/d/finevideo/dev/train-00000-of-01357.parquet"
B, H, W = 64, 384, 640


def t(f, n=1):
    torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(n): r = f()
    torch.cuda.synchronize(); return (time.perf_counter() - s) / n, r


col = pq.ParquetFile(PQ).read_row_group(0, columns=["mp4"]).column(0)
mp4 = col[2].as_py()

# 1. decode all frames, single thread
def dec():
    c = av.open(io.BytesIO(mp4)); s = c.streams.video[0]; s.thread_type = "NONE"
    n = 0
    for _ in c.decode(s): n += 1
    return n
dt, n = t(dec); print(f"decode: {n / dt:.0f} frames/s per core")

# 2. convert one frame to BGR ndarray (only sampled frames pay this)
c = av.open(io.BytesIO(mp4)); fr = next(c.decode(video=0))
dt, img = t(lambda: fr.to_ndarray(format="bgr24"), 200); print(f"to_ndarray bgr: {dt * 1e3:.2f} ms/frame")

# 3. IPC cost: pickle + unpickle a 16-frame chunk of letterboxed frames
chunk = [np.zeros((H, W, 3), np.uint8) for _ in range(16)]
dt, _ = t(lambda: pickle.loads(pickle.dumps(chunk, protocol=5)), 20)
print(f"pickle round trip: {dt / 16 * 1e3:.2f} ms/frame ({H * W * 3 / 1e6:.2f} MB/frame)")
dt, _ = t(lambda: np.stack(chunk * 4), 20); print(f"np.stack 64: {dt / 64 * 1e3:.3f} ms/frame")

# 4. model forward only, FP16
m = AutoBackend("yolo11l.pt", device=torch.device("cuda"), fp16=True, fuse=True); m.eval()
x = torch.rand(B, 3, H, W, device="cuda").half()
with torch.inference_mode():
    for _ in range(3): m(x)
    dt, y = t(lambda: m(x), 10); print(f"forward fp16 b{B}: {B / dt:.0f} img/s")
    torch.backends.cudnn.benchmark = True
    for _ in range(3): m(x)
    dt, _ = t(lambda: m(x), 10); print(f"forward fp16 b{B} cudnn.benchmark: {B / dt:.0f} img/s")
    # 5. NMS as used now (per-image loop inside ultralytics)
    pred = y[0] if isinstance(y, (list, tuple)) else y
    dt, _ = t(lambda: non_max_suppression(pred, 0.4, 0.7, classes=[0, 2, 16, 63, 67]), 10)
    print(f"ultralytics NMS b{B}: {dt / B * 1e3:.3f} ms/img")
    # 6. H2D copy, pageable vs pinned
    h = torch.from_numpy(np.stack(chunk * 4))
    dt, _ = t(lambda: h.cuda(), 10); print(f"H2D pageable: {h.numel() / dt / 1e9:.1f} GB/s")
    hp = h.pin_memory()
    dt, _ = t(lambda: hp.cuda(non_blocking=True), 10); print(f"H2D pinned: {hp.numel() / dt / 1e9:.1f} GB/s")
