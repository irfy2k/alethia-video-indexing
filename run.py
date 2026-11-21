"""Fast pipeline: python run.py --input DIR --output results.jsonl [--k 3] [--workers N] [--batch 64]
[--videos ids.txt]. Semantics: docs/PLAN.md. Decode workers (spawn) sample K frames/second and
letterbox them; the main process batches frames on the GPU (FP16), NMS, aggregates, writes."""
import argparse
import json
import multiprocessing as mp
import os
import queue
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from common import CLASS_IDS, iter_videos, video_lines  # noqa: E402
from decode import decode  # noqa: E402

IMGSZ, STRIDE = 640, 32
LB_AUTO = True  # model.predict() sets rect=True -> min-rect letterbox (640x384 for 640x360). False = 640x640 square.
CHUNK = 16  # frames per worker->main message


def letterbox(img, size=IMGSZ, auto=LB_AUTO):
    """Same maths/rounding as ultralytics LetterBox(size, auto, stride=32, center=True)."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = max(round(w * r), 1), max(round(h * r), 1)
    dw, dh = size - nw, size - nh
    if auto:
        dw, dh = dw % STRIDE, dh % STRIDE
    dw, dh = dw / 2, dh / 2
    if (w, h) != (nw, nh):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return cv2.copyMakeBorder(img, round(dh - 0.1), round(dh + 0.1), round(dw - 0.1),
                              round(dw + 0.1), cv2.BORDER_CONSTANT, value=(114, 114, 114))


def worker(tasks, results, k, threads):
    """No torch / CUDA here. Messages: ("f", vid, [(sec, arr)]), ("d", vid, last_sec, n, err), ("x",)."""
    cv2.setNumThreads(1)
    while True:
        item = tasks.get()
        if item is None:
            break
        vid, mp4 = item
        seen, buf, last, n, err = set(), [], 0, 0, None

        def keep(t):  # first frame of each of the k sub-intervals [s + j/k, s + (j+1)/k)
            s = int(t // 1)
            key = (s, min(int((t - s) * k), k - 1))
            if key in seen:
                return False
            seen.add(key)
            return True

        try:
            for t, img in decode(mp4, keep, threads):
                n += 1
                last = int(t // 1)
                if img is not None:
                    buf.append((last, letterbox(img)))
                    if len(buf) >= CHUNK:
                        results.put(("f", vid, buf))
                        buf = []
        except Exception as e:
            err = str(e)[:200] or type(e).__name__
        if n == 0 and not err:
            err = "no frames decoded"
        if buf:
            results.put(("f", vid, buf))
        results.put(("d", vid, last, n, err))
    results.put(("x",))


def load_model(path):
    import torch
    from ultralytics.nn.autobackend import AutoBackend
    m = AutoBackend(path, device=torch.device("cuda"), fp16=True, fuse=True)
    m.eval()
    return m


def infer(model, arr):
    """arr: stacked uint8 NHWC BGR -> list of class-id lists."""
    import torch
    from ultralytics.utils.nms import non_max_suppression
    with torch.inference_mode():
        x = torch.from_numpy(arr).pin_memory().cuda(non_blocking=True).permute(0, 3, 1, 2).contiguous().flip(1)
        x = x.half().div_(255)  # same as ultralytics preprocess with fp16
        out = non_max_suppression(model(x), 0.4, 0.7, classes=list(CLASS_IDS))
        return [o[:, 5].int().tolist() for o in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 3) - 2))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=1, help="codec threads per worker")
    ap.add_argument("--videos")
    ap.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo11l.pt"))
    a = ap.parse_args()
    only = set(open(a.videos).read().split()) if a.videos else None
    t_start = time.time()

    ctx = mp.get_context("spawn")
    tasks = ctx.Queue(maxsize=a.workers * 2)
    tasks.cancel_join_thread()  # never block exit on the feeder thread
    results = ctx.Queue(maxsize=48)  # <= 48*CHUNK frames in flight (~1 GB)
    procs = [ctx.Process(target=worker, args=(tasks, results, a.k, a.threads), daemon=True)
             for _ in range(a.workers)]
    for p in procs:
        p.start()

    pending, reader_errs = set(), []

    def reader():
        try:
            for vid, mp4 in iter_videos(a.input, only):
                pending.add(vid)
                tasks.put((vid, mp4))
        except Exception as e:
            reader_errs.append(str(e)[:200] or type(e).__name__)
        for _ in procs:
            tasks.put(None)

    threading.Thread(target=reader, daemon=True).start()
    model = load_model(a.model)
    infer(model, np.zeros((1, IMGSZ, IMGSZ, 3), np.uint8))  # warmup

    out = open(a.output, "w")
    states, bufs = {}, {}  # vid -> state; shape -> [(vid, sec, arr)]
    stats = dict(videos=0, decoded=0, inferred=0)

    def emit(obj):
        out.write(json.dumps(obj) + "\n")

    def finalize(vid):
        st = states.pop(vid)
        res = [(s, ids) for s, ids in st["res"] if s < st["fail"]]
        for line in video_lines(vid, res):
            emit(line)
        if st["err"]:
            emit({"video": vid, "error": st["err"]})
        out.flush()
        pending.discard(vid)
        stats["videos"] += 1

    def state(vid):
        return states.setdefault(vid, dict(res=[], recv=0, inf=0, done=False, fail=1 << 60, err=None))

    def check(vid):
        st = states[vid]
        if st["done"] and st["inf"] == st["recv"]:
            finalize(vid)

    todo, done = queue.Queue(maxsize=2), queue.Queue()  # ready batches -> inference thread -> results
    sent = got = 0

    def infer_loop():
        while True:
            job = todo.get()
            if job is None:
                return
            items, arr = job
            try:
                done.put((items, infer(model, arr), None))
            except Exception as e:
                done.put((items, None, e))

    threading.Thread(target=infer_loop, daemon=True).start()

    def handle(items, ids, e):  # main thread only: all state lives here
        if e is None:
            for (vid, sec, _), i in zip(items, ids):
                states[vid]["res"].append((sec, i))
        else:
            for vid, sec, _ in items:
                st = states[vid]
                st["fail"] = min(st["fail"], sec)
                st["err"] = st["err"] or ("inference failed: " + (str(e)[:150] or type(e).__name__))
        stats["inferred"] += len(items)
        counts = {}
        for x in items:
            counts[x[0]] = counts.get(x[0], 0) + 1
        for vid, c in counts.items():
            states[vid]["inf"] += c
            check(vid)

    def drain(block=False):
        nonlocal got
        while got < sent:
            try:
                handle(*done.get(timeout=1) if block else done.get_nowait())
            except queue.Empty:
                if not block:
                    return
                continue
            got += 1

    def flush(force):
        nonlocal sent
        for shape in list(bufs):
            b = bufs[shape]
            while len(b) >= a.batch or (force and b):
                items = b[:a.batch]
                del b[:a.batch]
                todo.put((items, np.stack([x[2] for x in items])))
                sent += 1
                drain()

    exits, last = 0, time.time()
    while exits < len(procs):
        try:
            msg = results.get(timeout=0.05)
        except queue.Empty:
            flush(True)  # GPU would idle otherwise
            drain()
            if time.time() - last > 300:  # hung worker: give up on the rest
                for p in procs:
                    p.terminate()
                break
            if any(p.is_alive() for p in procs):
                continue
            try:  # all workers gone: drain what is left, then stop
                msg = results.get(timeout=1)
            except queue.Empty:
                break
        last = time.time()
        if msg[0] == "x":
            exits += 1
        elif msg[0] == "f":
            _, vid, frames = msg
            state(vid)["recv"] += len(frames)
            for sec, arr in frames:
                bufs.setdefault(arr.shape, []).append((vid, sec, arr))
            flush(False)
            drain()
        else:
            _, vid, last_sec, n, err = msg
            st = state(vid)
            st.update(done=True, err=err)
            if err:
                st["fail"] = min(st["fail"], last_sec)
            stats["decoded"] += n
            check(vid)
    flush(True)
    drain(True)
    todo.put(None)
    for vid in list(states):  # worker died mid-video: last second may be partial
        st = states[vid]
        if st["res"]:
            st["fail"] = min(st["fail"], max(s for s, _ in st["res"]))
        st["done"] = True
        st["err"] = st["err"] or "worker died"
        check(vid)
    for vid in sorted(pending):  # never reported
        emit({"video": vid, "error": "worker died"})
    for e in reader_errs:
        emit({"video": "(input)", "error": "input read failed: " + e})
    out.close()
    for p in procs:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
    dt = time.time() - t_start
    print(f"videos {stats['videos']}  frames decoded {stats['decoded']} inferred {stats['inferred']}  "
          f"wall {dt:.1f}s  {stats['decoded'] / dt:.1f} decoded fps  {stats['inferred'] / dt:.1f} infer fps",
          file=sys.stderr)


if __name__ == "__main__":
    main()
    sys.stderr.flush()
    os._exit(0)  # never hang on CUDA/queue teardown
