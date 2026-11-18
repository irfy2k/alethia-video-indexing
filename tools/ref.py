"""Slow, obviously-correct reference: every frame through yolo11l.
python tools/ref.py --input DIR --output ref.jsonl [--limit-videos N] [--videos ids.txt]"""
import argparse
import json
import time

from common import CLASS_IDS, iter_videos, video_lines
from decode import DecodeError, decode


def infer(model, frames):
    rs = model.predict(frames, imgsz=640, conf=0.4, iou=0.7, classes=list(CLASS_IDS),
                       verbose=False)
    return [r.boxes.cls.int().tolist() for r in rs]


def process(model, mp4, batch_size):
    """Returns (results, fail_sec, error). Frames in seconds >= fail_sec are unreliable."""
    results, batch, last_sec, err = [], [], 0, None

    def flush():
        ids = infer(model, [b for _, b in batch])
        results.extend((t, i) for (t, _), i in zip(batch, ids))
        batch.clear()

    try:
        for t, img in decode(mp4):
            last_sec = int(t)
            batch.append((t, img))
            if len(batch) == batch_size:
                first_sec = int(batch[0][0])
                try:
                    flush()
                except Exception:
                    last_sec = first_sec  # inference failed: this batch's first second is incomplete
                    raise
        if batch:
            flush()
    except Exception as e:
        err = str(e)[:200] or type(e).__name__
        if batch and isinstance(e, DecodeError):  # decode died; still infer what was decoded
            try:
                flush()
            except Exception:
                pass
    return results, last_sec, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit-videos", type=int)
    ap.add_argument("--videos", help="text file of video ids to process")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO("yolo11l.pt")
    only = set(open(a.videos).read().split()) if a.videos else None
    n, frames, t0 = 0, 0, time.time()
    with open(a.output, "w") as out:
        for vid, mp4 in iter_videos(a.input, only):
            if a.limit_videos and n >= a.limit_videos:
                break
            results, fail_sec, err = process(model, mp4, a.batch)
            frames += len(results)
            if err:  # keep only seconds strictly before the one that failed
                results = [r for r in results if int(r[0]) < fail_sec]
            for line in video_lines(vid, results):
                out.write(json.dumps(line) + "\n")
            if err:
                out.write(json.dumps({"video": vid, "error": err}) + "\n")
            out.flush()
            n += 1
            print(f"{n} videos, {frames / (time.time() - t0):.1f} frames/s", flush=True)


if __name__ == "__main__":
    main()
