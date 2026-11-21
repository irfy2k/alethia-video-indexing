"""Shared helpers for ref.py / run.py. Semantics: docs/PLAN.md 'Fixed semantics'."""
import glob
import os


CLASS_IDS = {0: "person", 2: "car", 16: "dog", 63: "laptop", 67: "cell phone"}


def iter_videos(input_dir, only=None):
    """Yield (video_id, mp4_bytes); id = '<parquet basename>:<row>'.
    Reads one row group at a time, mp4 column only. `only` = optional set of ids to keep.
    Unreadable file / row group / cell -> (id, None) so the caller writes an error line."""
    import pyarrow.parquet as pq
    for path in sorted(glob.glob(os.path.join(input_dir, "*.parquet"))):
        name = os.path.basename(path)
        try:
            pf = pq.ParquetFile(path)
            groups = pf.num_row_groups
            has_mp4 = "mp4" in pf.schema_arrow.names
        except Exception:  # row count unknown: one placeholder row
            if only is None or any(v.startswith(name + ":") for v in only):
                yield f"{name}:0", None
            continue
        start = 0
        for g in range(groups):
            try:
                n = pf.metadata.row_group(g).num_rows
            except Exception:
                yield f"{name}:{start}", None
                break  # later row offsets unknown
            want = [i for i in range(n) if only is None or f"{name}:{start + i}" in only]
            col = None
            if want and has_mp4:
                try:
                    col = pf.read_row_group(g, columns=["mp4"]).column(0)
                except Exception:
                    col = None
            for i in want:
                try:
                    v = col[i].as_py() if col is not None else None
                except Exception:
                    v = None
                yield f"{name}:{start + i}", v
            del col
            start += n


def aggregate(frames):
    """frames: list of per-frame detections for ONE second; each = list of COCO class ids
    (one entry per detection). Returns (objects, people)."""
    n = len(frames)
    if n == 0:
        return [], 0
    objects = [name for cid, name in CLASS_IDS.items()
               if 2 * sum(cid in f for f in frames) >= n]
    counts = sorted(f.count(0) for f in frames)
    # median of ints is an int or x.5; floor of (a+b)/2 rounds the .5 case down
    people = (counts[(n - 1) // 2] + counts[n // 2]) // 2
    return objects, people


def video_lines(video_id, results):
    """results: iterable of (t_rel_seconds, class_id_list), one per frame.
    Returns output dicts for seconds 0..last second inclusive."""
    by_sec = {}
    for t, ids in results:
        by_sec.setdefault(max(0, int(t // 1)), []).append(ids)
    if not by_sec:
        return []
    out = []
    for s in range(max(by_sec) + 1):
        objects, people = aggregate(by_sec.get(s, []))
        out.append({"video": video_id, "second": s, "objects": objects, "people": people})
    return out
