"""python tools/probe.py --input DIR --output docs/data_stats.md  (also writes the .csv next to it)
Per-video container metadata + packet timestamps (demux, no decode); 1 frame decoded for rotation."""
import argparse
import csv
import glob
import io
import os
import statistics
from collections import Counter

import av
import pyarrow.parquet as pq

from decode import get_rotation

FIELDS = ["video", "file", "row", "row_group", "bytes", "codec", "width", "height", "fps",
          "duration", "n_frames", "rotation", "vfr", "start_time", "error"]


def probe_video(mp4):
    c = av.open(io.BytesIO(mp4))
    s = c.streams.video[0]
    pts = sorted(float(p.pts * s.time_base) for p in c.demux(s) if p.pts is not None)
    d = [b - a for a, b in zip(pts, pts[1:])]
    med = statistics.median(d) if d else 0
    vfr = bool(d) and med > 0 and (max(d) > 1.5 * med or min(d) < 0.5 * med)
    c.seek(0)
    rot = 0.0
    for f in c.decode(s):
        rot = get_rotation(s, f)
        break
    c.close()
    return dict(codec=s.codec_context.name, width=s.codec_context.width,
                height=s.codec_context.height, fps=round(float(s.average_rate or 0), 3),
                duration=round(pts[-1] - pts[0] + med, 3) if pts else 0, n_frames=len(pts),
                rotation=round(rot), vfr=int(vfr), start_time=round(pts[0], 3) if pts else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="docs/data_stats.md")
    a = ap.parse_args()
    rows, layout = [], []
    for path in sorted(glob.glob(os.path.join(a.input, "*.parquet"))):
        name = os.path.basename(path)
        pf = pq.ParquetFile(path)
        sizes = [pf.metadata.row_group(g).num_rows for g in range(pf.num_row_groups)]
        layout.append((name, sizes))
        start = 0
        for g, n in enumerate(sizes):
            col = pf.read_row_group(g, columns=["mp4"]).column(0)
            for i in range(n):
                v = col[i].as_py()
                r = dict(video=f"{name}:{start + i}", file=name, row=start + i, row_group=g,
                         bytes=len(v or b""), error="")
                try:
                    r.update(probe_video(v))
                except Exception as e:
                    r["error"] = str(e)[:80]
                rows.append(r)
            start += n
        print(name, len(rows), flush=True)
    csv_path = os.path.splitext(a.output)[0] + ".csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS, restval="")
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if not r["error"]]

    def cnt(key):
        return ", ".join(f"{k}: {v}" for k, v in Counter(key(r) for r in ok).most_common())

    durs = sorted(r["duration"] for r in ok)
    odd = [r for r in ok if r["rotation"] or r["vfr"] or r["codec"] != "h264" or r["duration"] > 600]
    sizes = Counter(n for _, s in layout for n in s)
    L = [f"# Data stats\n\n{len(rows)} videos in {len(layout)} files, {len(rows) - len(ok)} unprobeable, "
         f"total {sum(durs) / 3600:.1f} h\n",
         f"- codecs: {cnt(lambda r: r['codec'])}",
         f"- resolution: {cnt(lambda r: str(r['width']) + 'x' + str(r['height']))}",
         f"- fps: {cnt(lambda r: round(r['fps']))}",
         f"- rotation: {cnt(lambda r: r['rotation'])}",
         f"- VFR suspects: {sum(r['vfr'] for r in ok)}",
         f"- duration s: min {durs[0] if durs else 0}, median {statistics.median(durs) if durs else 0}, "
         f"max {durs[-1] if durs else 0}",
         f"- start_time != 0: {sum(1 for r in ok if r['start_time'])}",
         f"- row-group sizes (rows: number of groups): {dict(sizes)}",
         "", "## Row-group layout", ""]
    L += [f"- {n}: {s}" for n, s in layout]
    L += ["", "## Odd videos (rotation, VFR, non-h264, >600 s)", ""]
    L += [f"- {r['video']}: {r['codec']} {r['width']}x{r['height']} {r['fps']}fps {r['duration']}s "
          f"rot={r['rotation']} vfr={r['vfr']}" for r in odd]
    L += ["", "## Unprobeable", ""] + [f"- {r['video']}: {r['error']}" for r in rows if r["error"]]
    open(a.output, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
