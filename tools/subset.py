"""python tools/subset.py --stats docs/data_stats.csv --out /mnt/d/finevideo/subset --n 40
Chooses ~n diverse videos and writes <out>/videos.txt (original video ids, one per line).
No parquet copy: ids stay identical because ref.py takes --videos <file> as a filter."""
import argparse
import csv
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(a.stats)) if not r["error"]]
    for r in rows:
        r["dur"] = float(r["duration"])
        w, h = int(r["width"]), int(r["height"])
        m = min(w, h)
        r["bucket"] = (r["codec"], "<480" if m < 480 else "<720" if m < 720 else "<1080" if m < 1080
                       else ">=1080", "portrait" if h > w else "land")
    random.Random(0).shuffle(rows)
    pick = {}
    for r in [r for r in rows if r["rotation"] != "0" or r["vfr"] == "1"][:8]:
        pick[r["video"]] = r
    seen = set()
    for r in rows:  # one per (codec, resolution bucket, orientation)
        if r["bucket"] not in seen:
            seen.add(r["bucket"])
            pick[r["video"]] = r
    by_dur = sorted(rows, key=lambda r: r["dur"])
    for r in by_dur[:3] + by_dur[-3:]:
        pick[r["video"]] = r
    for r in rows:  # fill remaining slots from spread-out durations, then anything
        if len(pick) >= a.n:
            break
        pick.setdefault(r["video"], r)
    ids = sorted(pick)
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "videos.txt"), "w").write("\n".join(ids) + "\n")
    print(len(ids), "videos,", round(sum(pick[i]["dur"] for i in ids) / 60, 1), "min total")


if __name__ == "__main__":
    main()
