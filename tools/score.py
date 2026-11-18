"""python tools/score.py ref.jsonl cand.jsonl"""
import json
import sys


def load(path):
    rows, errors = {}, 0
    for line in open(path):
        d = json.loads(line)
        if "error" in d:
            errors += 1
        else:
            rows[(d["video"], d["second"])] = (set(d["objects"]), d["people"])
    return rows, errors


def main(ref_path, cand_path):
    ref, ref_err = load(ref_path)
    cand, cand_err = load(cand_path)
    tp = fp = fn = exact = within1 = 0
    for k, (ro, rp) in ref.items():
        co, cp = cand.get(k, (set(), 0))  # missing second = empty prediction
        tp += len(ro & co)
        fp += len(co - ro)
        fn += len(ro - co)
        exact += rp == cp
        within1 += abs(rp - cp) <= 1
    n = max(len(ref), 1)
    f1 = 1.0 if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
    ex, w1 = 100 * exact / n, 100 * within1 / n
    print(f"seconds in ref: {len(ref)}  missing in cand: {len(set(ref) - set(cand))}  "
          f"extra in cand: {len(set(cand) - set(ref))}")
    print(f"error lines: ref {ref_err}  cand {cand_err}")
    print(f"object F1 {f1:.4f} (>=0.95) tp={tp} fp={fp} fn={fn}")
    print(f"people exact {ex:.2f}% (>=85)  within1 {w1:.2f}% (>=95)")
    ok = f1 >= 0.95 and ex >= 85 and w1 >= 95
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(*sys.argv[1:3]) else 1)
