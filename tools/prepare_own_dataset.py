import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
NDJSON = ROOT / "data" / "foochi_labels.ndjson"
OUT = ROOT / "data" / "foochi_yolo"


def fetch(args):
    url, dest = args
    if dest.exists():
        return
    for attempt in range(5):
        try:
            dest.write_bytes(urlopen(url, timeout=120).read())
            return
        except Exception as e:
            print(f"retry {attempt + 1}/5 {dest.name}: {e}")
            time.sleep(2 * (attempt + 1))
    print(f"FAILED: {dest.name}")


def main():
    if not NDJSON.exists():
        sys.exit(f"Missing {NDJSON}")
    lines = [json.loads(l) for l in NDJSON.read_text().splitlines() if l.strip()]
    rows = [r for r in lines[1:] if r.get("type") == "image"]
    if not rows:
        sys.exit("No image rows found - unexpected file format. First line: " + str(lines[0])[:300])

    stats = defaultdict(lambda: {"images": 0, "with_ball": 0, "empty": 0, "multi": 0})
    warnings, jobs = [], []
    for r in rows:
        try:
            name, split, url = r["file"], r["split"], r["url"]
        except KeyError as e:
            sys.exit(f"Missing key {e}; row keys are: {list(r.keys())}")
        boxes = r.get("annotations", {}).get("boxes", [])
        s = stats[split]
        s["images"] += 1
        s["with_ball"] += len(boxes) > 0
        s["empty"] += len(boxes) == 0
        s["multi"] += len(boxes) > 1
        # leakage guard: test frames must come from foochi_test only
        if (split == "test") != name.startswith("foochi_test"):
            warnings.append(f"{split}: {name}")
        img_dir, lbl_dir = OUT / "images" / split, OUT / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        jobs.append((url, img_dir / name))
        (lbl_dir / (Path(name).stem + ".txt")).write_text(
            "".join(f"{int(c)} {x} {y} {w} {h}\n" for c, x, y, w, h in boxes))

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(fetch, jobs))

    (OUT / "data.yaml").write_text(
        f"path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: ball\n")

    print("\n=== SUMMARY ===")
    for split in ("train", "val", "test"):
        print(split, dict(stats[split]))
    if warnings:
        print(f"\nWARNING: {len(warnings)} files look misplaced (test must be foochi_test only):", warnings[:5])
    print(f"\nDone -> {OUT}")


if __name__ == "__main__":
    main()
