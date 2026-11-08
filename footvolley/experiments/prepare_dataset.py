import json
import zlib
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent          # footvolley/
NDJSON = ROOT / "data" / "fotevolly.ndjson"
OUT = ROOT / "data" / "fotevolly_yolo"
BLOCK = 50         # frames per block; whole blocks go to val so train/val don't share near-identical frames
VAL_EVERY = 5        # every 5th block -> val (~20%)


def split_for(name):
    m = re.search(r"(\d+)\.\w+$", name)
    if not m:
        return "train"
    group = re.sub(r"[_-]?\d+\.\w+$", "", name)
    block = int(m.group(1)) // BLOCK
    return "val" if zlib.crc32(f"{group}:{block}".encode()) % VAL_EVERY == 0 else "train"


def fetch(args):
    url, dest = args
    if not dest.exists():
        dest.write_bytes(urlopen(url, timeout=60).read())


def main():
    lines = [json.loads(l) for l in NDJSON.read_text().splitlines() if l.strip()]
    meta, images = lines[0], [r for r in lines[1:] if r.get("type") == "image"]

    jobs = []
    counts = {"train": 0, "val": 0}
    for r in images:
        split = split_for(r["file"])
        counts[split] += 1
        img_dir = OUT / "images" / split
        lbl_dir = OUT / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        jobs.append((r["url"], img_dir / r["file"]))
        boxes = r.get("annotations", {}).get("boxes", [])
        (lbl_dir / (Path(r["file"]).stem + ".txt")).write_text(
            "".join(f"{int(c)} {x} {y} {w} {h}\n" for c, x, y, w, h in boxes)
        )

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(fetch, jobs))

    (OUT / "data.yaml").write_text(
        f"path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n"
    )
    print(f"Done: {counts} -> {OUT}")


if __name__ == "__main__":
    main()