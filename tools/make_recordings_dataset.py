"""Turn our human/AI recordings into a Kaggle dataset folder with vs_labels.csv.

Expects the unzipped folders human_8khz/ and ai_8khz/ under kaggle_upload/raw/.
Language comes from the file name (eng / hindi / hing, ENG / HIN / HNG).
Each human file is a different person; each AI file is treated as its own voice.
About 30% of each (real/AI, language) group goes to test, the rest to train.

Run:  python tools/make_recordings_dataset.py
Then upload kaggle_upload/vs_recordings.zip to Kaggle as a new dataset.
"""

from __future__ import annotations

import csv
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "kaggle_upload" / "raw"
OUT = ROOT / "kaggle_upload" / "vs_recordings"
TEST_EVERY = 3            # every 3rd clip of each group -> test (~33%)


def language(name: str) -> str:
    n = name.lower()
    if re.search(r"hing|hng", n):
        return "hinglish"
    if re.search(r"hin", n):
        return "hi"
    if re.search(r"eng", n):
        return "en"
    raise ValueError(f"Can't tell the language of {name}")


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    rows = []
    for folder, label in [("human_8khz", "real"), ("ai_8khz", "fake")]:
        for i, f in enumerate(sorted((RAW / folder).glob("*.wav"))):
            lang = language(f.stem)
            num = re.search(r"\d+", f.stem).group()
            person = f"{'human' if label == 'real' else 'ai'}_{f.stem[:3].lower() if label == 'fake' else 'p'}{int(num):03d}"
            dst = Path(label) / f"{person}_{lang}.wav"
            (OUT / dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, OUT / dst)
            rows.append({"file": dst.as_posix(), "label": label, "language": lang, "speaker": person,
                         "tool": "" if label == "real" else "team_ai", "original": f.name})

    # stratified split: every TEST_EVERY-th clip of each (label, language) group goes to test
    seen: dict[tuple, int] = {}
    for r in rows:
        k = (r["label"], r["language"])
        seen[k] = seen.get(k, 0) + 1
        r["split"] = "test" if seen[k] % TEST_EVERY == 0 else "train"

    with open(OUT / "vs_labels.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "label", "language", "speaker", "split", "tool", "original"])
        w.writeheader()
        w.writerows(rows)

    zpath = OUT.with_suffix(".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(OUT).as_posix())

    from collections import Counter
    c = Counter((r["split"], r["label"], r["language"]) for r in rows)
    for k in sorted(c):
        print(f"{k[0]:5s} {k[1]:4s} {k[2]:8s} {c[k]:3d}")
    print(f"\n{len(rows)} clips -> {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
