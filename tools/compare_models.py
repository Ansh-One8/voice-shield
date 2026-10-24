"""Compare our detector with public HuggingFace deepfake detectors on the same clips.

Prints, per model: EER, AI clips caught and humans flagged at each model's own
0.5 cut-off, and the score on every unseen AI clip. Saves results/compare.json.
"""

from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vs import audio                       # noqa: E402
from vs.detector import Detector           # noqa: E402
from vs.metrics import compute_eer         # noqa: E402

HF = ["MelodyMachine/Deepfake-audio-detection-V2", "mo-thecreator/Deepfake-audio-detection",
      "garystafford/wav2vec2-deepfake-voice-detector", "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"]


def clips():
    out = [(f, "fake", "unseen") for f in sorted(glob.glob(str(ROOT / "test_set/unseen/*/*.wav")))]
    for r in csv.DictReader(open(ROOT / "kaggle_upload/vs_recordings/vs_labels.csv", encoding="utf-8")):
        if r["label"] == "real":
            out.append((str(ROOT / "kaggle_upload/vs_recordings" / r["file"]), "real", "human"))
    out += [(f, "fake", "tts") for f in sorted(glob.glob(str(ROOT / "test_set/tts/*.mp3")))]
    return out


def fake_index(model) -> int:
    labels = {i: l.lower() for i, l in model.config.id2label.items()}
    for i, l in labels.items():
        if any(k in l for k in ("fake", "spoof", "synthetic", "ai", "deepfake")):
            return i
    return 0


@torch.inference_mode()
def hf_scores(name, items):
    fe = AutoFeatureExtractor.from_pretrained(name)
    m = AutoModelForAudioClassification.from_pretrained(name).eval()
    fi = fake_index(m)
    print(f"{name}: labels {m.config.id2label}, fake index {fi}", flush=True)
    out = []
    for f, _, _ in items:
        x = audio.trim_silence(audio.load_audio(f))
        ps = []
        for w in audio.windows(x, hop=audio.SR * 2):
            inp = fe(w, sampling_rate=16000, return_tensors="pt")
            ps.append(float(torch.softmax(m(**inp).logits, -1)[0, fi]))
        out.append(float(np.mean(ps)))
    return out


def report(name, items, s):
    y = np.array([l == "fake" for _, l, _ in items]); s = np.array(s)
    eer, thr = compute_eer(s[~y], s[y])
    grp = np.array([g for _, _, g in items])
    caught = {g: f"{int((s[grp == g] >= 0.5).sum())}/{int((grp == g).sum())}" for g in ("unseen", "tts")}
    fa = f"{int((s[grp == 'human'] >= 0.5).sum())}/{int((grp == 'human').sum())}"
    print(f"\n## {name}\nEER {eer * 100:.1f}% | AI caught @0.5: unseen {caught['unseen']}, tts {caught['tts']} | humans flagged @0.5: {fa}")
    print("unseen scores:", [round(v, 2) for v in s[grp == "unseen"]])
    return {"eer": eer, "scores": s.tolist()}


def main():
    items = clips()
    print(len(items), "clips", flush=True)
    res = {}
    d = Detector()
    res["ours"] = report("ours (fine-tuned)", items, [d.score_clip(audio.load_audio(f)).p_fake for f, _, _ in items])
    for name in HF:
        try:
            res[name] = report(name, items, hf_scores(name, items))
        except Exception as e:
            print(f"\n## {name}\nfailed: {e}")
    res["_items"] = items
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/compare.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
