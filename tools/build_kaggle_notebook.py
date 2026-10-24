"""Builds the self-contained Kaggle notebooks in training/ (our code is embedded in them).

Run:  python tools/build_kaggle_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CELLS = [
    ("md", """# Voice Shield · 01 · Baseline on In-the-Wild

Scores the public **XLS-R + SLS** checkpoint on the In-the-Wild dataset with **no training**.
The checkpoint's author reports **7.84% EER** on In-the-Wild, so our number should land close to that.
If it does, our model port is correct and we can start fine-tuning.

**Before you press Run All**
1. Settings → Accelerator → **GPU T4 x2** (or P100).
2. Settings → **Internet on** (needs a phone-verified Kaggle account). The checkpoint downloads from HuggingFace.
3. Add Input → search **"in the wild audio deepfake"** and add a copy that contains `meta.csv` and the wav files.

All our code is inside this notebook (the `%%writefile` cells), so nothing else needs uploading.

It scores three ways:
- `paper`: exactly like the original code (first 4 s, silence kept). Should give about 7.84%.
- `trimmed`: silence cut at start and end, like our app does.
- `phone`: trimmed and then made to sound like a phone call (8 kHz, telephone band, G.711 μ-law)."""),
    ("code", """# Settings you can change
SUBSET = None        # e.g. 3000 for a quick test run, None for all ~31.8k clips
BATCH = 32
MODES = ["paper", "trimmed", "phone"]"""),
    ("code", """import sys, glob, os, time
sys.path.insert(0, os.getcwd())     # the vs/ package written by the cells above

meta = glob.glob('/kaggle/input/**/meta.csv', recursive=True)
assert meta, "Add an In-the-Wild dataset that has meta.csv."
META = meta[0]; WAV_DIR = os.path.dirname(META)
print("In-the-Wild meta:", META)"""),
    ("code", """import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from vs import audio
from vs.model import load_sls, download_checkpoint
from vs.metrics import compute_eer

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")
ckpt = download_checkpoint()
model = load_sls(ckpt, device)          # raises if the weight conversion does not match
print("Checkpoint loaded:", ckpt)"""),
    ("code", """df = pd.read_csv(META)
df["is_fake"] = (df["label"].str.lower() == "spoof").astype(int)
if SUBSET:
    df = df.groupby("is_fake", group_keys=False).apply(lambda g: g.sample(min(len(g), SUBSET // 2), random_state=0))
print(df["label"].value_counts())

class Clips(Dataset):
    def __init__(self, df, mode):
        self.df, self.mode = df.reset_index(drop=True), mode
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        x = audio.load_audio(os.path.join(WAV_DIR, self.df.file[i]))
        if self.mode in ("trimmed", "phone"):
            x = audio.trim_silence(x)
        if self.mode == "phone":
            x = audio.phone_simulate(x, rng=np.random.default_rng(i))
        x = audio.repeat_pad(x)
        if self.mode != "paper":
            x = audio.normalize_loudness(x)     # the app sets every window to -23 dBFS
        return torch.from_numpy(x), i"""),
    ("code", """results, all_scores = [], {}
for mode in MODES:
    dl = DataLoader(Clips(df, mode), batch_size=BATCH, num_workers=4)
    p_fake = np.zeros(len(df)); t0 = time.time()
    with torch.inference_mode(), torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
        for x, idx in dl:
            p_real = model(x.to(device)).float()[:, 1].exp()
            p_fake[idx.numpy()] = (1 - p_real).cpu().numpy()
    all_scores[mode] = p_fake
    y = df["is_fake"].values
    eer, thr = compute_eer(p_fake[y == 0], p_fake[y == 1])
    fa = float(np.mean(p_fake[y == 0] > 0.5))
    results.append({"mode": mode, "EER %": round(eer * 100, 2), "threshold": round(thr, 4),
                    "false alarms on real @0.5 %": round(fa * 100, 1), "minutes": round((time.time() - t0) / 60, 1)})
    print(results[-1])

res = pd.DataFrame(results); print(res.to_string(index=False))"""),
    ("code", """out = df[["file", "label"]].copy()
for m, s in all_scores.items():
    out[f"p_fake_{m}"] = np.round(s, 5)
out.to_csv("/kaggle/working/itw_baseline_scores.csv", index=False)
res.to_csv("/kaggle/working/itw_baseline_results.csv", index=False)
print("Saved. Copy the results table above into the chat.")"""),
]


FT_CELLS = [
    ("md", """# Voice Shield · 02 · Fine-tune for Hindi, English, real mics and phone calls

Starts from the public XLS-R + SLS checkpoint and fine-tunes it so it:
- stops calling real human voices "AI" when they come from ordinary mics, phones or quiet rooms
- works on **Hindi** as well as English
- works on **phone-quality audio** (8 kHz, μ-law, phone codecs)
- does not use volume, silence, noise or room echo as a shortcut

It prints a **before vs after** table at the end.

**AI voices used for training**
- English: ASVspoof 2019 (older tools) + **MLAAD**: 130+ recent tools, including ElevenLabs, MiniMax, F5-TTS,
  XTTS-v2, Chatterbox, VibeVoice, Qwen3-TTS, Fish, Kokoro and many more.
- Hindi: IndicSynth (FreeVC, XTTS-v2) + MLAAD Hindi (IndicF5, XTTS-v2, VoxCPM2, Indri and more).

**Newest tools kept out of training**, so the "newest AI tools never trained on" rows show how the model
does on a tool it has never seen: ElevenLabs v3, OpenAI TTS-1 HD, Cartesia Sonic-3, Gemini TTS,
MiniMax 2.8, Resemble.ai, Inworld TTS-2, Chatterbox-Turbo (English) and Veena, Ringg (Hindi).
Older versions from several of the same companies are trained on. Empty the holdout lists for the final model.

**Real voices used for training**: ASVspoof 2019, FLEURS (Hindi + English), LibriSpeech (English audiobooks,
to match MLAAD's audiobook-style fakes), **Svarah (Indian-accented English)** and Kathbath (Hindi).

**Splits**: train / dev / test, never sharing a speaker where the dataset says who is speaking.
Dev picks the best checkpoint. Test is only reported. In-the-Wild and the held-out tools are test only.

**Before you press Run All**
1. Settings → Accelerator → **GPU P100** or **GPU T4 x2** (only one GPU is used).
2. Settings → **Internet on**.
3. Add Input → search **"ASVspoof 2019"** and add one that contains `ASVspoof2019.LA.cm.train.trn.txt`.
4. Add Input → your **In-the-Wild** dataset from notebook 01 (optional, but gives the real-world test row).
5. **Strongly recommended: HuggingFace token.** Without it MLAAD (newest AI voices), Svarah and Kathbath are skipped.
   - log in to huggingface.co and accept the access terms on all three:
     **huggingface.co/datasets/mueller91/MLAAD**, **huggingface.co/datasets/ai4bharat/Svarah**,
     **huggingface.co/datasets/ai4bharat/Kathbath**
   - Settings → Access Tokens → new **Read** token
   - Kaggle: Add-ons → Secrets → add **HF_TOKEN** with that token, and switch it on for this notebook
6. Optional, our own recordings: upload them as a Kaggle dataset with a `vs_labels.csv` file
   (columns `file,label,language,speaker,split,tool`) and add it as an input. People marked `train` are
   trained on; people marked `test` become the "Our recordings" test row.

**Run A** (`RUN = "A"`, about 2–3 hours on a T4): proves generalization, saves `voice_shield_ft.pth`.
**Run B** (`RUN = "B"`, about 1–1.5 hours): attach run A's notebook output as an input, trains on every
tool starting from run A's model, saves `voice_shield_final.pth` (the demo model).
Use Save Version → Save & Run All so it keeps running with the browser closed, then download the `.pth`
file from the version's Output tab."""),
    ("code", """QUICK_TEST = False       # True = tiny dry run (~15 min) to check the setup
RUN = "A"                # "A" = keep the 10 newest tools out (proves generalization), start from the public model
                         # "B" = final model: train on every tool, continue from run A's model
                         #       (attach run A's output voice_shield_ft.pth as an input first)

STEPS, BATCH, ACCUM, EVAL_EVERY = 3000, 8, 2, 500
# Hindi IndicSynth shards are sorted by tool: about 0-54 FreeVC, 56-106 XTTS-v2. Take 6 of each.
INDICSYNTH_SHARDS = [0, 9, 18, 27, 36, 45, 56, 66, 76, 86, 96, 106]
PER_SHARD = 700
HOLDOUT_XTTS = False     # True = never train on XTTS-v2, report it as an unseen tool
FLEURS_TRAIN, FLEURS_EVAL = 2500, 400        # real speech per language
KATHBATH_SHARDS, KB_PER_SHARD = [0, 8, 16, 24], 800
SVARAH_TRAIN, SVARAH_TEST, SVARAH_PER_SHARD = [0, 1], [2], 1500   # Indian-accented English, 3 files in total
ASV_TRAIN_FAKE = 6000                         # all ~2.6k real ASVspoof clips are used
EVAL_PER_CLASS, ITW_PER_CLASS = 400, 1500
LIBRI = {"train.clean.100": [0, 7], "train.other.500": [0, 32], "test.clean": [0]}
LIBRI_PER_SHARD = 700
MLAAD_EN_PER_TOOL, MLAAD_HI_PER_TOOL, MLAAD_HOLDOUT_PER_TOOL = 40, 300, 60   # clips per tool
MLAAD_MAX_EN_TOOLS = None                     # None = every English tool
# Newest tools kept out of training -> "never trained on" test rows. Set to [] for the final model.
MLAAD_HOLDOUT_EN = ["ElevenLabs-v3", "OpenAI TTS-1 HD", "Cartesia.ai (Sonic-3)", "Gemini-3.1-Flash-TTS",
                    "MiniMax-Speech-2.8-Turbo", "Resemble.ai (April 12th, 2025)", "Inworld-TTS-2", "Chatterbox-Turbo"]
MLAAD_HOLDOUT_HI = ["Veena", "Ringg Squirrel TTS v1.0"]
DATA = "/tmp/vs_data"

if RUN == "B":
    MLAAD_HOLDOUT_EN, MLAAD_HOLDOUT_HI = [], []   # nothing held out: the demo model sees every tool
    STEPS, EVAL_EVERY = 1500, 500

if QUICK_TEST:
    STEPS, EVAL_EVERY = 40, 20
    INDICSYNTH_SHARDS, PER_SHARD = [0, 60], 150
    FLEURS_TRAIN, FLEURS_EVAL = 150, 60
    KATHBATH_SHARDS, KB_PER_SHARD = [0], 100
    SVARAH_TRAIN, SVARAH_TEST, SVARAH_PER_SHARD = [0], [2], 60
    ASV_TRAIN_FAKE, EVAL_PER_CLASS, ITW_PER_CLASS = 300, 60, 100
    LIBRI, LIBRI_PER_SHARD = {"train.clean.100": [0], "test.clean": [0]}, 100
    MLAAD_EN_PER_TOOL, MLAAD_HI_PER_TOOL, MLAAD_HOLDOUT_PER_TOOL, MLAAD_MAX_EN_TOOLS = 3, 10, 10, 12"""),
    ("code", """import sys, glob, os, random, shutil
sys.path.insert(0, os.getcwd())     # the vs/ package written by the cells above

import numpy as np, pandas as pd, torch
from vs import train_data as td
from vs.model import load_sls, download_checkpoint
from vs.train import evaluate, finetune

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "(no GPU: turn it on in Settings)")
from vs import audio as _a
print("ffmpeg:", shutil.which("ffmpeg"), "| AMR/GSM phone codecs:",
      "available" if _a._ffmpeg_codec(np.zeros(8000, "float32"), "amr") is not None else
      "not available (phone simulation uses telephone band + G.711 only)")
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    HF_TOKEN = None
print("HuggingFace token:", "found" if HF_TOKEN else
      "NOT SET: MLAAD (newest AI voices), Svarah and Kathbath will be skipped. See step 5 at the top.")
START = None
if RUN == "B":
    START = td.find_file("voice_shield_ft.pth")
    assert START, ("Run B needs run A's model: Add Input → Your Work → Notebooks → pick the run A notebook "
                   "(its output has voice_shield_ft.pth).")
print("Run", RUN, "| starting from:", START or "public XLS-R + SLS checkpoint")"""),
    ("md", """## 1 · Collect data (about 15–30 minutes the first time)"""),
    ("code", """random.seed(0)
rows = []

asv = td.asvspoof2019_la()
asv_train_real = [r for r in asv if r["split"] == "train" and r["label"] == "real"]
asv_train_fake = random.sample([r for r in asv if r["split"] == "train" and r["label"] == "fake"], ASV_TRAIN_FAKE)
rows += asv_train_real + asv_train_fake + [r for r in asv if r["split"] == "dev"]
print("ASVspoof:", len(rows))

for lang in ["hi_in", "en_us"]:
    rows += td.fleurs(lang, "train", DATA, FLEURS_TRAIN)
    rows += td.fleurs(lang, "validation", DATA, FLEURS_EVAL)
    rows += td.fleurs(lang, "test", DATA, FLEURS_EVAL)
print("+ FLEURS:", len(rows))
rows += td.librispeech(LIBRI, DATA, LIBRI_PER_SHARD)
print("+ LibriSpeech:", len(rows))

rows += td.indicsynth_hindi(INDICSYNTH_SHARDS, DATA, PER_SHARD, holdout_xtts=HOLDOUT_XTTS)
if HF_TOKEN:
    try:
        rows += td.kathbath_hindi(KATHBATH_SHARDS, DATA, KB_PER_SHARD, HF_TOKEN)
    except Exception as e:
        print("Kathbath skipped:", e)
    try:
        rows += td.svarah(SVARAH_TRAIN, SVARAH_TEST, DATA, SVARAH_PER_SHARD, HF_TOKEN)
    except Exception as e:
        print("Svarah skipped (did you accept its terms on huggingface.co?):", e)
    for lang, per_tool, holdout, max_tools in [("en", MLAAD_EN_PER_TOOL, MLAAD_HOLDOUT_EN, MLAAD_MAX_EN_TOOLS),
                                               ("hi", MLAAD_HI_PER_TOOL, MLAAD_HOLDOUT_HI, None)]:
        try:
            rows += td.mlaad(lang, DATA, per_tool, HF_TOKEN, holdout, max_tools=max_tools,
                             per_holdout_tool=MLAAD_HOLDOUT_PER_TOOL)
        except Exception as e:
            print(f"MLAAD {lang} skipped (did you accept its terms on huggingface.co?):", e)
else:
    print("No HF_TOKEN: skipping Kathbath, Svarah and MLAAD (newest AI voices).")
itw = td.in_the_wild()
rows += td.extra_labels()
print("Total clips:", len(rows), "| In-the-Wild available:", len(itw))
display(td.summary(rows))
du = shutil.disk_usage("/tmp")
print(f"Disk: {du.used / 1e9:.1f} GB used, {du.free / 1e9:.1f} GB free")
print(pd.Series([r["generator"] for r in rows if r["source"] == "indicsynth"]).value_counts())
for lang in ["en", "hi"]:
    m = [r for r in rows if r["source"] == "mlaad" and r["lang"] == lang]
    print(f"MLAAD {lang}: {len({r['generator'] for r in m})} tools, {len(m)} clips, "
          f"{sum(r['split'] == 'test' for r in m)} from held-out tools")"""),
    ("code", """def pick(rs, n, seed=0):
    rs = list(rs); random.Random(seed).shuffle(rs); return rs[:n]

def by(**kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]

def pick_each(rs, n, key="source"):
    # n clips spread equally over sources (or tools), so one big dataset can't fill a test group
    groups = {}
    for r in rs:
        groups.setdefault(r[key] if key != "tool" else (r["generator"] or r["source"]), []).append(r)
    out = []
    for i, k in enumerate(sorted(groups)):
        out += pick(groups[k], max(1, n // max(len(groups), 1)), seed=i)
    return out

train_rows = [r for r in rows if r["split"] == "train"]

half = EVAL_PER_CLASS // 2
dev_groups = {
    # English fakes: half older tools (ASVspoof), half newer tools (MLAAD) when available
    "English dev": pick_each(by(split="dev", lang="en", label="real"), EVAL_PER_CLASS * 2)
                   + pick(by(split="dev", source="asvspoof19", label="fake"), half)
                   + pick(by(split="dev", source="mlaad", lang="en"), half),
    "Hindi dev":   pick_each(by(split="dev", lang="hi", label="real"), EVAL_PER_CLASS)
                   + pick_each(by(split="dev", lang="hi", label="fake"), EVAL_PER_CLASS, key="tool"),
}
en_test_real = pick_each(by(split="test", lang="en", label="real"), EVAL_PER_CLASS)   # FLEURS + LibriSpeech + Svarah
test_groups = {
    "English real voices (FLEURS + LibriSpeech + Svarah test)": en_test_real,
    "Hindi real voices (FLEURS test)": pick(by(split="test", source="fleurs_hi"), EVAL_PER_CLASS),
}
if by(split="test", source="svarah"):
    test_groups["Indian-accented English real voices (Svarah test)"] = pick(by(split="test", source="svarah"), EVAL_PER_CLASS)
if by(split="test", source="mlaad", lang="en"):
    test_groups["English, newest AI tools never trained on"] = (
        en_test_real + pick_each(by(split="test", source="mlaad", lang="en"), EVAL_PER_CLASS, key="tool"))
if by(split="test", source="mlaad", lang="hi"):
    test_groups["Hindi, newest AI tools never trained on"] = (
        pick(by(split="test", source="fleurs_hi"), EVAL_PER_CLASS) + pick(by(split="test", source="mlaad", lang="hi"), EVAL_PER_CLASS))
if HOLDOUT_XTTS:
    test_groups["Hindi, unseen tool (XTTS-v2)"] = (pick(by(split="test", source="fleurs_hi"), EVAL_PER_CLASS)
                                                  + pick(by(split="test", generator="xtts_v2"), EVAL_PER_CLASS))
if itw:
    test_groups["English real world (In-the-Wild)"] = (pick([r for r in itw if r["label"] == "real"], ITW_PER_CLASS)
                                                      + pick([r for r in itw if r["label"] == "fake"], ITW_PER_CLASS))
ours = [r for r in rows if r["source"] == "ours" and r["split"] == "test"]
if ours:
    test_groups["Our recordings (all, 8 kHz)"] = ours
    test_groups["Our recordings, English"] = [r for r in ours if r["lang"] == "en"]
    test_groups["Our recordings, Hindi + Hinglish"] = [r for r in ours if r["lang"] == "hi"]
for name, g in {**dev_groups, **test_groups}.items():
    print(f"{name:40s} real {sum(r['label']=='real' for r in g):5d}  fake {sum(r['label']=='fake' for r in g):5d}")
print("Training clips:", len(train_rows))

# How training batches will be mixed: 25% per group, and an equal share per AI tool / real dataset inside each group
mix = td.mix_report(train_rows)
print(mix.groupby(["lang", "label"])[["clips", "share %"]].sum())
pd.set_option("display.max_rows", 300)
display(mix)"""),
    ("md", """## 2 · Before: the starting model (ready-made for run A, run A's model for run B)"""),
    ("code", """model = load_sls(START or download_checkpoint(), device)
before = evaluate(model, {**dev_groups, **test_groups}, device)
before.to_csv("/kaggle/working/before.csv", index=False)
before"""),
    ("md", """## 3 · Fine-tune"""),
    ("code", """OUT = "/kaggle/working/voice_shield_ft.pth" if RUN == "A" else "/kaggle/working/voice_shield_final.pth"
history = finetune(model, train_rows, dev_groups, device=device, out_path=OUT,
                   steps=STEPS, batch=BATCH, accum=ACCUM, eval_every=EVAL_EVERY)
pd.DataFrame(history)"""),
    ("md", """## 4 · After: our fine-tuned model, and the comparison"""),
    ("code", """del model; torch.cuda.empty_cache()
model = load_sls(OUT, device)
after = evaluate(model, {**dev_groups, **test_groups}, device)
after.to_csv("/kaggle/working/after.csv", index=False)

cols = ["EER %", "false alarms on real @0.5 %", "missed fakes @0.5 %"]
cmp = before[["group", "mode"] + [c for c in cols if c in before]].merge(
      after[["group", "mode"] + [c for c in cols if c in after]], on=["group", "mode"], suffixes=(" before", " after"))
cmp.to_csv("/kaggle/working/before_vs_after.csv", index=False)
print("Copy this table into the chat:")
print(cmp.to_string(index=False))"""),
]


VS_FILES = ["__init__.py", "audio.py", "metrics.py", "model.py", "augment.py", "train_data.py", "train.py"]


def code_cells(files: list[str]) -> list[tuple[str, str]]:
    """Cells that write our vs/ package into the Kaggle working folder."""
    cells = [("md", "## Our code\nRun these cells as they are. Each one writes one file of our `vs` package."),
             ("code", "import os\nos.makedirs('vs', exist_ok=True)")]
    for f in files:
        src = (ROOT / "vs" / f).read_text(encoding="utf-8")
        cells.append(("code", f"%%writefile vs/{f}\n{src}"))
    return cells


def _write_nb(cells, name: str, files: list[str]) -> Path:
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                     "language_info": {"name": "python"}},
        "cells": [],
    }
    cells = cells[:1] + code_cells(files) + cells[1:]     # code right after the intro
    for kind, src in cells:
        cell = {"cell_type": "markdown" if kind == "md" else "code", "metadata": {},
                "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell.update(outputs=[], execution_count=None)
        nb["cells"].append(cell)
    out = ROOT / "training" / name
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return out


def build_notebook() -> list[Path]:
    return [_write_nb(CELLS, "01_baseline_in_the_wild.ipynb", ["__init__.py", "audio.py", "metrics.py", "model.py"]),
            _write_nb(FT_CELLS, "02_finetune_hindi_phone.ipynb", VS_FILES)]


if __name__ == "__main__":
    for nb in build_notebook():
        print(nb)
