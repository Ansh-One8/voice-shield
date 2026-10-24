"""Training and evaluation data: download, extract, and list clips.

Every clip becomes one row (a dict) with:
    path, label ('real' | 'fake'), lang ('en' | 'hi'), source, speaker,
    generator (cloning tool, '' for real), split ('train' | 'dev' | 'test')

Splits never share speakers where the source tells us who is speaking, and
IndicSynth's XTTS-v2 clips are kept out of training entirely so we can measure
"a cloning tool the model has never seen".
"""

from __future__ import annotations

import io
import os
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from . import audio
from .augment import augment, eval_view


def find_file(name: str, root: str = "/kaggle/input") -> str | None:
    """First file called `name` under root, without walking into audio folders.

    glob('**') would list hundreds of thousands of audio files in ASVspoof.
    """
    skip = {"flac", "wav", "wavs", "audio"}
    for d, dirs, files in os.walk(root):
        for f in files:
            if f.lower() == name.lower():          # some uploads change the case
                return os.path.join(d, f)
        dirs[:] = [x for x in dirs if x.lower() not in skip]
    return None


def _speaker_split(speaker, dev_share: int = 10) -> str:
    """Stable train/dev split by speaker (about 1 in `dev_share` goes to dev)."""
    return "dev" if zlib.crc32(str(speaker).encode()) % dev_share == 0 else "train"


# ---------------------------------------------------------------- parquet extraction

def _audio_bytes(row: dict) -> bytes | None:
    """Encoded audio from a HuggingFace row, whatever the audio column is called."""
    a = row.get("audio")
    if isinstance(a, dict) and a.get("bytes"):
        return a["bytes"]
    for v in row.values():
        if isinstance(v, dict) and v.get("bytes"):
            return v["bytes"]
    return None


def _audio_name(row: dict) -> str:
    for v in row.values():
        if isinstance(v, dict) and v.get("path"):
            return str(v["path"])
    return ""


def extract_parquet(parquet_path: str, out_dir: str, prefix: str, max_clips: int,
                    row_fn, max_seconds: float = 8.0) -> list[dict]:
    """Decode audio from a HuggingFace parquet file into 16 kHz wav files.

    row_fn(row) returns the metadata dict for a row, or None to skip it.
    Rows are taken evenly across the file so we don't only get its first speaker.
    """
    import pyarrow.parquet as pq
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(parquet_path)
    print(f"  {Path(parquet_path).name}: {pf.metadata.num_rows} rows, columns {pf.schema_arrow.names}")
    step = max(1, pf.metadata.num_rows // max_clips)
    rows, idx = [], 0
    for batch in pf.iter_batches(batch_size=64):
        for row in batch.to_pylist():
            idx += 1
            if (idx - 1) % step or len(rows) >= max_clips:
                continue
            meta = row_fn(row)
            if meta is None:
                continue
            raw = _audio_bytes(row)
            if not raw:
                continue
            try:
                x, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
            except Exception:
                continue
            x = audio.resample(x.mean(axis=1), sr, audio.SR)[: int(max_seconds * audio.SR)]
            if len(x) < audio.SR:                       # skip clips under 1 s
                continue
            path = out / f"{prefix}_{idx:07d}.wav"
            sf.write(path, x, audio.SR, subtype="PCM_16")
            rows.append({**meta, "path": str(path)})
        if len(rows) >= max_clips:
            break
    return rows


def _hf_extract(repo: str, filename: str, out_dir: str, prefix: str, max_clips: int, row_fn,
                token: str | None = None) -> list[dict]:
    """Download one parquet file, extract clips from it, then delete it to save disk."""
    from huggingface_hub import hf_hub_download
    tmp = Path(out_dir) / "_download"
    path = hf_hub_download(repo_id=repo, repo_type="dataset", filename=filename, token=token,
                           local_dir=str(tmp))
    try:
        return extract_parquet(path, out_dir, prefix, max_clips, row_fn)
    finally:
        Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------- sources

def fleurs(lang: str, split: str, out_dir: str, max_clips: int) -> list[dict]:
    """Real speech from FLEURS. lang: 'hi_in' or 'en_us'. split: train | validation | test."""
    short = lang[:2]
    ours = {"train": "train", "validation": "dev", "test": "test"}[split]
    return _hf_extract("google/fleurs", f"parquet-data/{lang}/{split}-00000-of-00001.parquet",
                       f"{out_dir}/fleurs_{lang}_{split}", f"fleurs_{short}_{split}", max_clips,
                       lambda r: {"label": "real", "lang": short, "source": f"fleurs_{short}",
                                  "speaker": f"fleurs_{short}_{split}", "generator": "",
                                  "split": ours})


def indicsynth_hindi(shards: list[int], out_dir: str, per_shard: int,
                     holdout_xtts: bool = False) -> list[dict]:
    """Hindi AI speech from IndicSynth (107 shards, ~0.5 GB each).

    The Hindi shards are sorted by tool: roughly 0-54 are FreeVC (voice
    conversion) and 56-106 are XTTS-v2 (voice-cloning TTS). With holdout_xtts,
    every XTTS-v2 clip goes to test as an "unseen tool"; otherwise XTTS-v2 is
    trained on too, split by speaker like FreeVC.
    """
    def meta(r):
        gen = r.get("Generative Model") or "unknown"
        spk = r.get("Target Speaker ID")
        split = "test" if holdout_xtts and gen == "xtts_v2" else _speaker_split(spk)
        return {"label": "fake", "lang": "hi", "source": "indicsynth", "speaker": f"is_{spk}",
                "generator": gen, "split": split}
    rows = []
    for s in shards:
        rows += _hf_extract("vdivyasharma/IndicSynth", f"Hindi/train-{s:05d}-of-00107.parquet",
                            f"{out_dir}/indicsynth", f"is{s:03d}", per_shard, meta)
        print(f"  IndicSynth shard {s}: {len(rows)} clips so far")
    return rows


def kathbath_hindi(shards: list[int], out_dir: str, per_shard: int, token: str) -> list[dict]:
    """Real Hindi speech from Kathbath (needs a HuggingFace token; accept the terms on its page first)."""
    def meta(r):
        spk = r.get("speaker_id") or r.get("speaker") or r.get("client_id")
        if spk is None:
            # no speaker column: fall back to the file name so clips still spread over train and dev
            spk = _audio_name(r) or str(zlib.crc32(repr(sorted(k for k in r)).encode() + os.urandom(4)))
        return {"label": "real", "lang": "hi", "source": "kathbath", "speaker": f"kb_{spk}",
                "generator": "", "split": _speaker_split(spk)}
    rows = []
    for s in shards:
        rows += _hf_extract("ai4bharat/Kathbath", f"hindi/train-{s:05d}-of-00032.parquet",
                            f"{out_dir}/kathbath", f"kb{s:02d}", per_shard, meta, token)
    return rows


def svarah(train_shards: list[int], test_shards: list[int], out_dir: str, per_shard: int,
           token: str) -> list[dict]:
    """Real Indian-accented English (AI4Bharat Svarah, needs HF_TOKEN; 3 files of ~365 MB).

    Without it, the only real English is American/British, and real Indians
    speaking English could look "unusual" to the model and be flagged as AI.
    """
    rows = []
    for s in train_shards + test_shards:
        def meta(r, test=s in test_shards):
            spk = r.get("speaker_id") or r.get("speaker") or _audio_name(r).split("_")[0] or None
            if spk is None:
                spk = str(zlib.crc32(os.urandom(4)))
            return {"label": "real", "lang": "en", "source": "svarah", "speaker": f"sv_{spk}",
                    "generator": "", "split": "test" if test else _speaker_split(spk)}
        rows += _hf_extract("ai4bharat/Svarah", f"data/test-{s:05d}-of-00003.parquet",
                            f"{out_dir}/svarah", f"sv{s}", per_shard, meta, token)
    return rows


def librispeech(parts: dict[str, list[int]], out_dir: str, per_shard: int) -> list[dict]:
    """Real English audiobook speech (open, many speakers).

    MLAAD's English fakes are audiobook text read by AI voices, so we need real
    audiobook reading too, or "audiobook style" would look like a sign of fakes.
    parts: e.g. {"train.clean.100": [0, 7], "train.other.500": [0, 32], "test.clean": [0]}
    """
    rows = []
    for part, shards in parts.items():
        for s in shards:
            def meta(r, part=part):
                spk = r.get("speaker_id")
                split = "test" if part.startswith("test") else _speaker_split(spk)
                return {"label": "real", "lang": "en", "source": "librispeech", "speaker": f"ls_{spk}",
                        "generator": "", "split": split}
            rows += _hf_extract("openslr/librispeech_asr", f"all/{part}/{s:04d}.parquet",
                                f"{out_dir}/librispeech", f"ls_{part}_{s:02d}", per_shard, meta)
    return rows


def mlaad(lang: str, out_dir: str, per_tool: int, token: str, holdout: list[str],
          max_tools: int | None = None, workers: int = 8, per_holdout_tool: int | None = None) -> list[dict]:
    """AI voices from MLAAD: many recent TTS / voice-cloning tools (needs HF_TOKEN).

    lang: 'en' (143 tools incl. ElevenLabs, OpenAI, Cartesia, MiniMax, F5, XTTS...)
          or 'hi' (9 tools incl. IndicF5, Veena, XTTS-v2).
    Tools named in `holdout` are never trained on: they become the
    "new tools" test. Other tools: ~90% of files train, ~10% dev.
    """
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import HfApi, hf_hub_download
    repo = "mueller91/MLAAD"
    api = HfApi(token=token)
    tools = sorted(t.path.split("/")[-1] for t in api.list_repo_tree(repo, f"fake/{lang}", repo_type="dataset")
                   if not t.path.endswith((".csv", ".txt", ".json")))
    if max_tools:
        keep = [t for t in tools if t in holdout]
        rest = [t for t in tools if t not in holdout]
        tools = keep + rest[:: max(1, len(rest) // max_tools)][:max_tools]
    out = Path(out_dir) / f"mlaad_{lang}"
    out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for tool in tools:
        files = sorted(f.path for f in api.list_repo_tree(repo, f"fake/{lang}/{tool}", repo_type="dataset")
                       if f.path.endswith((".wav", ".mp3", ".flac")))
        n = per_holdout_tool if (tool in holdout and per_holdout_tool) else per_tool
        step = max(1, len(files) // n)
        jobs += [(tool, f) for f in files[::step][:n]]
    print(f"  MLAAD {lang}: {len(tools)} tools, {len(jobs)} files to download")

    def fetch(job):
        tool, f = job
        x = None
        for attempt in range(3):                  # retry: many small downloads can hit rate limits
            try:
                p = hf_hub_download(repo, f, repo_type="dataset", token=token, local_dir=str(out / "_download"))
                x = audio.load_audio(p)[: 8 * audio.SR]
                Path(p).unlink(missing_ok=True)
                break
            except Exception:
                time.sleep(2 + 4 * attempt)
        if x is None:
            return None
        if len(x) < audio.SR:
            return None
        dst = out / f"{zlib.crc32(f.encode()):08x}.wav"
        sf.write(dst, x, audio.SR, subtype="PCM_16")
        split = "test" if tool in holdout else _speaker_split(f)      # split by file within a tool
        return {"path": str(dst), "label": "fake", "lang": lang, "source": "mlaad",
                "speaker": f"mlaad_{tool}", "generator": tool, "split": split}

    with ThreadPoolExecutor(workers) as ex:
        rows = [r for r in ex.map(fetch, jobs) if r]
    print(f"  MLAAD {lang}: {len(rows)} clips ({sum(r['split'] == 'test' for r in rows)} from held-out tools)")
    return rows


def asvspoof2019_la(search_root: str = "/kaggle/input") -> list[dict]:
    """English real + fake from an attached ASVspoof 2019 LA dataset (train and dev)."""
    rows = []
    for part, proto_name, split in [("train", "ASVspoof2019.LA.cm.train.trn.txt", "train"),
                                    ("dev", "ASVspoof2019.LA.cm.dev.trl.txt", "dev")]:
        proto = find_file(proto_name, search_root)
        if not proto:
            raise FileNotFoundError(f"{proto_name} not found. Attach the ASVspoof 2019 DATASET "
                                    f"(not a notebook) that has the LA folder.")
        la_root = os.path.dirname(os.path.dirname(proto))          # .../LA
        flac = os.path.join(la_root, f"ASVspoof2019_LA_{part}", "flac")
        if not os.path.isdir(flac):
            # some uploads nest the folder once more, e.g. ASVspoof2019_LA_train/ASVspoof2019_LA_train/flac
            hits = [d for d, dirs, _ in os.walk(la_root) if d.endswith(os.path.join(f"ASVspoof2019_LA_{part}", "flac"))]
            if not hits:
                raise FileNotFoundError(f"No ASVspoof2019_LA_{part}/flac folder under {la_root}.")
            flac = hits[0]
        for line in open(proto):
            spk, key, _, system, label = line.split()
            rows.append({"path": f"{flac}/{key}.flac", "label": "real" if label == "bonafide" else "fake",
                         "lang": "en", "source": "asvspoof19", "speaker": spk,
                         "generator": "" if system == "-" else system, "split": split})
    return rows


def in_the_wild(search_root: str = "/kaggle/input") -> list[dict]:
    """English real-world deepfakes. Test only, never trained on."""
    meta = find_file("meta.csv", search_root)
    if not meta:
        return []
    df = pd.read_csv(meta)
    base = os.path.dirname(meta)
    return [{"path": f"{base}/{r.file}", "label": "fake" if str(r.label).lower() == "spoof" else "real",
             "lang": "en", "source": "in_the_wild", "speaker": f"itw_{r.speaker}", "generator": "",
             "split": "test"} for r in df.itertuples()]


def extra_labels(search_root: str = "/kaggle/input") -> list[dict]:
    """Optional: our own recordings, from a dataset that has vs_labels.csv.

    Columns: file,label,language,speaker,split[,tool]
      label     real | fake
      language  en | hi | hinglish   (Hinglish is grouped with Hindi)
      speaker   the PERSON (for a clone: the person whose voice was cloned)
      split     train | test   (the same person must never appear in both)
      tool      optional, e.g. xtts, elevenlabs, f5
    """
    found = find_file("vs_labels.csv", search_root)
    if not found:
        return []
    df = pd.read_csv(found)
    base = os.path.dirname(found)
    clash = set(df[df.split == "train"].speaker) & set(df[df.split == "test"].speaker)
    if clash:
        raise ValueError(f"These people are in both train and test, which makes the test meaningless: {sorted(clash)}")
    rows = []
    for r in df.itertuples():
        lang = "hi" if str(r.language).lower().startswith(("hi", "hinglish")) else "en"
        tool = getattr(r, "tool", "")
        rows.append({"path": f"{base}/{r.file}", "label": r.label, "lang": lang, "source": "ours",
                     "speaker": f"ours_{r.speaker}", "generator": str(tool) if pd.notna(tool) else "",
                     "split": r.split})
    return rows


def summary(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    return df.groupby(["split", "lang", "label", "source"]).size().unstack("label", fill_value=0)


# ---------------------------------------------------------------- torch datasets

class TrainSet(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        try:
            x = audio.load_audio(r["path"])
        except Exception:
            x = np.zeros(audio.SR, np.float32)
        x = augment(x, self.rng)
        return torch.from_numpy(x), 1 if r["label"] == "real" else 0    # model index 1 = real


class EvalSet(Dataset):
    def __init__(self, rows: list[dict], phone: bool):
        self.rows, self.phone = rows, phone

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        try:
            x = audio.load_audio(self.rows[i]["path"])
        except Exception:
            x = np.zeros(audio.SR, np.float32)
        return torch.from_numpy(eval_view(x, self.phone, seed=i)), i


def worker_init(worker_id: int):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2**32 + worker_id)


def _bucket(r: dict) -> str:
    """What gets an equal share inside a group: the AI tool for fakes, the dataset for real voices."""
    if r["source"] == "ours":
        return "ours"
    if r["label"] == "fake":
        gen = r.get("generator") or r["source"]
        return "xtts_v2" if "xtts_v2" in gen else gen      # same tool, named differently by two datasets
    return r["source"]


def balanced_weights(rows: list[dict], ours_share: float = 0.05) -> torch.Tensor:
    """Sampling weights for balanced batches.

    1. The 4 groups English-real, English-AI, Hindi-real, Hindi-AI are drawn
       equally often, so overall it is 50% real and 50% AI, 50% English and 50% Hindi.
    2. Inside an AI group every TOOL gets the same share (ElevenLabs, XTTS-v2,
       FreeVC, each ASVspoof system...), however many clips it has.
    3. Inside a real group every dataset gets the same share.
    4. Our own recordings get a fixed `ours_share` of each group, for real and AI
       alike: enough to matter, not so much that a few dozen clips are memorized.
    """
    from collections import Counter
    key = [(r["lang"], r["label"], _bucket(r)) for r in rows]
    per_bucket = Counter(key)
    others_in_group = Counter(k[:2] for k in per_bucket if k[2] != "ours")
    has_ours = {k[:2] for k in per_bucket if k[2] == "ours"}
    w = []
    for k in key:
        g = k[:2]
        if k[2] == "ours":
            share = ours_share if others_in_group[g] else 1.0
        else:
            share = (1.0 - (ours_share if g in has_ours else 0.0)) / others_in_group[g]
        w.append(share / per_bucket[k])
    return torch.tensor(w, dtype=torch.double)


def mix_report(rows: list[dict]) -> pd.DataFrame:
    """Share of training draws per group and per tool/dataset, as the sampler will see them."""
    w = balanced_weights(rows).numpy()
    df = pd.DataFrame({"lang": [r["lang"] for r in rows], "label": [r["label"] for r in rows],
                       "bucket": [_bucket(r) for r in rows], "w": w / w.sum()})
    t = df.groupby(["lang", "label", "bucket"]).agg(clips=("w", "size"), share=("w", "sum")).reset_index()
    t["share %"] = (t.pop("share") * 100).round(2)
    return t
