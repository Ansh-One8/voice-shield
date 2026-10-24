"""Fine-tuning loop and evaluation for the SLS detector."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from .metrics import compute_eer
from .train_data import EvalSet, TrainSet, balanced_weights, worker_init


@torch.inference_mode()
def predict(model, rows: list[dict], device: str, phone: bool, batch: int = 32,
            workers: int = 4) -> np.ndarray:
    """Fake-probability for each row."""
    was_training = model.training
    model.eval()
    p = np.zeros(len(rows))
    dl = DataLoader(EvalSet(rows, phone), batch_size=batch, num_workers=workers)
    with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
        for x, idx in dl:
            p[idx.numpy()] = 1 - model(x.to(device)).float()[:, 1].exp().cpu().numpy()
    model.train(was_training)
    return p


def evaluate(model, groups: dict[str, list[dict]], device: str, workers: int = 4) -> pd.DataFrame:
    """EER and false alarms for each named group, clean and phone mode."""
    out = []
    for name, rows in groups.items():
        y = np.array([r["label"] == "fake" for r in rows])
        for phone in (False, True):
            p = predict(model, rows, device, phone, workers=workers)
            row = {"group": name, "mode": "phone" if phone else "clean",
                   "real": int((~y).sum()), "fake": int(y.sum())}
            if y.any() and (~y).any():
                eer, thr = compute_eer(p[~y], p[y])
                row.update({"EER %": round(eer * 100, 2), "threshold": round(thr, 3)})
            if (~y).any():
                row["false alarms on real @0.5 %"] = round(float(np.mean(p[~y] > 0.5)) * 100, 1)
            if y.any():
                row["missed fakes @0.5 %"] = round(float(np.mean(p[y] <= 0.5)) * 100, 1)
            out.append(row)
    return pd.DataFrame(out)


def selection_score(table: pd.DataFrame, select_groups: list[str]) -> float:
    """Lower is better: mean EER over the dev groups, clean and phone."""
    t = table[table["group"].isin(select_groups)]
    return float(t["EER %"].mean())


def finetune(model, train_rows: list[dict], dev_groups: dict[str, list[dict]], *, device: str,
             out_path: str, steps: int = 2000, batch: int = 8, accum: int = 2,
             lr_ssl: float = 2e-6, lr_head: float = 5e-5, warmup: int = 150,
             eval_every: int = 400, workers: int = 4, log=print) -> list[dict]:
    """Fine-tune and keep the checkpoint with the best dev EER.

    steps counts optimizer updates; each update sees batch * accum clips.
    """
    model.ssl.freeze_feature_encoder()              # the CNN front end stays as pretrained
    head = [p for n, p in model.named_parameters() if not n.startswith("ssl.") and p.requires_grad]
    ssl = [p for n, p in model.named_parameters() if n.startswith("ssl.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": ssl, "lr": lr_ssl}, {"params": head, "lr": lr_head}],
                            weight_decay=1e-4)
    base = [lr_ssl, lr_head]
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    sampler = WeightedRandomSampler(balanced_weights(train_rows), num_samples=steps * batch * accum,
                                    replacement=True)
    dl = DataLoader(TrainSet(train_rows), batch_size=batch, sampler=sampler, num_workers=workers,
                    worker_init_fn=worker_init, drop_last=True, persistent_workers=workers > 0)

    history, best = [], math.inf
    select = list(dev_groups)
    model.train()
    t0, step, micro, run_loss = time.time(), 0, 0, 0.0
    for x, y in dl:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            loss = F.nll_loss(model(x).float(), y) / accum
        scaler.scale(loss).backward()
        run_loss += loss.item()
        micro += 1
        if micro % accum:
            continue

        step += 1
        scale = min(1.0, step / warmup) * 0.5 * (1 + math.cos(math.pi * min(step / steps, 1.0)))
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * max(scale, 0.02)
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        if step % 50 == 0:
            log(f"step {step}/{steps}  loss {run_loss / 50:.4f}  {(time.time() - t0) / 60:.1f} min")
            run_loss = 0.0
        if step % eval_every == 0 or step == steps:
            table = evaluate(model, dev_groups, device, workers)
            score = selection_score(table, select)
            history.append({"step": step, "dev_mean_eer": round(score, 2),
                            "minutes": round((time.time() - t0) / 60, 1)})
            log(table.to_string(index=False))
            # save on the first evaluation no matter what, so the next cell always has a file
            if score < best or not Path(out_path).exists():
                best = score if not math.isnan(score) else best
                save(model, out_path)
                log(f"  new best dev EER {score:.2f}% -> saved {out_path}")
            else:
                log(f"  dev EER {score:.2f}% (best {best:.2f}%)")
            model.train()
        if step >= steps:
            break
    Path(out_path).with_suffix(".history.json").write_text(json.dumps(history, indent=1))
    return history


def save(model, path: str) -> None:
    """Half-precision state dict (~680 MB). vs.model.load_sls loads it directly."""
    sd = {k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()}
    torch.save(sd, path)
