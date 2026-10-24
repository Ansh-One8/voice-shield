"""XLS-R + SLS deepfake detector, rebuilt on HuggingFace transformers.

The public checkpoint (Zhang et al. 2024, reproduced by Sukhdeve) was trained
with fairseq, which is hard to install on Windows and new Pythons. This file
rebuilds the same network with `transformers.Wav2Vec2Model` and converts the
fairseq weights on load, so the laptop and Kaggle run the same code.

Model input: (batch, 64600) raw 16 kHz waveform, no normalisation.
Model output: log-probabilities for [spoof, bonafide]  (index 1 = real voice).
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Config, Wav2Vec2Model

XLSR_NAME = "facebook/wav2vec2-xls-r-300m"
CKPT_REPO = "sukhdeveyash/XLS-R-SLS-Deepfake-Detection"
CKPT_FILE = "v1/epoch_2.pth"   # the repo's recommended checkpoint (7.84% EER on In-the-Wild)


def xlsr_config() -> Wav2Vec2Config:
    cfg = Wav2Vec2Config.from_pretrained(XLSR_NAME)
    # fairseq was called with mask=False and every layer must run, so turn off
    # SpecAugment masking and LayerDrop.
    cfg.apply_spec_augment = False
    cfg.mask_time_prob = 0.0
    cfg.mask_feature_prob = 0.0
    cfg.layerdrop = 0.0
    return cfg


class SLSModel(nn.Module):
    """XLS-R 300M -> Selective Layer Summarization -> small classifier head."""

    def __init__(self, cfg: Wav2Vec2Config | None = None):
        super().__init__()
        self.ssl = Wav2Vec2Model(cfg or xlsr_config())
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.fc0 = nn.Linear(1024, 1)
        self.fc1 = nn.Linear(22847, 1024)   # 341 * 67 after 3x3 max-pool of (201 frames, 1024 dims)
        self.fc3 = nn.Linear(1024, 2)

        # fairseq's layer_results are the raw outputs of all 24 transformer
        # layers. HF applies a final LayerNorm to its last hidden state, so we
        # hook the layers directly to get exactly the same tensors.
        self._layer_out: list[torch.Tensor] = []
        for layer in self.ssl.encoder.layers:
            layer.register_forward_hook(self._grab)

    def _grab(self, _module, _inp, out):
        self._layer_out.append(out[0] if isinstance(out, tuple) else out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._layer_out = []
        self.ssl(x)
        layers = self._layer_out                       # 24 x (B, T, 1024)
        self._layer_out = []

        pooled = torch.stack([h.mean(dim=1) for h in layers], dim=1)   # (B, 24, 1024)
        full = torch.stack(layers, dim=1)                              # (B, 24, T, 1024)
        weight = torch.sigmoid(self.fc0(pooled)).unsqueeze(-1)          # (B, 24, 1, 1)
        feat = (full * weight).sum(dim=1).unsqueeze(1)                 # (B, 1, T, 1024)

        x = self.selu(self.first_bn(feat))
        x = F.max_pool2d(x, (3, 3))
        x = torch.flatten(x, 1)
        x = self.selu(self.fc1(x))
        x = self.selu(self.fc3(x))
        return F.log_softmax(x, dim=1)


# ---------------------------------------------------------------- fairseq -> HF weights

_RULES = [
    (r"feature_extractor\.conv_layers\.(\d+)\.0\.", r"feature_extractor.conv_layers.\1.conv."),
    (r"feature_extractor\.conv_layers\.(\d+)\.2\.1\.", r"feature_extractor.conv_layers.\1.layer_norm."),
    (r"^layer_norm\.", "feature_projection.layer_norm."),
    (r"^post_extract_proj\.", "feature_projection.projection."),
    (r"encoder\.layers\.(\d+)\.self_attn\.", r"encoder.layers.\1.attention."),
    (r"encoder\.layers\.(\d+)\.self_attn_layer_norm\.", r"encoder.layers.\1.layer_norm."),
    (r"encoder\.layers\.(\d+)\.fc1\.", r"encoder.layers.\1.feed_forward.intermediate_dense."),
    (r"encoder\.layers\.(\d+)\.fc2\.", r"encoder.layers.\1.feed_forward.output_dense."),
]
# pretraining-only parts; mask_emb is unused because masking is off
_SKIP = ("quantizer.", "project_q.", "final_proj.", "label_embs", "target_glu.", "mask_emb")


def _pos_conv_keys(hf_keys: set[str]) -> tuple[str, str]:
    base = "ssl.encoder.pos_conv_embed.conv."
    if base + "parametrizations.weight.original0" in hf_keys:
        return base + "parametrizations.weight.original0", base + "parametrizations.weight.original1"
    return base + "weight_g", base + "weight_v"


def convert_fairseq_checkpoint(sd: dict, model: SLSModel) -> dict:
    """Rename a fairseq SLS state dict so it loads into SLSModel."""
    hf_keys = set(model.state_dict().keys())
    g_key, v_key = _pos_conv_keys(hf_keys)
    out = {}
    for k, v in sd.items():
        k = k.removeprefix("module.")
        if not k.startswith("ssl_model.model."):
            out[k] = v                          # classifier head: same names
            continue
        k = k.removeprefix("ssl_model.model.")
        if k.startswith(_SKIP):
            continue
        if k.startswith("encoder.pos_conv.0."):
            tail = k.removeprefix("encoder.pos_conv.0.")
            target = {"weight_g": g_key, "weight_v": v_key,
                      "parametrizations.weight.original0": g_key,
                      "parametrizations.weight.original1": v_key,
                      "bias": "ssl.encoder.pos_conv_embed.conv.bias"}[tail]
            out[target] = v
            continue
        for pat, rep in _RULES:
            k = re.sub(pat, rep, k)
        out["ssl." + k] = v
    return out


def load_sls(checkpoint_path: str, device: str = "cpu") -> SLSModel:
    model = SLSModel()
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    sd = convert_fairseq_checkpoint(raw, model) if any(
        k.removeprefix("module.").startswith("ssl_model.") for k in raw) else raw
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint did not match the model.\n"
                           f"Missing ({len(missing)}): {missing[:10]}\n"
                           f"Unexpected ({len(unexpected)}): {unexpected[:10]}")
    return model.to(device).eval()


def download_checkpoint(filename: str = CKPT_FILE, cache_dir: str | None = None) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=CKPT_REPO, filename=filename, cache_dir=cache_dir)
