# Results

All numbers come from our own runs: the Kaggle fine-tuning notebook (`training/02_finetune_hindi_phone.ipynb`, run A)
and local tests with `tools/score_folder.py` and `tools/compare_models.py`.

**Metric.** EER (Equal Error Rate) is the error at the cut-off where false alarms equal missed fakes. Lower is
better; it is the standard metric of the ASVspoof challenges. "False alarms" = real human voices called AI.

## 1. Before vs after fine-tuning (Kaggle test sets)

| Test set | Mode | EER before | EER after | False alarms on real, before | after |
|---|---|---|---|---|---|
| English dev | clean | 20.50% | **11.75%** | 35.9% | **0.1%** |
| English dev | phone | 24.75% | **12.25%** | 57.0% | **2.2%** |
| Hindi dev | clean | 25.55% | **15.56%** | 19.5% | **0.0%** |
| Hindi dev | phone | 28.59% | **12.35%** | 31.4% | **0.7%** |
| English real voices (FLEURS + LibriSpeech + Svarah test) | clean | — | — | 29.4% | **0.0%** |
| English real voices | phone | — | — | 53.2% | **1.3%** |
| Hindi real voices (FLEURS test) | clean | — | — | 31.8% | **0.0%** |
| Hindi real voices | phone | — | — | 46.0% | **1.2%** |
| Indian-accented English (Svarah test) | clean | — | — | 48.8% | **0.0%** |
| Indian-accented English | phone | — | — | 58.8% | **0.2%** |
| **English, AI tools never trained on** | clean | 33.52% | **22.95%** | 29.4% | **0.0%** |
| **English, AI tools never trained on** | phone | 41.97% | **20.29%** | 53.2% | **1.3%** |
| **Hindi, AI tools never trained on** | clean | 15.79% | **2.50%** | 31.8% | **0.0%** |
| **Hindi, AI tools never trained on** | phone | 23.96% | **2.50%** | 46.0% | **1.2%** |
| **Our recordings (people never trained on)** | clean | 61.11% | **5.56%** | 100.0% | **0.0%** |
| **Our recordings** | phone | 72.22% | **5.56%** | 94.4% | **0.0%** |

"Before" = the public XLS-R + SLS checkpoint (English ASVspoof 2019 only). "After" = our fine-tuned model.
Held-out tools: ElevenLabs v3, OpenAI TTS-1 HD, Cartesia Sonic-3, Gemini TTS, MiniMax 2.8, Resemble.ai,
Inworld TTS-2, Chatterbox-Turbo (English); Veena, Ringg (Hindi).

## 2. Laptop app, unseen clips (calibrated cut-off, full-clip scoring)

| Set | AI caught | Humans called real | Humans called AI |
|---|---|---|---|
| Our 36 held-out recordings | 18 / 18 | 17 / 18 (1 uncertain) | **0** |
| 10 new TTS clips (Google TTS, Microsoft Edge neural voices, other TTS) | 10 / 10 | — | — |

## 3. Comparison with public detectors (same 80 clips)

13 unseen 8 kHz AI clips + 60 of our real human recordings + 7 Google/Edge TTS clips.

| Model | EER | Unseen AI caught (@0.5) | TTS caught | Humans flagged as AI |
|---|---|---|---|---|
| **Voice Shield (ours)** | **10.8%** | 6 / 13 | **7 / 7** | **0 / 60** |
| MelodyMachine/Deepfake-audio-detection-V2 | 59.2% | 13 / 13 | 2 / 7 | 60 / 60 |
| mo-thecreator/Deepfake-audio-detection | 90.0% | 13 / 13 | 4 / 7 | 60 / 60 |
| garystafford/wav2vec2-deepfake-voice-detector | 45.0% | 0 / 13 | 6 / 7 | 0 / 60 |
| Gustking/wav2vec2-large-xlsr-deepfake-audio-classification | 30.0% | 0 / 13 | 2 / 7 | 0 / 60 |

Two public models catch all 13 AI clips only because they call every clip AI, including all 60 real humans.

## 4. Shortcuts found and removed

| Problem | Evidence | Fix |
|---|---|---|
| Loudness shortcut | Same real voice: 98–100% AI at -47 dB, 0–38% at normal level | Loudness normalised in training and inference |
| Splicing artifact (live stream) | Cutting pauses and joining speech pushed real voices to 92–99% AI | Score continuous audio; voice detection only decides when |

## 5. Known limitations

- AI audio played through a speaker into a microphone (replay) is called human. Replay needs replay training data.
- Newest English generators: ~20–23% EER on held-out tools; on a set of 13 unseen 8 kHz AI clips, 6 caught at 0.5.
- Our own test set is small (36 clips). Svarah has no speaker labels, so its split is by file.
