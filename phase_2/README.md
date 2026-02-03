# Phase 2: Audio Dataset Preparation and ASR Modeling

## Overview

Prepare audio data from the TORGO database, establish a Whisper baseline on dysarthric speech, and fine-tune Whisper using LoRA to improve transcription accuracy.

## Dataset: TORGO Database

The [TORGO database](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html) is loaded via Hugging Face (`abnerh/TORGO-database`).

**Dataset stats (from `--info`):**

| Group | Samples | Duration | Avg Length |
|-------|---------|----------|------------|
| Dysarthric | 5,574 | 5.46h | 3.53s |
| Healthy | 10,978 | 8.22h | 2.69s |
| **Total** | **16,552** | **13.68h** | **2.97s** |

**Columns:** `audio`, `transcription`, `speech_status`, `gender`, `duration`

**Splits (80/10/10, stratified by speech_status):**

| Split | Samples |
|-------|---------|
| Train | 13,240 |
| Val | 1,656 |
| Test | 1,656 |

## Deliverables

- [x] Downloaded and split TORGO audio dataset
- [ ] Baseline WER measurement (stock Whisper on dysarthric vs. healthy)
- [ ] Preprocessed audio (normalized, trimmed, augmented)
- [ ] Fine-tuned LoRA model on dysarthric speech
- [ ] WER improvement report (target: ≥5% relative improvement)

## Files in This Directory

| File | Purpose |
|------|---------|
| `data_loader.py` | Download TORGO from HF, create stratified train/val/test splits |
| `audio_preprocessing.py` | Resample to 16kHz, normalize, trim silence, augment |
| `baseline_evaluation.py` | Run stock Whisper (tiny/base/small), compute WER by group |
| `lora_finetuning.py` | Fine-tune Whisper with LoRA adapters using HF Seq2SeqTrainer |
| `training_config.yaml` | Hyperparameters for LoRA fine-tuning |
| `wer_analysis.py` | Compare baseline vs. fine-tuned WER, generate report |

## Workflow

### Step 1: Download & Split Dataset

```bash
# Preview dataset stats
python phase_2/data_loader.py --info

# Download and create train/val/test splits
python phase_2/data_loader.py --output ./data/audio/torgo
```

Output: `data/audio/torgo/torgo_dataset/` and `data/audio/torgo/splits.json`

### Step 2: Baseline Evaluation

Evaluate stock Whisper models on the test split **before any fine-tuning**:

```bash
python phase_2/baseline_evaluation.py --input ./data/audio/torgo
```

Tests whisper-tiny, whisper-base, and whisper-small. Reports:

| Metric | Description |
|--------|-------------|
| Overall WER | Across all test samples |
| WER by speech status | Dysarthric vs. healthy |
| Common error types | Substitutions, deletions, insertions |

Picks the best model size as the starting point for fine-tuning.

### Step 3: Preprocess Audio

```bash
python phase_2/audio_preprocessing.py --input ./data/audio/torgo --output ./data/audio/torgo_processed
```

Pipeline:
1. Resample to 16kHz
2. Normalize amplitude to [-1, 1]
3. Trim leading/trailing silence (-40 dB threshold)
4. Filter out samples < 0.5s or > 30s
5. Augment training data only (speed perturbation, noise, pitch shift)

### Step 4: LoRA Fine-Tuning

```bash
python phase_2/lora_finetuning.py --config phase_2/training_config.yaml
```

Key settings (in `training_config.yaml`):

```yaml
model: openai/whisper-small
lora: r=16, alpha=32, dropout=0.1
training: 10 epochs, lr=1e-4, early stopping (patience=3)
```

Saves best checkpoint to `models/whisper-lora-torgo/final/`.

### Step 5: Evaluation & Analysis

```bash
python phase_2/wer_analysis.py --model ./models/whisper-lora-torgo/final
```

Generates `wer_report.md` with:
- Baseline vs. fine-tuned WER comparison
- Breakdown by speech status (dysarthric vs. healthy)
- Sample transcription comparisons (before/after)
- Training curves and compute cost

## Checkpoint

- **Success Criteria:** WER improvement ≥ 5% (relative) on dysarthric test samples
- **If below target:**
  1. Increase LoRA rank (r=32 or r=64)
  2. Add more augmented data
  3. Try unfreezing more layers (encoder + decoder)
  4. Experiment with whisper-medium if compute allows
- **If above target:** Proceed to Phase 3 (multimodal CLIP integration)

## Dependencies

```
torch>=2.1.0
transformers>=4.35.0
datasets>=2.14.0
peft>=0.7.0
librosa>=0.10.0
jiwer>=3.0.1
soundfile>=0.12.0
accelerate>=0.24.0
pyyaml>=6.0
```

## References

- Rudzicz, F., Namasivayam, A.K. & Wolff, T. "The TORGO database of acoustic and articulatory speech from speakers with dysarthria." *Lang Resources & Evaluation* 46, 523–541 (2012).
- Radford, A. et al. "Robust Speech Recognition via Large-Scale Weak Supervision." (Whisper, 2022).
- Hu, E.J. et al. "LoRA: Low-Rank Adaptation of Large Language Models." (2021).
