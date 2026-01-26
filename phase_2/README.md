# Phase 2: Audio Dataset Preparation and ASR Modeling

## Overview
Prepare audio data, establish Whisper baseline, and fine-tune using LoRA.

## Deliverables
- [ ] Processed audio dataset from Project Euphonia
- [ ] Baseline WER measurement
- [ ] Fine-tuned LoRA model
- [ ] WER improvement report (target: ≥5%)

## Files in This Directory
- `audio_preprocessing.py` - Clean, normalize, and augment audio
- `baseline_evaluation.py` - Test stock Whisper model
- `lora_finetuning.py` - Fine-tune Whisper with LoRA
- `training_config.yaml` - Training hyperparameters
- `wer_analysis.py` - Detailed WER analysis

## Critical Steps
1. Apply for Project Euphonia dataset access (can take 1-2 weeks)
2. Preprocess audio files
3. Split into train/val/test
4. Evaluate baseline
5. Fine-tune with LoRA
6. Measure improvement

See `../IMPLEMENTATION_PLAN.md` for detailed code examples.

## Checkpoint
- **Success Criteria:** WER improvement ≥ 5%
- **If Failed:** Increase LoRA rank, add more training data
