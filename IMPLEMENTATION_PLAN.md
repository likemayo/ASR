# Consolidated Implementation Plan for ADI/O ASR Project

This document contains the detailed, phase-by-phase implementation guide with code examples, timelines, and checkpoints.

## Table of Contents
1. [Prerequisites](#prerequisites--foundational-skills)
2. [Phase 1: Imagery Dataset](#phase-1-imagery-dataset-and-visual-framework)
3. [Phase 2: Audio Processing & ASR](#phase-2-audio-dataset-preparation-and-asr-modeling)
4. [Phase 3: Multimodal Integration](#phase-3-multimodality-development)
5. [Phase 4: Prototype](#phase-4-therapy-application-prototype)
6. [Phase 5: Testing & Reporting](#phase-5-final-testing-and-reporting)
7. [Risk Mitigation](#risk-mitigation-checkpoints)

---

## Prerequisites & Foundational Skills

### Python Fundamentals
- Functions, classes, libraries
- Data structures (lists, dicts, numpy arrays)
- File I/O operations

### Data Handling
```python
import pandas as pd
import librosa
import soundfile as sf
from PIL import Image
import numpy as np
```

### Machine Learning Basics
- What fine-tuning is and why it works
- Train/test/validation splits (and when to use each)
- Metrics: WER, accuracy, loss functions

**Recommended Learning Time:** 1-2 weeks if student needs refresher

---

## Phase 1: Imagery Dataset and Visual Framework

**Timeline:** Complete by January 2026 (~7 weeks)  
**Status:** ✅ COMPLETE

See `phase_1/` directory for implementation.

---

## Phase 2: Audio Dataset Preparation and ASR Modeling

**Timeline:** ~8 weeks  
**Checkpoint:** WER improvement ≥ 5%

Implementation files:
- `phase_2/audio_preprocessing.py`
- `phase_2/baseline_evaluation.py`
- `phase_2/lora_finetuning.py`

---

## Phase 3: Multimodality Development

**Timeline:** ~8 weeks  
**Checkpoint:** Multimodal helps ≥ 50% of cases

Implementation files:
- `phase_3/clip_embeddings.py`
- `phase_3/multimodal_asr.py`
- `phase_3/fusion_tuning.py`

---

## Phase 4: Therapy Application Prototype

**Timeline:** ~4 weeks  
**Checkpoint:** Prototype stable and responsive

Implementation files:
- `phase_4/app.py`
- `modules/asr.py`
- `modules/scaffolding.py`
- `modules/evaluation.py`

---

## Phase 5: Final Testing and Reporting

**Timeline:** ~4 weeks  
**Deliverable:** Complete written report with visualizations

---

## Risk Mitigation Checkpoints

| After Phase | Check | Action if Failed |
|-------------|-------|------------------|
| Phase 2 | WER improvement ≥ 5% | Increase LoRA rank, add training data |
| Phase 3 | Multimodal helps ≥ 50% of cases | Implement transcript normalization |
| Phase 4 | Prototype stable | Simplify UI, reduce features |
| Phase 5 | Results reproducible | Document all configurations |

---

## Key Warnings & Dependencies

### Critical Path Items
1. **Project Euphonia Access** (Apply early in Phase 1)
2. **GPU Access** (For Phase 2 fine-tuning)
3. **API Keys** (OpenAI for Phase 3/4, AWS for compute)
4. **Validation Data** (Need 2-3 speakers with atypical speech for Phase 2)

### Potential Bottlenecks
- CLIP may not work well with conversational speech (Phase 3)
- N-best hypothesis generation requires HuggingFace transformers (not standard Whisper)
- Fine-tuning costs can exceed budget if not monitored

---

See individual phase directories for detailed code and notebooks.
