# ADI/O: Developing Robust ASR for Autism-Focused Reading Comprehension Therapy

A comprehensive AI-assisted therapeutic tool combining visual and verbal understanding through multimodal automatic speech recognition (ASR), designed to improve reading comprehension for children with autism.

## Project Overview

**Author:** Sidharth Bildikar  
**Organization:** Tesla STEM High School, Redmond, Washington  
**Status:** In Development (Phase 1 Complete as of January 2026)  

### Quick Links
- 📋 [Project Timeline](#timeline)
- 🎯 [Implementation Plan](IMPLEMENTATION_PLAN.md)
- 🏗️ [System Architecture](#architecture)
- 📚 [Learning Resources](#learning-resources)
- 🚀 [Getting Started](#getting-started)

---

## The Problem

Up to 73% of autistic children excel at decoding words but struggle with reading comprehension. Current solutions like Visualization & Verbalization (V/V) therapy are effective but inaccessible due to:
- Few trained therapists
- High cost barriers
- Limited geographic availability

**The Blocker:** ASR systems fail dramatically on atypical speech (31% word error rate), making digital therapy delivery impractical.

---

## The Solution

**ADI/O** combines five components:

1. **Curated Image Bank** (100+ high-quality images)
2. **Gold-Standard Descriptions** (LLM-generated with structure word mappings)
3. **Multimodal ASR** (Whisper + LoRA + CLIP visual grounding)
4. **Therapy Scaffolding** (LLM-guided questions)
5. **Performance Evaluation** (automated response scoring)

### Key Innovation: Multimodal Decoder Biasing

```
Audio → Whisper (multiple candidates)
         ↓
      CLIP embeddings of candidates
         ↓
    Image → CLIP image embedding
         ↓
   Cosine Similarity Scoring
         ↓
   Rescore candidates using combined speech + visual context
         ↓
   Return best transcription (phonetically & visually consistent)
```

---

## Architecture

```
┌─────────────────────────────────────────┐
│         USER INTERFACE (Streamlit)      │
│  Image Display | Audio Input | Feedback │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│      MULTIMODAL ASR PIPELINE             │
│  ┌──────────────────────────────────┐   │
│  │ Fine-tuned Whisper (LoRA)       │   │
│  └──────────────┬───────────────────┘   │
│                 │                        │
│  ┌──────────────▼───────────────────┐   │
│  │ CLIP-based Visual Rescoring      │   │
│  │ (Semantic similarity with image) │   │
│  └──────────────┬───────────────────┘   │
│                 │                        │
│  ┌──────────────▼───────────────────┐   │
│  │ LLM Scaffolding Module           │   │
│  │ (Generate follow-up questions)   │   │
│  └──────────────┬───────────────────┘   │
│                 │                        │
│  ┌──────────────▼───────────────────┐   │
│  │ Evaluation Module                │   │
│  │ (Score response vs gold standard)│   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
         │              │              │
         ▼              ▼              ▼
   [Data Layer: Images | Audio | Metadata]
```

---

## Timeline

| Phase | Goal | Duration | Status |
|-------|------|----------|--------|
| **Phase 1** | Imagery Dataset & Visual Framework | 7 weeks | ✅ Complete |
| **Phase 2** | Audio Processing & ASR Modeling | 8 weeks | 🔄 In Progress |
| **Phase 3** | Multimodality Development | 8 weeks | ⏳ Planned |
| **Phase 4** | Therapy Prototype & Testing | 4 weeks | ⏳ Planned |
| **Phase 5** | Final Testing & Reporting | 4 weeks | ⏳ Planned |
| **Total** | Full Implementation | ~32 weeks | |

---

## Getting Started

### Prerequisites
- Python 3.8+
- GPU access (strongly recommended for fine-tuning)
- ~50GB disk space for models and data

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/adi-o.git
   cd adi-o
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.template .env
   # Edit .env with your API keys
   ```

### Quick Test

```bash
# Run a simple Whisper transcription
python -c "
import whisper
model = whisper.load_model('small')
result = model.transcribe('test_audio.wav')
print(result['text'])
"
```

---

## Project Structure

```
adi-o/
├── phase_1/                  # Imagery dataset and visual framework
│   ├── image_curator.py
│   ├── structure_word_mapper.py
│   └── validation_check.py
├── phase_2/                  # Audio processing and ASR modeling
│   ├── audio_preprocessing.py
│   ├── baseline_evaluation.py
│   ├── lora_finetuning.py
│   └── training_config.yaml
├── phase_3/                  # Multimodal ASR development
│   ├── clip_embeddings.py
│   ├── multimodal_asr.py
│   ├── fusion_tuning.py
│   └── transcript_normalization.py
├── phase_4/                  # Prototype and integration
│   ├── app.py                # Main Streamlit app
│   └── integration_tests.py
├── modules/                  # Reusable components
│   ├── asr.py
│   ├── scaffolding.py
│   ├── evaluation.py
│   └── utils.py
├── data/
│   ├── images/               # Curated image dataset
│   ├── audio/                # Audio validation data
│   ├── metadata.json         # Image metadata & gold standards
│   └── embeddings.pkl        # Pre-computed CLIP embeddings
├── models/
│   └── whisper-lora/         # Fine-tuned Whisper model
├── notebooks/                # Jupyter notebooks for exploration
├── tests/                    # Unit and integration tests
├── requirements.txt
├── .env.template
├── .gitignore
└── README.md
```

---

## Key Dependencies

| Library | Purpose | Version |
|---------|---------|---------|
| **torch** | Deep learning framework | 2.1.0 |
| **transformers** | Whisper & LLM inference | 4.35.0 |
| **librosa** | Audio processing | 0.10.0 |
| **peft** | LoRA fine-tuning | 0.7.0 |
| **clip** | Multimodal embeddings | 1.0 |
| **streamlit** | Web UI | 1.28.1 |
| **jiwer** | WER calculation | 3.0.1 |

---

## Phase Progression

### Phase 1: Imagery Dataset ✅
- [x] Curate 100 high-quality images
- [x] Tag with metadata (scene type, complexity, subjects)
- [x] Map structure words (who, what, where, color, size, mood)
- [x] Generate gold-standard descriptions
- [x] Validate dataset quality

**Deliverable:** `data/metadata.json` with complete image catalog

### Phase 2: ASR Modeling 🔄
- [ ] Access Project Euphonia dataset
- [ ] Preprocess audio (normalize, augment, clean)
- [ ] Establish baseline Whisper WER
- [ ] Fine-tune with LoRA
- [ ] Document WER improvements

**Expected Progress:** 5-15% WER improvement

### Phase 3: Multimodal Integration ⏳
- [ ] Generate CLIP image embeddings
- [ ] Implement n-best hypothesis rescoring
- [ ] Tune fusion coefficient (alpha)
- [ ] Validate multimodal gains

**Expected Progress:** 5-10% additional WER improvement

### Phase 4: Prototype ⏳
- [ ] Build Streamlit interface
- [ ] Integrate all components
- [ ] Implement scaffolding logic
- [ ] Add evaluation module
- [ ] Pilot testing

### Phase 5: Final Evaluation ⏳
- [ ] End-to-end testing
- [ ] Generate results tables & visualizations
- [ ] Compile final report
- [ ] Present findings

---

## Key Metrics

### WER (Word Error Rate)
Percentage of words incorrectly recognized:
```
WER = (S + D + I) / N × 100%
where S=substitutions, D=deletions, I=insertions, N=total words
```

### Expected Performance Progression
| Stage | WER | Source |
|-------|-----|--------|
| Baseline Whisper Small | ~31% | Tobin et al. (2024) |
| After LoRA fine-tuning | ~18% | Target: 5-15% improvement |
| After multimodal rescoring | ~12-15% | Target: additional 5-10% |

---

## Learning Resources

### Essential Skills by Phase
- **Phase 1:** Python, data structures, metadata management
- **Phase 2:** Audio processing, fine-tuning, evaluation metrics
- **Phase 3:** Embeddings, cosine similarity, hyperparameter tuning
- **Phase 4:** Web development (Streamlit), API integration
- **Phase 5:** Data visualization, technical writing

### Recommended Reading
- Whisper Paper: [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)
- LoRA Paper: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- CLIP Paper: [Learning Transferable Models for Unsupervised Learning](https://arxiv.org/abs/2103.14030)
- Project Euphonia: [Google's speech recognition for disordered speech](https://ai.googleblog.com/2021/07/project-euphonia-speech-recognition-for.html)

---

## Budget & Resources

**Total Budget:** $1,000

| Category | Cost | Purpose |
|----------|------|---------|
| Fine-tuning compute | $200 | 25-30 GPU hours on cloud |
| Inference compute | $100 | Model evaluation & testing |
| API calls | $600 | AWS Bedrock, OpenAI/Claude |
| Hosting | $70 | Web deployment |
| Domain | $30 | Custom domain (optional) |

---

## Troubleshooting

### Low WER Improvement in Phase 2
- **Try:** Increase LoRA rank (r=16 or r=32)
- **Try:** Add more training data
- **Try:** Adjust learning rate or warmup steps
- **Check:** Data quality and preprocessing

### CLIP Not Helping in Phase 3
- **Try:** Implement transcript normalization (remove fillers)
- **Try:** Fine-tune CLIP text encoder on conversational speech
- **Fallback:** Use semantic similarity of n-best candidates instead

### Streamlit App Slow in Phase 4
- **Try:** Reduce model loading with `@st.cache_resource`
- **Try:** Pre-compute embeddings offline
- **Try:** Use smaller Whisper model (tiny/base)

---

## Contributing

When implementing phases:
1. Create a branch: `git checkout -b phase-2-dev`
2. Make commits: `git commit -m "Phase 2: [specific change]"`
3. Push to branch: `git push origin phase-2-dev`
4. Create Pull Request with test results

---

## License

MIT License - See LICENSE file for details

---

## Contact & Mentorship

**Project Author:** Sidharth Bildikar  
**Mentorship:** MIT CSAIL Spoken Language Systems Group  
**Email:** [your-email@example.com]  
**LinkedIn:** [your-linkedin-profile]

---

## Citation

If you use this project, please cite:

```bibtex
@thesis{bildikar2026adi-o,
  title={Developing Robust ASR for Autism-Focused Reading Comprehension Therapy},
  author={Bildikar, Sidharth},
  school={Tesla STEM High School},
  year={2026},
  address={Redmond, Washington}
}
```

---

## Acknowledgments

- Google's Project Euphonia team for speech disorder dataset
- OpenAI for Whisper model
- MIT CSAIL for mentorship and guidance
- Dual-coding theory researchers and V&V therapy pioneers

---

**Last Updated:** January 26, 2026  
**Phase Status:** Phase 1 Complete | Phase 2 In Progress
