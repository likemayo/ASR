# ADI/O Git Repository - Quick Start

## ✅ Repository Initialized

Your ADI/O project is now a fully initialized Git repository!

### What Was Created

```
adio/
├── README.md                    # Main project documentation
├── IMPLEMENTATION_PLAN.md       # Detailed phase-by-phase guide with code
├── requirements.txt             # Python dependencies
├── .gitignore                   # Excludes large files, models, data
├── .env.template               # API key template (copy to .env)
│
├── phase_1/                    # Phase 1: Imagery Dataset
│   └── README.md
├── phase_2/                    # Phase 2: Audio & ASR Modeling
│   └── README.md
├── phase_3/                    # Phase 3: Multimodal Integration
│   └── README.md
├── phase_4/                    # Phase 4: Prototype
│   └── README.md
│
├── modules/                    # Reusable Python modules
│   └── README.md
├── data/                       # Images, audio, metadata
│   ├── images/
│   ├── audio/
│   └── README.md
├── models/                     # Trained models (LoRA, embeddings)
│   └── README.md
├── notebooks/                  # Jupyter notebooks for learning
│   └── README.md
├── tests/                      # Unit & integration tests
│   └── README.md
│
└── .git/                       # Git repository (hidden)
```

### First Commit
- **Commit Hash:** `2bd1b32`
- **Files:** 14 (structure + documentation)
- **Message:** "Initial project setup: Structure, documentation, and planning for ADI/O ASR project"

---

## 🚀 Next Steps for Your Student

### 1. Clone/Setup (if on another machine)
```bash
cd /path/to/project
git clone /Users/wuyanlin/Documents/ASR adi-o
cd adi-o
```

### 2. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.template .env
# Edit .env with API keys
```

### 3. Start Phase 1 (Already Complete - Review)
```bash
cd phase_1
# Read README.md
# Review image dataset structure
```

### 4. Begin Phase 2 Implementation
```bash
cd ../phase_2
# Create audio_preprocessing.py
# Create baseline_evaluation.py
# Create lora_finetuning.py
```

---

## 📋 Git Workflow for Student

### For Each Phase

```bash
# 1. Create a branch for the phase
git checkout -b phase-2-dev

# 2. Work on implementation
# Edit phase_2/audio_preprocessing.py
# Add new functions to modules/asr.py
# Create tests in tests/test_asr_module.py

# 3. Track progress with commits
git add phase_2/audio_preprocessing.py
git commit -m "Phase 2: Implement audio normalization and augmentation"

git add modules/asr.py
git commit -m "Phase 2: Add baseline WER evaluation function"

# 4. When phase is complete, push branch
git push origin phase-2-dev

# 5. Create Pull Request (for review/documentation)
# Then merge back to main
git checkout main
git merge phase-2-dev
git push origin main
```

### Example Commit Messages
```
Phase 2: Implement audio preprocessing pipeline
Phase 2: Add baseline Whisper evaluation with WER calculation
Phase 2: Configure LoRA fine-tuning (r=8, rank adaptation)
Phase 2: Document WER improvement (31% → 25%)
Phase 3: Generate CLIP embeddings for all 100 images
Phase 3: Implement multimodal rescoring with cosine similarity
Phase 3: Grid search for optimal fusion coefficient (alpha=0.3)
Phase 4: Build Streamlit prototype with image display
Phase 4: Integrate multimodal ASR pipeline into app
Phase 4: Add automated feedback logic using gold standards
```

---

## 🔍 Viewing Progress

### Check Git History
```bash
git log --oneline
git log --oneline phase_2/
git log --pretty=fuller
```

### View Changes
```bash
git diff                    # Unstaged changes
git diff --staged           # Staged changes
git diff main..phase-2-dev  # Difference between branches
```

### See What Changed in a Commit
```bash
git show 2bd1b32           # Full details
git show --name-only 2bd1b32  # Just file names
```

---

## 📚 Key Files to Review

| File | Purpose |
|------|---------|
| `README.md` | Project overview, architecture, timeline |
| `IMPLEMENTATION_PLAN.md` | Detailed code & procedures for each phase |
| `requirements.txt` | All Python dependencies |
| `phase_*/README.md` | Phase-specific instructions |
| `modules/README.md` | Documentation for reusable components |

---

## ⚙️ Important Configuration

### .env File
Your student should create this after cloning:
```bash
cp .env.template .env
```

Then edit with:
- OpenAI API key (for scaffolding in Phase 4)
- AWS credentials (if using compute)
- Project paths (should auto-work, but can override)

### .gitignore
Already configured to ignore:
- `__pycache__/` (Python cache)
- `.env` (sensitive keys)
- Large data files (`images/*.jpg`, `audio/*.wav`)
- Model files (`models/*.bin`, `*.safetensors`)
- Temp files and notebooks (`*.ipynb_checkpoints`)

**Good practice:** Large files stay local, reference them via metadata.json in Git

---

## 💡 Teaching Tips

### Use Branches for Organization
```bash
# Main branch stays stable
git checkout -b phase-2-audio-preprocessing
git checkout -b phase-2-lora-finetuning
git checkout -b phase-3-clip-integration
```

### Weekly Checkpoints
```bash
# After each week, commit progress
git add .
git commit -m "Week X: [Brief summary of work done]"
git tag week-X   # Mark important milestones
```

### Code Review
```bash
# Before merging to main, review changes
git log main..phase-2-dev
git diff main phase-2-dev | head -100
```

---

## 🎯 Phase Checkpoints

Track progress by looking at commits per phase:

```bash
# How many commits in Phase 2?
git log --oneline phase_2/

# What was done in Phase 3 this week?
git log --since="1 week ago" --oneline phase_3/

# Compare progress between phases
git log --all --oneline -- phase_*/
```

---

## 🔧 Troubleshooting

### "I made a mistake, how do I undo?"

```bash
# Undo unstaged changes
git checkout -- phase_2/audio_preprocessing.py

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1

# View what was lost
git reflog
```

### "I want to see what I changed since yesterday"

```bash
git log --since="1 day ago" --oneline
git log --until="1 day ago" --oneline
```

### "How do I sync changes with another machine?"

```bash
# On original machine
git push origin main

# On another machine
git pull origin main
```

---

## 📖 Git Resources for Student

- [Git Tutorial](https://www.atlassian.com/git/tutorials)
- [GitHub's Git Handbook](https://guides.github.com/introduction/git-handbook/)
- [Oh My Git! Interactive Learning](https://ohmygit.org/)

---

## Next Meeting Checklist

- [ ] Student clones repository
- [ ] Virtual environment set up
- [ ] Dependencies installed
- [ ] .env file created (with mock values)
- [ ] Student makes a test commit
- [ ] Review README.md and IMPLEMENTATION_PLAN.md together

---

**Status:** ✅ Repository ready for Phase 2  
**Last Updated:** January 26, 2026  
**Commit:** 2bd1b32
