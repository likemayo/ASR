# Jupyter Notebooks

Exploration and learning notebooks for each phase:
- `phase_1_exploration.ipynb` - Image dataset exploration
- `phase_2_audio_exploration.ipynb` - Audio preprocessing walkthrough
- `phase_3_clip_exploration.ipynb` - Understanding CLIP embeddings
- `phase_4_prototype_demo.ipynb` - Interactive prototype demo

## Usage

```bash
jupyter notebook phase_1_exploration.ipynb
```

## Notebooks vs Scripts

**Notebooks are for:**
- Learning and experimentation
- Visualizing intermediate results
- Quick prototyping

**Scripts are for:**
- Production code (Phase 2-4)
- Training and evaluation
- Reproducible pipelines

Use notebooks to understand, scripts to implement.

## Running in Google Colab

```python
# Mount drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repo
!git clone https://github.com/yourusername/adi-o.git
%cd adi-o
!pip install -r requirements.txt
```

Then run notebooks cell by cell.
