# Project Data

This directory contains:
- `images/` - Curated 100+ image dataset (Phase 1)
- `audio/` - Audio validation data (Phase 2)
- `metadata.json` - Complete image metadata and gold standards
- `embeddings.pkl` - Pre-computed CLIP embeddings (Phase 3)

## Directory Structure

```
data/
├── images/           # Phase 1 deliverable
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── audio/            # Phase 2 validation data
│   ├── speaker_1/
│   ├── speaker_2/
│   └── ...
├── metadata.json     # Main dataset file
└── embeddings.pkl    # CLIP cache
```

## Important Notes

- `.gitignore` excludes large image/audio files (too large for version control)
- Store media files locally; reference in version control via metadata.json
- Pre-computed embeddings should be cached after first generation
- All audio should be at 16kHz sampling rate for Whisper

## Data Format: metadata.json

```json
{
  "image_001": {
    "file_path": "data/images/image_001.jpg",
    "scene_type": "outdoor",
    "complexity_level": 2,
    "primary_subjects": ["child", "umbrella"],
    "actions": ["carrying", "walking"],
    "structure_words": {
      "what": "carrying an umbrella, walking",
      "who": "a young child",
      "where": "on a rainy street",
      ...
    }
  }
}
```

See `IMPLEMENTATION_PLAN.md` for details on data collection and format.
