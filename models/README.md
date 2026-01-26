# Models

This directory stores trained models and checkpoints:
- Fine-tuned Whisper (LoRA adapters)
- CLIP embeddings cache
- Model configuration files

## Structure

```
models/
├── whisper-lora/         # Phase 2 deliverable
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   ├── config.json
│   └── README.md
├── clip_config.json      # Phase 3
└── training_logs/        # Tensorboard logs (optional)
```

## Storage Notes

**Large Models:** Due to size constraints in Git:
- Whisper base model (~500MB) is downloaded at runtime
- LoRA adapters (~50MB) are stored here
- CLIP model (~700MB) is downloaded at runtime
- Only store final trained artifacts, not intermediate checkpoints

To save space:
```bash
# Only commit LoRA adapters, not base model
git add models/whisper-lora/adapter*
git add models/whisper-lora/config.json
```

## Loading Models

```python
from peft import PeftModel
from transformers import WhisperForConditionalGeneration

# Load base + LoRA adapter
base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model = PeftModel.from_pretrained(base_model, "models/whisper-lora")
```

See Phase 2 and Phase 4 for usage examples.
