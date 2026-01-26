# Reusable Modules

This directory contains core components used across multiple phases.

## Files
- `asr.py` - Complete multimodal ASR pipeline (Whisper + LoRA + CLIP)
- `scaffolding.py` - LLM-based question generation for therapy
- `evaluation.py` - Response evaluation against gold standards
- `utils.py` - Helper functions (loading models, saving files, etc.)

## Usage Example

```python
from modules.asr import MultimodalASR
from modules.evaluation import evaluate_response

# Load pipeline
asr = MultimodalASR('models/whisper-lora', alpha=0.3)

# Transcribe with visual context
transcript, scored_hypotheses = asr.transcribe(
    'user_audio.wav', 
    'image.jpg'
)

# Evaluate response
evaluation = evaluate_response(
    transcript, 
    gold_standard_dict
)

print(f"Score: {evaluation['score']:.1%}")
print(f"Feedback: {evaluation['feedback']}")
```

## Module Dependencies
- Phase 2: `asr.py` (Whisper + LoRA)
- Phase 3: `asr.py` (+ CLIP integration)
- Phase 4: `asr.py`, `scaffolding.py`, `evaluation.py`
- Phase 5: All modules for final testing

See individual files for complete documentation.
