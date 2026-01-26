# Tests

Unit and integration tests for project components.

## Test Structure

```
tests/
├── test_audio_preprocessing.py    # Phase 2
├── test_asr_module.py             # Phase 2-3
├── test_clip_embeddings.py        # Phase 3
├── test_evaluation.py             # Phase 4
├── test_scaffolding.py            # Phase 4
└── test_integration.py            # End-to-end
```

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_asr_module.py

# Run with coverage
pytest --cov=modules tests/
```

## Writing Tests

Example:
```python
import pytest
from modules.asr import MultimodalASR

def test_multimodal_asr_initialization():
    asr = MultimodalASR('models/whisper-lora')
    assert asr is not None
    assert asr.device in ['cpu', 'cuda']

def test_transcription_output():
    asr = MultimodalASR('models/whisper-lora')
    transcript = asr.transcribe('test_audio.wav')
    assert isinstance(transcript, str)
    assert len(transcript) > 0
```

## Continuous Integration

As you progress, add tests to prevent regressions:
- After Phase 2: Test ASR module
- After Phase 3: Test multimodal integration
- After Phase 4: Test end-to-end prototype

Each phase should maintain ≥80% test coverage.
