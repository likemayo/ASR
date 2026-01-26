# Phase 4: Therapy Application Prototype

## Overview
Build interactive Streamlit prototype integrating all components.

## Deliverables
- [ ] Working Streamlit app
- [ ] Image display + audio input
- [ ] Transcription + feedback
- [ ] Scaffolding questions
- [ ] Response evaluation
- [ ] Pilot testing results

## Files in This Directory
- `app.py` - Main Streamlit application
- `integration_tests.py` - End-to-end testing
- `config.py` - App configuration

## Running the App
```bash
pip install -r requirements.txt
cp .env.template .env
# Edit .env with your API keys

streamlit run phase_4/app.py
```

## App Flow
1. Display image
2. Ask user to describe it
3. Record and transcribe speech
4. Provide encouraging feedback
5. Ask scaffolding questions for missing elements
6. Track progress

## Checkpoint
- **Success Criteria:** Prototype stable and responsive
- **If Failed:** Simplify UI, reduce features, improve error handling

See `../IMPLEMENTATION_PLAN.md` for detailed UI code.
