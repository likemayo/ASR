# Phase 3: Multimodality Development

## Overview
Integrate CLIP for visual context rescoring of ASR hypotheses.

## Deliverables
- [ ] CLIP embeddings for all images
- [ ] Multimodal rescoring pipeline
- [ ] Optimal fusion coefficient (alpha)
- [ ] Documented multimodal improvements

## Files in This Directory
- `clip_embeddings.py` - Generate and cache CLIP embeddings
- `multimodal_asr.py` - Complete ASR + CLIP pipeline
- `fusion_tuning.py` - Grid search for optimal alpha
- `transcript_normalization.py` - Preprocess transcripts for CLIP
- `analysis.py` - Analyze where multimodal helps vs hurts

## Critical Steps
1. Pre-compute CLIP image embeddings
2. Generate n-best hypotheses from Whisper
3. Implement cosine similarity rescoring
4. Tune fusion coefficient with grid search
5. Analyze results

## Checkpoint
- **Success Criteria:** Multimodal helps ≥ 50% of test cases
- **If Failed:** Implement transcript normalization, consider fine-tuning CLIP

## Known Issue
CLIP's text encoder is built for captions, not conversational speech. May need preprocessing or fine-tuning.

See `../IMPLEMENTATION_PLAN.md` for detailed code examples.
