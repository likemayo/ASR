# Phase 1: Imagery Dataset and Visual Framework

## Overview
Build a curated set of 100+ high-quality images with metadata, structure word mappings, and gold-standard descriptions.

## Deliverables
- [ ] 100+ curated images
- [ ] `image_metadata.json` with complete mappings
- [ ] Gold-standard descriptions for each image
- [ ] Validation report

## Files in This Directory
- `image_curator.py` - Tools for organizing and tagging images
- `structure_word_mapper.py` - Create structure word mappings
- `validation_check.py` - Validate dataset quality
- `gold_standard_generator.py` - Generate gold-standard descriptions using LLM

## Getting Started
1. Download 100+ images from royalty-free sources (Unsplash, Pexels)
2. Organize into `data/images/` directory
3. Run `python image_curator.py` to tag images
4. Generate gold-standard descriptions with `python gold_standard_generator.py`
5. Validate with `python validation_check.py`

See `../IMPLEMENTATION_PLAN.md` for detailed instructions.
