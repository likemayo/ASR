# Phase 1: Image Dataset Curation - CSV Based

## Objective
Curate 100 images with complete metadata + questions + answers for autism therapy reading comprehension training.

## Quick Start (3 Steps)

1. **Create CSV file** with image metadata + prompts + answers (see CSV Format section)
2. **Put images in folder** `phase_1/data/images/`
3. **Load CSV**
   ```bash
   python image_curator.py --load-csv your_images.csv
   ```

**Output:** 3 files in `phase_1/data/`:
- `metadata.json` - Image metadata
- `prompts.jsonl` - Questions (one per line)
- `gold_answers.jsonl` - Correct answers (one per line)

## Workflow Details

### Step 1: Create & Fill CSV

Create a spreadsheet file (Excel, Google Sheets, Numbers, etc.) with **29 columns** total (9 metadata + 6 structure + 7 prompts + 5 answers). Here's what each column contains:

#### Metadata Columns (9 required)
1. **image_id**: 001, 002, ... 100 (unique identifier)
2. **filename**: image_001.jpg, image_002.jpg, etc. (must exist in `data/images/`)
3. **scene_type**: indoor, outdoor, or activity
4. **complexity**: 1 (simple), 2 (medium), or 3 (complex)
5. **subjects**: child;book;dog (semicolon-separated subjects)
6. **actions**: reading;sitting (semicolon-separated actions)
7. **colors**: red;blue (semicolon-separated prominent colors)
8. **setting**: bedroom, park, classroom, etc. (location/context)
9. **gold_standard**: Brief description of ideal student response

#### Structure Columns (6 - these generate questions)
10. **structure_what**: Description of what's in the image
11. **structure_who**: Description of who is in the image
12. **structure_where**: Description of where the scene is
13. **structure_color**: Which colors are prominent
14. **structure_size**: Size description (big, small, medium, etc.)
15. **structure_mood**: Emotional tone or mood

#### Prompt Columns (7 - optional, auto-generated if blank)
16. **question_who**: "Who is in the image?" (auto-generated if blank)
17. **question_what**: "What is the child doing?" (auto-generated if blank)
18. **question_where**: "Where is the scene?" (auto-generated if blank)
19. **question_color**: "What colors do you see?" (auto-generated if blank)
20. **question_size**: "Is the object big or small?" (auto-generated if blank)
21. **question_action**: "What action is happening?" (auto-generated if blank)
22. **question_mood**: "What mood does this convey?" (auto-generated if blank)

#### Answer Columns (5 - optional, can be blank)
23. **answer_who**: Comma-separated answers, e.g., "Child, boy, person"
24. **answer_what**: Comma-separated answers, e.g., "Reading, studying, looking at book"
25. **answer_where**: Comma-separated answers, e.g., "Bedroom, indoors, home"
26. **answer_color**: Comma-separated answers, e.g., "Blue, blue and white"
27. **answer_size**: Comma-separated answers
28. **answer_action**: Comma-separated answers
29. **answer_mood**: Comma-separated answers

**Pro Tip:** Fill in the structure columns (columns 10-15) with descriptions. The system will auto-generate questions. Only customize columns 16-22 if you want different questions than the defaults. Answer columns (23-29) are optional.

Save as CSV and copy to `phase_1/data/` folder

### Step 2: Load CSV into Curator

```bash
cd phase_1
python image_curator.py --load-csv your_images.csv
```

This automatically:
- ✓ Loads all images and metadata from CSV
- ✓ Auto-generates questions from `structure_words` if prompt columns are blank
- ✓ Validates all required data is complete
- ✓ Creates THREE output files:

**Output files in `phase_1/data/`:**
1. **metadata.json** - Complete image metadata (one JSON object per image)
2. **prompts.jsonl** - All questions (one JSON per line, 6 questions × 100 images = 600 prompts)
3. **gold_answers.jsonl** - Correct answers (one JSON per line, matching prompts)

## CSV Format Details

### Important: Use Semicolons for Multiple Items

When an image has multiple subjects, actions, or colors, **separate with semicolons** (`;`) not commas:

| Field | Example |
|-------|---------|
| subjects | dog;dog;child |
| actions | playing;running |
| colors | green;brown;red |

This is because CSV uses commas to separate columns. Semicolons separate items within a field.

### Example Row

```
001,image_001.jpg,outdoor,2,dog;child,playing with a ball,green;brown,park,a child and a dog,a boy and his pet,in a sunny park,green grass and trees,medium-sized dog,happy,A child throws a ball and plays with a dog in the park
```

## Output Format Details

### metadata.json

Each image becomes a JSON object containing all 10 metadata fields:

```json
{
  "image_id": "001",
  "file_path": "data/images/image_001.jpg",
  "scene_type": "outdoor",
  "complexity_level": 2,
  "primary_subjects": "dog;child",
  "actions": "playing with a ball",
  "colors": "green;brown",
  "setting": "park",
  "structure_words": "WHO;WHAT;COLOR",
  "gold_standard": "A child throws a ball and plays with a dog in the park"
}
```

### prompts.jsonl

Each line is a complete JSON question object:

```json
{"prompt_id":"001_q_who","image_id":"001","structure_word":"WHO","question":"Who is in the image?","difficulty":1}
{"prompt_id":"001_q_what","image_id":"001","structure_word":"WHAT","question":"What is the child doing?","difficulty":2}
{"prompt_id":"001_q_color","image_id":"001","structure_word":"COLOR","question":"What colors do you see?","difficulty":1}
```

**Prompt ID format:** `{image_id}_q_{structure_word_lowercase}` (e.g., `001_q_who`)

**Difficulty levels:**
- WHO, WHERE, COLOR = 1 (easy - visual identification)
- WHAT, SIZE, ACTION = 2 (medium - understanding activity)
- MOOD = 3 (hard - emotional interpretation)

### gold_answers.jsonl

Each line maps a prompt_id to acceptable answers:

```json
{"prompt_id":"001_q_who","answers":["A child","A boy","A kid","A child and a dog"]}
{"prompt_id":"001_q_what","answers":["Playing with a ball","Throwing a ball","Playing","Throwing"]}
{"prompt_id":"001_q_color","answers":["Green and brown","Green","Brown and green"]}
```

**Note:** If you leave answer columns blank in CSV, this file won't contain entries for those prompts.



| Command | Purpose |
|---------|---------|
| `python image_curator.py --load-csv file.csv` | Load images from CSV |
| `python image_curator.py --validate` | Validate all 100 images have complete metadata |
| `python image_curator.py --export` | Export metadata to CSV for review |
| `python image_curator.py --status` | Show completion status |

## Outputs

After running `--load-csv` (all files are automatically created):

1. **data/metadata.json**
   - Format: One JSON object per line
   - Contains: All 10 metadata fields per image
   - Size: ~100 bytes per image (10 KB for 100 images)
   - Used by: Phase 2 for training context

2. **data/prompts.jsonl**
   - Format: One JSON per line (JSONL format)
   - Contains: 6 questions × 100 images = 600 prompts total
   - Each prompt has: prompt_id, image_id, structure_word, question, difficulty
   - Size: ~150 bytes per prompt (90 KB for 600 prompts)
   - Used by: Phase 2 as training/test questions

3. **data/gold_answers.jsonl**
   - Format: One JSON per line (JSONL format)
   - Contains: Correct answer variations for each prompt
   - Each answer has: prompt_id, answers (list of acceptable responses)
   - Size: ~100 bytes per prompt (60 KB for 600 prompts)
   - Used by: Phase 2 for answer validation

## Project Files

| File/Folder | Purpose |
|------|---------|
| `image_curator.py` | Main tool: CSV loading + metadata/prompts/answers management + CLI |
| `Phase_1_Tutorial.ipynb` | Student-friendly tutorial showing complete workflow |
| `README.md` | This guide |
| `data/images/` | Put your 100 images here (JPEG or PNG) |
| `data/` | Output folder (metadata.json, prompts.jsonl, gold_answers.jsonl created here) |

## Troubleshooting

**Q: "Image not found" error**
- A: Check filename in CSV matches actual file in `data/images/`

**Q: "Invalid complexity value"**
- A: Complexity must be 1, 2, or 3 only

**Q: Missing values error**
- A: All metadata fields must be filled (no empty cells in columns 1-10)

**Q: No prompts.jsonl or gold_answers.jsonl created**
- A: Make sure you have `structure_words` filled in CSV. Prompts are auto-generated from structure_words.

**Q: Prompts are auto-generated, but I want custom questions**
- A: Fill in the `question_who`, `question_what`, etc. columns in your CSV with custom text

**Q: I filled in answers but they're not in gold_answers.jsonl**
- A: Make sure you filled the `answer_who`, `answer_what`, etc. columns (separate multiple answers with commas)

**Q: Want to edit after loading**
- A: Edit your CSV again, save, then run `--load-csv` again (it will reload everything)

**Q: CSV not found**
- A: Make sure your CSV is in the `phase_1/data/` folder, or specify full path

## Success Criteria

✓ 100 images with complete metadata (all 10 metadata columns filled)  
✓ All images have `structure_words` defined (WHO, WHAT, WHERE, COLOR, SIZE, ACTION, or MOOD)  
✓ Questions auto-generated for each structure_word (or custom questions if provided)  
✓ Answers provided for key questions (optional - can be auto-generated)  
✓ **Three output files created successfully:**
  - ✓ `data/metadata.json` - All image metadata
  - ✓ `data/prompts.jsonl` - All 600 questions (6 per image)
  - ✓ `data/gold_answers.jsonl` - Answer variations for each question
✓ All JSONL lines are valid JSON (can be parsed)  
✓ Ready for Phase 2: Question Answering VQA Model Training

## Next Steps After Phase 1

1. Save metadata.json to version control
2. Create Git tag: `git tag -a phase-1-complete -m "100 images curated"`
3. Proceed to Phase 2
