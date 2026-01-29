"""
Phase 1: Image Curator Tool - CSV Manual Entry + Prompts
Manages curated image dataset with metadata, prompts (questions), and gold answers.

USAGE:
  As a library:
    from image_curator import ImageCurator
    curator = ImageCurator(data_dir="data")
    curator.load_from_csv("images.csv")
    curator.save_metadata()
    curator.save_prompts()
    curator.save_gold_answers()
  
  As a CLI tool:
    python image_curator.py --load-csv images.csv
    python image_curator.py --validate
    python image_curator.py --status
"""

import json
import os
import argparse
import sys
from pathlib import Path
from PIL import Image
import csv
from typing import Dict, List, Optional


class ImageMetadata:
    """Represents metadata for a single image."""
    
    def __init__(self, image_id: str, file_path: str):
        self.image_id = image_id
        self.file_path = file_path
        self.scene_type = None  # indoor, outdoor, activity
        self.complexity_level = None  # 1-3
        self.primary_subjects = []  # e.g., ["child", "dog"]
        self.actions = []  # e.g., ["playing", "running"]
        self.colors = []  # e.g., ["blue", "green"]
        self.setting = None  # e.g., "park", "beach"
        self.structure_words = {
            "what": "",
            "who": "",
            "where": "",
            "color": "",
            "size": "",
            "mood": ""
        }
        self.gold_standard = None  # Complete description
    
    def to_dict(self) -> Dict:
        """Convert metadata to dictionary for JSON storage."""
        return {
            "image_id": self.image_id,
            "file_path": self.file_path,
            "scene_type": self.scene_type,
            "complexity_level": self.complexity_level,
            "primary_subjects": self.primary_subjects,
            "actions": self.actions,
            "colors": self.colors,
            "setting": self.setting,
            "structure_words": self.structure_words,
            "gold_standard": self.gold_standard
        }
    
    @staticmethod
    def from_dict(data: Dict) -> 'ImageMetadata':
        """Create metadata object from dictionary."""
        metadata = ImageMetadata(data["image_id"], data["file_path"])
        metadata.scene_type = data.get("scene_type")
        metadata.complexity_level = data.get("complexity_level")
        metadata.primary_subjects = data.get("primary_subjects", [])
        metadata.actions = data.get("actions", [])
        metadata.colors = data.get("colors", [])
        metadata.setting = data.get("setting")
        metadata.structure_words = data.get("structure_words", {})
        metadata.gold_standard = data.get("gold_standard")
        return metadata


class Prompt:
    """Represents a question (prompt) tied to a structure word."""
    
    # Difficulty mapping: structure_word -> difficulty level
    DIFFICULTY_MAP = {
        "who": 1,
        "what": 2,
        "where": 1,
        "color": 1,
        "size": 2,
        "action": 2,
        "mood": 3
    }
    
    # Default prompt templates
    DEFAULT_TEMPLATES = {
        "who": "Who is in the image?",
        "what": "What do you see in the image?",
        "where": "Where is this scene taking place?",
        "color": "What colors are in the image?",
        "size": "How would you describe the size?",
        "action": "What is happening in the image?",
        "mood": "How does the scene make you feel?"
    }
    
    def __init__(self, prompt_id: str, image_id: str, structure_word: str, question: str = None, difficulty: int = None):
        self.prompt_id = prompt_id
        self.image_id = image_id
        self.structure_word = structure_word
        self.question = question or self.DEFAULT_TEMPLATES.get(structure_word.lower(), "")
        self.difficulty = difficulty or self.DIFFICULTY_MAP.get(structure_word.lower(), 1)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSONL export."""
        return {
            "prompt_id": self.prompt_id,
            "image_id": self.image_id,
            "structure_word": self.structure_word,
            "question": self.question,
            "difficulty": self.difficulty
        }


class GoldAnswer:
    """Represents gold standard answers for a prompt."""
    
    def __init__(self, prompt_id: str, answers: List[str]):
        self.prompt_id = prompt_id
        self.answers = answers
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSONL export."""
        return {
            "prompt_id": self.prompt_id,
            "answers": self.answers
        }


class ImageCurator:
    """Tool for managing the image dataset via CSV.
    
    Handles:
    - Image metadata (scene type, complexity, subjects, actions, colors, setting, structure words)
    - Prompts (questions tied to structure words)
    - Gold answers (correct answers for each prompt)
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.metadata_file = self.data_dir / "metadata.json"
        self.prompts_file = self.data_dir / "prompts.jsonl"
        self.answers_file = self.data_dir / "gold_answers.jsonl"
        self.images = {}  # Dict of image_id -> ImageMetadata
        self.prompts = {}  # Dict of prompt_id -> Prompt
        self.gold_answers = {}  # Dict of prompt_id -> GoldAnswer
        
        # Create directories if they don't exist
        self.images_dir.mkdir(parents=True, exist_ok=True)
    
    def add_image(self, image_id: str, file_path: str):
        """Add an image to the curator."""
        metadata = ImageMetadata(image_id, file_path)
        self.images[image_id] = metadata
    
    def set_scene_type(self, image_id: str, scene_type: str):
        """Set scene type (indoor/outdoor/activity)."""
        if image_id in self.images:
            self.images[image_id].scene_type = scene_type
    
    def set_complexity(self, image_id: str, complexity_level: int):
        """Set complexity level (1-3)."""
        if image_id in self.images:
            self.images[image_id].complexity_level = complexity_level
    
    def set_subjects(self, image_id: str, subjects: List[str]):
        """Set primary subjects."""
        if image_id in self.images:
            self.images[image_id].primary_subjects = subjects
    
    def set_actions(self, image_id: str, actions: List[str]):
        """Set actions."""
        if image_id in self.images:
            self.images[image_id].actions = actions
    
    def set_colors(self, image_id: str, colors: List[str]):
        """Set colors."""
        if image_id in self.images:
            self.images[image_id].colors = colors
    
    def set_setting(self, image_id: str, setting: str):
        """Set setting description."""
        if image_id in self.images:
            self.images[image_id].setting = setting
    
    def set_structure_words(self, image_id: str, structure_words: Dict[str, str]):
        """Set structure words (what/who/where/color/size/mood)."""
        if image_id in self.images:
            self.images[image_id].structure_words.update(structure_words)
    
    def set_gold_standard(self, image_id: str, gold_standard: str):
        """Set gold standard response."""
        if image_id in self.images:
            self.images[image_id].gold_standard = gold_standard
    
    def add_prompt(self, prompt: Prompt):
        """Add a prompt to the dataset."""
        self.prompts[prompt.prompt_id] = prompt
    
    def add_gold_answer(self, answer: GoldAnswer):
        """Add gold answers for a prompt."""
        self.gold_answers[answer.prompt_id] = answer
    
    def load_from_csv(self, csv_file: str = "image_input.csv"):
        """Load all image data, prompts, and answers from CSV file.
        
        CSV should have columns:
        image_id, filename, scene_type, complexity, subjects, actions, colors, setting,
        structure_what, structure_who, structure_where, structure_color, structure_size, structure_mood, gold_standard,
        question_who, question_what, question_where, question_color, question_size, question_action, question_mood,
        answer_who, answer_what, answer_where, answer_color, answer_size, answer_action, answer_mood
        
        Note: Use semicolons (;) to separate multiple items within a field
        Example: dog;child for multiple subjects
        
        Prompts/answers can be left blank to use defaults.
        """
        csv_path = self.data_dir / csv_file
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        loaded_count = 0
        error_count = 0
        
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            
            for row_num, row in enumerate(reader, start=2):
                try:
                    image_id = row['image_id'].strip()
                    
                    # Skip empty rows
                    if not image_id or not row['filename'].strip():
                        continue
                    
                    # Add image
                    self.add_image(image_id, f"data/images/{row['filename'].strip()}")
                    
                    # Set metadata fields
                    if row.get('scene_type', '').strip():
                        self.set_scene_type(image_id, row['scene_type'].strip())
                    
                    if row.get('complexity', '').strip():
                        self.set_complexity(image_id, int(row['complexity'].strip()))
                    
                    if row.get('subjects', '').strip():
                        subjects = [s.strip() for s in row['subjects'].split(';')]
                        self.set_subjects(image_id, subjects)
                    
                    if row.get('actions', '').strip():
                        actions = [a.strip() for a in row['actions'].split(';')]
                        self.set_actions(image_id, actions)
                    
                    if row.get('colors', '').strip():
                        colors = [c.strip() for c in row['colors'].split(';')]
                        self.set_colors(image_id, colors)
                    
                    if row.get('setting', '').strip():
                        self.set_setting(image_id, row['setting'].strip())
                    
                    # Set structure words
                    structure_words = {}
                    if row.get('structure_what', '').strip():
                        structure_words['what'] = row['structure_what'].strip()
                    if row.get('structure_who', '').strip():
                        structure_words['who'] = row['structure_who'].strip()
                    if row.get('structure_where', '').strip():
                        structure_words['where'] = row['structure_where'].strip()
                    if row.get('structure_color', '').strip():
                        structure_words['color'] = row['structure_color'].strip()
                    if row.get('structure_size', '').strip():
                        structure_words['size'] = row['structure_size'].strip()
                    if row.get('structure_mood', '').strip():
                        structure_words['mood'] = row['structure_mood'].strip()
                    
                    if structure_words:
                        self.set_structure_words(image_id, structure_words)
                    
                    # Set gold standard
                    if row.get('gold_standard', '').strip():
                        self.set_gold_standard(image_id, row['gold_standard'].strip())
                    
                    # Load prompts and answers (one per structure word)
                    # Auto-generates prompts from structure_words if not provided in CSV
                    structure_word_types = ['who', 'what', 'where', 'color', 'size', 'action', 'mood']
                    
                    for sw_type in structure_word_types:
                        question_col = f'question_{sw_type}'
                        answer_col = f'answer_{sw_type}'
                        structure_col = f'structure_{sw_type}'  # Maps to structure_* columns
                        
                        # Create prompt if this structure word exists in the image
                        # (structure_words dict is populated above if any structure_* columns have data)
                        if sw_type in structure_words:
                            prompt_id = f"{image_id}_q_{sw_type}"
                            
                            # Use custom question if provided in CSV, otherwise None (Prompt will auto-generate)
                            question = row.get(question_col, '').strip()
                            if not question:
                                question = None
                            
                            # Create prompt (uses DEFAULT_TEMPLATES if question is None)
                            prompt = Prompt(prompt_id, image_id, sw_type.upper(), question)
                            self.add_prompt(prompt)
                            
                            # Add gold answers - first try custom answers, then fall back to structure_* value
                            if answer_col in row and row[answer_col].strip():
                                # Use custom answers if provided
                                answers = [a.strip() for a in row[answer_col].split(';')]
                                gold_answer = GoldAnswer(prompt_id, answers)
                                self.add_gold_answer(gold_answer)
                            elif structure_col in row and row[structure_col].strip():
                                # Use structure_* column value as auto-generated answer
                                # Split by semicolon if multiple values
                                answers = [a.strip() for a in row[structure_col].split(';')]
                                gold_answer = GoldAnswer(prompt_id, answers)
                                self.add_gold_answer(gold_answer)
                    
                    loaded_count += 1
                
                except Exception as e:
                    error_count += 1
        
        return loaded_count, error_count
    
    def save_metadata(self):
        """Save metadata to JSON file."""
        metadata_dict = {}
        for image_id, metadata in self.images.items():
            metadata_dict[image_id] = metadata.to_dict()
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
    
    def save_prompts(self):
        """Save prompts to JSONL file (one prompt per line)."""
        with open(self.prompts_file, 'w') as f:
            for prompt_id in sorted(self.prompts.keys()):
                prompt = self.prompts[prompt_id]
                f.write(json.dumps(prompt.to_dict()) + '\n')
    
    def save_gold_answers(self):
        """Save gold answers to JSONL file (one answer set per line)."""
        with open(self.answers_file, 'w') as f:
            for prompt_id in sorted(self.gold_answers.keys()):
                answer = self.gold_answers[prompt_id]
                f.write(json.dumps(answer.to_dict()) + '\n')
    
    def export_to_csv(self, csv_file: str = "metadata_export.csv"):
        """Export current metadata to CSV for review/editing."""
        csv_path = self.data_dir / csv_file
        
        with open(csv_path, 'w', newline='') as f:
            fieldnames = [
                'image_id', 'filename', 'scene_type', 'complexity', 
                'subjects', 'actions', 'colors', 'setting',
                'structure_what', 'structure_who', 'structure_where', 
                'structure_color', 'structure_size', 'structure_mood', 'gold_standard',
                'question_who', 'question_what', 'question_where', 'question_color', 'question_size', 'question_action', 'question_mood',
                'answer_who', 'answer_what', 'answer_where', 'answer_color', 'answer_size', 'answer_action', 'answer_mood'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for image_id, metadata in sorted(self.images.items()):
                row = {
                    'image_id': metadata.image_id,
                    'filename': Path(metadata.file_path).name,
                    'scene_type': metadata.scene_type or '',
                    'complexity': metadata.complexity_level or '',
                    'subjects': ';'.join(metadata.primary_subjects),
                    'actions': ';'.join(metadata.actions),
                    'colors': ';'.join(metadata.colors),
                    'setting': metadata.setting or '',
                    'structure_what': metadata.structure_words.get('what', ''),
                    'structure_who': metadata.structure_words.get('who', ''),
                    'structure_where': metadata.structure_words.get('where', ''),
                    'structure_color': metadata.structure_words.get('color', ''),
                    'structure_size': metadata.structure_words.get('size', ''),
                    'structure_mood': metadata.structure_words.get('mood', ''),
                    'gold_standard': metadata.gold_standard or '',
                }
                
                # Add prompts and answers
                for sw_type in ['who', 'what', 'where', 'color', 'size', 'action', 'mood']:
                    prompt_id = f"{image_id}_q_{sw_type}"
                    
                    if prompt_id in self.prompts:
                        row[f'question_{sw_type}'] = self.prompts[prompt_id].question
                    else:
                        row[f'question_{sw_type}'] = ''
                    
                    if prompt_id in self.gold_answers:
                        row[f'answer_{sw_type}'] = ';'.join(self.gold_answers[prompt_id].answers)
                    else:
                        row[f'answer_{sw_type}'] = ''
                
                writer.writerow(row)
    
    def validate_dataset(self) -> List[str]:
        """Validate that all images have complete metadata."""
        errors = []
        for image_id, metadata in self.images.items():
            if not metadata.scene_type:
                errors.append(f"{image_id}: Missing scene_type")
            if metadata.complexity_level is None:
                errors.append(f"{image_id}: Missing complexity_level")
            if not metadata.primary_subjects:
                errors.append(f"{image_id}: Missing subjects")
            if not metadata.actions:
                errors.append(f"{image_id}: Missing actions")
            if not metadata.colors:
                errors.append(f"{image_id}: Missing colors")
            if not metadata.setting:
                errors.append(f"{image_id}: Missing setting")
            if not metadata.gold_standard:
                errors.append(f"{image_id}: Missing gold_standard")
        return errors
    
    def print_status(self):
        """Print status of all images."""
        print(f"\n{'='*70}")
        print(f"Dataset Status: {len(self.images)} images, {len(self.prompts)} prompts, {len(self.gold_answers)} answers")
        print(f"{'='*70}")
        
        for image_id, metadata in sorted(self.images.items()):
            complete = all([
                metadata.scene_type,
                metadata.complexity_level is not None,
                metadata.primary_subjects,
                metadata.actions,
                metadata.colors,
                metadata.setting,
                metadata.gold_standard
            ])
            
            status = "✓" if complete else "✗"
            print(f"\n[{status}] {image_id}: {Path(metadata.file_path).name}")
            print(f"    Scene: {metadata.scene_type}")
            print(f"    Complexity: {metadata.complexity_level}")
            print(f"    Subjects: {', '.join(metadata.primary_subjects)}")
            print(f"    Setting: {metadata.setting}")
            
            # Count prompts for this image
            image_prompts = [p for p in self.prompts.values() if p.image_id == image_id]
            print(f"    Prompts: {len(image_prompts)}")


def main():
    """CLI interface for Phase 1 workflow."""
    parser = argparse.ArgumentParser(
        description="Phase 1: Image Curator - Load images from CSV"
    )
    
    parser.add_argument(
        "--load-csv",
        type=str,
        help="Load images from CSV file (default: image_input.csv)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate that all images have complete metadata"
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export current metadata to CSV"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show completion status"
    )
    
    args = parser.parse_args()
    
    # Initialize curator
    curator = ImageCurator(data_dir="data")
    
    # Load from CSV (default or if no args)
    if args.load_csv or (len(sys.argv) == 1):
        csv_file = args.load_csv or "image_input.csv"
        print(f"\n{'='*70}")
        print(f"Loading from CSV: {csv_file}")
        print(f"{'='*70}")
        try:
            loaded, errors = curator.load_from_csv(csv_file)
            print(f"\n✓ Successfully loaded {loaded} images")
            
            if errors > 0:
                print(f"⚠️  {errors} rows had errors")
            
            # Validate
            validation_errors = curator.validate_dataset()
            if validation_errors:
                print(f"\n⚠️  Validation found {len(validation_errors)} issues:")
                for error in validation_errors[:10]:
                    print(f"  - {error}")
            else:
                print(f"\n✓ All images have complete metadata!")
            
            # Save
            curator.save_metadata()
            print(f"✓ Saved to data/metadata.json")
            
            curator.save_prompts()
            print(f"✓ Saved to data/prompts.jsonl")
            
            curator.save_gold_answers()
            print(f"✓ Saved to data/gold_answers.jsonl")
            
            # Export CSV
            curator.export_to_csv()
            print(f"✓ Exported to data/metadata_export.csv")
            
            curator.print_status()
            
        except FileNotFoundError as e:
            print(f"❌ {str(e)}")
            return
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return
    
    # Validate mode
    if args.validate:
        errors = curator.validate_dataset()
        if errors:
            print(f"❌ Validation failed: {len(errors)} errors")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"✓ Validation passed!")
    
    # Export mode
    if args.export:
        curator.export_to_csv()
        print(f"✓ Exported to data/metadata_export.csv")
    
    # Status mode
    if args.status:
        curator.print_status()


if __name__ == "__main__":
    main()
