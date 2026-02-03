"""
data_loader.py - Download and organize the TORGO database for ASR experiments.

The HuggingFace version (abnerh/TORGO-database) provides these columns:
  - audio: audio waveform
  - transcription: ground truth text
  - speech_status: "dysarthric" or "healthy"
  - gender: "M" or "F"
  - duration: audio length in seconds

Since individual speaker IDs are not available in this dataset version,
we split by speech_status and use stratified splitting to create train/val/test.

Usage:
    python data_loader.py                    # Download and save to default path
    python data_loader.py --output ./data    # Custom output directory
    python data_loader.py --info             # Print dataset info without downloading
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

from datasets import load_dataset, Audio, DatasetDict, ClassLabel


# Split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def load_torgo(sampling_rate: int = 16000):
    """Load the TORGO dataset from Hugging Face and cast audio to target sample rate."""
    print("Downloading TORGO database from Hugging Face...")
    dataset = load_dataset("abnerh/TORGO-database")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
    print(f"Loaded {sum(len(split) for split in dataset.values())} total samples.")
    return dataset


def get_speech_status(sample: dict) -> str:
    """Return 'dysarthric' or 'healthy' from the sample."""
    return sample.get("speech_status", "unknown")


def get_dataset_info(dataset) -> dict:
    """Compute summary statistics without decoding audio."""
    info = {
        "total_samples": 0,
        "by_status": defaultdict(int),
        "by_gender": defaultdict(int),
        "by_status_gender": defaultdict(int),
        "durations": defaultdict(list),
    }

    for split_name in dataset:
        # Use select_columns to avoid decoding audio
        subset = dataset[split_name].select_columns(
            ["transcription", "speech_status", "gender", "duration"]
        )
        for sample in subset:
            info["total_samples"] += 1
            status = sample.get("speech_status", "unknown")
            gender = sample.get("gender", "unknown")
            duration = sample.get("duration", 0)

            info["by_status"][status] += 1
            info["by_gender"][gender] += 1
            info["by_status_gender"][f"{status}_{gender}"] += 1
            info["durations"][status].append(duration)

    return info


def create_splits(dataset, seed: int = 42) -> DatasetDict:
    """Create stratified train/val/test splits based on speech_status.

    Since the HF dataset only has a 'train' split, we split it ourselves
    stratified by speech_status to keep dysarthric/healthy ratios consistent.
    """
    # The HF dataset comes as a single 'train' split
    full = dataset["train"] if "train" in dataset else dataset[list(dataset.keys())[0]]

    # Cast speech_status to ClassLabel so stratify works
    unique_statuses = sorted(set(full["speech_status"]))
    full = full.cast_column("speech_status", ClassLabel(names=unique_statuses))

    # First split: train+val vs test
    split1 = full.train_test_split(
        test_size=TEST_RATIO,
        seed=seed,
        stratify_by_column="speech_status",
    )

    # Second split: train vs val (from the train+val portion)
    val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    split2 = split1["train"].train_test_split(
        test_size=val_ratio_adjusted,
        seed=seed,
        stratify_by_column="speech_status",
    )

    return DatasetDict({
        "train": split2["train"],
        "val": split2["test"],
        "test": split1["test"],
    })


def save_dataset(dataset: DatasetDict, output_dir: Path):
    """Save the split dataset and metadata to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute split stats without decoding audio
    split_stats = {}
    for split_name in dataset:
        subset = dataset[split_name].select_columns(["speech_status"])
        status_counts = defaultdict(int)
        for sample in subset:
            status_counts[sample["speech_status"]] += 1
        split_stats[split_name] = {
            "total": len(dataset[split_name]),
            "by_status": dict(status_counts),
        }

    # Save split info
    splits_path = output_dir / "splits.json"
    with open(splits_path, "w") as f:
        json.dump({
            "ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
            "stats": split_stats,
        }, f, indent=2)
    print(f"Split info saved to {splits_path}")

    # Save dataset to disk
    dataset_path = output_dir / "torgo_dataset"
    dataset.save_to_disk(str(dataset_path))
    print(f"Dataset saved to {dataset_path}")

    return split_stats


def main():
    parser = argparse.ArgumentParser(description="Download and organize TORGO dataset")
    parser.add_argument(
        "--output", type=str, default="./data/audio/torgo",
        help="Output directory for processed dataset",
    )
    parser.add_argument(
        "--sampling-rate", type=int, default=16000,
        help="Target audio sampling rate (default: 16000)",
    )
    parser.add_argument(
        "--info", action="store_true",
        help="Print dataset info without saving to disk",
    )
    args = parser.parse_args()

    dataset = load_torgo(sampling_rate=args.sampling_rate)

    # Print dataset structure
    print(f"\nDataset structure: {dataset}")
    cols = dataset[list(dataset.keys())[0]].column_names
    print(f"Columns: {cols}")

    if args.info:
        info = get_dataset_info(dataset)
        print(f"\nTotal samples: {info['total_samples']}")

        print("\nBy speech status:")
        for status, count in sorted(info["by_status"].items()):
            durs = info["durations"][status]
            total_hrs = sum(durs) / 3600
            mean_dur = sum(durs) / len(durs) if durs else 0
            print(f"  {status:12s}: {count:>6} samples, {total_hrs:.2f}h total, {mean_dur:.2f}s avg duration")

        print("\nBy gender:")
        for gender, count in sorted(info["by_gender"].items()):
            print(f"  {gender}: {count}")

        print("\nBy status + gender:")
        for key, count in sorted(info["by_status_gender"].items()):
            print(f"  {key}: {count}")
        return

    # Create stratified splits and save
    print("\nCreating stratified train/val/test splits...")
    split_dataset = create_splits(dataset)

    for name in split_dataset:
        print(f"  {name}: {len(split_dataset[name])} samples")

    stats = save_dataset(split_dataset, Path(args.output))
    print(f"\nDone. Dataset organized in {args.output}")


if __name__ == "__main__":
    main()
