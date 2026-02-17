"""
audio_preprocessing.py - Clean, normalize, and augment TORGO audio for Whisper fine-tuning.

Usage:
    python audio_preprocessing.py --input ./data/audio/torgo --output ./data/audio/torgo_processed
"""

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from datasets import load_from_disk, Audio


def resample(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """Resample audio to target sample rate."""
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def normalize_amplitude(audio: np.ndarray) -> np.ndarray:
    """Normalize audio amplitude to [-1, 1]."""
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio


def trim_silence(audio: np.ndarray, sr: int, top_db: int = 40) -> np.ndarray:
    """Trim leading and trailing silence."""
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


def is_valid_duration(audio: np.ndarray, sr: int, min_sec: float = 0.5, max_sec: float = 30.0) -> bool:
    """Check if audio duration is within acceptable range."""
    duration = len(audio) / sr
    return min_sec <= duration <= max_sec


def augment_speed(audio: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """Apply speed perturbation (time-stretch without pitch change)."""
    return librosa.effects.time_stretch(audio, rate=rate)


def augment_pitch(audio: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """Shift pitch by n_steps semitones."""
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def augment_noise(audio: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """Add Gaussian noise at a given SNR."""
    signal_power = np.mean(audio ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    return audio + noise


def preprocess_sample(audio: np.ndarray, sr: int, target_sr: int = 16000) -> np.ndarray | None:
    """Apply full preprocessing pipeline to a single audio sample.

    Returns None if the sample should be filtered out.
    """
    audio = resample(audio, sr, target_sr)
    audio = normalize_amplitude(audio)
    audio = trim_silence(audio, target_sr)

    if not is_valid_duration(audio, target_sr):
        return None

    return audio


def augment_sample(audio: np.ndarray, sr: int) -> list[tuple[str, np.ndarray]]:
    """Generate augmented versions of a single audio sample.

    Returns list of (suffix, augmented_audio) tuples.
    """
    augmented = []

    # Speed perturbation
    augmented.append(("speed09", augment_speed(audio, sr, rate=0.9)))
    augmented.append(("speed11", augment_speed(audio, sr, rate=1.1)))

    # Background noise
    augmented.append(("noise20", augment_noise(audio, snr_db=20.0)))
    augmented.append(("noise15", augment_noise(audio, snr_db=15.0)))

    # Pitch shifting
    augmented.append(("pitch_down2", augment_pitch(audio, sr, n_steps=-2)))
    augmented.append(("pitch_up2", augment_pitch(audio, sr, n_steps=2)))

    return augmented


def process_dataset(input_dir: Path, output_dir: Path, target_sr: int = 16000, do_augment: bool = True):
    """Process the full TORGO dataset: preprocess and optionally augment.

    Expects the dataset saved via data_loader.py at input_dir/torgo_dataset.
    """
    dataset_path = input_dir / "torgo_dataset"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "Run data_loader.py first to download the dataset."
        )

    dataset = load_from_disk(str(dataset_path))
    dataset = dataset.cast_column("audio", Audio(sampling_rate=target_sr))

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load split info (for reference, not used for speaker filtering since
    # the TORGO HF dataset doesn't include individual speaker IDs)
    splits_path = input_dir / "splits.json"
    if splits_path.exists():
        with open(splits_path) as f:
            split_data = json.load(f)
            print(f"Dataset stats: {split_data.get('stats', {})}")

    stats = {"processed": 0, "filtered": 0, "augmented": 0}

    for split_name in dataset:
        print(f"\nProcessing split: {split_name}")
        split_dir = output_dir / split_name
        split_dir.mkdir(exist_ok=True)

        for i, sample in enumerate(dataset[split_name]):
            audio_data = sample["audio"]["array"]
            sr = sample["audio"]["sampling_rate"]

            processed = preprocess_sample(audio_data, sr, target_sr)
            if processed is None:
                stats["filtered"] += 1
                continue

            # Save processed audio
            out_path = split_dir / f"sample_{i:05d}.wav"
            sf.write(str(out_path), processed, target_sr)
            stats["processed"] += 1

            # Augment only training data
            if do_augment and split_name == "train":
                for suffix, aug_audio in augment_sample(processed, target_sr):
                    aug_path = split_dir / f"sample_{i:05d}_{suffix}.wav"
                    sf.write(str(aug_path), aug_audio, target_sr)
                    stats["augmented"] += 1

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1} samples...")

    print(f"\nDone. Processed: {stats['processed']}, Filtered: {stats['filtered']}, Augmented: {stats['augmented']}")

    # Save stats
    with open(output_dir / "preprocessing_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Preprocess TORGO audio for Whisper fine-tuning")
    parser.add_argument("--input", type=str, default="./data/audio/torgo", help="Input directory with TORGO dataset")
    parser.add_argument("--output", type=str, default="./data/audio/torgo_processed", help="Output directory")
    parser.add_argument("--sr", type=int, default=16000, help="Target sampling rate")
    parser.add_argument("--no-augment", action="store_true", help="Skip data augmentation")
    args = parser.parse_args()

    process_dataset(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        target_sr=args.sr,
        do_augment=not args.no_augment,
    )


if __name__ == "__main__":
    main()
