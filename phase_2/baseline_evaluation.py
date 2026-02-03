"""
baseline_evaluation.py - Evaluate stock Whisper models on the TORGO dataset.

Tests multiple Whisper model sizes and reports WER broken down by
speech status (dysarthric vs. healthy).

Usage:
    python baseline_evaluation.py --input ./data/audio/torgo
    python baseline_evaluation.py --models tiny base small
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
from jiwer import wer, cer, process_words
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_from_disk, Audio


WHISPER_MODELS = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
}


def transcribe_audio(model, processor, audio_array: np.ndarray, sr: int, device: str) -> str:
    """Transcribe a single audio sample using Whisper."""
    input_features = processor(
        audio_array, sampling_rate=sr, return_tensors="pt"
    ).input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription.strip().lower()


def evaluate_model(model_name: str, model_id: str, dataset, device: str) -> dict:
    """Evaluate a single Whisper model on the test split."""
    print(f"\nEvaluating {model_name} ({model_id})...")

    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
    model.eval()

    # Collect results grouped by speech_status
    group_results = defaultdict(lambda: {"refs": [], "hyps": []})

    test_data = dataset["test"] if "test" in dataset else dataset[list(dataset.keys())[0]]
    total = len(test_data)

    for i, sample in enumerate(test_data):
        reference = sample.get("transcription", "")
        if not reference:
            continue
        reference = reference.strip().lower()

        status = sample.get("speech_status", "unknown")
        # Handle ClassLabel encoding (int -> string)
        if isinstance(status, int):
            status_map = {0: "dysarthria", 1: "healthy"}
            status = status_map.get(status, f"unknown_{status}")
        audio = sample["audio"]

        hypothesis = transcribe_audio(
            model, processor, audio["array"], audio["sampling_rate"], device
        )

        group_results[status]["refs"].append(reference)
        group_results[status]["hyps"].append(hypothesis)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{total} samples...")

    # Compute metrics
    report = {"model": model_name, "model_id": model_id, "groups": {}}

    all_refs, all_hyps = [], []

    for status, data in group_results.items():
        refs, hyps = data["refs"], data["hyps"]
        all_refs.extend(refs)
        all_hyps.extend(hyps)

        group_wer = wer(refs, hyps)
        group_cer = cer(refs, hyps)

        # Error type breakdown
        output = process_words(refs, hyps)
        report["groups"][status] = {
            "wer": group_wer,
            "cer": group_cer,
            "substitutions": output.substitutions,
            "deletions": output.deletions,
            "insertions": output.insertions,
            "num_samples": len(refs),
        }

    # Overall metrics
    if all_refs:
        report["overall_wer"] = wer(all_refs, all_hyps)
        report["overall_cer"] = cer(all_refs, all_hyps)
        report["total_samples"] = len(all_refs)

    print(f"  Done. Overall WER: {report.get('overall_wer', 0)*100:.1f}%")
    return report


def print_report(reports: list[dict]):
    """Print formatted comparison of all models."""
    print("\n" + "=" * 70)
    print("BASELINE EVALUATION RESULTS")
    print("=" * 70)

    # Model comparison
    print(f"\n{'Model':>10} {'Overall WER':>12} {'Overall CER':>12} {'Dysarthric':>12} {'Healthy':>12}")
    print("-" * 60)
    for r in reports:
        dys_wer = r.get("groups", {}).get("dysarthria", {}).get("wer", 0)
        healthy_wer = r.get("groups", {}).get("healthy", {}).get("wer", 0)
        print(
            f"{r['model']:>10} "
            f"{r.get('overall_wer', 0)*100:>11.1f}% "
            f"{r.get('overall_cer', 0)*100:>11.1f}% "
            f"{dys_wer*100:>11.1f}% "
            f"{healthy_wer*100:>11.1f}%"
        )

    # Best model
    best = min(reports, key=lambda r: r.get("overall_wer", float("inf")))
    print(f"\nBest model: {best['model']} (WER: {best.get('overall_wer', 0)*100:.1f}%)")

    # Per-group details for best model
    print(f"\n{'Per-Group Details (best model: ' + best['model'] + ')':=^70}")
    print(f"  {'Group':<14} {'WER':>8} {'CER':>8} {'Samples':>8} {'Sub':>6} {'Del':>6} {'Ins':>6}")
    print("  " + "-" * 56)
    for group, data in sorted(best["groups"].items()):
        print(
            f"  {group:<14} "
            f"{data['wer']*100:>7.1f}% "
            f"{data['cer']*100:>7.1f}% "
            f"{data['num_samples']:>8} "
            f"{data['substitutions']:>6} "
            f"{data['deletions']:>6} "
            f"{data['insertions']:>6}"
        )

    # WER gap
    dys = best["groups"].get("dysarthria", {}).get("wer", 0)
    healthy = best["groups"].get("healthy", {}).get("wer", 0)
    if dys and healthy:
        print(f"\n  WER gap (dysarthric - healthy): {(dys - healthy)*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Evaluate stock Whisper on TORGO")
    parser.add_argument(
        "--models", nargs="+", default=["tiny", "base", "small"],
        choices=list(WHISPER_MODELS.keys()),
        help="Whisper model sizes to evaluate",
    )
    parser.add_argument("--input", type=str, default="./data/audio/torgo", help="Path to local dataset")
    parser.add_argument("--output", type=str, default="./baseline_results.json", help="Output JSON path")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset_path = Path(args.input) / "torgo_dataset"
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(str(dataset_path))
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    reports = []
    for model_name in args.models:
        report = evaluate_model(model_name, WHISPER_MODELS[model_name], dataset, device)
        reports.append(report)

    print_report(reports)

    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(reports, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
