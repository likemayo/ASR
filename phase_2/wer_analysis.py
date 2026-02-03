"""
wer_analysis.py - Compare baseline vs. fine-tuned Whisper on TORGO test set.

Generates a detailed WER report with per-speaker, per-severity breakdowns
and sample transcription comparisons.

Usage:
    python wer_analysis.py --baseline baseline_results.json --model ../models/whisper-lora-torgo/final
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
from jiwer import wer, cer, compute_measures
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

from data_loader import SPEAKER_INFO, load_torgo, get_speaker_id, assign_split


def load_finetuned_model(model_path: str, device: str):
    """Load the LoRA fine-tuned Whisper model."""
    processor = WhisperProcessor.from_pretrained(model_path)
    base_model = WhisperForConditionalGeneration.from_pretrained(model_path)
    # If saved as a PEFT model, load adapter
    try:
        model = PeftModel.from_pretrained(base_model, model_path)
    except Exception:
        model = base_model
    model = model.to(device)
    model.eval()
    return model, processor


def transcribe(model, processor, audio_array, sr, device):
    """Transcribe a single audio sample."""
    inputs = processor(audio_array, sampling_rate=sr, return_tensors="pt").input_features.to(device)
    with torch.no_grad():
        pred_ids = model.generate(inputs)
    return processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip().lower()


def evaluate_on_test(model, processor, dataset, device) -> dict:
    """Run model on test split, return per-speaker results."""
    results = defaultdict(lambda: {"refs": [], "hyps": [], "pairs": []})

    for split_name in dataset:
        for sample in dataset[split_name]:
            speaker = get_speaker_id(sample)
            if assign_split(speaker) != "test":
                continue

            ref = None
            for key in ("text", "sentence", "transcription", "label"):
                if key in sample and sample[key]:
                    ref = sample[key].strip().lower()
                    break
            if not ref:
                continue

            audio = sample["audio"]
            hyp = transcribe(model, processor, audio["array"], audio["sampling_rate"], device)

            results[speaker]["refs"].append(ref)
            results[speaker]["hyps"].append(hyp)
            results[speaker]["pairs"].append({"reference": ref, "hypothesis": hyp})

    return dict(results)


def generate_report(baseline_results: dict, finetuned_results: dict, output_path: Path):
    """Generate a markdown report comparing baseline vs. fine-tuned."""
    lines = [
        "# WER Analysis Report: Baseline vs. Fine-Tuned Whisper on TORGO",
        "",
        "## Overall Comparison",
        "",
        "| Model | Overall WER | Dysarthric WER | Control WER |",
        "|-------|-------------|----------------|-------------|",
    ]

    for label, results in [("Baseline", baseline_results), ("Fine-Tuned (LoRA)", finetuned_results)]:
        all_refs, all_hyps = [], []
        dys_refs, dys_hyps = [], []
        ctrl_refs, ctrl_hyps = [], []

        for speaker, data in results.items():
            meta = SPEAKER_INFO.get(speaker, {})
            all_refs.extend(data["refs"])
            all_hyps.extend(data["hyps"])
            if meta.get("group") == "dysarthric":
                dys_refs.extend(data["refs"])
                dys_hyps.extend(data["hyps"])
            else:
                ctrl_refs.extend(data["refs"])
                ctrl_hyps.extend(data["hyps"])

        overall = wer(all_refs, all_hyps) * 100 if all_refs else 0
        dys = wer(dys_refs, dys_hyps) * 100 if dys_refs else 0
        ctrl = wer(ctrl_refs, ctrl_hyps) * 100 if ctrl_refs else 0
        lines.append(f"| {label} | {overall:.1f}% | {dys:.1f}% | {ctrl:.1f}% |")

    # Improvement
    lines.extend(["", "## Per-Speaker Comparison", ""])
    lines.append("| Speaker | Group | Severity | Baseline WER | Fine-Tuned WER | Improvement |")
    lines.append("|---------|-------|----------|-------------|----------------|-------------|")

    for speaker in sorted(set(list(baseline_results.keys()) + list(finetuned_results.keys()))):
        meta = SPEAKER_INFO.get(speaker, {})
        b_data = baseline_results.get(speaker, {"refs": [], "hyps": []})
        f_data = finetuned_results.get(speaker, {"refs": [], "hyps": []})

        b_wer = wer(b_data["refs"], b_data["hyps"]) * 100 if b_data["refs"] else 0
        f_wer = wer(f_data["refs"], f_data["hyps"]) * 100 if f_data["refs"] else 0
        improvement = b_wer - f_wer

        lines.append(
            f"| {speaker} | {meta.get('group', '?')} | {meta.get('severity', '—')} | "
            f"{b_wer:.1f}% | {f_wer:.1f}% | {improvement:+.1f}% |"
        )

    # Sample comparisons
    lines.extend(["", "## Sample Transcription Comparisons", ""])
    for speaker in sorted(finetuned_results.keys()):
        meta = SPEAKER_INFO.get(speaker, {})
        b_pairs = baseline_results.get(speaker, {}).get("pairs", [])
        f_pairs = finetuned_results.get(speaker, {}).get("pairs", [])

        lines.append(f"### {speaker} ({meta.get('group', '?')}, {meta.get('severity', '—')})")
        lines.append("")

        # Show up to 5 examples
        for i, (bp, fp) in enumerate(zip(b_pairs[:5], f_pairs[:5])):
            lines.append(f"**Example {i+1}:**")
            lines.append(f"- Reference: \"{bp['reference']}\"")
            lines.append(f"- Baseline:  \"{bp['hypothesis']}\"")
            lines.append(f"- LoRA:      \"{fp['hypothesis']}\"")
            lines.append("")

    report_text = "\n".join(lines)
    output_path.write_text(report_text)
    print(f"Report saved to {output_path}")
    return report_text


def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs. fine-tuned Whisper on TORGO")
    parser.add_argument("--baseline", type=str, default="baseline_results.json", help="Baseline results JSON")
    parser.add_argument("--model", type=str, default="../models/whisper-lora-torgo/final", help="Fine-tuned model path")
    parser.add_argument("--output", type=str, default="wer_report.md", help="Output report path")
    parser.add_argument("--input", type=str, default=None, help="Path to local dataset")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load dataset
    if args.input:
        from datasets import load_from_disk, Audio
        dataset_path = Path(args.input) / "torgo_dataset"
        dataset = load_from_disk(str(dataset_path))
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    else:
        dataset = load_torgo()

    # Baseline: re-evaluate or load cached results
    baseline_path = Path(args.baseline)
    if baseline_path.exists():
        print("Loading baseline results from cache...")
        with open(baseline_path) as f:
            cached = json.load(f)
        # Use the best model from baseline evaluation
        best = min(cached, key=lambda r: r.get("overall_wer", float("inf")))
        base_model_id = best["model_id"]
        print(f"Re-evaluating baseline with best model: {best['model']}")
    else:
        base_model_id = "openai/whisper-small"
        print(f"No cached baseline found. Using {base_model_id}")

    # Run baseline on test
    base_processor = WhisperProcessor.from_pretrained(base_model_id)
    base_model = WhisperForConditionalGeneration.from_pretrained(base_model_id).to(device)
    base_model.eval()
    print("Evaluating baseline on test set...")
    baseline_results = evaluate_on_test(base_model, base_processor, dataset, device)
    del base_model  # free memory

    # Run fine-tuned on test
    print(f"Loading fine-tuned model from {args.model}...")
    ft_model, ft_processor = load_finetuned_model(args.model, device)
    print("Evaluating fine-tuned model on test set...")
    finetuned_results = evaluate_on_test(ft_model, ft_processor, dataset, device)

    # Generate report
    report = generate_report(baseline_results, finetuned_results, Path(args.output))
    print("\n" + report)


if __name__ == "__main__":
    main()
