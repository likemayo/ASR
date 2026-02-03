"""
lora_finetuning.py - Fine-tune Whisper with LoRA on the TORGO dataset.

Usage:
    python lora_finetuning.py
    python lora_finetuning.py --config training_config.yaml
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
import numpy as np
from jiwer import wer

from datasets import load_from_disk, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType

from data_loader import SPEAKER_INFO, get_speaker_id, assign_split, load_torgo


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


@dataclass
class DataCollatorSpeechSeq2Seq:
    """Custom data collator for Whisper fine-tuning."""
    processor: WhisperProcessor
    max_audio_length: int = 30

    def __call__(self, features: list[dict]) -> dict:
        audio_arrays = [f["audio"]["array"] for f in features]
        sampling_rates = [f["audio"]["sampling_rate"] for f in features]

        # Extract input features (mel spectrograms)
        input_features = self.processor(
            audio_arrays,
            sampling_rate=sampling_rates[0],
            return_tensors="pt",
        ).input_features

        # Get labels (tokenized transcriptions)
        texts = []
        for f in features:
            for key in ("text", "sentence", "transcription", "label"):
                if key in f and f[key]:
                    texts.append(f[key].strip().lower())
                    break

        labels = self.processor.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
        ).input_ids

        # Replace padding token id with -100 so it's ignored by loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"input_features": input_features, "labels": labels}


def filter_by_split(dataset, target_split: str):
    """Filter dataset to only include samples from a specific split."""
    def _filter(sample):
        speaker = get_speaker_id(sample)
        return assign_split(speaker) == target_split

    filtered = {}
    for split_name in dataset:
        filtered_split = dataset[split_name].filter(_filter)
        if len(filtered_split) > 0:
            filtered[split_name] = filtered_split

    # Concatenate all HF splits that match our target split
    from datasets import concatenate_datasets
    all_splits = list(filtered.values())
    if all_splits:
        return concatenate_datasets(all_splits)
    return None


def compute_wer_metric(pred, processor):
    """Compute WER for evaluation during training."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad token for decoding
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    pred_str = [p.strip().lower() for p in pred_str]
    label_str = [l.strip().lower() for l in label_str]

    return {"wer": wer(label_str, pred_str)}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper with LoRA on TORGO")
    parser.add_argument("--config", type=str, default="training_config.yaml", help="Config file path")
    args = parser.parse_args()

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model and processor
    model_id = config["model"]["name"]
    print(f"Loading {model_id}...")
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)

    # Force decoder settings
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=config["model"]["language"],
        task=config["model"]["task"],
    )
    model.config.suppress_tokens = []

    # Apply LoRA
    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"],
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    dataset_path = config["data"].get("dataset_path")
    if dataset_path and Path(dataset_path).exists():
        dataset = load_from_disk(dataset_path)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=config["data"]["sampling_rate"]))
    else:
        dataset = load_torgo(sampling_rate=config["data"]["sampling_rate"])

    # Split into train/val
    train_dataset = filter_by_split(dataset, "train")
    val_dataset = filter_by_split(dataset, "val")

    if train_dataset is None or val_dataset is None:
        raise ValueError("Could not create train/val splits. Check speaker assignments.")

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Data collator
    data_collator = DataCollatorSpeechSeq2Seq(
        processor=processor,
        max_audio_length=config["data"]["max_audio_length"],
    )

    # Training arguments
    output_dir = config["output"]["model_dir"]
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["training"]["epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"],
        learning_rate=config["training"]["learning_rate"],
        warmup_steps=config["training"]["warmup_steps"],
        fp16=config["training"]["fp16"] and device == "cuda",
        eval_strategy="steps",
        eval_steps=config["training"]["eval_steps"],
        save_steps=config["training"]["save_steps"],
        save_total_limit=3,
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        predict_with_generate=True,
        generation_max_length=225,
        logging_dir=config["output"]["log_dir"],
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model=config["training"]["metric_for_best_model"],
        greater_is_better=config["training"]["greater_is_better"],
        report_to="none",
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_wer_metric(pred, processor),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config["training"]["early_stopping_patience"]
            )
        ],
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final model
    final_path = Path(output_dir) / "final"
    model.save_pretrained(str(final_path))
    processor.save_pretrained(str(final_path))
    print(f"\nModel saved to {final_path}")

    # Final eval
    eval_results = trainer.evaluate()
    print(f"\nFinal validation WER: {eval_results.get('eval_wer', 'N/A')}")


if __name__ == "__main__":
    main()
