"""Generate enhanced waveforms from trained fold models.

Usage (from project root):
    python -m enhancement.src.generate \
        --fold_dir results/experiments/fold1 \
        --output_dir data/source/vibravox_enhanced_baseline \
        --device cuda
"""
import os
import json
import logging
import argparse

import torch
import soundfile as sf
from omegaconf import OmegaConf
from datasets import load_dataset, concatenate_datasets

from .train import create_kfold_splits
from .stft import mag_pha_stft, mag_pha_istft
from .utils import load_model, get_stft_args

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_best_model(fold_dir, device):
    """Load best model from fold directory."""
    config_path = os.path.join(fold_dir, ".hydra", "config.yaml")
    best_path = os.path.join(fold_dir, "best.th")

    config = OmegaConf.load(config_path)
    model_args = config.model

    model = load_model(
        model_args.model_lib,
        model_args.model_class,
        OmegaConf.to_container(model_args.param, resolve=True),
        device,
    )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    stft_args = get_stft_args(config)
    return model, config, stft_args


def get_test_dataset(config):
    """Recreate K-fold split and return the test set."""
    dataset = load_dataset(config.dataset)
    all_data = concatenate_datasets([
        dataset["train"], dataset["dev"], dataset["test"],
    ])

    cv = config.cv
    _, _, testset, fold_info = create_kfold_splits(
        dataset=all_data,
        fold_index=cv.fold_index,
        num_folds=cv.num_folds,
        seed=config.seed,
    )

    logger.info(
        f"Fold {cv.fold_index}: test set = {len(testset)} utterances, "
        f"{len(fold_info['test_speakers'])} speakers"
    )
    return testset, fold_info


def generate(model, dataset, stft_args, output_dir, fold_index, model_path, device):
    """Run inference and save enhanced waveforms + metadata."""
    metadata = []
    total = len(dataset)

    for i in range(total):
        item = dataset[i]
        speaker_id = item["speaker_id"]
        sentence_id = item["sentence_id"]

        throat_np = item["audio.throat_microphone"]["array"].astype("float32")
        throat_tensor = torch.tensor(throat_np).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            input_com = mag_pha_stft(throat_tensor, **stft_args)[2]
            est_mag, est_pha, _ = model(input_com)
            est_audio = mag_pha_istft(est_mag, est_pha, **stft_args)

        # Match length to original throat audio
        orig_len = throat_np.shape[-1]
        est_audio = est_audio.squeeze(0)[:orig_len].cpu().numpy()

        # Save wav
        speaker_dir = os.path.join(output_dir, speaker_id)
        os.makedirs(speaker_dir, exist_ok=True)
        filename = f"{speaker_id}_{sentence_id}_enhanced.wav"
        wav_path = os.path.join(speaker_dir, filename)
        sf.write(wav_path, est_audio, 16000)

        metadata.append({
            "audio_path": f"{speaker_id}/{filename}",
            "speaker_id": speaker_id,
            "sentence_id": sentence_id,
            "fold": fold_index,
            "model_path": model_path,
            "condition": "baseline",
        })

        if (i + 1) % 500 == 0 or (i + 1) == total:
            logger.info(f"  [{i + 1}/{total}] saved {filename}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Generate enhanced waveforms")
    parser.add_argument("--fold_dir", type=str, required=True,
                        help="Path to fold experiment directory (e.g. results/experiments/fold1)")
    parser.add_argument("--output_dir", type=str, default="data/source/vibravox_enhanced_baseline",
                        help="Output directory for enhanced waveforms")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    fold_dir = args.fold_dir

    # Load model
    logger.info(f"Loading model from {fold_dir}")
    model, config, stft_args = load_best_model(fold_dir, device)

    fold_index = config.cv.fold_index
    model_path = os.path.join(fold_dir, "best.th")

    # Get test set
    testset, fold_info = get_test_dataset(config)

    # Generate
    logger.info(f"Generating enhanced waveforms → {args.output_dir}")
    metadata = generate(
        model, testset, stft_args, args.output_dir,
        fold_index, model_path, device,
    )

    # Append to metadata.jsonl (safe for multiple folds)
    meta_path = os.path.join(args.output_dir, "metadata.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(meta_path, "a") as f:
        for entry in metadata:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Done: {len(metadata)} utterances, metadata appended to {meta_path}")


if __name__ == "__main__":
    main()
