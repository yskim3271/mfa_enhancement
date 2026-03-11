"""Generate enhanced waveforms from trained fold models.

Usage (from project root):
    # Full test set (speaker subdirs, metadata appended)
    python -m enhancement.src.generate \
        --fold_dir results/experiments/fold1 \
        --output_dir data/source/vibravox_enhanced_baseline

    # Random subset for inspection (flat directory)
    python -m enhancement.src.generate \
        --fold_dir results/experiments/fold1 \
        --output_dir results/samples \
        --num_samples 20 --seed 42
"""
import os
import json
import logging
import argparse
import random

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


def enhance_single_item(model, item, stft_args, device):
    """Run inference on a single dataset item, return enhanced numpy waveform."""
    throat_np = item["audio.throat_microphone"]["array"].astype("float32")
    throat_tensor = torch.from_numpy(throat_np).unsqueeze(0).to(device)
    with torch.no_grad():
        input_com = mag_pha_stft(throat_tensor, **stft_args)[2]
        est_mag, est_pha, _ = model(input_com)
        est_audio = mag_pha_istft(est_mag, est_pha, **stft_args)
    return est_audio.squeeze(0)[:throat_np.shape[-1]].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Generate enhanced waveforms")
    parser.add_argument("--fold_dir", type=str, required=True,
                        help="Path to fold experiment directory")
    parser.add_argument("--output_dir", type=str,
                        default="data/source/vibravox_enhanced_baseline",
                        help="Output directory for enhanced waveforms")
    parser.add_argument("--num_samples", type=int, default=0,
                        help="Number of random samples (0 = all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sample selection")
    parser.add_argument("--flat", action="store_true",
                        help="Flat output (no speaker subdirs). Auto-enabled when --num_samples > 0")
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
    testset, _ = get_test_dataset(config)
    total = len(testset)

    # Select indices
    if args.num_samples > 0:
        rng = random.Random(args.seed)
        indices = sorted(rng.sample(range(total), min(args.num_samples, total)))
        flat = True  # auto-enable flat for sample mode
    else:
        indices = list(range(total))
        flat = args.flat

    # Output directory: append exp_name in sample mode
    output_dir = args.output_dir
    if args.num_samples > 0:
        exp_name = os.path.basename(fold_dir.rstrip("/"))
        output_dir = os.path.join(output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Generating {len(indices)}/{total} utterances → {output_dir}")

    # Generate
    metadata = []
    for rank, idx in enumerate(indices):
        item = testset[idx]
        speaker_id = item["speaker_id"]
        sentence_id = item["sentence_id"]

        est_audio = enhance_single_item(model, item, stft_args, device)

        filename = f"{speaker_id}_{sentence_id}_enhanced.wav"
        if flat:
            wav_path = os.path.join(output_dir, filename)
            audio_path = filename
        else:
            speaker_dir = os.path.join(output_dir, speaker_id)
            os.makedirs(speaker_dir, exist_ok=True)
            wav_path = os.path.join(speaker_dir, filename)
            audio_path = f"{speaker_id}/{filename}"
        sf.write(wav_path, est_audio, 16000)

        metadata.append({
            "audio_path": audio_path,
            "speaker_id": speaker_id,
            "sentence_id": sentence_id,
            "test_index": idx,
            "fold": fold_index,
            "model_path": model_path,
        })

        # Progress logging: every item in sample mode, every 500 in full mode
        if args.num_samples > 0:
            logger.info(f"  [{rank + 1}/{len(indices)}] {filename}")
        elif (rank + 1) % 500 == 0 or (rank + 1) == len(indices):
            logger.info(f"  [{rank + 1}/{len(indices)}] saved {filename}")

    # Save metadata (append for full mode, overwrite for sample mode)
    meta_path = os.path.join(output_dir, "metadata.jsonl")
    mode = "w" if args.num_samples > 0 else "a"
    with open(meta_path, mode) as f:
        for entry in metadata:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Done: {len(metadata)} utterances, metadata → {meta_path}")


if __name__ == "__main__":
    main()
