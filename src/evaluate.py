import json
import os

import torch
import numpy as np
import logging
from typing import Dict, Optional, Any
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from .compute_metrics import compute_metrics
from .stft import mag_pha_stft, mag_pha_istft
from .utils import bold, LogProgress


def evaluate(
    args: DictConfig,
    model: torch.nn.Module,
    data_loader: DataLoader,
    logger: logging.Logger,
    epoch: Optional[int] = None,
    stft_args: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Model evaluation

    Args:
        args: Evaluation settings (OmegaConf)
        model: Model to evaluate
        data_loader: DataLoader for evaluation
        logger: Logger instance
        epoch: Current epoch (during training)
        stft_args: STFT parameters

    Returns:
        Dictionary of metrics
    """
    prefix = f"Epoch {epoch+1}, " if epoch is not None else ""

    model.eval()

    iterator = LogProgress(logger, data_loader, name="Evaluate")
    results  = []
    with torch.no_grad():
        for data in iterator:
            throat, acoustic, _, text = data

            input_com = mag_pha_stft(throat, **stft_args)[2].to(args.device)

            est_mag, est_pha, _ = model(input_com)

            est_audio = mag_pha_istft(est_mag, est_pha, **stft_args)

            acoustic = acoustic.squeeze().detach().cpu().numpy()
            est_audio = est_audio.squeeze().detach().cpu().numpy()

            results.append(compute_metrics(acoustic, est_audio))

    pesq, csig, cbak, covl, seg_snr, stoi = np.mean(results, axis=0)
    metrics = {
        "pesq": pesq,
        "stoi": stoi,
        "csig": csig,
        "cbak": cbak,
        "covl": covl,
        "seg_snr": seg_snr
    }
    logger.info(bold(f"{prefix}Performance: PESQ={pesq:.4f}, STOI={stoi:.4f}, CSIG={csig:.4f}, CBAK={cbak:.4f}, COVL={covl:.4f}"))

    return metrics


def run_standalone_evaluation(args):
    """Standalone evaluation entry point.

    Usage:
        # Fold-based evaluation (K-fold test set)
        python -m enhancement.src.evaluate --fold_dir results/experiments/emb_pdetach_w002

        # Legacy evaluation (HuggingFace fixed test split)
        python -m enhancement.src.evaluate --model_config path/to/config.yaml --chkpt_dir path/to/dir
    """
    from .data import VibravoxDataset
    from .generate import load_best_model, get_test_dataset
    from .utils import load_model, load_checkpoint, get_stft_args_from_config
    from omegaconf import OmegaConf

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    if args.fold_dir:
        # --- Fold-based mode ---
        fold_dir = args.fold_dir
        exp_name = os.path.basename(fold_dir.rstrip("/"))
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        logger.info(f"[{exp_name}] Loading model from {fold_dir}")
        model, config, stft_args = load_best_model(fold_dir, device)
        config.device = str(device)

        testset, _ = get_test_dataset(config)

        ev_dataset = VibravoxDataset(
            datapair_list=testset,
            sampling_rate=16000,
            with_id=True,
            with_text=True,
        )
    else:
        # --- Legacy mode ---
        from datasets import load_dataset

        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        exp_name = os.path.basename(args.chkpt_dir.rstrip("/"))

        config = OmegaConf.load(args.model_config)
        config.device = str(device)
        model_args = config.model

        model = load_model(
            model_args.model_lib, model_args.model_class,
            model_args.param, str(device),
        )
        model = load_checkpoint(model, args.chkpt_dir, args.chkpt_file, str(device))
        stft_args = get_stft_args_from_config(model_args)

        testset = load_dataset("yskim3271/vibravox_16k", split="test")
        ev_dataset = VibravoxDataset(
            datapair_list=testset,
            sampling_rate=16000,
            with_id=True,
            with_text=True,
        )

    ev_loader = DataLoader(
        dataset=ev_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    logger.info(f"[{exp_name}] Test set: {len(ev_dataset)} utterances, device: {device}")

    metrics = evaluate(
        args=config,
        model=model,
        data_loader=ev_loader,
        logger=logger,
        epoch=None,
        stft_args=stft_args,
    )

    # Save results
    out_path = args.output or os.path.join(args.fold_dir or args.chkpt_dir, "eval_metrics.json")
    with open(out_path, "w") as f:
        json.dump({"experiment": exp_name, "num_samples": len(ev_dataset), **metrics}, f, indent=2)
    logger.info(f"[{exp_name}] Metrics saved to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate enhancement model")
    # Fold-based mode
    parser.add_argument("--fold_dir", type=str, default=None,
                        help="Path to fold experiment directory (e.g. results/experiments/emb_pdetach_w002)")
    # Legacy mode
    parser.add_argument("--model_config", type=str, default=None,
                        help="Path to the model config file (legacy mode)")
    parser.add_argument("--chkpt_dir", type=str, default=".",
                        help="Path to the checkpoint directory (legacy mode)")
    parser.add_argument("--chkpt_file", type=str, default="best.th")
    # Common
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=5)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: {fold_dir}/eval_metrics.json)")

    args = parser.parse_args()

    if not args.fold_dir and not args.model_config:
        parser.error("Either --fold_dir or --model_config is required")

    run_standalone_evaluation(args)
