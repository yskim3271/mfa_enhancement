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
    """Standalone evaluation entry point."""
    from .data import VibravoxDataset
    from .utils import load_model, load_checkpoint, get_stft_args_from_config
    from omegaconf import OmegaConf
    from datasets import load_dataset

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    conf = OmegaConf.load(args.model_config)
    conf.device = args.device

    model_args = conf.model
    model_lib = model_args.model_lib
    model_class_name = model_args.model_class

    model = load_model(model_lib, model_class_name, model_args.param, args.device)
    model = load_checkpoint(model, args.chkpt_dir, args.chkpt_file, args.device)

    # Load Vibravox test set
    testset = load_dataset("yskim3271/vibravox_16k", split="test")
    stft_args = get_stft_args_from_config(model_args)

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

    logger.info(f"Model: {model_class_name}")
    logger.info(f"Checkpoint: {args.chkpt_dir}")
    logger.info(f"Device: {args.device}")

    evaluate(
        args=conf,
        model=model,
        data_loader=ev_loader,
        logger=logger,
        epoch=None,
        stft_args=stft_args,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True, help="Path to the model config file.")
    parser.add_argument("--chkpt_dir", type=str, default='.', help="Path to the checkpoint directory.")
    parser.add_argument("--chkpt_file", type=str, default="best.th", help="Checkpoint file name.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=5)
    parser.add_argument("--log_file", type=str, default="output.log")

    args = parser.parse_args()
    run_standalone_evaluation(args)
