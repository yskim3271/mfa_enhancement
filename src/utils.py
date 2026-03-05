import os
import time
import torch
import numpy as np
import logging
import importlib
from typing import Dict, Any
from contextlib import contextmanager
from joblib import Parallel, delayed
from pesq import pesq


# Reusable joblib Parallel pool (loky backend)
_JOBLIB_PARALLEL = None
_JOBLIB_WORKERS = None

def _get_joblib_parallel(workers: int):
    """Return a reusable joblib Parallel instance; recreate if worker size changed."""
    global _JOBLIB_PARALLEL, _JOBLIB_WORKERS
    if _JOBLIB_PARALLEL is None or _JOBLIB_WORKERS != workers:
        if _JOBLIB_PARALLEL is not None:
            try:
                _JOBLIB_PARALLEL._terminate_pool()
            except Exception:
                pass
        _JOBLIB_PARALLEL = Parallel(n_jobs=workers, backend="loky", prefer="processes")
        _JOBLIB_WORKERS = workers
    return _JOBLIB_PARALLEL


def pesq_loss(clean, noisy, sr=16000):
    try:
        pesq_score = pesq(sr, clean, noisy, "wb")
    except Exception:
        pesq_score = -1
    return pesq_score


def batch_pesq(clean, noisy, workers=8, normalize=False):
    """Compute PESQ scores in parallel.

    Args:
        normalize: If True, return per-sample normalized tensor (for MetricGAN).
                   If False (default), return mean score as float.
    Returns:
        normalize=True: FloatTensor of per-sample scores in [0,1], or None if any failed.
        normalize=False: float mean score, or None if any failed.
    """
    parallel = _get_joblib_parallel(workers)
    scores = parallel(delayed(pesq_loss)(c, n) for c, n in zip(clean, noisy))
    scores = np.array(scores)
    if -1 in scores:
        return None
    if normalize:
        scores = (scores - 1) / 3.5
        return torch.FloatTensor(scores)
    return float(scores.mean())

def anti_wrapping_function(x):
    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)

def phase_losses(phase_r, phase_g):
    ip_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=1) - torch.diff(phase_g, dim=1)))
    iaf_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=2) - torch.diff(phase_g, dim=2)))

    return ip_loss + gd_loss + iaf_loss


def copy_state(state):
    return {k: v.cpu().clone() for k, v in state.items()}


@contextmanager
def swap_state(model, state):
    """Context manager that swaps the state of a model."""
    old_state = copy_state(model.state_dict())
    model.load_state_dict(state)
    try:
        yield
    finally:
        model.load_state_dict(old_state)


def pull_metric(history, name):
    out = []
    for metrics in history:
        if name in metrics:
            out.append(metrics[name])
    return out


class LogProgress:
    """
    Sort of like tqdm but using log lines and not as real time.
    """
    def __init__(self,
                 logger,
                 iterable,
                 updates=5,
                 total=None,
                 name="LogProgress",
                 level=logging.INFO):
        self.iterable = iterable
        self.total = total or len(iterable)
        self.updates = updates
        self.name = name
        self.logger = logger
        self.level = level

    def update(self, **infos):
        self._infos = infos

    def append(self, **infos):
        self._infos.update(**infos)

    def _append(self, info):
        self._infos.update(info)

    def __iter__(self):
        self._iterator = iter(self.iterable)
        self._index = -1
        self._infos = {}
        self._begin = time.time()
        return self

    def __next__(self):
        self._index += 1
        try:
            value = next(self._iterator)
        except StopIteration:
            raise
        else:
            return value
        finally:
            log_every = max(1, self.total // self.updates)
            if self._index >= 1 and self._index % log_every == 0:
                self._log()

    def _log(self):
        self._speed = (1 + self._index) / (time.time() - self._begin)
        infos = " | ".join(f"{k} {v}" for k, v in self._infos.items())
        if self._speed < 1e-4:
            speed = "oo sec/it"
        elif self._speed < 0.1:
            speed = f"{1/self._speed:.1f} sec/it"
        else:
            speed = f"{self._speed:.1f} it/sec"
        out = f"{self.name} | {self._index}/{self.total} | {speed}"
        if infos:
            out += " | " + infos
        self.logger.log(self.level, out)


def colorize(text, color):
    code = f"\033[{color}m"
    restore = "\033[0m"
    return "".join([code, text, restore])


def bold(text):
    return colorize(text, "1")


# ============================================================================
# Model and Checkpoint Utilities
# ============================================================================

def load_model(model_lib: str, model_class_name: str, model_params: Dict[str, Any], device: str = 'cuda'):
    """Load model dynamically from models directory."""
    module = importlib.import_module(f".models.{model_lib}", package=__package__)
    model_class = getattr(module, model_class_name)
    model = model_class(**model_params)
    return model.to(device)


def load_checkpoint(model: torch.nn.Module, chkpt_dir: str, chkpt_file: str, device: str = 'cuda'):
    """Load model checkpoint."""
    chkpt_path = os.path.join(chkpt_dir, chkpt_file)
    chkpt = torch.load(chkpt_path, map_location=device, weights_only=False)
    model.load_state_dict(chkpt['model'])
    return model


def get_stft_args(args) -> Dict[str, Any]:
    """Extract STFT arguments from flat config."""
    return {
        "n_fft": args.n_fft,
        "hop_size": args.hop_size,
        "win_size": args.win_size,
        "compress_factor": args.compress_factor
    }


def get_stft_args_from_config(model_args) -> Dict[str, Any]:
    """Extract STFT arguments from model configuration."""
    fft_len = model_args.param.fft_len
    return {
        "n_fft": fft_len,
        "hop_size": model_args.param.get("hop_size", fft_len // 4),
        "win_size": model_args.param.get("win_size", fft_len),
        "compress_factor": model_args.param.get("compress_factor", 1.0)
    }
