import os
import sys
import logging
import hydra
import random
import shutil
import torch
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets

from .data import VibravoxDataset
from .solver import Solver
from .utils import load_model

torch.backends.cudnn.benchmark = True


def create_kfold_splits(dataset, fold_index, num_folds=5, seed=2039):
    """N-group rotation K-Fold CV with speaker-disjoint splits.

    Splits speakers into N balanced groups using sorted-greedy bin packing
    (by utterance count), then rotates test/valid/train assignments:
        Fold i (1-based): test=group[(i-1)%N], valid=group[i%N], train=rest
    """
    from collections import defaultdict

    total = len(dataset)
    speaker_ids = dataset["speaker_id"]

    speaker_to_indices = defaultdict(list)
    for idx, spk in enumerate(speaker_ids):
        speaker_to_indices[spk].append(idx)

    speakers = list(speaker_to_indices.keys())
    total_speakers = len(speakers)

    # Sorted-greedy bin packing: largest speakers first → smallest-sum group
    speakers.sort(key=lambda s: len(speaker_to_indices[s]), reverse=True)

    groups = [[] for _ in range(num_folds)]
    group_sizes = [0] * num_folds

    for spk in speakers:
        spk_size = len(speaker_to_indices[spk])
        smallest = min(range(num_folds), key=lambda g: group_sizes[g])
        groups[smallest].append(spk)
        group_sizes[smallest] += spk_size

    # Group rotation (fold_index is 1-based)
    test_group = (fold_index - 1) % num_folds
    valid_group = fold_index % num_folds
    train_groups = [g for g in range(num_folds)
                    if g != test_group and g != valid_group]

    test_spks = groups[test_group]
    valid_spks = groups[valid_group]
    train_spks = [spk for g in train_groups for spk in groups[g]]

    test_indices = [i for spk in test_spks for i in speaker_to_indices[spk]]
    valid_indices = [i for spk in valid_spks for i in speaker_to_indices[spk]]
    train_indices = [i for spk in train_spks for i in speaker_to_indices[spk]]

    trainset = dataset.select(train_indices)
    devset = dataset.select(valid_indices)
    testset = dataset.select(test_indices)

    speaker_to_group = {}
    for gi in range(num_folds):
        for spk in groups[gi]:
            speaker_to_group[spk] = gi

    fold_info = {
        "fold_index": fold_index,
        "num_folds": num_folds,
        "seed": seed,
        "total_samples": total,
        "total_speakers": total_speakers,
        "train_size": len(trainset),
        "valid_size": len(devset),
        "test_size": len(testset),
        "group_sizes": group_sizes,
        "group_speaker_counts": [len(g) for g in groups],
        "test_group": test_group,
        "valid_group": valid_group,
        "train_groups": train_groups,
        "test_speakers": sorted(test_spks),
        "valid_speakers": sorted(valid_spks),
        "train_speakers": sorted(train_spks),
        "speaker_to_group": speaker_to_group,
    }

    return trainset, devset, testset, fold_info


def setup_logger(name):
    """Set up logger"""
    hydra_conf = OmegaConf.load(".hydra/hydra.yaml")
    logging.config.dictConfig(OmegaConf.to_container(hydra_conf.hydra.job_logging, resolve=True))
    return logging.getLogger(name)


def run(args):
    logger = setup_logger("train")

    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_args = args.model
    model_lib = model_args.model_lib
    model_class_name = model_args.model_class

    # Prepare model params (skip pretrained loading if resuming from checkpoint)
    model_params = OmegaConf.to_container(model_args.param, resolve=True)
    if args.continue_from is not None and 'load_pretrained_weights' in model_params:
        model_params['load_pretrained_weights'] = False
        logger.info("[Resume] Skipping pretrained weights loading (will load from checkpoint)")

    model = load_model(model_lib, model_class_name, model_params, device)

    logger.info(f"Selected model: {model_lib}.{model_class_name}")
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model's parameters: {total_params / 1_000_000:.2f} M")

    if args.save_code:
        scripts_dir = os.path.dirname(hydra.utils.to_absolute_path(__file__))
        project_root = os.path.dirname(scripts_dir)
        src = os.path.join(project_root, "src", "models", f"{model_lib}.py")
        dest = f"./{model_lib}.py"
        if os.path.exists(src):
            shutil.copy2(src, dest)
            logger.info(f"Copied {src} to {dest}")
        else:
            logger.warning(f"Model file not found: {src}")

    if args.optim == "adam":
        optim_class = torch.optim.Adam
    elif args.optim in ("adamW", "adamw"):
        optim_class = torch.optim.AdamW

    optim = optim_class(model.parameters(), lr=args.lr, betas=args.betas)

    # MetricGAN discriminator (optional)
    discriminator = None
    optim_disc = None
    scheduler_disc = None

    metricgan_cfg = args.get("metricgan", {})
    if metricgan_cfg.get("enabled", False):
        from .models.discriminator import MetricGAN_Discriminator
        discriminator = MetricGAN_Discriminator().to(device)
        optim_disc = optim_class(discriminator.parameters(), lr=args.lr, betas=args.betas)
        disc_params = sum(p.numel() for p in discriminator.parameters())
        logger.info(f"MetricGAN discriminator enabled: {disc_params / 1_000_000:.2f} M params")

    # Scheduler
    scheduler = None
    if args.lr_decay is not None:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=args.lr_decay, last_epoch=-1)
        if discriminator is not None:
            scheduler_disc = torch.optim.lr_scheduler.ExponentialLR(optim_disc, gamma=args.lr_decay, last_epoch=-1)

    # Load Vibravox dataset
    logger.info(f"Loading dataset: {args.dataset}")
    _dataset = load_dataset(args.dataset)

    # K-Fold Cross Validation
    cv_config = args.get("cv", {})
    if cv_config.get("enabled", False):
        all_data = concatenate_datasets([
            _dataset['train'], _dataset['dev'], _dataset['test']
        ])
        logger.info(f"[K-Fold CV] Fold {cv_config.fold_index}/{cv_config.num_folds}")
        logger.info(f"[K-Fold CV] Total samples: {len(all_data)}")

        trainset, validset, testset, fold_info = create_kfold_splits(
            dataset=all_data,
            fold_index=cv_config.fold_index,
            num_folds=cv_config.num_folds,
            seed=args.seed
        )

        logger.info(f"[K-Fold CV] Train: {len(trainset)}, Dev: {len(validset)}, Test: {len(testset)}")

        fold_info_path = os.path.join(os.getcwd(), "fold_info.yaml")
        OmegaConf.save(OmegaConf.create(fold_info), fold_info_path)
        logger.info(f"[K-Fold CV] Fold info saved to: {fold_info_path}")
    else:
        trainset = _dataset['train']
        validset = _dataset['dev']
        testset = _dataset['test']

    # Set up dataset and dataloader
    tr_dataset = VibravoxDataset(
        datapair_list=trainset,
        sampling_rate=args.sampling_rate,
        segment=args.segment,
        stride=args.stride,
    )
    tr_loader = DataLoader(
        dataset=tr_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    va_dataset = VibravoxDataset(
        datapair_list=validset,
        sampling_rate=args.sampling_rate,
    )
    va_loader = DataLoader(
        dataset=va_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    ev_dataset = VibravoxDataset(
        datapair_list=testset,
        sampling_rate=args.sampling_rate,
        with_id=True,
        with_text=True,
    )
    ev_loader = DataLoader(
        dataset=ev_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    dataloader = {
        "tr_loader": tr_loader,
        "va_loader": va_loader,
        "ev_loader": ev_loader,
    }

    # Embedding MetricGAN (optional)
    disc_emb = None
    optim_disc_emb = None
    scheduler_disc_emb = None
    emb_extractor = None

    emb_metricgan_cfg = args.get("embedding_metricgan", {})
    if emb_metricgan_cfg.get("enabled", False):
        from .embedding_extractor import EmbeddingQualityExtractor
        from .models.discriminator import FrameLevelEmbeddingCritic

        silence_cfg = emb_metricgan_cfg.get("silence_mask", {})
        silence_db = silence_cfg.get("silence_db") if silence_cfg.get("enabled", False) else None

        emb_extractor = EmbeddingQualityExtractor(
            model_name=emb_metricgan_cfg.model_name,
            layers=list(emb_metricgan_cfg.layers),
            silence_db=silence_db,
        ).to(device)

        disc_emb = FrameLevelEmbeddingCritic(ndf=16).to(device)
        optim_disc_emb = optim_class(disc_emb.parameters(), lr=args.lr, betas=args.betas)
        if args.lr_decay is not None:
            scheduler_disc_emb = torch.optim.lr_scheduler.ExponentialLR(
                optim_disc_emb, gamma=args.lr_decay, last_epoch=-1)

        disc_emb_params = sum(p.numel() for p in disc_emb.parameters())
        logger.info(f"Embedding MetricGAN enabled: {disc_emb_params / 1_000_000:.2f} M params")

    # Solver
    solver = Solver(
        data=dataloader,
        model=model,
        optim=optim,
        scheduler=scheduler,
        args=args,
        logger=logger,
        device=device,
        discriminator=discriminator,
        optim_disc=optim_disc,
        scheduler_disc=scheduler_disc,
        disc_emb=disc_emb,
        optim_disc_emb=optim_disc_emb,
        scheduler_disc_emb=scheduler_disc_emb,
        emb_extractor=emb_extractor,
    )
    solver.train()
    sys.exit(0)


def _main(args):
    logger = setup_logger("main")
    logger.info("For logs, checkpoints and samples check %s", os.getcwd())
    logger.debug(args)
    run(args)


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(args):
    logger = setup_logger("main")
    try:
        _main(args)
    except KeyboardInterrupt:
        logger.info("Training stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Error occurred in main: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
