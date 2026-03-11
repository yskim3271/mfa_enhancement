import os
import time
import shutil
import torch
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from .evaluate import evaluate
from .stft import mag_pha_istft, mag_pha_stft
from .utils import copy_state, swap_state, phase_losses, LogProgress, pull_metric, get_stft_args, batch_pesq


def _masked_mse(pred, target, mask):
    """MSE loss with optional boolean mask. Falls back to unmasked if mask is None."""
    if mask is not None:
        assert pred.shape == mask.shape, f"shape mismatch: pred {pred.shape} vs mask {mask.shape}"
        return F.mse_loss(pred[mask], target[mask])
    return F.mse_loss(pred, target)


class Solver(object):
    def __init__(
        self,
        data,
        model,
        optim,
        scheduler,
        args,
        logger,
        device=None,
        discriminator=None,
        optim_disc=None,
        scheduler_disc=None,
        disc_emb=None,
        optim_disc_emb=None,
        scheduler_disc_emb=None,
        emb_extractor=None,
    ):
        # Dataloaders
        self.tr_loader = data['tr_loader']
        self.va_loader = data['va_loader']
        self.ev_loader = data['ev_loader']

        self.model = model
        self.optim = optim
        self.scheduler = scheduler

        # MetricGAN discriminator (optional)
        self.discriminator = discriminator
        self.optim_disc = optim_disc
        self.scheduler_disc = scheduler_disc

        # Embedding MetricGAN (optional)
        self.disc_emb = disc_emb
        self.optim_disc_emb = optim_disc_emb
        self.scheduler_disc_emb = scheduler_disc_emb
        self.emb_extractor = emb_extractor

        # loss weights
        self.loss = args.loss

        # logger
        self.logger = logger

        # STFT params
        self.stft_args = get_stft_args(args)

        # Basic config
        self.max_grad_norm = getattr(args, 'max_grad_norm', 5.0)
        self.device = device or torch.device(args.device)

        self.epochs = args.epochs
        self.continue_from = args.continue_from
        self.eval_every = args.eval_every

        self.writer = None
        self.best_state = None
        self.best_pesq = 0.0
        self.history = []
        self.log_dir = args.log_dir
        self.num_prints = args.num_prints
        self.num_workers = args.num_workers
        self.args = args

        # Initialize or resume
        self._reset()

    def _serialize(self):
        """ Save states checkpoint. """
        package = {}
        package['model'] = copy_state(self.model.state_dict())
        package['best_state'] = self.best_state
        package['optimizer'] = self.optim.state_dict()
        package['scheduler'] = self.scheduler.state_dict() if self.scheduler is not None else None
        if self.discriminator is not None:
            package['discriminator'] = copy_state(self.discriminator.state_dict())
            package['optimizer_disc'] = self.optim_disc.state_dict()
            package['scheduler_disc'] = self.scheduler_disc.state_dict() if self.scheduler_disc is not None else None
        if self.disc_emb is not None:
            package['disc_emb'] = copy_state(self.disc_emb.state_dict())
            package['optimizer_disc_emb'] = self.optim_disc_emb.state_dict()
            package['scheduler_disc_emb'] = (
                self.scheduler_disc_emb.state_dict()
                if self.scheduler_disc_emb is not None else None)
        package['args'] = self.args
        package['history'] = self.history
        package['best_pesq'] = self.best_pesq
        tmp_path = "checkpoint.tmp"
        torch.save(package, tmp_path)
        os.rename(tmp_path, "checkpoint.th")

        best_path = "best.tmp"
        best_package = {**self.best_state, 'args': self.args}
        torch.save(best_package, best_path)
        os.rename(best_path, "best.th")

    def _reset(self):
        """Load checkpoint if 'continue_from' is specified, or create a fresh writer if not."""
        if self.continue_from is not None:
            self.logger.info(f'Loading checkpoint model: {self.continue_from}')
            if not os.path.exists(self.continue_from):
                raise FileNotFoundError(f"Checkpoint directory {self.continue_from} not found.")

            src_tb_dir = os.path.join(self.continue_from, 'tensorbd')
            dst_tb_dir = self.log_dir

            if os.path.exists(src_tb_dir):
                if not os.path.exists(dst_tb_dir):
                    shutil.copytree(src_tb_dir, dst_tb_dir)
                else:
                    self.logger.warning(f"TensorBoard log dir {dst_tb_dir} already exists. Skipping copy.")
                self.writer = SummaryWriter(log_dir=dst_tb_dir)

            ckpt_path = os.path.join(self.continue_from, 'checkpoint.th')
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint file {ckpt_path} not found.")
            self.logger.info(f"Loading checkpoint from {ckpt_path}")
            package = torch.load(ckpt_path, map_location='cpu', weights_only=False)

            self.model.load_state_dict(package['model'])
            self.optim.load_state_dict(package['optimizer'])

            scheduler_state = package.get('scheduler', None)
            if self.scheduler is not None and scheduler_state is not None:
                self.scheduler.load_state_dict(scheduler_state)

            if self.discriminator is not None:
                disc_state = package.get('discriminator', None)
                if disc_state is not None:
                    self.discriminator.load_state_dict(disc_state)
                optim_disc_state = package.get('optimizer_disc', None)
                if optim_disc_state is not None:
                    self.optim_disc.load_state_dict(optim_disc_state)
                scheduler_disc_state = package.get('scheduler_disc', None)
                if self.scheduler_disc is not None and scheduler_disc_state is not None:
                    self.scheduler_disc.load_state_dict(scheduler_disc_state)

            if self.disc_emb is not None:
                disc_emb_state = package.get('disc_emb', None)
                if disc_emb_state is not None:
                    self.disc_emb.load_state_dict(disc_emb_state)
                optim_disc_emb_state = package.get('optimizer_disc_emb', None)
                if optim_disc_emb_state is not None:
                    self.optim_disc_emb.load_state_dict(optim_disc_emb_state)
                scheduler_disc_emb_state = package.get('scheduler_disc_emb', None)
                if self.scheduler_disc_emb is not None and scheduler_disc_emb_state is not None:
                    self.scheduler_disc_emb.load_state_dict(scheduler_disc_emb_state)

            self.best_pesq = package.get('best_pesq', 0.0)
            self.best_state = package.get('best_state', None)
            self.history = package.get('history', [])

        else:
            self.writer = SummaryWriter(log_dir=self.log_dir)

    def train(self):

        if self.history:
            self.logger.info("Replaying metrics from previous run")
            for epoch, metrics in enumerate(self.history):
                info = " ".join(f"{k.capitalize()}={v:.5f}" for k, v in metrics.items())
                self.logger.info(f"Epoch {epoch + 1}: {info}")

        self.logger.info(f"Training for {self.epochs} epochs")

        for epoch in range(len(self.history), self.epochs):

            self.model.train()

            start = time.time()
            self.logger.info('-' * 70)
            self.logger.info("Training...")
            self.logger.info(f"Train | Epoch {epoch + 1} | Learning Rate {self.optim.param_groups[0]['lr']:.6f}")
            train_loss = self._run_one_step(epoch)

            self.logger.info(
                f"Train Summary | End of Epoch {epoch + 1} | Time {time.time() - start:.2f}s | Train Loss {train_loss:.5f}"
            )

            if self.discriminator is not None:
                self.discriminator.eval()
            if self.disc_emb is not None:
                self.disc_emb.eval()

            start = time.time()
            self.logger.info('-' * 70)
            self.logger.info('Validation...')
            with torch.no_grad():
                valid_pesq = self._run_validation(epoch)

            self.logger.info(
                f"Valid Summary | End of Epoch {epoch + 1} | Time {time.time() - start:.2f}s | Valid PESQ {valid_pesq:.4f}"
            )

            best_pesq = max(pull_metric(self.history, 'valid_pesq') + [valid_pesq])
            metrics = {'train': train_loss, 'valid_pesq': valid_pesq, 'best_pesq': best_pesq}
            self.history.append(metrics)
            info = " | ".join(f"{k} {v:.5f}" for k, v in metrics.items())
            self.best_pesq = max(self.best_pesq, best_pesq)
            self.logger.info('-' * 70)
            self.logger.info(f"Overall Summary | Epoch {epoch + 1} | {info}")

            if valid_pesq == best_pesq:
                self.logger.info(f'New best valid PESQ {valid_pesq:.4f}')
                self.best_state = {'model': copy_state(self.model.state_dict())}

            self._serialize()

            if (epoch + 1) % self.eval_every == 0:
                self.logger.info('-' * 70)
                self.logger.info('Evaluating on the test set...')
                with swap_state(self.model, self.best_state['model']):
                    ev_metrics = evaluate(
                        args=self.args,
                        model=self.model,
                        data_loader=self.ev_loader,
                        logger=self.logger,
                        epoch=epoch,
                        stft_args=self.stft_args)

                for k, v in ev_metrics.items():
                    self.writer.add_scalar(f"test/{k}", v, epoch)

        self.logger.info("-" * 70)
        self.logger.info("Training Completed")
        self.logger.info("-" * 70)
        self.writer.close()

    def _run_validation(self, epoch):
        """Run validation using PESQ as the selection metric."""
        clean_list = []
        enhanced_list = []

        name = f"Valid | Epoch {epoch + 1}"
        logprog = LogProgress(self.logger, self.va_loader, updates=self.num_prints, name=name)

        for data in logprog:
            throat, acoustic = data
            input_com = mag_pha_stft(throat, **self.stft_args)[2].to(self.device)

            est_mag, est_pha, _ = self.model(input_com)
            est_audio = mag_pha_istft(est_mag, est_pha, **self.stft_args)

            # Align lengths
            min_len = min(acoustic.shape[-1], est_audio.shape[-1])

            clean_list.append(acoustic[..., :min_len].squeeze().cpu().numpy())
            enhanced_list.append(est_audio[..., :min_len].squeeze().detach().cpu().numpy())

        pesq_score = batch_pesq(clean_list, enhanced_list, workers=self.num_workers)
        if pesq_score is None:
            pesq_score = 0.0

        self.writer.add_scalar("Validation/PESQ", pesq_score, epoch)

        return pesq_score

    def _run_one_step(self, epoch):

        total_loss = 0.0
        name = f"Train | Epoch {epoch + 1}"

        logprog = LogProgress(self.logger, self.tr_loader, updates=self.num_prints, name=name)

        for i, data in enumerate(logprog):

            throat, acoustic = data
            throat_mag, _, input_com = mag_pha_stft(throat, **self.stft_args)
            input_com = input_com.to(self.device)
            if self.disc_emb is not None:
                throat_mag = throat_mag.to(self.device)

            target_mag, target_pha, target_com = mag_pha_stft(acoustic, **self.stft_args)
            target_mag = target_mag.to(self.device)
            target_pha = target_pha.to(self.device)
            target_com = target_com.to(self.device)

            est_mag, est_pha, est_com = self.model(input_com)

            est_audio = mag_pha_istft(est_mag, est_pha, **self.stft_args)
            est_mag_con, _, est_com_con = mag_pha_stft(est_audio, **self.stft_args)
            est_mag_con = est_mag_con.to(self.device)

            one_labels = torch.ones(throat.shape[0], device=self.device)

            # --- MetricGAN: Discriminator training ---
            loss_disc_scalar = 0.0
            if self.discriminator is not None:
                self.discriminator.train()

                clean_list = list(acoustic.cpu().numpy())
                enhanced_list = list(est_audio.detach().cpu().numpy())
                batch_pesq_score = batch_pesq(clean_list, enhanced_list,
                                              workers=self.num_workers, normalize=True)

                self.optim_disc.zero_grad()

                metric_r = self.discriminator(
                    target_mag.unsqueeze(1), target_mag.unsqueeze(1))
                metric_g = self.discriminator(
                    target_mag.unsqueeze(1), est_mag_con.detach().unsqueeze(1))

                loss_disc_r = F.mse_loss(one_labels, metric_r.flatten())

                if batch_pesq_score is not None:
                    loss_disc_g = F.mse_loss(
                        batch_pesq_score.to(self.device), metric_g.flatten())
                else:
                    loss_disc_g = 0

                loss_disc = loss_disc_r + loss_disc_g
                loss_disc.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(), max_norm=self.max_grad_norm)

                self.optim_disc.step()
                loss_disc_scalar = loss_disc.item() if isinstance(loss_disc, torch.Tensor) else loss_disc

            # --- Embedding MetricGAN: Discriminator training ---
            loss_disc_emb_scalar = 0.0
            n_frames = None
            silence_mask = None
            if self.disc_emb is not None:
                self.disc_emb.train()

                # Teacher targets: extract acoustic once, reuse for both pairs
                acoustic_dev = acoustic.to(self.device)
                z_ac = self.emb_extractor.extract_embeddings(acoustic_dev)
                q_throat = self.emb_extractor.compute_frame_quality(
                    acoustic_dev, throat.to(self.device), z_ac=z_ac)
                q_enhanced = self.emb_extractor.compute_frame_quality(
                    acoustic_dev, est_audio.detach(), z_ac=z_ac)
                n_frames = q_throat.shape[1]
                silence_mask = self.emb_extractor.compute_silence_mask(
                    acoustic_dev, n_frames)
                # Pre-resolve mask.any() once to avoid repeated GPU syncs
                if silence_mask is not None and not silence_mask.any():
                    silence_mask = None

                self.optim_disc_emb.zero_grad()

                metric_emb_r = self.disc_emb(
                    target_mag.unsqueeze(1), target_mag.unsqueeze(1),
                    n_frames=n_frames)
                metric_emb_th = self.disc_emb(
                    target_mag.unsqueeze(1), throat_mag.unsqueeze(1),
                    n_frames=n_frames)
                metric_emb_g = self.disc_emb(
                    target_mag.unsqueeze(1), est_mag_con.detach().unsqueeze(1),
                    n_frames=n_frames)

                loss_disc_emb = (
                    _masked_mse(metric_emb_r, torch.ones_like(q_throat), silence_mask) +
                    _masked_mse(metric_emb_th, q_throat, silence_mask) +
                    _masked_mse(metric_emb_g, q_enhanced, silence_mask))

                loss_disc_emb.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.disc_emb.parameters(), max_norm=self.max_grad_norm)
                self.optim_disc_emb.step()
                loss_disc_emb_scalar = loss_disc_emb.item()

            # --- Generator training ---
            loss_magnitude = F.mse_loss(target_mag, est_mag)
            loss_phase = phase_losses(target_pha, est_pha)
            loss_complex = F.mse_loss(target_com, est_com) * 2
            loss_consistency = F.mse_loss(est_com, est_com_con) * 2

            loss = loss_complex * self.loss.complex + \
                   loss_consistency * self.loss.consistency + \
                   loss_magnitude * self.loss.magnitude + \
                   loss_phase * self.loss.phase

            # MetricGAN: generator adversarial loss
            loss_metric = torch.tensor(0.0, device=self.device)
            if self.discriminator is not None:
                metric_g = self.discriminator(
                    target_mag.unsqueeze(1), est_mag_con.unsqueeze(1))
                loss_metric = F.mse_loss(metric_g.flatten(), one_labels)
                loss = loss + loss_metric * self.loss.get("metric", 0.0)

            # Embedding MetricGAN: generator adversarial loss
            loss_metric_emb = torch.tensor(0.0, device=self.device)
            if self.disc_emb is not None:
                metric_emb_g = self.disc_emb(
                    target_mag.unsqueeze(1), est_mag_con.unsqueeze(1),
                    n_frames=n_frames)
                loss_metric_emb = _masked_mse(
                    metric_emb_g, torch.ones_like(metric_emb_g), silence_mask)
                loss = loss + loss_metric_emb * self.loss.get("metric_emb", 0.0)

            loss_dict = {
                "Magnitude_Loss": loss_magnitude,
                "Phase_Loss": loss_phase,
                "Complex_Loss": loss_complex,
                "Consistency_Loss": loss_consistency,
                "Metric_Loss": loss_metric,
                "EmbMetric_Loss": loss_metric_emb,
                "Disc_Loss": loss_disc_scalar,
                "DiscEmb_Loss": loss_disc_emb_scalar,
                "Total_Loss": loss
            }

            loss_dict_scalar = {k: v.item() if isinstance(v, torch.Tensor) else v
                               for k, v in loss_dict.items()}

            self.optim.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)

            self.optim.step()

            for k, v in loss_dict_scalar.items():
                if v != 0.0:
                    logprog.append(**{f"{k}": format(v, "4.5f")})
                    if i % (self.num_prints * 10) == 0:
                        self.writer.add_scalar(f"Train/{k}", v, epoch * len(self.tr_loader) + i)

            total_loss += loss.item()

        if self.scheduler is not None:
            self.scheduler.step()
        if self.scheduler_disc is not None:
            self.scheduler_disc.step()
        if self.scheduler_disc_emb is not None:
            self.scheduler_disc_emb.step()

        return total_loss / len(self.tr_loader)
