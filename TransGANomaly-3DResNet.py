import argparse
import bisect
import glob
import json
import os
import pickle
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import signal
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from utils.eval_utils import save_evaluation_curves


DEFAULTS = {
    "seed": 42,
    "epochs": 50,
    "batch_size": 64,
    "lr_g": 2e-4,
    "lr_d": 2e-5,
    "weight_decay_g": 1e-5,
    "weight_decay_d": 1e-6,
    "num_workers": 0,
    "use_amp": True,
    "max_grad_norm_g": 1.0,
    "max_grad_norm_d": 1.0,
    "fea_dim": 128,
    "mem_dim": 512,
    "mem_temperature": 0.07,
    "mem_shrink_thr": 0.0025,
    "w_recon_motion": 1.0,
    "w_pred_frame": 1.0,
    "w_grad": 0.10,
    "w_compact": 0.02,
    "w_entropy": 0.001,
    "w_adv": 0.001,
    "motion_score_weight": 0.5,
    "frame_score_weight": 0.5,
    "disc_score_weight": 0.0,
    "val_ratio": 0.10,
    "split_unit": "frame",
    "kfold": 5,
    "early_stop_patience": 5,
    "early_stop_min_delta": 0.0,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="TransGANomaly-inspired 3DResNet-HF2VAD for video anomaly detection"
    )
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--dataset_name", choices=["ped2", "avenue", "shanghaitech"], default="ped2")
    parser.add_argument("--dataset_base_dir", default="./data")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--lr_g", type=float, default=DEFAULTS["lr_g"])
    parser.add_argument("--lr_d", type=float, default=DEFAULTS["lr_d"])
    parser.add_argument("--weight_decay_g", type=float, default=DEFAULTS["weight_decay_g"])
    parser.add_argument("--weight_decay_d", type=float, default=DEFAULTS["weight_decay_d"])
    parser.add_argument("--num_workers", type=int, default=DEFAULTS["num_workers"])
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument(
        "--max_cache_chunks",
        type=int,
        default=2,
        help="Max chunked_samples pkl files cached per dataset/worker; <=0 caches all chunks",
    )
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--stats_every", type=int, default=1)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--val_ratio", type=float, default=DEFAULTS["val_ratio"])
    parser.add_argument("--split_unit", choices=["frame", "sample"], default=DEFAULTS["split_unit"])
    parser.add_argument("--kfold", type=int, default=DEFAULTS["kfold"], help="Number of folds; default is 5")
    parser.add_argument("--fold", type=int, default=None, help="Run/test a single fold index, 0-based")
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=DEFAULTS["early_stop_patience"],
        help="Stop after this many validation checks without improvement; <=0 disables",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=DEFAULTS["early_stop_min_delta"],
        help="Minimum val_total_g decrease required to reset early stopping",
    )
    parser.add_argument("--eval_test_during_train", action="store_true")
    parser.add_argument(
        "--detach_recon_motion",
        action="store_true",
        help="Stop prediction/adversarial losses from updating the motion reconstruction branch",
    )

    parser.add_argument("--w_recon_motion", type=float, default=DEFAULTS["w_recon_motion"])
    parser.add_argument("--w_pred_frame", type=float, default=DEFAULTS["w_pred_frame"])
    parser.add_argument("--w_grad", type=float, default=DEFAULTS["w_grad"])
    parser.add_argument("--w_compact", type=float, default=DEFAULTS["w_compact"])
    parser.add_argument("--w_entropy", type=float, default=DEFAULTS["w_entropy"])
    parser.add_argument("--w_adv", type=float, default=DEFAULTS["w_adv"])
    parser.add_argument("--motion_score_weight", type=float, default=DEFAULTS["motion_score_weight"])
    parser.add_argument("--frame_score_weight", type=float, default=DEFAULTS["frame_score_weight"])
    parser.add_argument(
        "--disc_score_weight",
        type=float,
        default=DEFAULTS["disc_score_weight"],
        help="Optional discriminator realism error weight in anomaly score. Default 0 uses generator only.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, record: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def sync_if_cuda(device):
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def snapshot_path(save_dir: Path, epoch: int) -> Path:
    return save_dir / f"model.pth-{epoch}"


def training_stats_path(save_dir: Path, epoch: int) -> Path:
    return save_dir / f"training_stats.npy-{epoch}"


def latest_snapshot_path(save_dir: Path) -> Optional[Path]:
    paths = [Path(p) for p in glob.glob(str(save_dir / "model.pth-*"))]
    if not paths:
        return None

    def epoch_key(path: Path):
        try:
            return int(path.name.rsplit("-", 1)[-1])
        except ValueError:
            return -1

    return max(paths, key=epoch_key)


def latest_training_stats_path(save_dir: Path) -> Optional[Path]:
    paths = [Path(p) for p in glob.glob(str(save_dir / "training_stats.npy-*"))]
    if not paths:
        return None

    def epoch_key(path: Path):
        try:
            return int(path.name.rsplit("-", 1)[-1])
        except ValueError:
            return -1

    return max(paths, key=epoch_key)


class ChunkedSamplesDataset(Dataset):
    def __init__(self, chunk_dir: Path, max_cache_chunks: int = 2):
        self.chunk_dir = Path(chunk_dir)
        self.chunk_files = sorted(self.chunk_dir.glob("chunked_samples_*.pkl"), key=lambda p: p.name)
        if not self.chunk_files:
            raise FileNotFoundError(f"No chunked_samples_*.pkl found in {self.chunk_dir}")

        self.chunk_lengths: List[int] = []
        self.cum_lengths: List[int] = []
        self.cache: Dict[int, Dict[str, np.ndarray]] = {}
        self.max_cache_chunks = max_cache_chunks if max_cache_chunks > 0 else len(self.chunk_files)

        total = 0
        for chunk_file in self.chunk_files:
            payload = joblib.load(chunk_file, mmap_mode="r")
            chunk_len = len(payload["sample_id"])
            self.chunk_lengths.append(chunk_len)
            total += chunk_len
            self.cum_lengths.append(total)

        self.total_len = total
        self._pred_frames: Optional[np.ndarray] = None
        print(
            f"[ChunkedSamplesDataset] {self.chunk_dir} | files={len(self.chunk_files)} "
            f"| samples={self.total_len} | max_cache_chunks={self.max_cache_chunks}"
        )

    def __len__(self):
        return self.total_len

    def _resolve_index(self, idx: int) -> Tuple[int, int]:
        chunk_idx = bisect.bisect_right(self.cum_lengths, idx)
        prev_cum = 0 if chunk_idx == 0 else self.cum_lengths[chunk_idx - 1]
        return chunk_idx, idx - prev_cum

    def _load_chunk(self, chunk_idx: int):
        if chunk_idx not in self.cache:
            if len(self.cache) >= self.max_cache_chunks:
                self.cache.pop(next(iter(self.cache)))
            self.cache[chunk_idx] = joblib.load(self.chunk_files[chunk_idx], mmap_mode="r")
        return self.cache[chunk_idx]

    def get_pred_frames(self) -> np.ndarray:
        if self._pred_frames is None:
            frames = []
            for chunk_file in self.chunk_files:
                payload = joblib.load(chunk_file, mmap_mode="r")
                frames.append(np.asarray(payload["pred_frame"]).reshape(-1).astype(np.int64))
            self._pred_frames = np.concatenate(frames, axis=0)
        return self._pred_frames

    def __getitem__(self, idx: int):
        chunk_idx, local_idx = self._resolve_index(idx)
        payload = self._load_chunk(chunk_idx)

        appearance = np.array(payload["appearance"][local_idx], dtype=np.float32, copy=True) / 255.0
        motion = np.array(payload["motion"][local_idx], dtype=np.float32, copy=True)
        bbox = np.array(payload["bbox"][local_idx], dtype=np.float32, copy=True)
        pred_frame = payload["pred_frame"][local_idx]

        if appearance.shape[0] < 5 or motion.shape[0] < 5:
            raise ValueError(f"Expected 5 appearance/motion steps, got {appearance.shape} and {motion.shape}")

        observed_app = torch.from_numpy(appearance[:4]).permute(0, 3, 1, 2).contiguous()
        target_app = torch.from_numpy(appearance[4]).permute(2, 0, 1).contiguous()
        motion = torch.from_numpy(motion[1:5]).permute(0, 3, 1, 2).contiguous()
        pred_frame = int(np.asarray(pred_frame).reshape(-1)[-1])
        return observed_app, motion, target_app, torch.from_numpy(bbox), torch.tensor(pred_frame, dtype=torch.long)


class ResBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride=(1, 1, 1)):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.downsample = None
        if stride != (1, 1, 1) or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet3DEncoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=32, out_ch=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, base_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            ResBlock3D(base_ch, base_ch, stride=(1, 1, 1)),
            ResBlock3D(base_ch, base_ch, stride=(1, 1, 1)),
        )
        self.layer2 = nn.Sequential(
            ResBlock3D(base_ch, base_ch * 2, stride=(1, 2, 2)),
            ResBlock3D(base_ch * 2, base_ch * 2, stride=(1, 1, 1)),
        )
        self.layer3 = nn.Sequential(
            ResBlock3D(base_ch * 2, out_ch, stride=(1, 2, 2)),
            ResBlock3D(out_ch, out_ch, stride=(1, 1, 1)),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class MemoryModule(nn.Module):
    def __init__(self, mem_dim, fea_dim, temperature, shrink_thres):
        super().__init__()
        self.temperature = temperature
        self.shrink_thres = shrink_thres
        self.memory = nn.Parameter(torch.randn(mem_dim, fea_dim))
        nn.init.xavier_uniform_(self.memory)

    def forward(self, x):
        b, c, t, h, w = x.shape
        query = x.permute(0, 2, 3, 4, 1).contiguous().view(-1, c)
        query_norm = F.normalize(query, dim=1)
        memory_norm = F.normalize(self.memory, dim=1)
        logits = torch.mm(query_norm, memory_norm.t()) / self.temperature
        att = F.softmax(logits, dim=1)
        if self.shrink_thres > 0:
            att = hard_shrink_relu(att, self.shrink_thres)
            att = att / (att.sum(dim=1, keepdim=True) + 1e-12)
        mem_read = torch.mm(att, self.memory)
        mem_out = mem_read.view(b, t, h, w, c).permute(0, 4, 1, 2, 3).contiguous()
        return mem_out, att, query, mem_read


class MotionDecoder3D(nn.Module):
    def __init__(self, in_ch=128, out_ch=2):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(in_ch, 128, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, out_ch, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.decoder(x)


class FrameDecoder2D(nn.Module):
    def __init__(self, in_ch=128, out_ch=3):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_ch, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_ch, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(x)


class TemporalAttentionPool2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Conv3d(channels, 1, kernel_size=1)

    def forward(self, x):
        weights = self.score(x)
        weights = torch.softmax(weights, dim=2)
        return (x * weights).sum(dim=2)


class TransGANomaly3DResNetGenerator(nn.Module):
    """
    HF2-VAD-like generator:
    memory-augmented flow reconstruction followed by reconstructed-flow-guided
    future RGB crop prediction.
    """

    def __init__(self, fea_dim, mem_dim, mem_temperature, mem_shrink_thr, detach_recon_motion=False):
        super().__init__()
        self.detach_recon_motion = detach_recon_motion

        self.motion_encoder = ResNet3DEncoder(in_ch=2, out_ch=fea_dim)
        self.memory = MemoryModule(mem_dim, fea_dim, mem_temperature, mem_shrink_thr)
        self.motion_decoder = MotionDecoder3D(in_ch=fea_dim, out_ch=2)

        self.app_encoder = ResNet3DEncoder(in_ch=3, out_ch=fea_dim)
        self.recon_motion_encoder = ResNet3DEncoder(in_ch=2, out_ch=fea_dim)
        self.pred_fuse = nn.Sequential(
            nn.Conv3d(fea_dim * 2, fea_dim, kernel_size=1, bias=False),
            nn.BatchNorm3d(fea_dim),
            nn.ReLU(inplace=True),
            ResBlock3D(fea_dim, fea_dim),
        )
        self.temporal_pool = TemporalAttentionPool2D(fea_dim)
        self.frame_decoder = FrameDecoder2D(in_ch=fea_dim, out_ch=3)

    def forward(self, observed_app, motion):
        motion_latent = self.motion_encoder(motion)
        mem_motion, att, query, mem_read = self.memory(motion_latent)
        recon_motion = self.motion_decoder(mem_motion)

        motion_for_pred = recon_motion.detach() if self.detach_recon_motion else recon_motion
        app_features = self.app_encoder(observed_app)
        recon_motion_features = self.recon_motion_encoder(motion_for_pred)
        pred_features = self.pred_fuse(torch.cat([app_features, recon_motion_features], dim=1))
        pred_frame = self.frame_decoder(self.temporal_pool(pred_features))

        aux = {"att": att, "query": query, "mem_read": mem_read}
        return recon_motion, pred_frame, aux


class Conditional3DPatchDiscriminator(nn.Module):
    """
    Conditional 3D PatchGAN.
    Input is a RGB sequence [previous T crops, real-or-predicted target crop],
    shaped as B x 3 x (T+1) x H x W. No sigmoid is used for LS-GAN.
    """

    def __init__(self, in_ch=3, base_ch=32):
        super().__init__()

        def block(cin, cout, stride, use_bn=True):
            layers = [nn.Conv3d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=not use_bn)]
            if use_bn:
                layers.append(nn.BatchNorm3d(cout))
            layers.append(nn.LeakyReLU(0.3, inplace=True))
            return nn.Sequential(*layers)

        self.net = nn.Sequential(
            block(in_ch, base_ch, stride=(1, 2, 2), use_bn=False),
            block(base_ch, base_ch * 2, stride=(1, 2, 2)),
            block(base_ch * 2, base_ch * 4, stride=(1, 2, 2)),
            block(base_ch * 4, base_ch * 4, stride=(1, 1, 1)),
            nn.Conv3d(base_ch * 4, 1, kernel_size=3, stride=1, padding=1),
        )
        self.apply(init_gan_weights)

    def forward(self, x):
        return self.net(x)


def init_gan_weights(module):
    if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.normal_(module.weight, mean=1.0, std=0.02)
        nn.init.zeros_(module.bias)


def hard_shrink_relu(inp, lambd, eps=1e-12):
    return (F.relu(inp - lambd) * inp) / (torch.abs(inp - lambd) + eps)


def gradient_loss_2d(pred, target):
    pred_gx = pred[..., :, 1:] - pred[..., :, :-1]
    pred_gy = pred[..., 1:, :] - pred[..., :-1, :]
    tgt_gx = target[..., :, 1:] - target[..., :, :-1]
    tgt_gy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_gx, tgt_gx) + F.l1_loss(pred_gy, tgt_gy)


def make_rgb_sequence(observed_app, final_frame):
    return torch.cat([observed_app, final_frame.unsqueeze(2)], dim=2)


def set_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad_(requires_grad)


def lsgan_loss(prediction, target_value: float):
    target = torch.full_like(prediction, float(target_value))
    return F.mse_loss(prediction, target)


def compute_generator_losses(generator, discriminator, observed_app, motion, target_app, args):
    recon_motion, pred_frame, aux = generator(observed_app, motion)
    loss_recon_motion = F.mse_loss(recon_motion, motion)
    loss_pred_frame = F.mse_loss(pred_frame, target_app)
    loss_grad = gradient_loss_2d(pred_frame, target_app)
    compact_loss = F.mse_loss(aux["query"], aux["mem_read"])
    att = aux["att"].clamp_min(1e-12)
    entropy_loss = -(att * torch.log(att)).sum(dim=1).mean()
    fake_seq = make_rgb_sequence(observed_app, pred_frame)
    adv_loss = 0.5 * lsgan_loss(discriminator(fake_seq), 1.0)
    total_loss = (
        args.w_recon_motion * loss_recon_motion
        + args.w_pred_frame * loss_pred_frame
        + args.w_grad * loss_grad
        + args.w_compact * compact_loss
        + args.w_entropy * entropy_loss
        + args.w_adv * adv_loss
    )
    loss_dict = {
        "total_g": float(total_loss.detach().item()),
        "recon_motion": float(loss_recon_motion.detach().item()),
        "pred_frame": float(loss_pred_frame.detach().item()),
        "grad": float(loss_grad.detach().item()),
        "compact": float(compact_loss.detach().item()),
        "entropy": float(entropy_loss.detach().item()),
        "adv_g": float(adv_loss.detach().item()),
    }
    return total_loss, loss_dict


@torch.no_grad()
def compute_sample_errors(generator, discriminator, observed_app, motion, target_app, use_discriminator=False):
    recon_motion, pred_frame, _ = generator(observed_app, motion)
    motion_err = ((recon_motion - motion) ** 2).mean(dim=(1, 2, 3, 4))
    frame_err = ((pred_frame - target_app) ** 2).mean(dim=(1, 2, 3))
    disc_err = None
    if use_discriminator:
        fake_seq = make_rgb_sequence(observed_app, pred_frame)
        disc_map = discriminator(fake_seq)
        disc_err = ((disc_map - 1.0) ** 2).mean(dim=(1, 2, 3, 4))
    return motion_err, frame_err, disc_err


@torch.no_grad()
def collect_train_score_stats(generator, discriminator, loader, device, keep_scores=False, use_discriminator=False):
    generator.eval()
    discriminator.eval()
    motion_errs = []
    frame_errs = []
    disc_errs = []
    for observed_app, motion, target_app, _, _ in tqdm(loader, desc="collect train stats", leave=False):
        observed_app = observed_app.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
        motion = motion.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
        target_app = target_app.to(device, non_blocking=True)
        motion_err, frame_err, disc_err = compute_sample_errors(
            generator,
            discriminator,
            observed_app,
            motion,
            target_app,
            use_discriminator=use_discriminator,
        )
        motion_errs.extend(motion_err.cpu().numpy().tolist())
        frame_errs.extend(frame_err.cpu().numpy().tolist())
        if disc_err is not None:
            disc_errs.extend(disc_err.cpu().numpy().tolist())

    motion_errs = np.asarray(motion_errs, dtype=np.float32)
    frame_errs = np.asarray(frame_errs, dtype=np.float32)
    disc_errs = np.asarray(disc_errs, dtype=np.float32)
    stats = {
        "motion_mean": float(motion_errs.mean()) if len(motion_errs) else 0.0,
        "motion_std": float(motion_errs.std() + 1e-8) if len(motion_errs) else 1.0,
        "frame_mean": float(frame_errs.mean()) if len(frame_errs) else 0.0,
        "frame_std": float(frame_errs.std() + 1e-8) if len(frame_errs) else 1.0,
        "disc_mean": float(disc_errs.mean()) if len(disc_errs) else 0.0,
        "disc_std": float(disc_errs.std() + 1e-8) if len(disc_errs) else 1.0,
    }
    if keep_scores:
        stats["motion_training_stats"] = motion_errs
        stats["frame_training_stats"] = frame_errs
        stats["disc_training_stats"] = disc_errs
    return stats


def save_training_stats(generator, discriminator, loader, device, path: Path, use_discriminator=False):
    stats = collect_train_score_stats(
        generator,
        discriminator,
        loader,
        device,
        keep_scores=True,
        use_discriminator=use_discriminator,
    )
    hf_stats = {
        "of_training_stats": stats["motion_training_stats"],
        "frame_training_stats": stats["frame_training_stats"],
        "disc_training_stats": stats["disc_training_stats"],
        "motion_mean": stats["motion_mean"],
        "motion_std": stats["motion_std"],
        "frame_mean": stats["frame_mean"],
        "frame_std": stats["frame_std"],
        "disc_mean": stats["disc_mean"],
        "disc_std": stats["disc_std"],
        "saved_at_utc": utc_now_iso(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(hf_stats, path)
    return hf_stats


def load_training_stats(path: Path):
    stats = torch.load(path, map_location="cpu", weights_only=False)
    if "motion_mean" not in stats:
        of_scores = np.asarray(stats["of_training_stats"], dtype=np.float32)
        frame_scores = np.asarray(stats["frame_training_stats"], dtype=np.float32)
        disc_scores = np.asarray(stats.get("disc_training_stats", []), dtype=np.float32)
        stats["motion_mean"] = float(of_scores.mean()) if len(of_scores) else 0.0
        stats["motion_std"] = float(of_scores.std() + 1e-8) if len(of_scores) else 1.0
        stats["frame_mean"] = float(frame_scores.mean()) if len(frame_scores) else 0.0
        stats["frame_std"] = float(frame_scores.std() + 1e-8) if len(frame_scores) else 1.0
        stats["disc_mean"] = float(disc_scores.mean()) if len(disc_scores) else 0.0
        stats["disc_std"] = float(disc_scores.std() + 1e-8) if len(disc_scores) else 1.0
    return {
        "motion_mean": float(stats["motion_mean"]),
        "motion_std": float(stats["motion_std"]),
        "frame_mean": float(stats["frame_mean"]),
        "frame_std": float(stats["frame_std"]),
        "disc_mean": float(stats.get("disc_mean", 0.0)),
        "disc_std": float(stats.get("disc_std", 1.0)),
    }


def _gt_sort_key(k):
    name = Path(str(k)).stem
    parts = name.split("_")
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return int(parts[0]), int(parts[1])
    return int(name), 0


def _sorted_gt_items(gt):
    return sorted(gt.items(), key=lambda item: _gt_sort_key(item[0]))


def load_pickle_compat(path: Path):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError as exc:
            if not str(exc).endswith("'numpy._core.numeric'"):
                raise
            f.seek(0)

            class NumpyCompatUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if module.startswith("numpy._core"):
                        module = module.replace("numpy._core", "numpy.core", 1)
                    return super().find_class(module, name)

            return NumpyCompatUnpickler(f).load()


def load_gt_labels(dataset_base_dir: Path, dataset_name: str):
    gt_dir = dataset_base_dir / dataset_name / "ground_truth_demo"
    candidates = ["gt_label_12fps.json", "gt_label.json"] if dataset_name == "shanghaitech" else ["gt_label.json"]
    gt_path = next((gt_dir / name for name in candidates if (gt_dir / name).exists()), None)
    if gt_path is None:
        raise FileNotFoundError(f"No ground-truth file found in {gt_dir}")

    gt = load_pickle_compat(gt_path)
    gt_items = _sorted_gt_items(gt)
    testing_frame_counts = [len(np.asarray(labels)) for _, labels in gt_items]
    gt_concat = np.concatenate([np.asarray(labels, dtype=np.float32) for _, labels in gt_items], axis=0)
    return gt_concat, testing_frame_counts, gt_path


def smooth_scores_by_video(scores: np.ndarray, testing_frame_counts: List[int], trim: int = 4) -> np.ndarray:
    smoothed = []
    start = 0
    for video_len in testing_frame_counts:
        cur = np.asarray(scores[start + trim:start + video_len], dtype=np.float32)
        if len(cur) >= 17:
            cur = signal.medfilt(cur, kernel_size=17)
        smoothed.append(cur)
        start += video_len
    return np.concatenate(smoothed, axis=0)


@torch.no_grad()
def evaluate_frame_auc(
    generator,
    discriminator,
    loader,
    gt_concat,
    testing_frame_counts,
    device,
    save_dir=None,
    suffix="test",
    motion_w=0.5,
    frame_w=0.5,
    disc_w=0.0,
    train_stats=None,
):
    eval_started_at = utc_now_iso()
    eval_start = time.perf_counter()
    forward_time_sec = 0.0
    num_batches = 0
    num_samples = 0
    generator.eval()
    discriminator.eval()
    total_frames = int(np.sum(testing_frame_counts))
    frame_bbox_scores = [dict() for _ in range(total_frames)]

    motion_mean = train_stats["motion_mean"] if train_stats else 0.0
    motion_std = train_stats["motion_std"] if train_stats else 1.0
    frame_mean = train_stats["frame_mean"] if train_stats else 0.0
    frame_std = train_stats["frame_std"] if train_stats else 1.0
    disc_mean = train_stats["disc_mean"] if train_stats else 0.0
    disc_std = train_stats["disc_std"] if train_stats else 1.0
    use_discriminator = disc_w > 0

    obj_idx = 0
    for observed_app, motion, target_app, _, pred_frame in tqdm(loader, desc="Eval", leave=False):
        batch_size = int(observed_app.shape[0])
        observed_app = observed_app.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
        motion = motion.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
        target_app = target_app.to(device, non_blocking=True)

        sync_if_cuda(device)
        forward_start = time.perf_counter()
        motion_err, frame_err, disc_err = compute_sample_errors(
            generator,
            discriminator,
            observed_app,
            motion,
            target_app,
            use_discriminator=use_discriminator,
        )
        sync_if_cuda(device)
        forward_time_sec += time.perf_counter() - forward_start
        num_batches += 1
        num_samples += batch_size

        motion_err = motion_err.cpu().numpy()
        frame_err = frame_err.cpu().numpy()
        motion_z = (motion_err - motion_mean) / max(motion_std, 1e-8)
        frame_z = (frame_err - frame_mean) / max(frame_std, 1e-8)
        scores = motion_w * motion_z + frame_w * frame_z
        if use_discriminator and disc_err is not None:
            disc_z = (disc_err.cpu().numpy() - disc_mean) / max(disc_std, 1e-8)
            scores = scores + disc_w * disc_z
        pred_frame = pred_frame.cpu().numpy()

        for i, score in enumerate(scores):
            frame_id = int(pred_frame[i])
            if 0 <= frame_id < total_frames:
                frame_bbox_scores[frame_id][obj_idx] = float(score)
            obj_idx += 1

    frame_scores = np.empty(total_frames, dtype=np.float32)
    empty_score = (
        motion_w * ((0.0 - motion_mean) / max(motion_std, 1e-8))
        + frame_w * ((0.0 - frame_mean) / max(frame_std, 1e-8))
        + disc_w * ((0.0 - disc_mean) / max(disc_std, 1e-8))
    )
    for i, item in enumerate(frame_bbox_scores):
        frame_scores[i] = max(item.values()) if item else empty_score

    trimmed_gt = []
    trimmed_scores = []
    start = 0
    for video_len in testing_frame_counts:
        trimmed_gt.append(gt_concat[start:start + video_len][4:])
        trimmed_scores.append(frame_scores[start:start + video_len][4:])
        start += video_len

    gt_eval = np.concatenate(trimmed_gt, axis=0)
    scores_eval = np.concatenate(trimmed_scores, axis=0)
    smoothed_scores = smooth_scores_by_video(frame_scores, testing_frame_counts, trim=4)
    raw_auc = float(roc_auc_score(gt_eval, scores_eval))
    auc = float(roc_auc_score(gt_eval, smoothed_scores))

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(frame_scores, save_dir / f"frame_scores_{suffix}.pkl")
        curves_dir = save_dir / f"anomaly_curves_{suffix}"
        curve_auc = save_evaluation_curves(scores_eval, gt_eval, str(curves_dir), np.asarray(testing_frame_counts) - 4)
        auc = float(curve_auc)

    eval_duration_sec = time.perf_counter() - eval_start
    timing = {
        "started_at_utc": eval_started_at,
        "ended_at_utc": utc_now_iso(),
        "duration_sec": eval_duration_sec,
        "model_forward_sec": forward_time_sec,
        "num_batches": num_batches,
        "num_samples": num_samples,
        "samples_per_sec": num_samples / eval_duration_sec if eval_duration_sec > 0 else None,
        "model_forward_samples_per_sec": num_samples / forward_time_sec if forward_time_sec > 0 else None,
    }

    return {"auc": auc, "raw_auc": raw_auc, "frame_scores": frame_scores, "timing": timing}


@torch.no_grad()
def evaluate_loss(generator, discriminator, loader, device, args):
    generator.eval()
    discriminator.eval()
    keys = ["total_g", "recon_motion", "pred_frame", "grad", "compact", "entropy", "adv_g"]
    running = {key: 0.0 for key in keys}
    for observed_app, motion, target_app, _, _ in tqdm(loader, desc="val", leave=False):
        observed_app = observed_app.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
        motion = motion.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
        target_app = target_app.to(device, non_blocking=True)
        total_loss, loss_dict = compute_generator_losses(generator, discriminator, observed_app, motion, target_app, args)
        running["total_g"] += float(total_loss.item())
        for key in keys[1:]:
            running[key] += float(loss_dict[key])

    n = max(len(loader), 1)
    return {key: value / n for key, value in running.items()}


def train_one_epoch(
    generator,
    discriminator,
    loader,
    optimizer_g,
    optimizer_d,
    scaler_g,
    scaler_d,
    device,
    use_amp,
    args,
):
    generator.train()
    discriminator.train()
    keys = [
        "total_g",
        "total_d",
        "d_real",
        "d_fake",
        "recon_motion",
        "pred_frame",
        "grad",
        "compact",
        "entropy",
        "adv_g",
    ]
    running = {key: 0.0 for key in keys}
    pbar = tqdm(loader, desc="train", leave=False)
    for observed_app, motion, target_app, _, _ in pbar:
        observed_app = observed_app.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
        motion = motion.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
        target_app = target_app.to(device, non_blocking=True)

        optimizer_g.zero_grad(set_to_none=True)
        optimizer_d.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            recon_motion, pred_frame, aux = generator(observed_app, motion)
            real_seq = make_rgb_sequence(observed_app, target_app)
            fake_seq_detached = make_rgb_sequence(observed_app, pred_frame.detach())
            d_real_loss = 0.5 * lsgan_loss(discriminator(real_seq), 1.0)
            d_fake_loss = 0.5 * lsgan_loss(discriminator(fake_seq_detached), 0.0)
            d_loss = d_real_loss + d_fake_loss

        scaler_d.scale(d_loss).backward()
        scaler_d.unscale_(optimizer_d)
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), DEFAULTS["max_grad_norm_d"])
        scaler_d.step(optimizer_d)
        scaler_d.update()

        optimizer_g.zero_grad(set_to_none=True)
        set_requires_grad(discriminator, False)
        discriminator.eval()
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            loss_recon_motion = F.mse_loss(recon_motion, motion)
            loss_pred_frame = F.mse_loss(pred_frame, target_app)
            loss_grad = gradient_loss_2d(pred_frame, target_app)
            compact_loss = F.mse_loss(aux["query"], aux["mem_read"])
            att = aux["att"].clamp_min(1e-12)
            entropy_loss = -(att * torch.log(att)).sum(dim=1).mean()
            fake_seq = make_rgb_sequence(observed_app, pred_frame)
            adv_loss = 0.5 * lsgan_loss(discriminator(fake_seq), 1.0)
            g_loss = (
                args.w_recon_motion * loss_recon_motion
                + args.w_pred_frame * loss_pred_frame
                + args.w_grad * loss_grad
                + args.w_compact * compact_loss
                + args.w_entropy * entropy_loss
                + args.w_adv * adv_loss
            )
        scaler_g.scale(g_loss).backward()
        scaler_g.unscale_(optimizer_g)
        torch.nn.utils.clip_grad_norm_(generator.parameters(), DEFAULTS["max_grad_norm_g"])
        scaler_g.step(optimizer_g)
        scaler_g.update()
        set_requires_grad(discriminator, True)
        discriminator.train()

        values = {
            "total_g": float(g_loss.detach().item()),
            "total_d": float(d_loss.detach().item()),
            "d_real": float(d_real_loss.detach().item()),
            "d_fake": float(d_fake_loss.detach().item()),
            "recon_motion": float(loss_recon_motion.detach().item()),
            "pred_frame": float(loss_pred_frame.detach().item()),
            "grad": float(loss_grad.detach().item()),
            "compact": float(compact_loss.detach().item()),
            "entropy": float(entropy_loss.detach().item()),
            "adv_g": float(adv_loss.detach().item()),
        }
        for key in keys:
            running[key] += values[key]
        pbar.set_postfix(g=f"{values['total_g']:.4f}", d=f"{values['total_d']:.4f}")

    n = max(len(loader), 1)
    return {key: value / n for key, value in running.items()}


def build_save_dir(args):
    if args.save_dir is not None:
        return Path(args.save_dir)
    return Path("./outputs") / f"TransGANomaly-3DResNet_{args.dataset_name}"


def save_hf_snapshot(
    save_dir: Path,
    generator,
    discriminator,
    optimizer_g,
    optimizer_d,
    epoch: int,
    step: int,
    best_metric=None,
    metric_name="val_total_g",
    max_to_save: int = 5,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "model.pth"
    total_models = glob.glob(str(model_path) + "*")
    if len(total_models) >= max_to_save:
        total_models.sort()
        os.remove(total_models[0])

    path = snapshot_path(save_dir, epoch)
    torch.save(
        {
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "model_state_dict": generator.state_dict(),
            "optimizer_g_state_dict": optimizer_g.state_dict(),
            "optimizer_d_state_dict": optimizer_d.state_dict(),
            "step": step,
            "epoch": epoch,
            "best_metric": best_metric,
            "best_metric_name": metric_name,
            "saved_at_utc": utc_now_iso(),
        },
        path,
    )
    print(f"models {path} save successfully!")
    return path


def save_best_model(save_dir: Path, generator, discriminator) -> Path:
    path = save_dir / "best.pth"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "model_state_dict": generator.state_dict(),
        },
        path,
    )
    print(f"models {path} save successfully!")
    return path


def load_checkpoint(path: Path, generator, discriminator=None, optimizer_g=None, optimizer_d=None, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    gen_state = ckpt.get("generator_state_dict", ckpt.get("model_state_dict"))
    if gen_state is None:
        raise KeyError(f"No generator/model state dict found in {path}")
    generator.load_state_dict(gen_state)
    if discriminator is not None and "discriminator_state_dict" in ckpt:
        discriminator.load_state_dict(ckpt["discriminator_state_dict"])
    if optimizer_g is not None and "optimizer_g_state_dict" in ckpt:
        optimizer_g.load_state_dict(ckpt["optimizer_g_state_dict"])
    if optimizer_d is not None and "optimizer_d_state_dict" in ckpt:
        optimizer_d.load_state_dict(ckpt["optimizer_d_state_dict"])
    return ckpt


def make_loader(dataset, batch_size, shuffle, num_workers, pin_memory, persistent_workers=False, prefetch_factor=2):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)


def make_loader_from_args(dataset, args, shuffle, device):
    return make_loader(
        dataset,
        args.batch_size,
        shuffle,
        args.num_workers,
        device.type == "cuda",
        args.persistent_workers,
        args.prefetch_factor,
    )


def init_models_optimizers(device, args):
    generator = TransGANomaly3DResNetGenerator(
        fea_dim=DEFAULTS["fea_dim"],
        mem_dim=DEFAULTS["mem_dim"],
        mem_temperature=DEFAULTS["mem_temperature"],
        mem_shrink_thr=DEFAULTS["mem_shrink_thr"],
        detach_recon_motion=args.detach_recon_motion,
    ).to(device)
    discriminator = Conditional3DPatchDiscriminator(in_ch=3, base_ch=32).to(device)
    optimizer_g = optim.Adam(
        generator.parameters(),
        lr=args.lr_g,
        betas=(0.5, 0.999),
        eps=1e-6,
        weight_decay=args.weight_decay_g,
    )
    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=args.lr_d,
        betas=(0.5, 0.999),
        eps=1e-6,
        weight_decay=args.weight_decay_d,
    )
    scaler_g = torch.amp.GradScaler(device.type, enabled=DEFAULTS["use_amp"] and device.type == "cuda")
    scaler_d = torch.amp.GradScaler(device.type, enabled=DEFAULTS["use_amp"] and device.type == "cuda")
    return generator, discriminator, optimizer_g, optimizer_d, scaler_g, scaler_d


def split_holdout_indices(dataset: ChunkedSamplesDataset, val_ratio: float, seed: int, split_unit: str):
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("--val_ratio must be in (0, 1) for training")

    rng = np.random.default_rng(seed)
    n = len(dataset)
    if split_unit == "sample":
        indices = np.arange(n)
        rng.shuffle(indices)
        val_size = max(1, int(round(n * val_ratio)))
        val_idx = np.sort(indices[:val_size])
        train_idx = np.sort(indices[val_size:])
        return train_idx, val_idx

    groups = dataset.get_pred_frames()
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)
    val_group_count = max(1, int(round(len(unique_groups) * val_ratio)))
    val_groups = unique_groups[:val_group_count]
    val_mask = np.isin(groups, val_groups)
    val_idx = np.flatnonzero(val_mask)
    train_idx = np.flatnonzero(~val_mask)
    return train_idx, val_idx


def iter_kfold_indices(dataset: ChunkedSamplesDataset, n_splits: int, seed: int, split_unit: str):
    if split_unit == "sample":
        all_indices = np.arange(len(dataset))
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, val_idx in splitter.split(all_indices):
            yield np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)
        return

    groups = dataset.get_pred_frames()
    unique_groups = np.unique(groups)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_group_idx, val_group_idx in splitter.split(unique_groups):
        train_groups = unique_groups[train_group_idx]
        val_groups = unique_groups[val_group_idx]
        train_idx = np.flatnonzero(np.isin(groups, train_groups))
        val_idx = np.flatnonzero(np.isin(groups, val_groups))
        yield train_idx.astype(np.int64), val_idx.astype(np.int64)


def save_split_indices(save_dir: Path, train_idx: np.ndarray, val_idx: np.ndarray):
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "train_indices.npy", train_idx.astype(np.int64))
    np.save(save_dir / "val_indices.npy", val_idx.astype(np.int64))


def maybe_training_stats_subset(train_dataset, save_dir: Path):
    index_path = save_dir / "train_indices.npy"
    if index_path.exists():
        return Subset(train_dataset, np.load(index_path).astype(np.int64).tolist())
    return train_dataset


def run_train(args, device, use_amp, train_dataset, test_dataset, gt_concat, testing_frame_counts, save_dir):
    events_path = save_dir / "events.jsonl"
    train_index_path = save_dir / "train_indices.npy"
    val_index_path = save_dir / "val_indices.npy"
    if train_index_path.exists() and val_index_path.exists():
        train_idx = np.load(train_index_path).astype(np.int64)
        val_idx = np.load(val_index_path).astype(np.int64)
        print(f"[SPLIT] Reusing {train_index_path} and {val_index_path}")
    else:
        train_idx, val_idx = split_holdout_indices(train_dataset, args.val_ratio, args.seed, args.split_unit)
        save_split_indices(save_dir, train_idx, val_idx)

    append_jsonl(
        events_path,
        {
            "event": "train_start",
            "timestamp_utc": utc_now_iso(),
            "epochs": args.epochs,
            "resume": bool(args.resume),
            "checkpoint": args.checkpoint,
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
            "split_unit": args.split_unit,
            "gan": "LS-GAN conditional 3D PatchGAN",
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
        },
    )

    train_subset = Subset(train_dataset, train_idx.tolist())
    val_subset = Subset(train_dataset, val_idx.tolist())
    train_loader = make_loader_from_args(train_subset, args, True, device)
    train_stats_loader = make_loader_from_args(train_subset, args, False, device)
    val_loader = make_loader_from_args(val_subset, args, False, device)
    test_loader = None
    if args.eval_test_during_train:
        if test_dataset is None or gt_concat is None or testing_frame_counts is None:
            raise ValueError("--eval_test_during_train requires testing data and ground truth")
        test_loader = make_loader_from_args(test_dataset, args, False, device)

    generator, discriminator, optimizer_g, optimizer_d, scaler_g, scaler_d = init_models_optimizers(device, args)
    best_path = save_dir / "best.pth"
    start_epoch = 0
    step = 0
    best_val_loss = float("inf")
    best_epoch = None
    bad_val_checks = 0
    stopped_early = False
    stop_epoch = None

    resume_path = Path(args.checkpoint) if args.checkpoint is not None else latest_snapshot_path(save_dir)
    if args.resume or args.checkpoint is not None:
        if resume_path is None or not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found for resume in {save_dir}")
        ckpt = load_checkpoint(
            resume_path,
            generator,
            discriminator,
            optimizer_g if args.resume else None,
            optimizer_d if args.resume else None,
            device=device,
        )
        best_val_loss = float(ckpt.get("best_metric", best_val_loss))
        summary_path = save_dir / "train_summary.json"
        if summary_path.exists():
            try:
                best_epoch = json.loads(summary_path.read_text()).get("best_epoch")
            except json.JSONDecodeError:
                best_epoch = None
        step = int(ckpt.get("step", 0))
        if args.resume:
            start_epoch = int(ckpt.get("epoch", 0))
            print(f"[RESUME] {resume_path} epoch={start_epoch} best_val_loss={best_val_loss:.6f}")

    for epoch in range(start_epoch, args.epochs):
        epoch_started_at = utc_now_iso()
        epoch_start = time.perf_counter()
        train_start = time.perf_counter()
        train_metrics = train_one_epoch(
            generator,
            discriminator,
            train_loader,
            optimizer_g,
            optimizer_d,
            scaler_g,
            scaler_d,
            device,
            use_amp,
            args,
        )
        step += len(train_loader)
        train_duration_sec = time.perf_counter() - train_start
        result = {"epoch": epoch + 1, "train": train_metrics, "best_checkpoint_updated": False}
        stop_after_epoch = False

        if (epoch + 1) % args.eval_every == 0:
            val_start = time.perf_counter()
            val_metrics = evaluate_loss(generator, discriminator, val_loader, device, args)
            val_duration_sec = time.perf_counter() - val_start
            result["val"] = val_metrics
            result["val_duration_sec"] = val_duration_sec

            if args.eval_test_during_train:
                audit_start = time.perf_counter()
                train_score_stats = collect_train_score_stats(
                    generator,
                    discriminator,
                    train_stats_loader,
                    device,
                    use_discriminator=args.disc_score_weight > 0,
                )
                audit_metrics = evaluate_frame_auc(
                    generator,
                    discriminator,
                    test_loader,
                    gt_concat,
                    testing_frame_counts,
                    device,
                    save_dir=None,
                    motion_w=args.motion_score_weight,
                    frame_w=args.frame_score_weight,
                    disc_w=args.disc_score_weight,
                    train_stats=train_score_stats,
                )
                result["audit_test_auc"] = audit_metrics["auc"]
                result["audit_raw_test_auc"] = audit_metrics["raw_auc"]
                result["audit_timing"] = audit_metrics["timing"]
                result["audit_duration_sec"] = time.perf_counter() - audit_start

            if val_metrics["total_g"] < best_val_loss - args.early_stop_min_delta:
                best_val_loss = val_metrics["total_g"]
                best_epoch = epoch + 1
                bad_val_checks = 0
                save_best_model(save_dir, generator, discriminator)
                print(f"[BEST] epoch={epoch + 1} val_total_g={best_val_loss:.6f}")
                result["best_checkpoint_updated"] = True
            else:
                bad_val_checks += 1

            result["early_stop"] = {
                "bad_val_checks": bad_val_checks,
                "patience": args.early_stop_patience,
                "min_delta": args.early_stop_min_delta,
            }
            if args.early_stop_patience > 0 and bad_val_checks >= args.early_stop_patience:
                stopped_early = True
                stop_epoch = epoch + 1
                stop_after_epoch = True
                print(
                    f"[EARLY STOP] epoch={epoch + 1} no val_total_g improvement "
                    f"for {bad_val_checks} validation checks"
                )

        snapshot = save_hf_snapshot(
            save_dir,
            generator,
            discriminator,
            optimizer_g,
            optimizer_d,
            epoch + 1,
            step,
            best_val_loss,
            metric_name="val_total_g",
            max_to_save=5,
        )
        if args.stats_every > 0 and (epoch + 1) % args.stats_every == 0:
            stats_path = training_stats_path(save_dir, epoch + 1)
            train_stats = save_training_stats(
                generator,
                discriminator,
                train_stats_loader,
                device,
                stats_path,
                use_discriminator=args.disc_score_weight > 0,
            )
            result["training_stats"] = str(stats_path)
            result["training_stats_summary"] = {
                "motion_mean": train_stats["motion_mean"],
                "motion_std": train_stats["motion_std"],
                "frame_mean": train_stats["frame_mean"],
                "frame_std": train_stats["frame_std"],
                "disc_mean": train_stats["disc_mean"],
                "disc_std": train_stats["disc_std"],
            }
        result["train_duration_sec"] = train_duration_sec
        result["epoch_duration_sec"] = time.perf_counter() - epoch_start
        result["epoch_started_at_utc"] = epoch_started_at
        result["epoch_ended_at_utc"] = utc_now_iso()
        result["checkpoint"] = str(snapshot)
        result["best_val_loss_so_far"] = best_val_loss
        append_jsonl(events_path, {"event": "epoch_end", **result})
        print(json.dumps(result, ensure_ascii=True))
        if stop_after_epoch:
            break

    summary = {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_path),
        "latest_checkpoint": str(latest_snapshot_path(save_dir)) if latest_snapshot_path(save_dir) else None,
        "train_samples": int(len(train_subset)),
        "val_samples": int(len(val_subset)),
        "checkpoint_selection": "validation generator loss from training split only",
        "stopped_early": stopped_early,
        "stop_epoch": stop_epoch,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "latest_training_stats": str(latest_training_stats_path(save_dir)) if latest_training_stats_path(save_dir) else None,
        "events_log": str(events_path),
        "completed_at_utc": utc_now_iso(),
    }
    with open(save_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, ensure_ascii=True))


def run_test(args, device, train_dataset, test_dataset, gt_concat, testing_frame_counts, save_dir):
    events_path = save_dir / "events.jsonl"
    test_started_at = utc_now_iso()
    test_start = time.perf_counter()
    append_jsonl(
        events_path,
        {
            "event": "test_start",
            "timestamp_utc": test_started_at,
            "checkpoint": args.checkpoint,
            "disc_score_weight": args.disc_score_weight,
        },
    )
    test_loader = make_loader_from_args(test_dataset, args, False, device)

    generator, discriminator, _, _, _, _ = init_models_optimizers(device, args)
    ckpt_path = Path(args.checkpoint) if args.checkpoint is not None else save_dir / "best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    load_checkpoint(ckpt_path, generator, discriminator, device=device)
    stats_path = None
    summary_path = save_dir / "train_summary.json"
    if summary_path.exists():
        try:
            best_epoch = json.loads(summary_path.read_text()).get("best_epoch")
            if best_epoch is not None:
                stats_path = training_stats_path(save_dir, int(best_epoch))
        except json.JSONDecodeError:
            stats_path = None
    if stats_path is None:
        stats_path = latest_training_stats_path(save_dir) or training_stats_path(save_dir, 0)
    if stats_path.exists():
        train_score_stats = load_training_stats(stats_path)
    else:
        stats_dataset = maybe_training_stats_subset(train_dataset, save_dir)
        train_stats_loader = make_loader_from_args(stats_dataset, args, False, device)
        train_score_stats = save_training_stats(
            generator,
            discriminator,
            train_stats_loader,
            device,
            stats_path,
            use_discriminator=args.disc_score_weight > 0,
        )
    eval_metrics = evaluate_frame_auc(
        generator,
        discriminator,
        test_loader,
        gt_concat,
        testing_frame_counts,
        device,
        save_dir=save_dir,
        suffix="test",
        motion_w=args.motion_score_weight,
        frame_w=args.frame_score_weight,
        disc_w=args.disc_score_weight,
        train_stats=train_score_stats,
    )
    summary = {
        "checkpoint": str(ckpt_path),
        "test_auc": eval_metrics["auc"],
        "raw_test_auc": eval_metrics["raw_auc"],
        "training_stats": str(stats_path),
        "score_weights": {
            "motion": args.motion_score_weight,
            "frame": args.frame_score_weight,
            "disc": args.disc_score_weight,
        },
        "timing": eval_metrics["timing"],
        "total_test_duration_sec": time.perf_counter() - test_start,
        "started_at_utc": test_started_at,
        "ended_at_utc": utc_now_iso(),
    }
    with open(save_dir / "test_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    append_jsonl(events_path, {"event": "test_end", **summary})
    print(json.dumps(summary, ensure_ascii=True))


def run_kfold_training(args, device, use_amp, train_dataset, test_dataset, gt_concat, testing_frame_counts, save_dir):
    if args.checkpoint is not None and args.fold is None:
        raise ValueError("--checkpoint with --kfold train is only supported together with --fold")

    folds = list(iter_kfold_indices(train_dataset, args.kfold, args.seed, args.split_unit))
    requested_folds = [args.fold] if args.fold is not None else list(range(args.kfold))
    test_loader = None
    if args.eval_test_during_train:
        if test_dataset is None or gt_concat is None or testing_frame_counts is None:
            raise ValueError("--eval_test_during_train requires testing data and ground truth")
        test_loader = make_loader_from_args(test_dataset, args, False, device)

    fold_summaries = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        if fold_idx not in requested_folds:
            continue

        fold_dir = save_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        events_path = fold_dir / "events.jsonl"
        train_index_path = fold_dir / "train_indices.npy"
        val_index_path = fold_dir / "val_indices.npy"
        if train_index_path.exists() and val_index_path.exists():
            train_idx = np.load(train_index_path).astype(np.int64)
            val_idx = np.load(val_index_path).astype(np.int64)
            print(f"[SPLIT] fold={fold_idx} reusing saved split indices")
        else:
            save_split_indices(fold_dir, train_idx, val_idx)

        append_jsonl(
            events_path,
            {
                "event": "train_start",
                "timestamp_utc": utc_now_iso(),
                "mode": "kfold",
                "fold": fold_idx,
                "kfold": args.kfold,
                "epochs": args.epochs,
                "resume": bool(args.resume),
                "checkpoint": args.checkpoint,
                "train_samples": int(len(train_idx)),
                "val_samples": int(len(val_idx)),
                "split_unit": args.split_unit,
                "gan": "LS-GAN conditional 3D PatchGAN",
                "early_stop_patience": args.early_stop_patience,
                "early_stop_min_delta": args.early_stop_min_delta,
            },
        )

        train_subset = Subset(train_dataset, train_idx.tolist())
        val_subset = Subset(train_dataset, val_idx.tolist())
        train_loader = make_loader_from_args(train_subset, args, True, device)
        train_stats_loader = make_loader_from_args(train_subset, args, False, device)
        val_loader = make_loader_from_args(val_subset, args, False, device)

        generator, discriminator, optimizer_g, optimizer_d, scaler_g, scaler_d = init_models_optimizers(device, args)
        best_path = fold_dir / "best.pth"
        best_val_loss = float("inf")
        best_audit_test_auc = None
        start_epoch = 0
        step = 0
        best_epoch = None
        bad_val_checks = 0
        stopped_early = False
        stop_epoch = None

        fold_resume_path = (
            Path(args.checkpoint)
            if args.checkpoint is not None and args.fold == fold_idx
            else latest_snapshot_path(fold_dir)
        )
        if args.resume:
            if fold_resume_path is not None and fold_resume_path.exists():
                ckpt = load_checkpoint(
                    fold_resume_path,
                    generator,
                    discriminator,
                    optimizer_g,
                    optimizer_d,
                    device=device,
                )
                start_epoch = int(ckpt.get("epoch", 0))
                best_val_loss = float(ckpt.get("best_metric", best_val_loss))
                step = int(ckpt.get("step", 0))
                summary_path = fold_dir / "train_summary.json"
                if summary_path.exists():
                    try:
                        best_epoch = json.loads(summary_path.read_text()).get("best_epoch")
                    except json.JSONDecodeError:
                        best_epoch = None
                print(
                    f"[KFOLD RESUME] fold={fold_idx} {fold_resume_path} "
                    f"epoch={start_epoch} best_val_loss={best_val_loss:.6f}"
                )
            else:
                print(f"[KFOLD RESUME] fold={fold_idx} no checkpoint at {fold_resume_path}; starting fresh")
        elif args.checkpoint is not None and args.fold == fold_idx:
            ckpt = load_checkpoint(Path(args.checkpoint), generator, discriminator, device=device)
            best_val_loss = float(ckpt.get("best_metric", best_val_loss))
            print(f"[KFOLD LOAD] fold={fold_idx} loaded model from {args.checkpoint}")

        print(f"[KFOLD] fold={fold_idx} train_samples={len(train_subset)} val_samples={len(val_subset)}")

        for epoch in range(start_epoch, args.epochs):
            epoch_started_at = utc_now_iso()
            epoch_start = time.perf_counter()
            train_start = time.perf_counter()
            train_metrics = train_one_epoch(
                generator,
                discriminator,
                train_loader,
                optimizer_g,
                optimizer_d,
                scaler_g,
                scaler_d,
                device,
                use_amp,
                args,
            )
            step += len(train_loader)
            train_duration_sec = time.perf_counter() - train_start
            result = {
                "fold": fold_idx,
                "epoch": epoch + 1,
                "train": train_metrics,
                "best_checkpoint_updated": False,
            }
            stop_after_epoch = False

            if (epoch + 1) % args.eval_every == 0:
                val_start = time.perf_counter()
                val_metrics = evaluate_loss(generator, discriminator, val_loader, device, args)
                val_duration_sec = time.perf_counter() - val_start
                result["val"] = val_metrics
                result["val_duration_sec"] = val_duration_sec

                if args.eval_test_during_train:
                    audit_start = time.perf_counter()
                    train_score_stats = collect_train_score_stats(
                        generator,
                        discriminator,
                        train_stats_loader,
                        device,
                        use_discriminator=args.disc_score_weight > 0,
                    )
                    audit_metrics = evaluate_frame_auc(
                        generator,
                        discriminator,
                        test_loader,
                        gt_concat,
                        testing_frame_counts,
                        device,
                        save_dir=None,
                        motion_w=args.motion_score_weight,
                        frame_w=args.frame_score_weight,
                        disc_w=args.disc_score_weight,
                        train_stats=train_score_stats,
                    )
                    result["audit_test_auc"] = audit_metrics["auc"]
                    result["audit_raw_test_auc"] = audit_metrics["raw_auc"]
                    result["audit_timing"] = audit_metrics["timing"]
                    result["audit_duration_sec"] = time.perf_counter() - audit_start

                if val_metrics["total_g"] < best_val_loss - args.early_stop_min_delta:
                    best_val_loss = val_metrics["total_g"]
                    best_epoch = epoch + 1
                    best_audit_test_auc = result.get("audit_test_auc")
                    bad_val_checks = 0
                    save_best_model(fold_dir, generator, discriminator)
                    print(f"[KFOLD BEST] fold={fold_idx} epoch={epoch + 1} val_total_g={best_val_loss:.6f}")
                    result["best_checkpoint_updated"] = True
                else:
                    bad_val_checks += 1

                result["early_stop"] = {
                    "bad_val_checks": bad_val_checks,
                    "patience": args.early_stop_patience,
                    "min_delta": args.early_stop_min_delta,
                }
                if args.early_stop_patience > 0 and bad_val_checks >= args.early_stop_patience:
                    stopped_early = True
                    stop_epoch = epoch + 1
                    stop_after_epoch = True
                    print(
                        f"[KFOLD EARLY STOP] fold={fold_idx} epoch={epoch + 1} "
                        f"no val_total_g improvement for {bad_val_checks} validation checks"
                    )

            snapshot = save_hf_snapshot(
                fold_dir,
                generator,
                discriminator,
                optimizer_g,
                optimizer_d,
                epoch + 1,
                step,
                best_val_loss,
                metric_name="val_total_g",
                max_to_save=5,
            )
            if args.stats_every > 0 and (epoch + 1) % args.stats_every == 0:
                stats_path = training_stats_path(fold_dir, epoch + 1)
                train_stats = save_training_stats(
                    generator,
                    discriminator,
                    train_stats_loader,
                    device,
                    stats_path,
                    use_discriminator=args.disc_score_weight > 0,
                )
                result["training_stats"] = str(stats_path)
                result["training_stats_summary"] = {
                    "motion_mean": train_stats["motion_mean"],
                    "motion_std": train_stats["motion_std"],
                    "frame_mean": train_stats["frame_mean"],
                    "frame_std": train_stats["frame_std"],
                    "disc_mean": train_stats["disc_mean"],
                    "disc_std": train_stats["disc_std"],
                }
            result["train_duration_sec"] = train_duration_sec
            result["epoch_duration_sec"] = time.perf_counter() - epoch_start
            result["epoch_started_at_utc"] = epoch_started_at
            result["epoch_ended_at_utc"] = utc_now_iso()
            result["checkpoint"] = str(snapshot)
            result["best_val_loss_so_far"] = best_val_loss
            append_jsonl(events_path, {"event": "epoch_end", **result})
            print(json.dumps(result, ensure_ascii=True))
            if stop_after_epoch:
                break

        fold_summary = {
            "fold": fold_idx,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "audit_test_auc_at_best_val": best_audit_test_auc,
            "best_checkpoint": str(best_path),
            "latest_checkpoint": str(latest_snapshot_path(fold_dir)) if latest_snapshot_path(fold_dir) else None,
            "train_samples": int(len(train_subset)),
            "val_samples": int(len(val_subset)),
            "checkpoint_selection": "validation generator loss from training fold only",
            "stopped_early": stopped_early,
            "stop_epoch": stop_epoch,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            "latest_training_stats": str(latest_training_stats_path(fold_dir)) if latest_training_stats_path(fold_dir) else None,
            "events_log": str(events_path),
            "completed_at_utc": utc_now_iso(),
        }
        with open(fold_dir / "train_summary.json", "w", encoding="utf-8") as f:
            json.dump(fold_summary, f, indent=2)
        fold_summaries.append(fold_summary)
        print(json.dumps(fold_summary, ensure_ascii=True))

    if not fold_summaries:
        raise ValueError("No fold was executed. Check --fold against --kfold.")

    summary = {
        "kfold": args.kfold,
        "executed_folds": [item["fold"] for item in fold_summaries],
        "mean_best_val_loss": float(np.mean([item["best_val_loss"] for item in fold_summaries])),
        "std_best_val_loss": float(np.std([item["best_val_loss"] for item in fold_summaries])),
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "folds": fold_summaries,
    }
    with open(save_dir / "kfold_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, ensure_ascii=True))


def run_kfold_test(args, device, train_dataset, test_dataset, gt_concat, testing_frame_counts, save_dir):
    if args.checkpoint is not None and args.fold is None:
        raise ValueError("--checkpoint with --kfold test is only supported together with --fold")

    events_path = save_dir / "events.jsonl"
    test_started_at = utc_now_iso()
    test_start = time.perf_counter()
    append_jsonl(
        events_path,
        {
            "event": "test_start",
            "timestamp_utc": test_started_at,
            "mode": "kfold",
            "kfold": args.kfold,
            "fold": args.fold,
            "checkpoint": args.checkpoint,
            "disc_score_weight": args.disc_score_weight,
        },
    )

    folds = list(iter_kfold_indices(train_dataset, args.kfold, args.seed, args.split_unit))
    requested_folds = [args.fold] if args.fold is not None else list(range(args.kfold))
    test_loader = make_loader_from_args(test_dataset, args, False, device)
    frame_scores_list = []
    fold_results = []

    for fold_idx, (train_idx, _) in enumerate(folds):
        if fold_idx not in requested_folds:
            continue

        fold_dir = save_dir / f"fold_{fold_idx}"
        ckpt_path = Path(args.checkpoint) if args.checkpoint is not None and args.fold == fold_idx else fold_dir / "best.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found for fold {fold_idx}: {ckpt_path}")

        split_train_path = fold_dir / "train_indices.npy"
        if split_train_path.exists():
            train_idx = np.load(split_train_path).astype(np.int64)

        generator, discriminator, _, _, _, _ = init_models_optimizers(device, args)
        load_checkpoint(ckpt_path, generator, discriminator, device=device)
        stats_path = None
        summary_path = fold_dir / "train_summary.json"
        if summary_path.exists():
            try:
                best_epoch = json.loads(summary_path.read_text()).get("best_epoch")
                if best_epoch is not None:
                    stats_path = training_stats_path(fold_dir, int(best_epoch))
            except json.JSONDecodeError:
                stats_path = None
        if stats_path is None:
            stats_path = latest_training_stats_path(fold_dir) or training_stats_path(fold_dir, 0)
        if stats_path.exists():
            train_score_stats = load_training_stats(stats_path)
        else:
            train_subset = Subset(train_dataset, train_idx.tolist())
            train_stats_loader = make_loader_from_args(train_subset, args, False, device)
            train_score_stats = save_training_stats(
                generator,
                discriminator,
                train_stats_loader,
                device,
                stats_path,
                use_discriminator=args.disc_score_weight > 0,
            )
        eval_metrics = evaluate_frame_auc(
            generator,
            discriminator,
            test_loader,
            gt_concat,
            testing_frame_counts,
            device,
            save_dir=fold_dir,
            suffix="test",
            motion_w=args.motion_score_weight,
            frame_w=args.frame_score_weight,
            disc_w=args.disc_score_weight,
            train_stats=train_score_stats,
        )
        frame_scores_list.append(eval_metrics["frame_scores"])
        fold_results.append(
            {
                "fold": fold_idx,
                "checkpoint": str(ckpt_path),
                "test_auc": eval_metrics["auc"],
                "raw_test_auc": eval_metrics["raw_auc"],
                "training_stats": str(stats_path),
                "timing": eval_metrics["timing"],
            }
        )
        print(json.dumps(fold_results[-1], ensure_ascii=True))

    if not frame_scores_list:
        raise ValueError("No fold checkpoint evaluated. Check --fold and checkpoint paths.")

    ensemble_scores = np.mean(np.stack(frame_scores_list, axis=0), axis=0)
    trimmed_gt = []
    trimmed_scores = []
    start = 0
    for video_len in testing_frame_counts:
        trimmed_gt.append(gt_concat[start:start + video_len][4:])
        trimmed_scores.append(ensemble_scores[start:start + video_len][4:])
        start += video_len
    gt_eval = np.concatenate(trimmed_gt, axis=0)
    scores_eval = np.concatenate(trimmed_scores, axis=0)
    smoothed_scores = smooth_scores_by_video(ensemble_scores, testing_frame_counts, trim=4)

    joblib.dump(ensemble_scores, save_dir / "frame_scores_test_ensemble.pkl")
    curves_dir = save_dir / "anomaly_curves_test_ensemble"
    ensemble_auc = float(save_evaluation_curves(scores_eval, gt_eval, str(curves_dir), np.asarray(testing_frame_counts) - 4))
    raw_ensemble_auc = float(roc_auc_score(gt_eval, scores_eval))
    smoothed_ensemble_auc = float(roc_auc_score(gt_eval, smoothed_scores))

    summary = {
        "kfold": args.kfold,
        "executed_folds": [item["fold"] for item in fold_results],
        "ensemble_test_auc": ensemble_auc,
        "raw_ensemble_test_auc": raw_ensemble_auc,
        "smoothed_ensemble_test_auc": smoothed_ensemble_auc,
        "score_weights": {
            "motion": args.motion_score_weight,
            "frame": args.frame_score_weight,
            "disc": args.disc_score_weight,
        },
        "fold_results": fold_results,
        "total_test_duration_sec": time.perf_counter() - test_start,
        "started_at_utc": test_started_at,
        "ended_at_utc": utc_now_iso(),
    }
    with open(save_dir / "test_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    append_jsonl(events_path, {"event": "test_end", **summary})
    print(json.dumps(summary, ensure_ascii=True))


def main():
    args = parse_args()
    set_seed(args.seed)

    dataset_base_dir = Path(args.dataset_base_dir)
    train_dir = dataset_base_dir / args.dataset_name / "training" / "chunked_samples"
    test_dir = dataset_base_dir / args.dataset_name / "testing" / "chunked_samples"
    save_dir = build_save_dir(args)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or str(args.device) == "cpu" else "cpu")
    use_amp = DEFAULTS["use_amp"] and device.type == "cuda"

    train_dataset = ChunkedSamplesDataset(train_dir, max_cache_chunks=args.max_cache_chunks)
    test_dataset = None
    gt_concat = None
    testing_frame_counts = None
    if args.mode == "test" or args.eval_test_during_train:
        test_dataset = ChunkedSamplesDataset(test_dir, max_cache_chunks=args.max_cache_chunks)
        gt_concat, testing_frame_counts, gt_path = load_gt_labels(dataset_base_dir, args.dataset_name)
        print(f"[GT] {gt_path} | frames={int(np.sum(testing_frame_counts))} | videos={len(testing_frame_counts)}")

    if args.kfold < 1:
        raise ValueError("--kfold must be >= 1")
    if args.fold is not None and not (0 <= args.fold < args.kfold):
        raise ValueError("--fold must be in [0, kfold-1]")

    if args.mode == "train":
        if args.kfold > 1:
            run_kfold_training(args, device, use_amp, train_dataset, test_dataset, gt_concat, testing_frame_counts, save_dir)
        else:
            run_train(args, device, use_amp, train_dataset, test_dataset, gt_concat, testing_frame_counts, save_dir)
    else:
        if args.kfold > 1:
            run_kfold_test(args, device, train_dataset, test_dataset, gt_concat, testing_frame_counts, save_dir)
        else:
            run_test(args, device, train_dataset, test_dataset, gt_concat, testing_frame_counts, save_dir)


if __name__ == "__main__":
    main()
