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
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "num_workers": 0,
    "use_amp": True,
    "max_grad_norm": 1.0,
    "fea_dim": 128,
    "mem_dim": 512,
    "mem_temperature": 0.07,
    "mem_shrink_thr": 0.0025,
    "w_recon_motion": 1.0,
    "w_pred_frame": 1.0,
    "w_compact": 0.02,
    "w_entropy": 0.001,
    "w_grad": 0.10,
    "motion_score_weight": 0.5,
    "frame_score_weight": 0.5,
    "val_ratio": 0.10,
    "split_unit": "frame",
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="HF2VAD-like reconstruction + prediction using 3D ResNet on chunked_samples"
    )
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--dataset_name", choices=["ped2", "avenue", "shanghaitech"], default="shanghaitech")
    parser.add_argument("--dataset_base_dir", default="./data")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--weight_decay", type=float, default=DEFAULTS["weight_decay"])
    parser.add_argument("--num_workers", type=int, default=DEFAULTS["num_workers"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path for test mode or resume")
    parser.add_argument("--resume", action="store_true", help="Resume optimizer/model from --checkpoint in train mode")
    parser.add_argument("--kfold", type=int, default=1, help="Number of folds for cross-validation training")
    parser.add_argument("--fold", type=int, default=None, help="Run/test a single fold index (0-based)")
    parser.add_argument("--val_ratio", type=float, default=DEFAULTS["val_ratio"])
    parser.add_argument("--split_unit", choices=["frame", "sample"], default=DEFAULTS["split_unit"])
    parser.add_argument(
        "--eval_test_during_train",
        action="store_true",
        help="Log test AUC during train; never used for checkpoint selection",
    )
    parser.add_argument(
        "--detach_recon_motion",
        action="store_true",
        help="Stop prediction loss from updating the motion reconstruction branch",
    )

    parser.add_argument("--w_recon_motion", type=float, default=DEFAULTS["w_recon_motion"])
    parser.add_argument("--w_pred_frame", type=float, default=DEFAULTS["w_pred_frame"])
    parser.add_argument("--w_grad", type=float, default=DEFAULTS["w_grad"])
    parser.add_argument("--w_compact", type=float, default=DEFAULTS["w_compact"])
    parser.add_argument("--w_entropy", type=float, default=DEFAULTS["w_entropy"])
    parser.add_argument("--motion_score_weight", type=float, default=DEFAULTS["motion_score_weight"])
    parser.add_argument("--frame_score_weight", type=float, default=DEFAULTS["frame_score_weight"])
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
    def __init__(self, chunk_dir: Path):
        self.chunk_dir = Path(chunk_dir)
        self.chunk_files = sorted(self.chunk_dir.glob("chunked_samples_*.pkl"), key=lambda p: p.name)
        if not self.chunk_files:
            raise FileNotFoundError(f"No chunked_samples_*.pkl found in {self.chunk_dir}")

        self.chunk_lengths: List[int] = []
        self.cum_lengths: List[int] = []
        self.cache: Dict[int, Dict[str, np.ndarray]] = {}

        total = 0
        for chunk_file in self.chunk_files:
            payload = joblib.load(chunk_file, mmap_mode="r")
            chunk_len = len(payload["sample_id"])
            self.chunk_lengths.append(chunk_len)
            total += chunk_len
            self.cum_lengths.append(total)

        self.total_len = total
        self._pred_frames: Optional[np.ndarray] = None
        print(f"[ChunkedSamplesDataset] {self.chunk_dir} | files={len(self.chunk_files)} | samples={self.total_len}")

    def __len__(self):
        return self.total_len

    def _resolve_index(self, idx: int) -> Tuple[int, int]:
        chunk_idx = bisect.bisect_right(self.cum_lengths, idx)
        prev_cum = 0 if chunk_idx == 0 else self.cum_lengths[chunk_idx - 1]
        return chunk_idx, idx - prev_cum

    def _load_chunk(self, chunk_idx: int):
        if chunk_idx not in self.cache:
            if len(self.cache) >= 2:
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


class HF2VADLike3DResNet(nn.Module):
    """
    HF2VAD-like flow:
    1. reconstruct motion through a memory autoencoder branch;
    2. predict the next frame from observed appearance plus reconstructed motion.
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


def hard_shrink_relu(inp, lambd, eps=1e-12):
    return (F.relu(inp - lambd) * inp) / (torch.abs(inp - lambd) + eps)


def gradient_loss_2d(pred, target):
    pred_gx = pred[..., :, 1:] - pred[..., :, :-1]
    pred_gy = pred[..., 1:, :] - pred[..., :-1, :]
    tgt_gx = target[..., :, 1:] - target[..., :, :-1]
    tgt_gy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_gx, tgt_gx) + F.l1_loss(pred_gy, tgt_gy)


def compute_losses(model, observed_app, motion, target_app, args):
    recon_motion, pred_frame, aux = model(observed_app, motion)
    loss_recon_motion = F.mse_loss(recon_motion, motion)
    loss_pred_frame = F.mse_loss(pred_frame, target_app)
    loss_grad = gradient_loss_2d(pred_frame, target_app)
    compact_loss = F.mse_loss(aux["query"], aux["mem_read"])
    att = aux["att"].clamp_min(1e-12)
    entropy_loss = -(att * torch.log(att)).sum(dim=1).mean()
    total_loss = (
        args.w_recon_motion * loss_recon_motion
        + args.w_pred_frame * loss_pred_frame
        + args.w_grad * loss_grad
        + args.w_compact * compact_loss
        + args.w_entropy * entropy_loss
    )
    loss_dict = {
        "total": float(total_loss.detach().item()),
        "recon_motion": float(loss_recon_motion.detach().item()),
        "pred_frame": float(loss_pred_frame.detach().item()),
        "grad": float(loss_grad.detach().item()),
        "compact": float(compact_loss.detach().item()),
        "entropy": float(entropy_loss.detach().item()),
    }
    return total_loss, loss_dict


@torch.no_grad()
def compute_sample_errors(model, observed_app, motion, target_app):
    recon_motion, pred_frame, _ = model(observed_app, motion)
    motion_err = ((recon_motion - motion) ** 2).mean(dim=(1, 2, 3, 4))
    frame_err = ((pred_frame - target_app) ** 2).mean(dim=(1, 2, 3))
    return motion_err, frame_err


@torch.no_grad()
def collect_train_score_stats(model, loader, device, keep_scores=False):
    model.eval()
    motion_errs = []
    frame_errs = []
    for observed_app, motion, target_app, _, _ in tqdm(loader, desc="collect train stats", leave=False):
        observed_app = observed_app.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
        motion = motion.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
        target_app = target_app.to(device, non_blocking=True)
        motion_err, frame_err = compute_sample_errors(model, observed_app, motion, target_app)
        motion_errs.extend(motion_err.cpu().numpy().tolist())
        frame_errs.extend(frame_err.cpu().numpy().tolist())

    motion_errs = np.asarray(motion_errs, dtype=np.float32)
    frame_errs = np.asarray(frame_errs, dtype=np.float32)
    stats = {
        "motion_mean": float(motion_errs.mean()) if len(motion_errs) else 0.0,
        "motion_std": float(motion_errs.std() + 1e-8) if len(motion_errs) else 1.0,
        "frame_mean": float(frame_errs.mean()) if len(frame_errs) else 0.0,
        "frame_std": float(frame_errs.std() + 1e-8) if len(frame_errs) else 1.0,
    }
    if keep_scores:
        stats["motion_training_stats"] = motion_errs
        stats["frame_training_stats"] = frame_errs
    return stats


def save_training_stats(model, loader, device, path: Path):
    stats = collect_train_score_stats(model, loader, device, keep_scores=True)
    hf_stats = {
        "of_training_stats": stats["motion_training_stats"],
        "frame_training_stats": stats["frame_training_stats"],
        "motion_mean": stats["motion_mean"],
        "motion_std": stats["motion_std"],
        "frame_mean": stats["frame_mean"],
        "frame_std": stats["frame_std"],
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
        stats["motion_mean"] = float(of_scores.mean()) if len(of_scores) else 0.0
        stats["motion_std"] = float(of_scores.std() + 1e-8) if len(of_scores) else 1.0
        stats["frame_mean"] = float(frame_scores.mean()) if len(frame_scores) else 0.0
        stats["frame_std"] = float(frame_scores.std() + 1e-8) if len(frame_scores) else 1.0
    return {
        "motion_mean": float(stats["motion_mean"]),
        "motion_std": float(stats["motion_std"]),
        "frame_mean": float(stats["frame_mean"]),
        "frame_std": float(stats["frame_std"]),
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
        raise FileNotFoundError(f"No ground-truth file found in {gt_dir}; tried {candidates}")

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
    model,
    loader,
    gt_concat,
    testing_frame_counts,
    device,
    save_dir=None,
    suffix="test",
    motion_w=0.5,
    frame_w=0.5,
    train_stats=None,
):
    eval_started_at = utc_now_iso()
    eval_start = time.perf_counter()
    forward_time_sec = 0.0
    num_batches = 0
    num_samples = 0
    model.eval()
    total_frames = int(np.sum(testing_frame_counts))
    frame_bbox_scores = [dict() for _ in range(total_frames)]

    motion_mean = train_stats["motion_mean"] if train_stats else 0.0
    motion_std = train_stats["motion_std"] if train_stats else 1.0
    frame_mean = train_stats["frame_mean"] if train_stats else 0.0
    frame_std = train_stats["frame_std"] if train_stats else 1.0

    obj_idx = 0
    for observed_app, motion, target_app, _, pred_frame in tqdm(loader, desc="Eval", leave=False):
        batch_size = int(observed_app.shape[0])
        observed_app = observed_app.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
        motion = motion.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
        target_app = target_app.to(device, non_blocking=True)

        sync_if_cuda(device)
        forward_start = time.perf_counter()
        motion_err, frame_err = compute_sample_errors(model, observed_app, motion, target_app)
        sync_if_cuda(device)
        forward_time_sec += time.perf_counter() - forward_start
        num_batches += 1
        num_samples += batch_size

        motion_err = motion_err.cpu().numpy()
        frame_err = frame_err.cpu().numpy()
        motion_z = (motion_err - motion_mean) / max(motion_std, 1e-8)
        frame_z = (frame_err - frame_mean) / max(frame_std, 1e-8)
        scores = motion_w * motion_z + frame_w * frame_z
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
def evaluate_loss(model, loader, device, args):
    model.eval()
    running = {"total": 0.0, "recon_motion": 0.0, "pred_frame": 0.0, "grad": 0.0, "compact": 0.0, "entropy": 0.0}
    for observed_app, motion, target_app, _, _ in tqdm(loader, desc="val", leave=False):
        observed_app = observed_app.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
        motion = motion.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
        target_app = target_app.to(device, non_blocking=True)
        total_loss, loss_dict = compute_losses(model, observed_app, motion, target_app, args)
        running["total"] += float(total_loss.item())
        for key in ["recon_motion", "pred_frame", "grad", "compact", "entropy"]:
            running[key] += float(loss_dict[key])

    n = max(len(loader), 1)
    return {key: value / n for key, value in running.items()}


def train_one_epoch(model, loader, optimizer, scaler, device, use_amp, max_grad_norm, args):
    model.train()
    running = {"total": 0.0, "recon_motion": 0.0, "pred_frame": 0.0, "grad": 0.0, "compact": 0.0, "entropy": 0.0}
    pbar = tqdm(loader, desc="train", leave=False)
    for observed_app, motion, target_app, _, _ in pbar:
        observed_app = observed_app.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
        motion = motion.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
        target_app = target_app.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            total_loss, loss_dict = compute_losses(model, observed_app, motion, target_app, args)
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        running["total"] += float(total_loss.item())
        for key in ["recon_motion", "pred_frame", "grad", "compact", "entropy"]:
            running[key] += float(loss_dict[key])
        pbar.set_postfix(loss=f"{total_loss.item():.4f}")

    n = max(len(loader), 1)
    return {key: value / n for key, value in running.items()}


def build_save_dir(args):
    if args.save_dir is not None:
        return Path(args.save_dir)
    return Path("./outputs") / f"3dresnet2_{args.dataset_name}"


def save_checkpoint(path: Path, model, optimizer, epoch, best_metric, args, metric_name="val_loss"):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "best_metric_name": metric_name,
            "saved_at_utc": utc_now_iso(),
            "args": vars(args),
        },
        path,
    )


def save_hf_snapshot(
    save_dir: Path,
    model,
    optimizer,
    epoch: int,
    step: int,
    best_metric=None,
    metric_name="val_loss",
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
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
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


def save_best_model(save_dir: Path, model) -> Path:
    path = save_dir / "best.pth"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, path)
    print(f"models {path} save successfully!")
    return path


def load_checkpoint(path: Path, model, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


def make_loader(dataset, batch_size, shuffle, num_workers, pin_memory):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)


def init_model_optimizer(device, args):
    model = HF2VADLike3DResNet(
        fea_dim=DEFAULTS["fea_dim"],
        mem_dim=DEFAULTS["mem_dim"],
        mem_temperature=DEFAULTS["mem_temperature"],
        mem_shrink_thr=DEFAULTS["mem_shrink_thr"],
        detach_recon_motion=args.detach_recon_motion,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=DEFAULTS["use_amp"] and device.type == "cuda")
    return model, optimizer, scaler


def split_holdout_indices(dataset: ChunkedSamplesDataset, val_ratio: float, seed: int, split_unit: str):
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("--val_ratio must be in (0, 1) for standard training")

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
            yield train_idx, val_idx
        return

    groups = dataset.get_pred_frames()
    unique_groups = np.unique(groups)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_group_idx, val_group_idx in splitter.split(unique_groups):
        train_groups = unique_groups[train_group_idx]
        val_groups = unique_groups[val_group_idx]
        train_idx = np.flatnonzero(np.isin(groups, train_groups))
        val_idx = np.flatnonzero(np.isin(groups, val_groups))
        yield train_idx, val_idx


def save_split_indices(save_dir: Path, train_idx: np.ndarray, val_idx: np.ndarray):
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "train_indices.npy", train_idx.astype(np.int64))
    np.save(save_dir / "val_indices.npy", val_idx.astype(np.int64))


def maybe_training_stats_subset(train_dataset, save_dir: Path):
    index_path = save_dir / "train_indices.npy"
    if index_path.exists():
        return Subset(train_dataset, np.load(index_path).astype(np.int64).tolist())
    return train_dataset


def run_standard_training(args, device, use_amp, train_dataset, test_dataset, gt_concat, testing_frame_counts, save_dir):
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
            "mode": "standard",
            "epochs": args.epochs,
            "resume": bool(args.resume),
            "checkpoint": args.checkpoint,
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
            "split_unit": args.split_unit,
        },
    )

    train_subset = Subset(train_dataset, train_idx.tolist())
    val_subset = Subset(train_dataset, val_idx.tolist())
    train_loader = make_loader(train_subset, args.batch_size, True, args.num_workers, device.type == "cuda")
    train_stats_loader = make_loader(train_subset, args.batch_size, False, args.num_workers, device.type == "cuda")
    val_loader = make_loader(val_subset, args.batch_size, False, args.num_workers, device.type == "cuda")
    test_loader = None
    if args.eval_test_during_train:
        if test_dataset is None or gt_concat is None or testing_frame_counts is None:
            raise ValueError("--eval_test_during_train requires testing data and ground truth")
        test_loader = make_loader(test_dataset, args.batch_size, False, args.num_workers, device.type == "cuda")

    model, optimizer, scaler = init_model_optimizer(device, args)
    best_path = save_dir / "best.pth"
    start_epoch = 0
    step = 0
    best_val_loss = float("inf")
    best_epoch = None

    resume_path = Path(args.checkpoint) if args.checkpoint is not None else latest_snapshot_path(save_dir)
    if args.resume or args.checkpoint is not None:
        if resume_path is None or not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found for resume in {save_dir}")
        ckpt = load_checkpoint(resume_path, model, optimizer if args.resume else None, device=device)
        best_val_loss = float(ckpt.get("best_metric", best_val_loss))
        summary_path = save_dir / "train_summary.json"
        if summary_path.exists():
            try:
                best_epoch = json.loads(summary_path.read_text()).get("best_epoch")
            except json.JSONDecodeError:
                best_epoch = None
        step = int(ckpt.get("step", 0))
        if args.resume:
            if "epoch" in ckpt:
                start_epoch = int(ckpt["epoch"])
            else:
                start_epoch = int(Path(resume_path).name.rsplit("-", 1)[-1])
            print(f"[RESUME] {resume_path} epoch={start_epoch} best_val_loss={best_val_loss:.6f}")
            append_jsonl(
                events_path,
                {
                    "event": "resume",
                    "timestamp_utc": utc_now_iso(),
                    "checkpoint": str(resume_path),
                    "start_epoch": start_epoch,
                    "best_val_loss": best_val_loss,
                },
            )

    history = []
    for epoch in range(start_epoch, args.epochs):
        epoch_started_at = utc_now_iso()
        epoch_start = time.perf_counter()
        train_start = time.perf_counter()
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, use_amp, DEFAULTS["max_grad_norm"], args)
        step += len(train_loader)
        train_duration_sec = time.perf_counter() - train_start
        result = {"epoch": epoch + 1, "train": train_metrics, "best_checkpoint_updated": False}

        if (epoch + 1) % args.eval_every == 0:
            val_start = time.perf_counter()
            val_metrics = evaluate_loss(model, val_loader, device, args)
            val_duration_sec = time.perf_counter() - val_start
            result["val"] = val_metrics
            result["val_duration_sec"] = val_duration_sec

            if args.eval_test_during_train:
                audit_start = time.perf_counter()
                train_score_stats = collect_train_score_stats(model, train_stats_loader, device)
                audit_metrics = evaluate_frame_auc(
                    model,
                    test_loader,
                    gt_concat,
                    testing_frame_counts,
                    device,
                    save_dir=None,
                    motion_w=args.motion_score_weight,
                    frame_w=args.frame_score_weight,
                    train_stats=train_score_stats,
                )
                result["audit_test_auc"] = audit_metrics["auc"]
                result["audit_raw_test_auc"] = audit_metrics["raw_auc"]
                result["audit_timing"] = audit_metrics["timing"]
                result["audit_duration_sec"] = time.perf_counter() - audit_start

            if val_metrics["total"] < best_val_loss:
                best_val_loss = val_metrics["total"]
                best_epoch = epoch + 1
                save_best_model(save_dir, model)
                print(f"[BEST] epoch={epoch + 1} val_loss={best_val_loss:.6f}")
                result["best_checkpoint_updated"] = True

        snapshot = save_hf_snapshot(save_dir, model, optimizer, epoch + 1, step, best_val_loss, metric_name="val_loss", max_to_save=5)
        stats_path = training_stats_path(save_dir, epoch + 1)
        train_stats = save_training_stats(model, train_stats_loader, device, stats_path)
        result["training_stats"] = str(stats_path)
        result["training_stats_summary"] = {
            "motion_mean": train_stats["motion_mean"],
            "motion_std": train_stats["motion_std"],
            "frame_mean": train_stats["frame_mean"],
            "frame_std": train_stats["frame_std"],
        }
        result["train_duration_sec"] = train_duration_sec
        result["epoch_duration_sec"] = time.perf_counter() - epoch_start
        result["epoch_started_at_utc"] = epoch_started_at
        result["epoch_ended_at_utc"] = utc_now_iso()
        result["checkpoint"] = str(snapshot)
        result["best_val_loss_so_far"] = best_val_loss
        history.append(result)
        append_jsonl(events_path, {"event": "epoch_end", **result})
        print(json.dumps(result, ensure_ascii=True))

    summary = {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_path),
        "latest_checkpoint": str(latest_snapshot_path(save_dir)) if latest_snapshot_path(save_dir) else None,
        "train_samples": int(len(train_subset)),
        "val_samples": int(len(val_subset)),
        "checkpoint_selection": "validation loss from training split only",
        "latest_training_stats": str(latest_training_stats_path(save_dir)) if latest_training_stats_path(save_dir) else None,
        "events_log": str(events_path),
        "completed_at_utc": utc_now_iso(),
    }
    with open(save_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, ensure_ascii=True))


def run_kfold_training(args, device, use_amp, train_dataset, test_dataset, gt_concat, testing_frame_counts, save_dir):
    if args.checkpoint is not None and args.fold is None:
        raise ValueError("--checkpoint with --kfold train is only supported together with --fold")

    n_splits = args.kfold
    folds = list(iter_kfold_indices(train_dataset, n_splits, args.seed, args.split_unit))
    test_loader = None
    if args.eval_test_during_train:
        if test_dataset is None or gt_concat is None or testing_frame_counts is None:
            raise ValueError("--eval_test_during_train requires testing data and ground truth")
        test_loader = make_loader(test_dataset, args.batch_size, False, args.num_workers, device.type == "cuda")

    requested_folds = [args.fold] if args.fold is not None else list(range(n_splits))
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
                "kfold": n_splits,
                "epochs": args.epochs,
                "resume": bool(args.resume),
                "checkpoint": args.checkpoint,
                "train_samples": int(len(train_idx)),
                "val_samples": int(len(val_idx)),
                "split_unit": args.split_unit,
            },
        )

        train_subset = Subset(train_dataset, train_idx.tolist())
        val_subset = Subset(train_dataset, val_idx.tolist())
        train_loader = make_loader(train_subset, args.batch_size, True, args.num_workers, device.type == "cuda")
        train_stats_loader = make_loader(train_subset, args.batch_size, False, args.num_workers, device.type == "cuda")
        val_loader = make_loader(val_subset, args.batch_size, False, args.num_workers, device.type == "cuda")

        model, optimizer, scaler = init_model_optimizer(device, args)
        best_path = fold_dir / "best.pth"
        best_val_loss = float("inf")
        best_audit_test_auc = None
        start_epoch = 0
        step = 0
        best_epoch = None

        fold_resume_path = Path(args.checkpoint) if args.checkpoint is not None and args.fold == fold_idx else latest_snapshot_path(fold_dir)
        if args.resume:
            if fold_resume_path is not None and fold_resume_path.exists():
                ckpt = load_checkpoint(fold_resume_path, model, optimizer, device=device)
                if "epoch" in ckpt:
                    start_epoch = int(ckpt["epoch"])
                else:
                    start_epoch = int(Path(fold_resume_path).name.rsplit("-", 1)[-1])
                best_val_loss = float(ckpt.get("best_metric", best_val_loss))
                summary_path = fold_dir / "train_summary.json"
                if summary_path.exists():
                    try:
                        best_epoch = json.loads(summary_path.read_text()).get("best_epoch")
                    except json.JSONDecodeError:
                        best_epoch = None
                step = int(ckpt.get("step", 0))
                print(f"[KFOLD RESUME] fold={fold_idx} {fold_resume_path} epoch={start_epoch} best_val_loss={best_val_loss:.6f}")
                append_jsonl(
                    events_path,
                    {
                        "event": "resume",
                        "timestamp_utc": utc_now_iso(),
                        "checkpoint": str(fold_resume_path),
                        "fold": fold_idx,
                        "start_epoch": start_epoch,
                        "best_val_loss": best_val_loss,
                    },
                )
            else:
                print(f"[KFOLD RESUME] fold={fold_idx} no checkpoint at {fold_resume_path}; starting fresh")
        elif args.checkpoint is not None and args.fold == fold_idx:
            ckpt = load_checkpoint(Path(args.checkpoint), model, device=device)
            best_val_loss = float(ckpt.get("best_metric", best_val_loss))
            print(f"[KFOLD LOAD] fold={fold_idx} loaded model from {args.checkpoint}")

        print(f"[KFOLD] fold={fold_idx} train_samples={len(train_subset)} val_samples={len(val_subset)}")

        for epoch in range(start_epoch, args.epochs):
            epoch_started_at = utc_now_iso()
            epoch_start = time.perf_counter()
            train_start = time.perf_counter()
            train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, use_amp, DEFAULTS["max_grad_norm"], args)
            step += len(train_loader)
            train_duration_sec = time.perf_counter() - train_start
            result = {"fold": fold_idx, "epoch": epoch + 1, "train": train_metrics, "best_checkpoint_updated": False}

            if (epoch + 1) % args.eval_every == 0:
                val_start = time.perf_counter()
                val_metrics = evaluate_loss(model, val_loader, device, args)
                val_duration_sec = time.perf_counter() - val_start
                result["val"] = val_metrics
                result["val_duration_sec"] = val_duration_sec

                if args.eval_test_during_train:
                    audit_start = time.perf_counter()
                    train_score_stats = collect_train_score_stats(model, train_stats_loader, device)
                    audit_metrics = evaluate_frame_auc(
                        model,
                        test_loader,
                        gt_concat,
                        testing_frame_counts,
                        device,
                        save_dir=None,
                        motion_w=args.motion_score_weight,
                        frame_w=args.frame_score_weight,
                        train_stats=train_score_stats,
                    )
                    result["audit_test_auc"] = audit_metrics["auc"]
                    result["audit_raw_test_auc"] = audit_metrics["raw_auc"]
                    result["audit_timing"] = audit_metrics["timing"]
                    result["audit_duration_sec"] = time.perf_counter() - audit_start

                if val_metrics["total"] < best_val_loss:
                    best_val_loss = val_metrics["total"]
                    best_epoch = epoch + 1
                    best_audit_test_auc = result.get("audit_test_auc")
                    save_best_model(fold_dir, model)
                    print(f"[KFOLD BEST] fold={fold_idx} epoch={epoch + 1} val_loss={best_val_loss:.6f}")
                    result["best_checkpoint_updated"] = True

            snapshot = save_hf_snapshot(fold_dir, model, optimizer, epoch + 1, step, best_val_loss, metric_name="val_loss", max_to_save=5)
            stats_path = training_stats_path(fold_dir, epoch + 1)
            train_stats = save_training_stats(model, train_stats_loader, device, stats_path)
            result["training_stats"] = str(stats_path)
            result["training_stats_summary"] = {
                "motion_mean": train_stats["motion_mean"],
                "motion_std": train_stats["motion_std"],
                "frame_mean": train_stats["frame_mean"],
                "frame_std": train_stats["frame_std"],
            }
            result["train_duration_sec"] = train_duration_sec
            result["epoch_duration_sec"] = time.perf_counter() - epoch_start
            result["epoch_started_at_utc"] = epoch_started_at
            result["epoch_ended_at_utc"] = utc_now_iso()
            result["checkpoint"] = str(snapshot)
            result["best_val_loss_so_far"] = best_val_loss
            append_jsonl(events_path, {"event": "epoch_end", **result})
            print(json.dumps(result, ensure_ascii=True))

        fold_summary = {
            "fold": fold_idx,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "audit_test_auc_at_best_val": best_audit_test_auc,
            "best_checkpoint": str(best_path),
            "latest_checkpoint": str(latest_snapshot_path(fold_dir)) if latest_snapshot_path(fold_dir) else None,
            "train_samples": int(len(train_subset)),
            "val_samples": int(len(val_subset)),
            "checkpoint_selection": "validation loss from training fold only",
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
        "kfold": n_splits,
        "executed_folds": [item["fold"] for item in fold_summaries],
        "mean_best_val_loss": float(np.mean([item["best_val_loss"] for item in fold_summaries])),
        "std_best_val_loss": float(np.std([item["best_val_loss"] for item in fold_summaries])),
        "folds": fold_summaries,
    }
    with open(save_dir / "kfold_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, ensure_ascii=True))


def run_standard_test(args, device, train_dataset, test_dataset, gt_concat, testing_frame_counts, save_dir):
    events_path = save_dir / "events.jsonl"
    test_started_at = utc_now_iso()
    test_start = time.perf_counter()
    append_jsonl(
        events_path,
        {
            "event": "test_start",
            "timestamp_utc": test_started_at,
            "checkpoint": args.checkpoint,
            "mode": "standard",
        },
    )
    test_loader = make_loader(test_dataset, args.batch_size, False, args.num_workers, device.type == "cuda")

    model, _, _ = init_model_optimizer(device, args)
    ckpt_path = Path(args.checkpoint) if args.checkpoint is not None else save_dir / "best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    load_checkpoint(ckpt_path, model, device=device)
    stats_path = None
    summary_path = save_dir / "train_summary.json"
    if summary_path.exists():
        try:
            best_epoch = json.loads(summary_path.read_text()).get("best_epoch")
            if best_epoch is not None:
                candidate = training_stats_path(save_dir, int(best_epoch))
                if candidate.exists():
                    stats_path = candidate
        except json.JSONDecodeError:
            stats_path = None
    if stats_path is None:
        stats_path = latest_training_stats_path(save_dir) or training_stats_path(save_dir, 0)
    if stats_path.exists():
        train_score_stats = load_training_stats(stats_path)
    else:
        stats_dataset = maybe_training_stats_subset(train_dataset, save_dir)
        train_stats_loader = make_loader(stats_dataset, args.batch_size, False, args.num_workers, device.type == "cuda")
        train_score_stats = save_training_stats(model, train_stats_loader, device, stats_path)
    eval_metrics = evaluate_frame_auc(
        model,
        test_loader,
        gt_concat,
        testing_frame_counts,
        device,
        save_dir=save_dir,
        suffix="test",
        motion_w=args.motion_score_weight,
        frame_w=args.frame_score_weight,
        train_stats=train_score_stats,
    )
    summary = {
        "checkpoint": str(ckpt_path),
        "test_auc": eval_metrics["auc"],
        "raw_test_auc": eval_metrics["raw_auc"],
        "training_stats": str(stats_path),
        "timing": eval_metrics["timing"],
        "total_test_duration_sec": time.perf_counter() - test_start,
        "started_at_utc": test_started_at,
        "ended_at_utc": utc_now_iso(),
    }
    with open(save_dir / "test_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    append_jsonl(events_path, {"event": "test_end", **summary})
    print(json.dumps(summary, ensure_ascii=True))


def run_kfold_test(args, device, train_dataset, test_dataset, gt_concat, testing_frame_counts, save_dir):
    events_path = save_dir / "events.jsonl"
    test_started_at = utc_now_iso()
    test_start = time.perf_counter()
    append_jsonl(
        events_path,
        {
            "event": "test_start",
            "timestamp_utc": test_started_at,
            "checkpoint": args.checkpoint,
            "mode": "kfold",
            "kfold": args.kfold,
            "fold": args.fold,
        },
    )
    n_splits = args.kfold
    folds = list(iter_kfold_indices(train_dataset, n_splits, args.seed, args.split_unit))
    test_loader = make_loader(test_dataset, args.batch_size, False, args.num_workers, device.type == "cuda")

    requested_folds = [args.fold] if args.fold is not None else list(range(n_splits))
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

        model, _, _ = init_model_optimizer(device, args)
        load_checkpoint(ckpt_path, model, device=device)
        stats_path = None
        summary_path = fold_dir / "train_summary.json"
        if summary_path.exists():
            try:
                best_epoch = json.loads(summary_path.read_text()).get("best_epoch")
                if best_epoch is not None:
                    candidate = training_stats_path(fold_dir, int(best_epoch))
                    if candidate.exists():
                        stats_path = candidate
            except json.JSONDecodeError:
                stats_path = None
        if stats_path is None:
            stats_path = latest_training_stats_path(fold_dir) or training_stats_path(fold_dir, 0)
        if stats_path.exists():
            train_score_stats = load_training_stats(stats_path)
        else:
            train_subset = Subset(train_dataset, train_idx.tolist())
            train_stats_loader = make_loader(train_subset, args.batch_size, False, args.num_workers, device.type == "cuda")
            train_score_stats = save_training_stats(model, train_stats_loader, device, stats_path)
        eval_metrics = evaluate_frame_auc(
            model,
            test_loader,
            gt_concat,
            testing_frame_counts,
            device,
            save_dir=fold_dir,
            suffix="test",
            motion_w=args.motion_score_weight,
            frame_w=args.frame_score_weight,
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

    joblib.dump(ensemble_scores, save_dir / "frame_scores_test_ensemble.pkl")
    curves_dir = save_dir / "anomaly_curves_test_ensemble"
    ensemble_auc = float(save_evaluation_curves(scores_eval, gt_eval, str(curves_dir), np.asarray(testing_frame_counts) - 4))
    raw_ensemble_auc = float(roc_auc_score(gt_eval, scores_eval))

    summary = {
        "kfold": n_splits,
        "executed_folds": [item["fold"] for item in fold_results],
        "ensemble_test_auc": ensemble_auc,
        "raw_ensemble_test_auc": raw_ensemble_auc,
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

    train_dataset = ChunkedSamplesDataset(train_dir)
    test_dataset = None
    gt_concat = None
    testing_frame_counts = None
    if args.mode == "test" or args.eval_test_during_train:
        test_dataset = ChunkedSamplesDataset(test_dir)
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
            run_standard_training(args, device, use_amp, train_dataset, test_dataset, gt_concat, testing_frame_counts, save_dir)
    else:
        if args.kfold > 1:
            run_kfold_test(args, device, train_dataset, test_dataset, gt_concat, testing_frame_counts, save_dir)
        else:
            run_standard_test(args, device, train_dataset, test_dataset, gt_concat, testing_frame_counts, save_dir)


if __name__ == "__main__":
    main()