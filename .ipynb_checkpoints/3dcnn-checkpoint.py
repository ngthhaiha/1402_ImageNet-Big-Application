import argparse
import bisect
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
}


def parse_args():
    parser = argparse.ArgumentParser(description="3D CNN train/test on HF2VAD chunked_samples")
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--dataset_name", choices=["ped2", "avenue", "shanghaitech"], default="avenue")
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
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


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

    def __getitem__(self, idx: int):
        chunk_idx, local_idx = self._resolve_index(idx)
        payload = self._load_chunk(chunk_idx)

        appearance = np.array(payload["appearance"][local_idx], dtype=np.float32, copy=True) / 255.0
        motion = np.array(payload["motion"][local_idx], dtype=np.float32, copy=True)
        bbox = np.array(payload["bbox"][local_idx], dtype=np.float32, copy=True)
        pred_frame = payload["pred_frame"][local_idx]

        observed_app = torch.from_numpy(appearance[:4]).permute(0, 3, 1, 2).contiguous()
        target_app = torch.from_numpy(appearance[4]).permute(2, 0, 1).contiguous()
        motion = torch.from_numpy(motion[1:]).permute(0, 3, 1, 2).contiguous()

        pred_frame = np.asarray(pred_frame).reshape(-1)
        pred_frame = int(pred_frame[-1])

        return observed_app, motion, target_app, torch.from_numpy(bbox), torch.tensor(pred_frame, dtype=torch.long)


class Conv3DEncoder(nn.Module):
    def __init__(self, in_ch=3, out_ch=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_ch, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 96, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.Conv3d(96, out_ch, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.encoder(x)


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
    def __init__(self, out_ch=2):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 128, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)),
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


class FlowReconPredict3DCNN(nn.Module):
    def __init__(self, fea_dim, mem_dim, mem_temperature, mem_shrink_thr):
        super().__init__()
        self.app_encoder = Conv3DEncoder(in_ch=3, out_ch=fea_dim)
        self.mot_encoder = Conv3DEncoder(in_ch=2, out_ch=fea_dim)
        self.fuse = nn.Sequential(
            nn.Conv3d(fea_dim * 2, fea_dim, kernel_size=1),
            nn.BatchNorm3d(fea_dim),
            nn.ReLU(inplace=True),
        )
        self.memory = MemoryModule(mem_dim, fea_dim, mem_temperature, mem_shrink_thr)
        self.motion_decoder = MotionDecoder3D(out_ch=2)
        self.frame_decoder = FrameDecoder2D(in_ch=fea_dim, out_ch=3)

    def forward(self, observed_app, motion):
        z_app = self.app_encoder(observed_app)
        z_mot = self.mot_encoder(motion)
        latent = self.fuse(torch.cat([z_app, z_mot], dim=1))
        mem_out, att, query, mem_read = self.memory(latent)
        recon_motion = self.motion_decoder(mem_out)
        pred_frame = self.frame_decoder(mem_out.mean(dim=2))
        return recon_motion, pred_frame, {"att": att, "query": query, "mem_read": mem_read}


def hard_shrink_relu(inp, lambd, eps=1e-12):
    return (F.relu(inp - lambd) * inp) / (torch.abs(inp - lambd) + eps)


def gradient_loss_2d(pred, target):
    pred_gx = pred[..., :, 1:] - pred[..., :, :-1]
    pred_gy = pred[..., 1:, :] - pred[..., :-1, :]
    tgt_gx = target[..., :, 1:] - target[..., :, :-1]
    tgt_gy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_gx, tgt_gx) + F.l1_loss(pred_gy, tgt_gy)


def compute_losses(model, observed_app, motion, target_app, args=None):
    recon_motion, pred_frame, aux = model(observed_app, motion)
    loss_recon_motion = F.mse_loss(recon_motion, motion)
    loss_pred_frame = F.mse_loss(pred_frame, target_app)
    loss_grad = gradient_loss_2d(pred_frame, target_app)
    compact_loss = F.mse_loss(aux["query"], aux["mem_read"])
    att = aux["att"].clamp_min(1e-12)
    entropy_loss = -(att * torch.log(att)).sum(dim=1).mean()
    total_loss = (
        DEFAULTS["w_recon_motion"] * loss_recon_motion
        + DEFAULTS["w_pred_frame"] * loss_pred_frame
        + DEFAULTS["w_grad"] * loss_grad
        + DEFAULTS["w_compact"] * compact_loss
        + DEFAULTS["w_entropy"] * entropy_loss
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
def collect_train_score_stats(model, loader, device):
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
    return {
        "motion_mean": float(motion_errs.mean()) if len(motion_errs) else 0.0,
        "motion_std": float(motion_errs.std() + 1e-8) if len(motion_errs) else 1.0,
        "frame_mean": float(frame_errs.mean()) if len(frame_errs) else 0.0,
        "frame_std": float(frame_errs.std() + 1e-8) if len(frame_errs) else 1.0,
    }


@torch.no_grad()
def evaluate_frame_auc(model, loader, gt_concat, testing_frame_counts, device, save_dir=None, suffix="test", motion_w=0.5, frame_w=0.5, train_stats=None):
    model.eval()
    total_frames = int(np.sum(testing_frame_counts))
    frame_bbox_scores = [dict() for _ in range(total_frames)]

    motion_mean = train_stats["motion_mean"] if train_stats else 0.0
    motion_std = train_stats["motion_std"] if train_stats else 1.0
    frame_mean = train_stats["frame_mean"] if train_stats else 0.0
    frame_std = train_stats["frame_std"] if train_stats else 1.0

    obj_idx = 0
    for observed_app, motion, target_app, _, pred_frame in tqdm(loader, desc="Eval", leave=False):
        observed_app = observed_app.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
        motion = motion.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
        target_app = target_app.to(device, non_blocking=True)

        motion_err, frame_err = compute_sample_errors(model, observed_app, motion, target_app)
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
    empty_score = motion_w * ((0.0 - motion_mean) / max(motion_std, 1e-8)) + frame_w * ((0.0 - frame_mean) / max(frame_std, 1e-8))
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
    auc = float(roc_auc_score(gt_eval, scores_eval))

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(frame_scores, save_dir / f"frame_scores_{suffix}.json")
        curves_dir = save_dir / f"anomaly_curves_{suffix}"
        save_evaluation_curves(scores_eval, gt_eval, str(curves_dir), np.asarray(testing_frame_counts) - 4)

    return {"auc": auc, "frame_scores": frame_scores}


@torch.no_grad()
def evaluate_loss(model, loader, device):
    model.eval()
    running = {"total": 0.0, "recon_motion": 0.0, "pred_frame": 0.0, "grad": 0.0, "compact": 0.0, "entropy": 0.0}
    for observed_app, motion, target_app, _, _ in tqdm(loader, desc="val", leave=False):
        observed_app = observed_app.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
        motion = motion.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
        target_app = target_app.to(device, non_blocking=True)
        total_loss, loss_dict = compute_losses(model, observed_app, motion, target_app, None)
        running["total"] += float(total_loss.item())
        for key in ["recon_motion", "pred_frame", "grad", "compact", "entropy"]:
            running[key] += float(loss_dict[key])

    n = max(len(loader), 1)
    return {key: value / n for key, value in running.items()}


def train_one_epoch(model, loader, optimizer, scaler, device, use_amp, max_grad_norm):
    model.train()
    running = {"total": 0.0, "recon_motion": 0.0, "pred_frame": 0.0, "grad": 0.0, "compact": 0.0, "entropy": 0.0}
    pbar = tqdm(loader, desc="train", leave=False)
    for observed_app, motion, target_app, _, _ in pbar:
        observed_app = observed_app.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
        motion = motion.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
        target_app = target_app.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            total_loss, loss_dict = compute_losses(model, observed_app, motion, target_app, None)
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


def load_gt_labels(dataset_base_dir: Path, dataset_name: str):
    gt_path = dataset_base_dir / dataset_name / "ground_truth_demo" / "gt_label.json"
    gt = joblib.load(gt_path) if gt_path.suffix == ".joblib" else None
    if gt is None:
        with open(gt_path, "rb") as f:
            import pickle
            gt = pickle.load(f)
    gt_items = sorted(gt.items(), key=lambda item: int(item[0]))
    testing_frame_counts = [len(np.asarray(labels)) for _, labels in gt_items]
    gt_concat = np.concatenate([np.asarray(labels, dtype=np.float32) for _, labels in gt_items], axis=0)
    return gt_concat, testing_frame_counts


def build_save_dir(args):
    if args.save_dir is not None:
        return Path(args.save_dir)
    suffix = f"_kfold{args.kfold}" if args.kfold and args.kfold > 1 else ""
    return Path("./outputs") / f"3dcnn_{args.dataset_name}{suffix}"


def save_checkpoint(path: Path, model, optimizer, epoch, best_metric, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(path: Path, model, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


def make_loader(dataset, batch_size, shuffle, num_workers, pin_memory):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)


def init_model_optimizer(device, args):
    model = FlowReconPredict3DCNN(
        fea_dim=DEFAULTS["fea_dim"],
        mem_dim=DEFAULTS["mem_dim"],
        mem_temperature=DEFAULTS["mem_temperature"],
        mem_shrink_thr=DEFAULTS["mem_shrink_thr"],
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=DEFAULTS["use_amp"] and device.type == "cuda")
    return model, optimizer, scaler


def run_standard_training(args, device, use_amp, train_dataset, test_dataset, gt_concat, testing_frame_counts, save_dir):
    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers, device.type == "cuda")
    train_stats_loader = make_loader(train_dataset, args.batch_size, False, args.num_workers, device.type == "cuda")
    test_loader = make_loader(test_dataset, args.batch_size, False, args.num_workers, device.type == "cuda")

    model, optimizer, scaler = init_model_optimizer(device, args)
    best_path = save_dir / "best.pt"
    last_path = save_dir / "last.pt"
    start_epoch = 0
    best_auc = -1.0

    if args.checkpoint is not None:
        ckpt = load_checkpoint(Path(args.checkpoint), model, optimizer if args.resume else None, device=device)
        best_auc = float(ckpt.get("best_metric", ckpt.get("best_auc", -1.0)))
        if args.resume:
            start_epoch = int(ckpt.get("epoch", 0))
            print(f"[RESUME] epoch={start_epoch} best_auc={best_auc:.4f}")

    history = []
    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, use_amp, DEFAULTS["max_grad_norm"])
        result = {"epoch": epoch + 1, "train": train_metrics}

        if (epoch + 1) % args.eval_every == 0:
            train_score_stats = collect_train_score_stats(model, train_stats_loader, device)
            eval_metrics = evaluate_frame_auc(
                model,
                test_loader,
                gt_concat,
                testing_frame_counts,
                device,
                save_dir=save_dir,
                suffix=str(epoch + 1),
                motion_w=DEFAULTS["motion_score_weight"],
                frame_w=DEFAULTS["frame_score_weight"],
                train_stats=train_score_stats,
            )
            result["test_auc"] = eval_metrics["auc"]
            if eval_metrics["auc"] > best_auc:
                best_auc = eval_metrics["auc"]
                save_checkpoint(best_path, model, optimizer, epoch + 1, best_auc, args)
                print(f"[BEST] epoch={epoch + 1} auc={best_auc:.4f}")

        save_checkpoint(last_path, model, optimizer, epoch + 1, best_auc, args)
        history.append(result)
        print(json.dumps(result, ensure_ascii=True))

    summary = {"best_auc": best_auc, "best_checkpoint": str(best_path), "last_checkpoint": str(last_path)}
    with open(save_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, ensure_ascii=True))


def run_kfold_training(args, device, use_amp, train_dataset, test_dataset, gt_concat, testing_frame_counts, save_dir):
    if args.resume:
        raise ValueError("--resume is not supported with --kfold > 1")
    if args.checkpoint is not None:
        raise ValueError("--checkpoint is not used with --kfold > 1 train mode")

    n_splits = args.kfold
    all_indices = np.arange(len(train_dataset))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    test_loader = make_loader(test_dataset, args.batch_size, False, args.num_workers, device.type == "cuda")

    requested_folds = [args.fold] if args.fold is not None else list(range(n_splits))
    fold_summaries = []

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(all_indices)):
        if fold_idx not in requested_folds:
            continue

        fold_dir = save_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_subset = Subset(train_dataset, train_idx.tolist())
        val_subset = Subset(train_dataset, val_idx.tolist())
        train_loader = make_loader(train_subset, args.batch_size, True, args.num_workers, device.type == "cuda")
        train_stats_loader = make_loader(train_subset, args.batch_size, False, args.num_workers, device.type == "cuda")
        val_loader = make_loader(val_subset, args.batch_size, False, args.num_workers, device.type == "cuda")

        model, optimizer, scaler = init_model_optimizer(device, args)
        best_path = fold_dir / "best.pt"
        last_path = fold_dir / "last.pt"
        best_val_loss = float("inf")
        best_test_auc = -1.0
        history = []

        print(f"[KFOLD] fold={fold_idx} train_samples={len(train_subset)} val_samples={len(val_subset)}")

        for epoch in range(args.epochs):
            train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, use_amp, DEFAULTS["max_grad_norm"])
            result = {"fold": fold_idx, "epoch": epoch + 1, "train": train_metrics}

            if (epoch + 1) % args.eval_every == 0:
                val_metrics = evaluate_loss(model, val_loader, device)
                result["val"] = val_metrics
                if val_metrics["total"] < best_val_loss:
                    best_val_loss = val_metrics["total"]
                    train_score_stats = collect_train_score_stats(model, train_stats_loader, device)
                    eval_metrics = evaluate_frame_auc(
                        model,
                        test_loader,
                        gt_concat,
                        testing_frame_counts,
                        device,
                        save_dir=fold_dir,
                        suffix=str(epoch + 1),
                        motion_w=DEFAULTS["motion_score_weight"],
                        frame_w=DEFAULTS["frame_score_weight"],
                        train_stats=train_score_stats,
                    )
                    best_test_auc = eval_metrics["auc"]
                    result["test_auc"] = best_test_auc
                    save_checkpoint(best_path, model, optimizer, epoch + 1, best_val_loss, args)
                    print(f"[KFOLD BEST] fold={fold_idx} epoch={epoch + 1} val_loss={best_val_loss:.6f} test_auc={best_test_auc:.4f}")

            save_checkpoint(last_path, model, optimizer, epoch + 1, best_val_loss, args)
            history.append(result)
            print(json.dumps(result, ensure_ascii=True))

        fold_summary = {
            "fold": fold_idx,
            "best_val_loss": best_val_loss,
            "best_test_auc": best_test_auc,
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
        }
        with open(fold_dir / "train_summary.json", "w", encoding="utf-8") as f:
            json.dump(fold_summary, f, indent=2)
        fold_summaries.append(fold_summary)
        print(json.dumps(fold_summary, ensure_ascii=True))

    if not fold_summaries:
        raise ValueError("No fold was executed. Check --fold against --kfold.")

    aucs = [item["best_test_auc"] for item in fold_summaries if item["best_test_auc"] >= 0]
    summary = {
        "kfold": n_splits,
        "executed_folds": [item["fold"] for item in fold_summaries],
        "mean_test_auc": float(np.mean(aucs)) if aucs else None,
        "std_test_auc": float(np.std(aucs)) if aucs else None,
        "folds": fold_summaries,
    }
    with open(save_dir / "kfold_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, ensure_ascii=True))


def run_standard_test(args, device, train_dataset, test_dataset, gt_concat, testing_frame_counts, save_dir):
    train_stats_loader = make_loader(train_dataset, args.batch_size, False, args.num_workers, device.type == "cuda")
    test_loader = make_loader(test_dataset, args.batch_size, False, args.num_workers, device.type == "cuda")

    model, _, _ = init_model_optimizer(device, args)
    ckpt_path = Path(args.checkpoint) if args.checkpoint is not None else save_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    load_checkpoint(ckpt_path, model, device=device)
    train_score_stats = collect_train_score_stats(model, train_stats_loader, device)
    eval_metrics = evaluate_frame_auc(
        model,
        test_loader,
        gt_concat,
        testing_frame_counts,
        device,
        save_dir=save_dir,
        suffix="test",
        motion_w=DEFAULTS["motion_score_weight"],
        frame_w=DEFAULTS["frame_score_weight"],
        train_stats=train_score_stats,
    )
    summary = {"checkpoint": str(ckpt_path), "test_auc": eval_metrics["auc"]}
    with open(save_dir / "test_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, ensure_ascii=True))


def run_kfold_test(args, device, train_dataset, test_dataset, gt_concat, testing_frame_counts, save_dir):
    n_splits = args.kfold
    all_indices = np.arange(len(train_dataset))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    test_loader = make_loader(test_dataset, args.batch_size, False, args.num_workers, device.type == "cuda")

    requested_folds = [args.fold] if args.fold is not None else list(range(n_splits))
    frame_scores_list = []
    fold_results = []

    for fold_idx, (train_idx, _) in enumerate(splitter.split(all_indices)):
        if fold_idx not in requested_folds:
            continue

        fold_dir = save_dir / f"fold_{fold_idx}"
        ckpt_path = Path(args.checkpoint) if args.checkpoint is not None and args.fold == fold_idx else fold_dir / "best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found for fold {fold_idx}: {ckpt_path}")

        train_subset = Subset(train_dataset, train_idx.tolist())
        train_stats_loader = make_loader(train_subset, args.batch_size, False, args.num_workers, device.type == "cuda")
        model, _, _ = init_model_optimizer(device, args)
        load_checkpoint(ckpt_path, model, device=device)
        train_score_stats = collect_train_score_stats(model, train_stats_loader, device)
        eval_metrics = evaluate_frame_auc(
            model,
            test_loader,
            gt_concat,
            testing_frame_counts,
            device,
            save_dir=fold_dir,
            suffix="test",
            motion_w=DEFAULTS["motion_score_weight"],
            frame_w=DEFAULTS["frame_score_weight"],
            train_stats=train_score_stats,
        )
        frame_scores_list.append(eval_metrics["frame_scores"])
        fold_results.append({"fold": fold_idx, "checkpoint": str(ckpt_path), "test_auc": eval_metrics["auc"]})
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
    ensemble_auc = float(roc_auc_score(gt_eval, scores_eval))

    joblib.dump(ensemble_scores, save_dir / "frame_scores_test_ensemble.json")
    curves_dir = save_dir / "anomaly_curves_test_ensemble"
    save_evaluation_curves(scores_eval, gt_eval, str(curves_dir), np.asarray(testing_frame_counts) - 4)

    summary = {
        "kfold": n_splits,
        "executed_folds": [item["fold"] for item in fold_results],
        "ensemble_test_auc": ensemble_auc,
        "fold_results": fold_results,
    }
    with open(save_dir / "test_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
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
    test_dataset = ChunkedSamplesDataset(test_dir)
    gt_concat, testing_frame_counts = load_gt_labels(dataset_base_dir, args.dataset_name)

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
