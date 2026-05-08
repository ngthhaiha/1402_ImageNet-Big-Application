import argparse
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BASE_PATH = Path(__file__).resolve().with_name("graph_3dresnet2.py")
BASE_SPEC = importlib.util.spec_from_file_location("graph_3dresnet2_base", BASE_PATH)
base = importlib.util.module_from_spec(BASE_SPEC)
sys.modules[BASE_SPEC.name] = base
BASE_SPEC.loader.exec_module(base)


DEFAULTS = {
    **base.DEFAULTS,
    "topo_layers": 2,
    "topo_heads": 4,
    "topo_dropout": 0.10,
    "topo_diffusion_steps": 2,
    "topo_temporal_window": 2,
    "topo_temporal_weight": 0.35,
    "topo_direction_weight": 0.20,
    "topo_speed_weight": 0.15,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="TopoGraph-Diffusion Transformer + 3DResNet2 for video anomaly detection"
    )
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
    parser.add_argument("--val_ratio", type=float, default=DEFAULTS["val_ratio"])
    parser.add_argument("--split_unit", choices=["frame", "sample"], default=DEFAULTS["split_unit"])
    parser.add_argument("--eval_test_during_train", action="store_true")
    parser.add_argument("--detach_recon_motion", action="store_true")

    parser.add_argument("--w_recon_motion", type=float, default=DEFAULTS["w_recon_motion"])
    parser.add_argument("--w_pred_frame", type=float, default=DEFAULTS["w_pred_frame"])
    parser.add_argument("--w_grad", type=float, default=DEFAULTS["w_grad"])
    parser.add_argument("--w_compact", type=float, default=DEFAULTS["w_compact"])
    parser.add_argument("--w_entropy", type=float, default=DEFAULTS["w_entropy"])
    parser.add_argument("--motion_score_weight", type=float, default=DEFAULTS["motion_score_weight"])
    parser.add_argument("--frame_score_weight", type=float, default=DEFAULTS["frame_score_weight"])
    parser.add_argument("--relation_score_weight", type=float, default=DEFAULTS["relation_score_weight"])

    parser.add_argument("--disable_graph", action="store_true", help="Disable TopoGraph and run the original 3DResNet2 branch")
    parser.add_argument("--disable_frame_grouping", action="store_true")
    parser.add_argument("--graph_alpha", type=float, default=DEFAULTS["graph_alpha"])
    parser.add_argument("--graph_topk", type=int, default=DEFAULTS["graph_topk"])
    parser.add_argument("--graph_proximity_weight", type=float, default=DEFAULTS["graph_proximity_weight"])

    parser.add_argument("--topo_layers", type=int, default=DEFAULTS["topo_layers"])
    parser.add_argument("--topo_heads", type=int, default=DEFAULTS["topo_heads"])
    parser.add_argument("--topo_dropout", type=float, default=DEFAULTS["topo_dropout"])
    parser.add_argument("--topo_diffusion_steps", type=int, default=DEFAULTS["topo_diffusion_steps"])
    parser.add_argument("--topo_temporal_window", type=int, default=DEFAULTS["topo_temporal_window"])
    parser.add_argument("--topo_temporal_weight", type=float, default=DEFAULTS["topo_temporal_weight"])
    parser.add_argument("--topo_direction_weight", type=float, default=DEFAULTS["topo_direction_weight"])
    parser.add_argument("--topo_speed_weight", type=float, default=DEFAULTS["topo_speed_weight"])
    return parser.parse_args()


def safe_unit(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(eps)


def motion_trajectory_stats(motion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flow_per_t = motion.mean(dim=(3, 4)).permute(0, 2, 1).contiguous()
    mean_flow = flow_per_t.mean(dim=1)
    speed = torch.linalg.norm(mean_flow, dim=1, keepdim=True)
    direction = safe_unit(mean_flow)
    return mean_flow, speed, direction


def topological_edge_features(
    boxes: torch.Tensor,
    pred_frame: torch.Tensor,
    motion: torch.Tensor,
    temporal_window: int,
    proximity_weight: float,
    temporal_weight: float,
    direction_weight: float,
    speed_weight: float,
    topk: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = boxes.shape[0]
    boxes = boxes.float()
    pred_frame = pred_frame.reshape(-1).float()
    _, speed, direction = motion_trajectory_stats(motion)

    x1 = torch.minimum(boxes[:, 0], boxes[:, 2])
    y1 = torch.minimum(boxes[:, 1], boxes[:, 3])
    x2 = torch.maximum(boxes[:, 0], boxes[:, 2])
    y2 = torch.maximum(boxes[:, 1], boxes[:, 3])
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

    centers = torch.stack(((x1 + x2) * 0.5, (y1 + y2) * 0.5), dim=1)
    wh = torch.stack(((x2 - x1).clamp_min(1.0), (y2 - y1).clamp_min(1.0)), dim=1)
    area = (wh[:, 0] * wh[:, 1]).clamp_min(1.0)
    diag = torch.linalg.norm(wh, dim=1).clamp_min(1.0)

    rel = centers[None, :, :] - centers[:, None, :]
    scale = ((diag[:, None] + diag[None, :]) * 0.5).clamp_min(1.0)
    dist = torch.linalg.norm(rel, dim=-1)
    dist_norm = dist / scale
    proximity = torch.exp(-dist_norm)
    rel_unit = safe_unit(rel)

    iou = base.bbox_iou_matrix(boxes_xyxy, eps=eps)
    frame_delta = torch.abs(pred_frame[:, None] - pred_frame[None, :])
    same_frame = frame_delta.eq(0)
    temporal_valid = frame_delta.le(float(max(temporal_window, 0)))
    temporal_decay = torch.exp(-frame_delta / max(float(temporal_window), 1.0))

    speed_i = speed[:, None, 0]
    speed_j = speed[None, :, 0]
    rel_speed = torch.abs(speed_i - speed_j)
    speed_affinity = torch.exp(-rel_speed)

    dir_i = direction[:, None, :]
    dir_j = direction[None, :, :]
    direction_cos = (dir_i * dir_j).sum(dim=-1)
    approach_i_to_j = (dir_i * rel_unit).sum(dim=-1)
    approach_j_to_i = (dir_j * (-rel_unit)).sum(dim=-1)
    interaction = torch.relu(approach_i_to_j) * torch.relu(approach_j_to_i)

    area_ratio = torch.log((area[None, :] / area[:, None]).clamp_min(eps)).clamp(-4.0, 4.0) / 4.0
    rel_xy = (rel / scale[..., None]).clamp(-4.0, 4.0) / 4.0

    valid = temporal_valid
    eye = torch.eye(n, device=boxes.device, dtype=torch.bool)
    valid = valid | eye

    prior = (
        iou
        + proximity_weight * proximity
        + temporal_weight * temporal_decay
        + direction_weight * interaction
        + speed_weight * speed_affinity
    ) * valid.float()
    prior = torch.maximum(prior, eye.float())

    if topk is not None and topk > 0 and topk + 1 < n:
        _, nn_idx = torch.topk(prior, k=topk + 1, dim=1)
        keep = torch.zeros_like(prior)
        keep.scatter_(1, nn_idx, 1.0)
        prior = prior * keep
        valid = keep.bool() | eye

    adj = prior / prior.sum(dim=1, keepdim=True).clamp_min(eps)
    edge = torch.stack(
        [
            same_frame.float(),
            temporal_decay,
            iou,
            proximity,
            rel_xy[..., 0],
            rel_xy[..., 1],
            speed_i.expand_as(rel_speed),
            speed_j.expand_as(rel_speed),
            rel_speed,
            direction_cos,
            approach_i_to_j,
            approach_j_to_i,
            area_ratio,
        ],
        dim=-1,
    )
    return edge, adj, valid


class TopoDiffusionTransformerLayer(nn.Module):
    def __init__(self, channels: int, heads: int, edge_dim: int, dropout: float):
        super().__init__()
        if channels % heads != 0:
            raise ValueError("channels must be divisible by topo_heads")
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.edge_bias = nn.Sequential(
            nn.LayerNorm(edge_dim),
            nn.Linear(edge_dim, heads),
        )
        self.out = nn.Linear(channels, channels)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout),
        )
        self.diffusion_logit = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x: torch.Tensor, edge: torch.Tensor, adj: torch.Tensor, valid: torch.Tensor, diffusion_steps: int):
        beta = torch.sigmoid(self.diffusion_logit)
        diffused = x
        for _ in range(max(0, diffusion_steps)):
            diffused = (1.0 - beta) * diffused + beta * torch.matmul(adj, diffused)

        h = self.norm1(diffused)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        n = x.shape[0]
        q = q.view(n, self.heads, self.head_dim).transpose(0, 1)
        k = k.view(n, self.heads, self.head_dim).transpose(0, 1)
        v = v.view(n, self.heads, self.head_dim).transpose(0, 1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + self.edge_bias(edge).permute(2, 0, 1)
        scores = scores.masked_fill(~valid.unsqueeze(0), torch.finfo(scores.dtype).min)

        attn = torch.softmax(scores, dim=-1)
        msg = torch.matmul(self.drop(attn), v).transpose(0, 1).contiguous().view(n, self.channels)
        x = x + self.drop(self.out(msg))
        x = x + self.ffn(self.norm2(x))
        return x, attn


class TopoGraphDiffusionTransformer(nn.Module):
    def __init__(
        self,
        channels: int,
        layers: int,
        heads: int,
        dropout: float,
        alpha_init: float,
        topk: int,
        proximity_weight: float,
        temporal_window: int,
        temporal_weight: float,
        direction_weight: float,
        speed_weight: float,
        diffusion_steps: int,
    ):
        super().__init__()
        self.topk = topk
        self.proximity_weight = proximity_weight
        self.temporal_window = temporal_window
        self.temporal_weight = temporal_weight
        self.direction_weight = direction_weight
        self.speed_weight = speed_weight
        self.diffusion_steps = diffusion_steps

        self.node_extra = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, channels),
        )
        edge_dim = 13
        self.layers = nn.ModuleList(
            [TopoDiffusionTransformerLayer(channels, heads, edge_dim, dropout) for _ in range(layers)]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
        )
        alpha_init = float(np.clip(alpha_init, 1e-4, 1.0 - 1e-4))
        self.alpha_logit = nn.Parameter(torch.logit(torch.tensor(alpha_init, dtype=torch.float32)))

    def forward(self, latent: torch.Tensor, boxes: torch.Tensor | None, pred_frame: torch.Tensor | None, motion: torch.Tensor):
        if boxes is None or pred_frame is None or latent.shape[0] <= 1:
            return latent, {"graph_alpha": 0.0, "graph_edge_density": 0.0, "topo_attn_entropy": 0.0}

        b, c, _, _, _ = latent.shape
        edge, adj, valid = topological_edge_features(
            boxes=boxes,
            pred_frame=pred_frame,
            motion=motion,
            temporal_window=self.temporal_window,
            proximity_weight=self.proximity_weight,
            temporal_weight=self.temporal_weight,
            direction_weight=self.direction_weight,
            speed_weight=self.speed_weight,
            topk=self.topk,
        )

        pooled = latent.mean(dim=(2, 3, 4))
        mean_flow, speed, _ = motion_trajectory_stats(motion)
        area = ((boxes[:, 2] - boxes[:, 0]).abs() * (boxes[:, 3] - boxes[:, 1]).abs()).clamp_min(1.0)
        area_log = torch.log1p(area).unsqueeze(1) / 10.0
        frame_scaled = pred_frame.reshape(-1, 1).float() / pred_frame.reshape(-1).float().max().clamp_min(1.0)
        node_extra = torch.cat([mean_flow, speed, area_log, frame_scaled], dim=1)

        h = pooled + self.node_extra(node_extra)
        attn_entropy = 0.0
        for layer in self.layers:
            h, attn = layer(h, edge, adj, valid, self.diffusion_steps)
            attn_entropy = float((-(attn.clamp_min(1e-12) * torch.log(attn.clamp_min(1e-12))).sum(dim=-1).mean()).detach().cpu().item())

        delta = self.output(h).view(b, c, 1, 1, 1)
        alpha = torch.sigmoid(self.alpha_logit).to(dtype=latent.dtype, device=latent.device)
        enhanced = latent + alpha * delta
        stats = {
            "graph_alpha": float(alpha.detach().cpu().item()),
            "graph_edge_density": float(valid.float().mean().detach().cpu().item()),
            "topo_attn_entropy": attn_entropy,
        }
        return enhanced, stats


class TopoGraphDiffusion3DResNet(base.HF2VADLike3DResNet):
    def __init__(
        self,
        fea_dim,
        mem_dim,
        mem_temperature,
        mem_shrink_thr,
        detach_recon_motion=False,
        use_graph=True,
        graph_alpha=0.10,
        graph_topk=8,
        graph_proximity_weight=0.25,
        topo_layers=2,
        topo_heads=4,
        topo_dropout=0.10,
        topo_diffusion_steps=2,
        topo_temporal_window=2,
        topo_temporal_weight=0.35,
        topo_direction_weight=0.20,
        topo_speed_weight=0.15,
    ):
        super().__init__(
            fea_dim=fea_dim,
            mem_dim=mem_dim,
            mem_temperature=mem_temperature,
            mem_shrink_thr=mem_shrink_thr,
            detach_recon_motion=detach_recon_motion,
            use_graph=False,
        )
        self.use_graph = use_graph
        self.motion_graph = TopoGraphDiffusionTransformer(
            channels=fea_dim,
            layers=topo_layers,
            heads=topo_heads,
            dropout=topo_dropout,
            alpha_init=graph_alpha,
            topk=graph_topk,
            proximity_weight=graph_proximity_weight,
            temporal_window=topo_temporal_window,
            temporal_weight=topo_temporal_weight,
            direction_weight=topo_direction_weight,
            speed_weight=topo_speed_weight,
            diffusion_steps=topo_diffusion_steps,
        ) if use_graph else None

    def forward(self, observed_app, motion, bbox=None, pred_frame=None):
        motion_latent = self.motion_encoder(motion)
        graph_stats = {"graph_alpha": 0.0, "graph_edge_density": 0.0, "topo_attn_entropy": 0.0}
        if self.motion_graph is not None:
            motion_latent, graph_stats = self.motion_graph(motion_latent, bbox, pred_frame, motion)

        mem_motion, att, query, mem_read = self.memory(motion_latent)
        recon_motion = self.motion_decoder(mem_motion)

        motion_for_pred = recon_motion.detach() if self.detach_recon_motion else recon_motion
        app_features = self.app_encoder(observed_app)
        recon_motion_features = self.recon_motion_encoder(motion_for_pred)
        pred_features = self.pred_fuse(torch.cat([app_features, recon_motion_features], dim=1))
        pred_frame_out = self.frame_decoder(self.temporal_pool(pred_features))

        aux = {
            "att": att,
            "query": query,
            "mem_read": mem_read,
            "relation_anomaly": base.relation_anomaly_from_attention(att, motion_latent.shape),
            **graph_stats,
        }
        return recon_motion, pred_frame_out, aux


def init_model_optimizer(device, args):
    model = TopoGraphDiffusion3DResNet(
        fea_dim=base.DEFAULTS["fea_dim"],
        mem_dim=base.DEFAULTS["mem_dim"],
        mem_temperature=base.DEFAULTS["mem_temperature"],
        mem_shrink_thr=base.DEFAULTS["mem_shrink_thr"],
        detach_recon_motion=args.detach_recon_motion,
        use_graph=not args.disable_graph,
        graph_alpha=args.graph_alpha,
        graph_topk=args.graph_topk,
        graph_proximity_weight=args.graph_proximity_weight,
        topo_layers=args.topo_layers,
        topo_heads=args.topo_heads,
        topo_dropout=args.topo_dropout,
        topo_diffusion_steps=args.topo_diffusion_steps,
        topo_temporal_window=args.topo_temporal_window,
        topo_temporal_weight=args.topo_temporal_weight,
        topo_direction_weight=args.topo_direction_weight,
        topo_speed_weight=args.topo_speed_weight,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=base.DEFAULTS["use_amp"] and device.type == "cuda")
    return model, optimizer, scaler


def build_save_dir(args):
    if args.save_dir is not None:
        return Path(args.save_dir)
    return Path("./outputs") / f"topograph_diffusion_3dresnet2_{args.dataset_name}"


def main():
    base.parse_args = parse_args
    base.init_model_optimizer = init_model_optimizer
    base.build_save_dir = build_save_dir
    base.main()


if __name__ == "__main__":
    main()
