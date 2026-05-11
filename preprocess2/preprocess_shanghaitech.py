#!/usr/bin/env python3
"""
Local preprocessing pipeline for ShanghaiTech.

This is a local, script-friendly port of the two Kaggle notebooks in this
folder.  It runs the three preprocessing stages needed by HF2VAD:

1. RAFT optical flow extraction.
2. Faster R-CNN person boxes plus foreground-motion boxes.
3. Chunked STC sample generation.

Default input:
    data/shanghaitech/testing/frames

Default output:
    data/shanghaitech2/testing/{frames,flows,chunked_samples}
    data/shanghaitech2/shanghaitech_bboxes_test.npy
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import joblib
import numpy as np
import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from torchvision.transforms import functional as TF
from tqdm import tqdm


DATASET_NAME = "shanghaitech"

DATASET_CFG = {
    "conf_thr": 0.5,
    "min_area": 8 * 8,
    "cover_thr": 0.65,
    "binary_thr": 15,
    "gauss_mask_size": 5,
    "contour_min_area": 40 * 40,
}

SPLIT_ALIASES = {
    "test": ("test", "testing"),
    "testing": ("test", "testing"),
    "train": ("train", "training"),
    "training": ("train", "training"),
}

FRAME_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
DEFAULT_RAFT_HW = (640, 1024)
DEFAULT_PATCH_SIZE = 32
DEFAULT_CONTEXT = 4


@dataclass(frozen=True)
class VideoFrames:
    name: str
    frames: tuple[Path, ...]


def parse_split(value: str) -> tuple[str, str]:
    key = value.lower().strip()
    if key not in SPLIT_ALIASES:
        raise ValueError("split must be one of: testing, test, training, train")
    return SPLIT_ALIASES[key]


def parse_stages(value: str) -> set[str]:
    allowed = {"all", "link", "flows", "bboxes", "samples"}
    stages = {x.strip().lower() for x in value.split(",") if x.strip()}
    unknown = stages - allowed
    if unknown:
        raise ValueError(f"unknown stage(s): {sorted(unknown)}")
    if not stages or "all" in stages:
        return {"link", "flows", "bboxes", "samples"}
    return stages


def scan_videos(frames_root: Path, limit_videos: int | None = None) -> list[VideoFrames]:
    if not frames_root.exists():
        raise FileNotFoundError(f"frames root does not exist: {frames_root}")

    video_dirs = sorted(
        p for p in frames_root.iterdir()
        if p.is_dir() and not p.name.startswith(".") and not p.name.endswith("_gt")
    )
    videos: list[VideoFrames] = []

    if video_dirs:
        for video_dir in video_dirs:
            frames = tuple(
                sorted(
                    p for p in video_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in FRAME_EXTS
                )
            )
            if frames:
                videos.append(VideoFrames(video_dir.name, frames))
    else:
        frames = tuple(
            sorted(
                p for p in frames_root.iterdir()
                if p.is_file() and p.suffix.lower() in FRAME_EXTS
            )
        )
        if frames:
            videos.append(VideoFrames(frames_root.name, frames))

    if limit_videos is not None:
        videos = videos[:limit_videos]
    if not videos:
        raise RuntimeError(f"no videos/frames found under {frames_root}")
    return videos


def total_frames(videos: Iterable[VideoFrames]) -> int:
    return sum(len(v.frames) for v in videos)


def safe_symlink_or_copy(src: Path, dst: Path, copy: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def prepare_output_tree(
    input_root: Path,
    output_root: Path,
    split_dir: str,
    videos: list[VideoFrames],
    copy_frames: bool,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    out_frames_root = output_root / split_dir / "frames"
    out_frames_root.mkdir(parents=True, exist_ok=True)

    for video in videos:
        safe_symlink_or_copy(video.frames[0].parent, out_frames_root / video.name, copy=copy_frames)

    gt_src = input_root / "ground_truth_demo"
    gt_dst = output_root / "ground_truth_demo"
    if gt_src.exists():
        safe_symlink_or_copy(gt_src, gt_dst, copy=copy_frames)

    return out_frames_root


def read_frame_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        from PIL import Image

        return np.array(Image.open(path).convert("RGB"))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_frame_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        from PIL import Image

        rgb = np.array(Image.open(path).convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return img


def extract_flow_raft(
    img1_rgb: np.ndarray,
    img2_rgb: np.ndarray,
    raft_model: torch.nn.Module,
    raft_transforms,
    target_hw: tuple[int, int],
    device: torch.device,
) -> np.ndarray:
    target_h, target_w = target_hw
    orig_h, orig_w = img1_rgb.shape[:2]

    img1_resized = cv2.resize(img1_rgb, (target_w, target_h))
    img2_resized = cv2.resize(img2_rgb, (target_w, target_h))

    t1 = TF.to_tensor(img1_resized).unsqueeze(0).to(device)
    t2 = TF.to_tensor(img2_resized).unsqueeze(0).to(device)
    t1, t2 = raft_transforms(t1, t2)

    with torch.no_grad():
        flow_pred = raft_model(t1, t2)[-1]

    flow_np = flow_pred[0].permute(1, 2, 0).detach().cpu().numpy()
    flow_np = cv2.resize(flow_np, (orig_w, orig_h))
    flow_np[..., 0] *= orig_w / target_w
    flow_np[..., 1] *= orig_h / target_h
    return flow_np.astype(np.float32)


def load_raft(device: torch.device):
    print("Loading RAFT-Large...")
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=True).to(device).eval()
    return model, weights.transforms()


def save_zero_flow(frame_path: Path, save_path: Path) -> None:
    img = read_frame_rgb(frame_path)
    zero = np.zeros((img.shape[0], img.shape[1], 2), dtype=np.float32)
    np.save(str(save_path), zero)


def extract_flows(
    videos: list[VideoFrames],
    output_root: Path,
    split_dir: str,
    device: torch.device,
    raft_hw: tuple[int, int],
    overwrite: bool,
    flow_name_mode: str,
) -> None:
    flows_root = output_root / split_dir / "flows"
    flows_root.mkdir(parents=True, exist_ok=True)
    raft_model, raft_transforms = load_raft(device)

    pair_count = sum(max(0, len(video.frames) - 1) for video in videos)
    pbar = tqdm(total=pair_count, desc="Extracting RAFT flows")

    for video in videos:
        video_flow_dir = flows_root / video.name
        video_flow_dir.mkdir(parents=True, exist_ok=True)
        frames = video.frames
        if flow_name_mode == "target" and frames:
            first_flow_path = video_flow_dir / f"{frames[0].name}.npy"
            if overwrite or not first_flow_path.exists():
                save_zero_flow(frames[0], first_flow_path)

        for idx in range(len(frames) - 1):
            f1 = frames[idx]
            f2 = frames[idx + 1]
            flow_frame = f2 if flow_name_mode == "target" else f1
            save_path = video_flow_dir / f"{flow_frame.name}.npy"
            if save_path.exists() and not overwrite:
                pbar.update(1)
                continue

            img1 = read_frame_rgb(f1)
            img2 = read_frame_rgb(f2)
            flow = extract_flow_raft(img1, img2, raft_model, raft_transforms, raft_hw, device)
            np.save(str(save_path), flow)
            pbar.update(1)

    pbar.close()
    del raft_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def get_obj_bboxes(
    img_rgb: np.ndarray,
    detector: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    tensor = TF.to_tensor(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = detector(tensor)[0]

    boxes = pred["boxes"].detach().cpu().numpy()
    scores = pred["scores"].detach().cpu().numpy()
    labels = pred["labels"].detach().cpu().numpy()

    keep = (labels == 1) & (scores >= DATASET_CFG["conf_thr"])
    boxes = boxes[keep]
    if boxes.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)

    areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    boxes = boxes[areas >= DATASET_CFG["min_area"]]
    if boxes.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return boxes.astype(np.float32)


def del_cover_bboxes(bboxes: np.ndarray) -> np.ndarray:
    if bboxes.shape[0] == 0:
        return bboxes

    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = areas.argsort()
    keep: list[int] = []

    for pos, small_idx in enumerate(order):
        larger = order[pos + 1:]
        if len(larger) == 0:
            keep.append(small_idx)
            continue
        ix1 = np.maximum(x1[small_idx], x1[larger])
        iy1 = np.maximum(y1[small_idx], y1[larger])
        ix2 = np.minimum(x2[small_idx], x2[larger])
        iy2 = np.minimum(y2[small_idx], y2[larger])
        inter = np.maximum(0, ix2 - ix1 + 1) * np.maximum(0, iy2 - iy1 + 1)
        if not np.any(inter / areas[small_idx] > DATASET_CFG["cover_thr"]):
            keep.append(small_idx)

    if not keep:
        return np.zeros((0, 4), dtype=np.float32)
    return bboxes[keep].astype(np.float32)


def get_fg_bboxes(frames_bgr: list[np.ndarray], obj_bboxes: np.ndarray) -> np.ndarray:
    area_thr = DATASET_CFG["contour_min_area"]
    binary_thr = DATASET_CFG["binary_thr"]
    gauss_k = DATASET_CFG["gauss_mask_size"]
    extend = 2

    sum_grad = np.zeros_like(frames_bgr[0], dtype=np.float32)
    for idx in range(len(frames_bgr) - 1):
        img1 = cv2.GaussianBlur(frames_bgr[idx].astype(np.float32), (gauss_k, gauss_k), 0)
        img2 = cv2.GaussianBlur(frames_bgr[idx + 1].astype(np.float32), (gauss_k, gauss_k), 0)
        sum_grad += cv2.absdiff(img1, img2)

    _, binary = cv2.threshold(sum_grad.astype(np.uint8), binary_thr, 255, cv2.THRESH_BINARY)
    for bbox in obj_bboxes:
        x1, y1, x2, y2 = bbox.astype(np.int32)
        y1e = max(0, y1 - extend)
        y2e = min(y2 + extend, binary.shape[0] - 1)
        x1e = max(0, x1 - extend)
        x2e = min(x2 + extend, binary.shape[1] - 1)
        binary[y1e:y2e + 1, x1e:x2e + 1] = 0

    gray = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY) if binary.ndim == 3 else binary
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fg_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (w + 1) * (h + 1) > area_thr and w > 0 and h > 0 and w / h < 10 and h / w < 10:
            fg_boxes.append(
                [
                    max(0, x - extend),
                    max(0, y - extend),
                    min(x + w + extend, gray.shape[1] - 1),
                    min(y + h + extend, gray.shape[0] - 1),
                ]
            )

    if not fg_boxes:
        return np.zeros((0, 4), dtype=np.float32)
    return np.array(fg_boxes, dtype=np.float32)


def context_indices(length: int, idx: int, context: int, border_mode: str) -> list[int]:
    if border_mode == "predict":
        start = max(0, idx - context)
        end = idx
        need = context + 1
    else:
        start = max(0, idx - context)
        end = min(length - 1, idx + context)
        need = 2 * context + 1

    indices = list(range(start, end + 1))
    pad = need - len(indices)
    if pad > 0:
        if start == 0:
            indices = [indices[0]] * pad + indices
        else:
            indices = indices + [indices[-1]] * pad
    return indices


def load_detector(device: torch.device):
    print("Loading Faster R-CNN ResNet50 FPN...")
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    return fasterrcnn_resnet50_fpn(weights=weights).to(device).eval()


def extract_bboxes(
    videos: list[VideoFrames],
    output_root: Path,
    mode: str,
    device: torch.device,
    overwrite: bool,
) -> np.ndarray:
    save_path = output_root / f"{DATASET_NAME}_bboxes_{mode}.npy"
    if save_path.exists() and not overwrite:
        print(f"Loading existing bboxes: {save_path}")
        return np.load(str(save_path), allow_pickle=True)

    detector = load_detector(device)
    all_bboxes: list[np.ndarray] = []

    for video in videos:
        frames = video.frames
        for idx in tqdm(range(len(frames)), desc=f"BBoxes {video.name}", leave=False):
            ctx = context_indices(len(frames), idx, context=1, border_mode="hard")
            imgs_bgr = [read_frame_bgr(frames[i]) for i in ctx]
            cur_rgb = cv2.cvtColor(imgs_bgr[len(imgs_bgr) // 2], cv2.COLOR_BGR2RGB)

            obj_boxes = del_cover_bboxes(get_obj_bboxes(cur_rgb, detector, device))
            fg_boxes = get_fg_bboxes(imgs_bgr, obj_boxes)

            if obj_boxes.shape[0] > 0 and fg_boxes.shape[0] > 0:
                boxes = np.concatenate([obj_boxes, fg_boxes], axis=0)
            elif fg_boxes.shape[0] > 0:
                boxes = fg_boxes
            else:
                boxes = obj_boxes
            all_bboxes.append(boxes)

    arr = np.array(all_bboxes, dtype=object)
    output_root.mkdir(parents=True, exist_ok=True)
    np.save(str(save_path), arr)
    non_empty = sum(1 for boxes in arr if len(boxes) > 0)
    print(f"Saved bboxes: {save_path} ({len(arr)} frames, {non_empty} non-empty)")

    del detector
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return arr


def clip_bbox_to_image(bbox: np.ndarray, height: int, width: int) -> tuple[int, int, int, int] | None:
    x1 = int(np.ceil(bbox[0]))
    y1 = int(np.ceil(bbox[1]))
    x2 = int(np.ceil(bbox[2]))
    y2 = int(np.ceil(bbox[3]))
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def crop_resize_hwc(arr: np.ndarray, bbox: np.ndarray, patch_size: int) -> np.ndarray | None:
    h, w = arr.shape[:2]
    clipped = clip_bbox_to_image(bbox, h, w)
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    crop = arr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (patch_size, patch_size))


def flow_path_for_frame(flows_root: Path, video_name: str, frame_path: Path) -> Path:
    return flows_root / video_name / f"{frame_path.name}.npy"


def load_flow_for_frame(flows_root: Path, video: VideoFrames, local_idx: int) -> np.ndarray:
    path = flow_path_for_frame(flows_root, video.name, video.frames[local_idx])
    if path.exists():
        flow = np.load(str(path))
    elif local_idx > 0:
        prev_path = flow_path_for_frame(flows_root, video.name, video.frames[local_idx - 1])
        if not prev_path.exists():
            raise FileNotFoundError(f"missing flow file: {path}")
        flow = np.load(str(prev_path))
    else:
        img = read_frame_bgr(video.frames[local_idx])
        flow = np.zeros((img.shape[0], img.shape[1], 2), dtype=np.float32)

    if flow.ndim == 2:
        flow = flow[..., np.newaxis]
    if flow.shape[2] > 2:
        flow = flow[..., :2]
    if flow.shape[2] == 1:
        flow = np.concatenate([flow, flow], axis=2)
    return flow.astype(np.float32)


def dump_chunk(buffer: dict[str, list], save_dir: Path, chunk_id: int) -> None:
    for key in ("sample_id", "appearance", "motion", "bbox", "pred_frame"):
        buffer[key] = np.asarray(buffer[key])

    out_path = save_dir / f"chunked_samples_{chunk_id:02d}.pkl"
    joblib.dump(buffer, str(out_path))
    print(f"Saved chunk {chunk_id}: {len(buffer['sample_id'])} samples -> {out_path}")
    gc.collect()


def chunk_size_for(mode: str) -> int:
    return 20000 if mode == "test" else 20000


def build_chunked_samples(
    videos: list[VideoFrames],
    output_root: Path,
    split_dir: str,
    all_bboxes: np.ndarray,
    chunk_size: int,
    patch_size: int,
    overwrite: bool,
    chunk_layout: str,
) -> Path:
    save_dir = output_root / split_dir / "chunked_samples"
    save_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(save_dir.glob("chunked_samples_*.pkl"))
    if existing and not overwrite:
        print(f"Chunk files already exist in {save_dir}; skipping samples stage.")
        return save_dir
    if overwrite:
        for path in existing:
            path.unlink()

    expected_frames = total_frames(videos)
    if len(all_bboxes) != expected_frames:
        raise ValueError(f"bbox length mismatch: {len(all_bboxes)} bboxes for {expected_frames} frames")

    flows_root = output_root / split_dir / "flows"
    if not flows_root.exists():
        raise FileNotFoundError(f"flows root does not exist: {flows_root}")

    buffer = {"sample_id": [], "appearance": [], "motion": [], "bbox": [], "pred_frame": []}
    sample_id = 0
    chunk_id = 0
    global_base = 0

    for video in tqdm(videos, desc="Building chunked samples"):
        frames = video.frames

        for local_idx, frame_path in enumerate(frames):
            global_idx = global_base + local_idx
            boxes = all_bboxes[global_idx]
            if len(boxes) == 0:
                continue

            ctx = context_indices(len(frames), local_idx, context=DEFAULT_CONTEXT, border_mode="predict")
            frame_clip = [read_frame_bgr(frames[i]) for i in ctx]
            flow_clip = [load_flow_for_frame(flows_root, video, i) for i in ctx]

            for bbox in boxes:
                app_clip = []
                mot_clip = []
                valid = True
                for app_frame, flow_frame in zip(frame_clip, flow_clip):
                    app_patch = crop_resize_hwc(app_frame, bbox, patch_size)
                    mot_patch = crop_resize_hwc(flow_frame, bbox, patch_size)
                    if app_patch is None or mot_patch is None:
                        valid = False
                        break
                    app_clip.append(app_patch)
                    mot_clip.append(mot_patch[..., :2])

                if not valid:
                    continue

                app_np = np.array(app_clip)
                mot_np = np.array(mot_clip)
                if chunk_layout == "notebook":
                    app_np = np.transpose(app_np, (0, 3, 1, 2))
                    mot_np = np.transpose(mot_np, (0, 3, 1, 2))

                buffer["sample_id"].append(sample_id)
                buffer["appearance"].append(app_np)
                buffer["motion"].append(mot_np.astype(np.float32))
                buffer["bbox"].append(np.asarray(bbox, dtype=np.float32))
                buffer["pred_frame"].append([global_idx])
                sample_id += 1

                if len(buffer["sample_id"]) == chunk_size:
                    dump_chunk(buffer, save_dir, chunk_id)
                    buffer = {"sample_id": [], "appearance": [], "motion": [], "bbox": [], "pred_frame": []}
                    chunk_id += 1

        global_base += len(frames)

    if buffer["sample_id"]:
        dump_chunk(buffer, save_dir, chunk_id)

    print(f"Done: {sample_id} samples in {save_dir}")
    return save_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess ShanghaiTech testing/training locally with RAFT.")
    parser.add_argument("--input-root", type=Path, default=Path("data/shanghaitech"))
    parser.add_argument("--output-root", type=Path, default=Path("data/shanghaitech2"))
    parser.add_argument("--split", default="testing", help="testing/test or training/train; default: testing")
    parser.add_argument(
        "--stages",
        default="all",
        help="comma-separated subset of: link,flows,bboxes,samples; default: all",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--raft-height", type=int, default=DEFAULT_RAFT_HW[0])
    parser.add_argument("--raft-width", type=int, default=DEFAULT_RAFT_HW[1])
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--limit-videos", type=int, default=None, help="debug only: process first N videos")
    parser.add_argument("--overwrite", action="store_true", help="overwrite generated flows/bboxes/chunks")
    parser.add_argument("--copy-frames", action="store_true", help="copy frames instead of symlinking them")
    parser.add_argument(
        "--flow-name-mode",
        choices=("target", "source"),
        default="target",
        help=(
            "target saves flow i->i+1 as frame_{i+1}.jpg.npy plus a zero first flow, "
            "which is aligned with HF2VAD; source matches the first notebook's pair naming."
        ),
    )
    parser.add_argument(
        "--chunk-layout",
        choices=("repo", "notebook"),
        default="repo",
        help="repo stores [T,H,W,C] chunks for datasets/dataset.py; notebook stores [T,C,H,W].",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    mode, split_dir = parse_split(args.split)
    stages = parse_stages(args.stages)
    device = torch.device(args.device)
    raft_hw = (args.raft_height, args.raft_width)
    chunk_size = args.chunk_size if args.chunk_size is not None else chunk_size_for(mode)

    input_frames_root = args.input_root / split_dir / "frames"
    videos = scan_videos(input_frames_root, limit_videos=args.limit_videos)

    print("=" * 72)
    print("ShanghaiTech local preprocessing")
    print(f"Input frames : {input_frames_root}")
    print(f"Output root  : {args.output_root}")
    print(f"Split        : {split_dir} ({mode})")
    print(f"Videos       : {len(videos)}")
    print(f"Frames       : {total_frames(videos)}")
    print(f"Stages       : {', '.join(sorted(stages))}")
    print(f"Device       : {device}")
    print(f"Chunk size   : {chunk_size}")
    print("=" * 72)

    if "link" in stages:
        prepare_output_tree(args.input_root, args.output_root, split_dir, videos, copy_frames=args.copy_frames)

    if "flows" in stages:
        extract_flows(
            videos=videos,
            output_root=args.output_root,
            split_dir=split_dir,
            device=device,
            raft_hw=raft_hw,
            overwrite=args.overwrite,
            flow_name_mode=args.flow_name_mode,
        )

    bbox_path = args.output_root / f"{DATASET_NAME}_bboxes_{mode}.npy"
    all_bboxes = None
    if "bboxes" in stages:
        all_bboxes = extract_bboxes(videos, args.output_root, mode, device, overwrite=args.overwrite)
    elif "samples" in stages:
        if not bbox_path.exists():
            raise FileNotFoundError(f"missing bbox file for samples stage: {bbox_path}")
        all_bboxes = np.load(str(bbox_path), allow_pickle=True)

    if "samples" in stages:
        if all_bboxes is None:
            all_bboxes = np.load(str(bbox_path), allow_pickle=True)
        build_chunked_samples(
            videos=videos,
            output_root=args.output_root,
            split_dir=split_dir,
            all_bboxes=all_bboxes,
            chunk_size=chunk_size,
            patch_size=args.patch_size,
            overwrite=args.overwrite,
            chunk_layout=args.chunk_layout,
        )

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
