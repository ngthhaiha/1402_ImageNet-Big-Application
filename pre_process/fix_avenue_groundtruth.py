import argparse
import glob
import os
import pickle
import shutil
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Resample Avenue frame-level groundtruth labels so they match the "
            "actual number of extracted test frames."
        )
    )
    parser.add_argument(
        "--gt-path",
        default="data/avenue/ground_truth_demo/gt_label.json",
        help="Path to the original Avenue groundtruth pickle file.",
    )
    parser.add_argument(
        "--frames-dir",
        default="data/avenue/testing/frames",
        help="Directory containing extracted test frames, one subdirectory per video.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help=(
            "Path for the resampled groundtruth pickle. "
            "Default: <gt-path stem>_resampled<suffix>."
        ),
    )
    parser.add_argument(
        "--frame-ext",
        default=".jpg",
        help="Frame filename extension used in --frames-dir.",
    )
    parser.add_argument(
        "--strategy",
        choices=("nearest",),
        default="nearest",
        help=(
            "Resampling strategy. 'nearest' maps each extracted frame to the "
            "closest source groundtruth frame on the normalized timeline."
        ),
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite --gt-path after saving a backup next to it.",
    )
    parser.add_argument(
        "--backup-suffix",
        default=".bak",
        help="Suffix used when --inplace is enabled.",
    )
    return parser.parse_args()


def load_gt(gt_path):
    with open(gt_path, "rb") as f:
        gt = pickle.load(f)
    if not isinstance(gt, dict):
        raise TypeError(f"Expected dict groundtruth, got {type(gt)}")
    return gt


def collect_frame_counts(frames_dir, frame_ext):
    video_dirs = sorted(
        [path for path in Path(frames_dir).iterdir() if path.is_dir()],
        key=lambda path: path.name,
    )
    frame_counts = []
    for video_dir in video_dirs:
        count = len(sorted(video_dir.glob(f"*{frame_ext}")))
        frame_counts.append((video_dir.name, count))
    return frame_counts


def resample_1d_labels(labels, target_len):
    src = np.asarray(labels)
    if src.ndim != 1:
        raise ValueError(f"Expected 1D labels, got shape={src.shape}")
    if target_len <= 0:
        raise ValueError(f"target_len must be positive, got {target_len}")
    if len(src) == target_len:
        return src.copy()

    indices = np.rint(np.linspace(0, len(src) - 1, num=target_len)).astype(np.int64)
    indices = np.clip(indices, 0, len(src) - 1)
    return src[indices]


def make_default_output_path(gt_path):
    gt_path = Path(gt_path)
    return gt_path.with_name(f"{gt_path.stem}_resampled{gt_path.suffix}")


def main():
    args = parse_args()

    gt_path = Path(args.gt_path)
    output_path = Path(args.output_path) if args.output_path else make_default_output_path(gt_path)

    gt = load_gt(gt_path)
    frame_counts = collect_frame_counts(args.frames_dir, args.frame_ext)

    if len(gt) != len(frame_counts):
        raise ValueError(
            f"Groundtruth video count ({len(gt)}) does not match frame folders ({len(frame_counts)})."
        )

    fixed_gt = {}
    print("video  gt_frames  input_frames  ratio")
    print("-------------------------------------")
    for video_idx, (video_name, input_len) in enumerate(frame_counts):
        if video_idx not in gt:
            raise KeyError(f"Missing video id {video_idx} in groundtruth.")
        src = np.asarray(gt[video_idx])
        dst = resample_1d_labels(src, input_len)
        fixed_gt[video_idx] = dst.astype(src.dtype, copy=False)
        ratio = len(src) / float(input_len)
        print(f"{video_name:>5} {len(src):>10} {input_len:>12} {ratio:>6.3f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(fixed_gt, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSaved resampled groundtruth to: {output_path}")

    if args.inplace:
        backup_path = gt_path.with_name(f"{gt_path.name}{args.backup_suffix}")
        if backup_path.exists():
            raise FileExistsError(
                f"Backup file already exists: {backup_path}. Remove it or change --backup-suffix."
            )
        shutil.copy2(gt_path, backup_path)
        shutil.copy2(output_path, gt_path)
        print(f"Backed up original groundtruth to: {backup_path}")
        print(f"Overwrote original groundtruth at: {gt_path}")


if __name__ == "__main__":
    main()
