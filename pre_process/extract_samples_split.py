import argparse
import os
import numpy as np
import joblib
from datasets.dataset import (
    get_dataset,
    img_batch_tensor2numpy,
    normalize_dataset_name,
    resolve_dataset_dir_name,
)

SPLIT_TO_DIR = {
    "train": "training",
    "val": "validation",
    "test": "testing"
}


def get_default_chunk_size(dataset_name, mode):
    dataset_name = normalize_dataset_name(dataset_name)
    if dataset_name == "ped2":
        return 100000
    elif dataset_name == "avenue":
        return 200000 if mode == "test" else 20000
    elif dataset_name == "shanghaitech":
        return 300000 if mode == "test" else 50000
    else:
        raise NotImplementedError("dataset name should be one of ped2, UCSDped2, avenue or shanghaitech!")


def dump_chunk(chunked_samples, save_dir, chunk_id):
    chunked_samples["sample_id"] = np.array(chunked_samples["sample_id"])
    chunked_samples["appearance"] = np.array(chunked_samples["appearance"])
    chunked_samples["motion"] = np.array(chunked_samples["motion"])
    chunked_samples["bbox"] = np.array(chunked_samples["bbox"])
    chunked_samples["pred_frame"] = np.array(chunked_samples["pred_frame"])

    out_path = os.path.join(save_dir, f"chunked_samples_{chunk_id:02d}.pkl")
    joblib.dump(chunked_samples, out_path)
    print(f"Chunk {chunk_id} file saved! -> {out_path}")


def samples_extraction(
    dataset_root,
    dataset_name,
    mode,
    all_bboxes,
    save_dir,
    start_idx=0,
    end_idx=None,
    chunk_id_offset=0,
    num_samples_each_chunk=None
):
    num_predicted_frame = 1
    dataset_logic_name = normalize_dataset_name(dataset_name)
    dataset_dir_name = resolve_dataset_dir_name(dataset_root, dataset_name)

    if num_samples_each_chunk is None:
        num_samples_each_chunk = get_default_chunk_size(dataset_logic_name, mode)

    # frames dataset
    dataset = get_dataset(
        dataset_name=dataset_logic_name,
        dir=os.path.join(dataset_root, dataset_dir_name),
        context_frame_num=4,
        mode=mode,
        border_mode="predict",
        all_bboxes=all_bboxes,
        patch_size=32,
        of_dataset=False
    )

    # flows dataset
    flow_dataset = get_dataset(
        dataset_name=dataset_logic_name,
        dir=os.path.join(dataset_root, dataset_dir_name),
        context_frame_num=4,
        mode=mode,
        border_mode="predict",
        all_bboxes=all_bboxes,
        patch_size=32,
        of_dataset=True
    )

    os.makedirs(save_dir, exist_ok=True)

    total_len = len(dataset)
    if end_idx is None:
        end_idx = total_len

    start_idx = max(0, start_idx)
    end_idx = min(end_idx, total_len)

    if start_idx >= end_idx:
        raise ValueError(f"Invalid range: start_idx={start_idx}, end_idx={end_idx}, dataset_len={total_len}")

    print(f"Running samples extraction for {dataset_name}/{mode}")
    print(f"Frame range: [{start_idx}, {end_idx}) out of {total_len}")
    print(f"Chunk size: {num_samples_each_chunk}")
    print(f"Chunk ID offset: {chunk_id_offset}")

    global_sample_id = 0
    cnt = 0
    chunk_id = chunk_id_offset
    chunked_samples = dict(sample_id=[], appearance=[], motion=[], bbox=[], pred_frame=[])

    for idx in range(start_idx, end_idx):
        if (idx - start_idx) % 1000 == 0:
            print(
                "Extracting foreground in {}-th frame, range [{}:{}) / total {}".format(
                    idx + 1, start_idx, end_idx, total_len
                )
            )

        frame_range = dataset._context_range(idx)

        # [num_bboxes, clip_len, C, patch_size, patch_size]
        batch, _ = dataset.__getitem__(idx)
        flow_batch, _ = flow_dataset.__getitem__(idx)

        # all the bboxes in current frame
        cur_bboxes = all_bboxes[idx]
        if len(cur_bboxes) > 0:
            batch = img_batch_tensor2numpy(batch)
            flow_batch = img_batch_tensor2numpy(flow_batch)

            # each STC treated as a sample
            for idx_box in range(cur_bboxes.shape[0]):
                chunked_samples["sample_id"].append(global_sample_id)
                chunked_samples["appearance"].append(batch[idx_box])
                chunked_samples["motion"].append(flow_batch[idx_box])
                chunked_samples["bbox"].append(cur_bboxes[idx_box])
                chunked_samples["pred_frame"].append(frame_range[-num_predicted_frame:])
                global_sample_id += 1
                cnt += 1

                if cnt == num_samples_each_chunk:
                    dump_chunk(chunked_samples, save_dir, chunk_id)

                    chunk_id += 1
                    cnt = 0
                    del chunked_samples
                    chunked_samples = dict(
                        sample_id=[],
                        appearance=[],
                        motion=[],
                        bbox=[],
                        pred_frame=[]
                    )

    # save the remaining samples
    if len(chunked_samples["sample_id"]) != 0:
        dump_chunk(chunked_samples, save_dir, chunk_id)

    print("All samples have been saved!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj_root", type=str, default="/home/liuzhian/hdd4T/code/hf2vad", help="project root path")
    parser.add_argument("--dataset_name", type=str, default="ped2", help="dataset name")
    parser.add_argument("--mode", type=str, default="train", help="train / val / test data")

    parser.add_argument("--start_idx", type=int, default=0, help="start frame index (inclusive)")
    parser.add_argument("--end_idx", type=int, default=None, help="end frame index (exclusive)")
    parser.add_argument("--chunk_id_offset", type=int, default=0, help="starting chunk id")
    parser.add_argument(
        "--num_samples_each_chunk",
        type=int,
        default=None,
        help="override default chunk size"
    )

    args = parser.parse_args()

    if args.mode not in SPLIT_TO_DIR:
        raise ValueError("mode must be one of: train, val, test")

    dataset_root = os.path.join(args.proj_root, "data")
    dataset_logic_name = normalize_dataset_name(args.dataset_name)
    dataset_dir_name = resolve_dataset_dir_name(dataset_root, args.dataset_name)
    all_bboxes = np.load(
        os.path.join(
            dataset_root,
            dataset_dir_name,
            f"{dataset_logic_name}_bboxes_{args.mode}.npy"
        ),
        allow_pickle=True
    )

    save_dir = os.path.join(
        dataset_root,
        dataset_dir_name,
        SPLIT_TO_DIR[args.mode],
        "chunked_samples"
    )

    samples_extraction(
        dataset_root=dataset_root,
        dataset_name=args.dataset_name,
        mode=args.mode,
        all_bboxes=all_bboxes,
        save_dir=save_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        chunk_id_offset=args.chunk_id_offset,
        num_samples_each_chunk=args.num_samples_each_chunk
    )
