import os
import torch
from torch.utils.data import DataLoader
import cv2
import torch.nn as nn
import numpy as np
import yaml
import joblib
import pickle
import gc
from tqdm import tqdm

from datasets.dataset import Chunked_sample_dataset, normalize_dataset_name, resolve_dataset_dir_name
from models.ml_memAE_sc import ML_MemAE_SC
from utils.eval_utils import save_evaluation_curves

METADATA = {
    "ped2": {
        "testing_video_num": 12,
        "testing_frames_cnt": [180, 180, 150, 180, 150, 180, 180, 180, 120, 150,
                               180, 180]
    },
    "avenue": {
        "testing_video_num": 21,
        "testing_frames_cnt": [1439, 1211, 923, 947, 1007, 1283, 605, 36, 1175, 841,
                               472, 1271, 549, 507, 1001, 740, 426, 294, 248, 273,
                               76],
    },
    "shanghaitech": {
        "testing_video_num": 107,
        "testing_frames_cnt": [265, 433, 337, 601, 505, 409, 457, 313, 409, 337,
                               337, 457, 577, 313, 529, 193, 289, 289, 265, 241,
                               337, 289, 265, 217, 433, 409, 529, 313, 217, 241,
                               313, 193, 265, 317, 457, 337, 361, 529, 409, 313,
                               385, 457, 481, 457, 433, 385, 241, 553, 937, 865,
                               505, 313, 361, 361, 529, 337, 433, 481, 649, 649,
                               409, 337, 769, 433, 241, 217, 265, 265, 217, 265,
                               409, 385, 481, 457, 313, 601, 241, 481, 313, 337,
                               457, 217, 241, 289, 337, 313, 337, 265, 265, 337,
                               361, 433, 241, 433, 601, 505, 337, 601, 265, 313,
                               241, 289, 361, 385, 217, 337, 265]
    },

}


def _gt_sort_key(k):
    name = os.path.splitext(str(k))[0]
    parts = name.split("_")
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return int(parts[0]), int(parts[1])
    return int(name), 0


def _sorted_gt_items(gt):
    return sorted(gt.items(), key=lambda item: _gt_sort_key(item[0]))


def _get_testing_frame_counts(dataset_name, gt=None):
    dataset_name = normalize_dataset_name(dataset_name)
    if isinstance(gt, dict) and len(gt) > 0:
        return [len(np.asarray(labels)) for _, labels in _sorted_gt_items(gt)]
    return METADATA[dataset_name]["testing_frames_cnt"]


def _get_dataset_dir_name(config):
    return config.get("dataset_dir_name") or resolve_dataset_dir_name(
        config["dataset_base_dir"],
        config["dataset_name"],
    )


def _get_gt_path(config):
    dataset_logic_name = normalize_dataset_name(config["dataset_name"])
    dataset_dir_name = _get_dataset_dir_name(config)
    gt_file = "gt_label_12fps.json" if dataset_logic_name == "shanghaitech" else "gt_label.json"

    candidates = [
        os.path.join(config["dataset_base_dir"], dataset_dir_name, "ground_truth_demo", gt_file),
        os.path.join(config["dataset_base_dir"], dataset_dir_name, "ground_truth_demo", "gt_label.json"),
        os.path.join(config["dataset_base_dir"], dataset_logic_name, "ground_truth_demo", gt_file),
        os.path.join(config["dataset_base_dir"], dataset_logic_name, "ground_truth_demo", "gt_label.json"),
    ]
    for gt_path in candidates:
        if os.path.exists(gt_path):
            return gt_path
    raise FileNotFoundError("No ground-truth file found. Tried: %s" % ", ".join(candidates))


def _list_test_chunk_files(testing_chunked_samples_path):
    if os.path.isdir(testing_chunked_samples_path):
        files = [
            os.path.join(testing_chunked_samples_path, f)
            for f in sorted(os.listdir(testing_chunked_samples_path))
            if f.startswith("chunked_samples_") and f.endswith(".pkl")
        ]
        if not files:
            raise FileNotFoundError("No chunked sample files found in %s" % testing_chunked_samples_path)
        return files
    if os.path.isfile(testing_chunked_samples_path):
        return [testing_chunked_samples_path]
    raise FileNotFoundError("Path not found: %s" % testing_chunked_samples_path)


def evaluate(config, ckpt_path, testing_chunked_samples_file, suffix):
    dataset_name = normalize_dataset_name(config["dataset_name"])
    device = config["device"]
    num_workers = config["num_workers"]
    eval_dir = os.path.join(config["eval_root"], config["exp_name"])

    gt_path = _get_gt_path(config)
    gt = pickle.load(open(gt_path, "rb"))
    testing_frame_counts = _get_testing_frame_counts(dataset_name, gt=gt)
    testset_num_frames = int(np.sum(testing_frame_counts))

    os.makedirs(eval_dir, exist_ok=True)

    model = ML_MemAE_SC(num_in_ch=config["model_paras"]["motion_channels"],
                        seq_len=config["model_paras"]["num_flows"],
                        features_root=config["model_paras"]["feature_root"],
                        num_slots=config["model_paras"]["num_slots"],
                        shrink_thres=config["model_paras"]["shrink_thres"],
                        mem_usage=config["model_paras"]["mem_usage"],
                        skip_ops=config["model_paras"]["skip_ops"]).to(device).eval()

    # load weights
    model_weights = torch.load(ckpt_path, weights_only=False)["model_state_dict"]
    model.load_state_dict(model_weights)
    print("load pre-trained success!")

    score_func = nn.MSELoss(reduction="none")

    eval_batch_size = min(config.get("eval_batchsize", config.get("batchsize", 64)), 32)
    testing_chunk_files = _list_test_chunk_files(testing_chunked_samples_file)

    # bbox anomaly scores for each frame
    frame_bbox_scores = [{} for i in range(testset_num_frames)]
    bbox_uid = 0

    for chunk_idx, chunk_file in enumerate(testing_chunk_files):
        dataset_test = Chunked_sample_dataset(chunk_file, last_flow=True)
        dataloader_test = DataLoader(dataset=dataset_test, batch_size=eval_batch_size, num_workers=num_workers, shuffle=False)

        for ii, test_data in tqdm(enumerate(dataloader_test),
                                  desc="Eval chunk %d/%d" % (chunk_idx + 1, len(testing_chunk_files)),
                                  total=len(dataloader_test)):
            _, sample_ofs_test, bbox_test, pred_frame_test, indices_test = test_data
            sample_ofs_test = sample_ofs_test.to(device)

            out_test = model(sample_ofs_test)
            loss_of_test = score_func(out_test["recon"], sample_ofs_test).cpu().data.numpy()
            scores = np.sum(np.sum(np.sum(loss_of_test, axis=3), axis=2), axis=1)

            # anomaly scores for each sample
            for i in range(len(scores)):
                frame_id = int(pred_frame_test[i][-1].item())
                frame_bbox_scores[frame_id][bbox_uid] = scores[i]
                bbox_uid += 1

        del dataloader_test, dataset_test
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    joblib.dump(frame_bbox_scores, os.path.join(config["eval_root"], config["exp_name"],
                                                "frame_bbox_scores_%s.json" % suffix))

    # frame_bbox_scores = joblib.load(os.path.join(config["eval_root"], config["exp_name"],
    #                                              "frame_bbox_scores_%s.json" % suffix))

    # frame-level anomaly score (i.e., the maximum anomaly scores of all the objects in it)
    frame_scores = np.empty(len(frame_bbox_scores))
    for i in range(len(frame_scores)):
        if len(frame_bbox_scores[i].items()) == 0:
            frame_scores[i] = 0  # assign ZERO when no object exists
        else:
            frame_scores[i] = np.max(list(frame_bbox_scores[i].values()))

    joblib.dump(frame_scores,
                os.path.join(config["eval_root"], config["exp_name"], "frame_scores_%s.json" % suffix))

    # frame_scores = joblib.load(
    #     os.path.join(config["eval_root"], config["exp_name"], "frame_scores_%s.json" % suffix)
    # )

    # ================== Calculate AUC ==============================
    # load gt labels
    gt_concat = np.concatenate([np.asarray(labels) for _, labels in _sorted_gt_items(gt)], axis=0)

    new_gt = np.array([])
    new_frame_scores = np.array([])

    start_idx = 0
    for cur_video_len in testing_frame_counts:

        gt_each_video = gt_concat[start_idx:start_idx + cur_video_len][4:]
        scores_each_video = frame_scores[start_idx:start_idx + cur_video_len][4:]

        start_idx += cur_video_len

        new_gt = np.concatenate((new_gt, gt_each_video), axis=0)
        new_frame_scores = np.concatenate((new_frame_scores, scores_each_video), axis=0)

    gt_concat = new_gt
    frame_scores = new_frame_scores

    curves_save_path = os.path.join(config["eval_root"], config["exp_name"], 'anomaly_curves_%s' % suffix)
    auc = save_evaluation_curves(frame_scores, gt_concat, curves_save_path,
                                 np.array(testing_frame_counts) - 4)

    return auc


if __name__ == '__main__':
    model_save_path = "./pretrained_ckpts/sh_ML_MemAE_SC.pth"
    cfg_file = "./pretrained_ckpts/sh_ML_MemAE_SC_cfg.yaml"

    config = yaml.safe_load(open(cfg_file))
    dataset_base_dir = config["dataset_base_dir"]
    dataset_name = normalize_dataset_name(config["dataset_name"])
    dataset_dir_name = _get_dataset_dir_name(config)

    if dataset_name == "shanghaitech":
        testing_chunked_samples_file = os.path.join("./data", dataset_dir_name, "testing/chunked_samples")
    else:
        testing_chunked_samples_file = os.path.join("./data", dataset_dir_name,
                                                    "testing/chunked_samples/chunked_samples_00.pkl")

    with torch.no_grad():
        auc = evaluate(config, model_save_path, testing_chunked_samples_file, suffix="best")
        print(auc)
