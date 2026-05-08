import numpy as np
from mmcv.transforms import Compose
from mmdet.apis import init_detector as mmdet_init_detector
from mmdet.apis import inference_detector as mmdet_inference_detector


def init_detector(config, checkpoint=None, device='cuda:0'):
    """
    Initialize detector with MMDetection 3.x API.

    Args:
        config (str): path to config file
        checkpoint (str | None): checkpoint path
        device (str): e.g. 'cuda:0' or 'cpu'

    Returns:
        model: initialized detector
    """
    model = mmdet_init_detector(config, checkpoint, device=device)
    model.eval()
    return model


def inference_detector(model, img):
    """
    Compatible wrapper for old HF2VAD code.

    Old code expects:
        result = inference_detector(model, img)
        bbox_result = result
        bboxes = np.vstack(bbox_result)

    So this function returns:
        list[np.ndarray], one ndarray per class
        each ndarray shape: [N, 5] with columns [x1, y1, x2, y2, score]
    """
    if isinstance(img, np.ndarray):
        # Build test pipeline for ndarray input
        pipeline_cfg = model.cfg.test_dataloader.dataset.pipeline

        # Make a shallow copy of pipeline config so we do not permanently mutate original cfg
        pipeline_cfg = [dict(step) for step in pipeline_cfg]

        if len(pipeline_cfg) > 0 and pipeline_cfg[0].get('type', '') != 'LoadImageFromNDArray':
            pipeline_cfg[0]['type'] = 'LoadImageFromNDArray'

        test_pipeline = Compose(pipeline_cfg)
        result = mmdet_inference_detector(model, img, test_pipeline=test_pipeline)
    else:
        result = mmdet_inference_detector(model, img)

    pred_instances = result.pred_instances

    bboxes = pred_instances.bboxes.detach().cpu().numpy()
    scores = pred_instances.scores.detach().cpu().numpy()
    labels = pred_instances.labels.detach().cpu().numpy()

    # Determine class count
    if hasattr(model, 'dataset_meta') and isinstance(model.dataset_meta, dict) and 'classes' in model.dataset_meta:
        num_classes = len(model.dataset_meta['classes'])
    else:
        num_classes = int(labels.max()) + 1 if labels.size > 0 else 1

    # Return empty per-class arrays if no detections
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for _ in range(num_classes)]

    dets = np.concatenate([bboxes, scores[:, None]], axis=1).astype(np.float32)

    bbox_result = [dets[labels == i] for i in range(num_classes)]
    return bbox_result