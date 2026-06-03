import sys
import os
from pathlib import Path
import cv2
import torch
from torchvision import transforms

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
PHATLAM_DIR = PROJECT_ROOT / "phatlam" / "phatlam_pipeline_paper_topk"

if str(PHATLAM_DIR) not in sys.path:
    sys.path.append(str(PHATLAM_DIR))

try:
    from phase1.extract_nasnet_gmm_topk_features import gmm_select_global_topk_frames, NASNetMobileFeatureExtractor
    from phase1.models.transformer import TransformerAnomalyModel
    from phase2.models import build_phase2_model
except ImportError as e:
    print(f"Warning: Could not import AI models. {e}")

# Constants
PHASE1_CKPT = str(PROJECT_ROOT / "UCF-Crime" / "outputs_phase1_cv" / "top30_transformer_fold0" / "best_auc.pth")
PHASE2_CKPT = str(PROJECT_ROOT / "UCF-Crime" / "outputs_phase2_cv_transformer_fold0_noleak" / "topk8_transformer_fold0_convnext_tiny_fold1" / "best.pth")

# Alphabetical 13 classes used in Phase 2
UCF_CRIME_13_CLASSES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism"
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_extractor = None
_phase1_model = None
_phase2_model = None
_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

_phase2_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])


def _load_models():
    global _extractor, _phase1_model, _phase2_model
    if _extractor is not None:
        return

    print("Loading AI models...")
    _extractor = NASNetMobileFeatureExtractor('nasnetamobile', pretrained=False).to(device).eval()

    ckpt1 = torch.load(PHASE1_CKPT, map_location="cpu", weights_only=False)
    cfg1 = ckpt1.get("config", {})
    _phase1_model = TransformerAnomalyModel(
        input_dim=int(cfg1.get("input_dim", 1056)),
        seq_len=int(cfg1.get("seq_len", 30)),
        hidden_dim=int(cfg1.get("hidden_dim", 256)),
        num_layers=int(cfg1.get("num_layers", 4)),
        num_heads=int(cfg1.get("num_heads", 4)),
        dropout=float(cfg1.get("dropout", 0.3)),
    ).to(device).eval()
    _phase1_model.load_state_dict(ckpt1["model_state"], strict=True)

    ckpt2 = torch.load(PHASE2_CKPT, map_location="cpu", weights_only=False)
    _phase2_model = build_phase2_model("convnext_tiny", num_classes=13, pretrained=False).to(device).eval()
    _phase2_model.load_state_dict(ckpt2["model_state"], strict=True)
    print("Models loaded successfully.")


def run_phase1(video_path: str) -> list[dict]:
    """
    Returns list of segments: [{start_time, end_time, anomaly_score, _cache_frames}]
    """
    _load_models()
    
    frames, sel_idx, motion, bounds, pad_mask, fps, total, width, height = gmm_select_global_topk_frames(
        Path(video_path),
        top_k=30,
        image_size=224,
        frame_stride=1,
        warmup_frames=5,
    )
    
    batch = torch.stack([_transform(f) for f in frames], dim=0)
    feats = []
    with torch.no_grad():
        for start in range(0, batch.shape[0], 64):
            x = batch[start:start+64].to(device, non_blocking=True)
            y = _extractor(x)
            feats.append(y)
        features = torch.cat(feats, dim=0).unsqueeze(0) # [1, 30, 1056]
        
        out = _phase1_model(features)
        segment_logits = out["segment_logits"].squeeze(0) # [30]
        segment_scores = torch.sigmoid(segment_logits).cpu().numpy()
        
    segments = []
    fps_f = float(fps)
    for i in range(len(bounds)):
        s_frame, e_frame = bounds[i]
        start_time = float(s_frame) / fps_f
        end_time = float(e_frame) / fps_f
        if end_time <= start_time:
            end_time = start_time + (1.0 / fps_f)
        score = float(segment_scores[i])
        segments.append({
            "start_time": start_time,
            "end_time": end_time,
            "anomaly_score": score,
            "_s_frame": int(s_frame),
            "_e_frame": int(e_frame),
        })
        
    segments.sort(key=lambda x: x["start_time"])
    merged = []
    for seg in segments:
        if not merged:
            merged.append(seg)
        else:
            last = merged[-1]
            if seg["start_time"] <= last["end_time"]:
                last["end_time"] = max(last["end_time"], seg["end_time"])
                last["anomaly_score"] = max(last["anomaly_score"], seg["anomaly_score"])
                last["_s_frame"] = min(last["_s_frame"], seg["_s_frame"])
                last["_e_frame"] = max(last["_e_frame"], seg["_e_frame"])
            else:
                merged.append(seg)
                
    return merged


def run_phase2(video_path: str, segment: dict) -> dict:
    """
    Returns: {predicted_class, confidence_score}
    """
    _load_models()
    
    s_frame = segment.get("_s_frame")
    e_frame = segment.get("_e_frame")
    
    if s_frame is None or e_frame is None:
        fps = 30.0
        s_frame = int(segment["start_time"] * fps)
        e_frame = int(segment["end_time"] * fps)
        
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    duration = e_frame - s_frame + 1
    if duration <= 0:
        duration = 1
    
    # Extract 16 frames uniformly
    idxs = [int(x) for x in torch.linspace(0, duration - 1, 16)]
    
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, s_frame + idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    
    if not frames:
        return {"predicted_class": "Normal", "confidence_score": 1.0}
        
    while len(frames) < 16:
        frames.append(frames[-1].copy())
        
    batch = torch.stack([_phase2_transform(f) for f in frames], dim=0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = _phase2_model(batch)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        
    score, class_idx = probs.max(dim=0)
    class_idx = int(class_idx.cpu())
    confidence_score = float(score.cpu())
    
    predicted_class = UCF_CRIME_13_CLASSES[class_idx]
    
    return {
        "predicted_class": predicted_class,
        "confidence_score": confidence_score,
    }
