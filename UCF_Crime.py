# ======================= IMPORT & CONFIG =======================
import os, cv2, torch, random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

SRC_ROOT = '/kaggle/input/anomaly-detection-dataset/Anomaly-Detection-Dataset'
CLIP_ROOT = '/kaggle/working/clips_all'
TRAIN_LIST = 'Anomaly_Train.txt'

CLIP_LEN = 16
NUM_CLIPS_PER_VIDEO = 3    
RESIZE = (112, 112)
BATCH_SIZE = 4
EPOCHS = 10                  # ↑ tăng nhẹ
FOLDS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ANOMALY_CLASSES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion",
    "Fighting", "Robbery", "Shooting", "Shoplifting", "Stealing",
    "Vandalism", "RoadAccidents"
]
ALL_CLASSES = ANOMALY_CLASSES + ["Normal"]
LABEL_MAP = {name: i for i, name in enumerate(ALL_CLASSES)}
NUM_CLASSES = len(ALL_CLASSES)

# ======================= VIDEO MAP =======================
def build_video_map(root_dir):
    video_map = {}
    for d, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith(('.avi', '.mp4')):
                video_map[f] = os.path.join(d, f)
    return video_map

video_map = build_video_map(SRC_ROOT)

def find_video_path(rel_path):
    return video_map.get(os.path.basename(rel_path), None)

# ======================= CLIP EXTRACTION =======================
def extract_multiple_clips(video_path, num_clips=NUM_CLIPS_PER_VIDEO):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < CLIP_LEN:
        cap.release()
        return []

    clips = []
    for _ in range(num_clips):
        start = random.randint(0, max(0, frame_count - CLIP_LEN))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(CLIP_LEN):
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, RESIZE)
            frames.append(frame)
        if len(frames) == CLIP_LEN:
            clips.append(np.stack(frames, axis=0))
    cap.release()
    return clips

# ======================= PRECOMPUTE CLIPS =======================
os.makedirs(CLIP_ROOT, exist_ok=True)
with open(os.path.join(SRC_ROOT, TRAIN_LIST)) as f:
    lines = f.read().splitlines()

for rel in tqdm(lines, desc="Precomputing clips"):
    parts = rel.split('/')
    label_name = next((p for p in parts if p in LABEL_MAP), "Normal")
    label = LABEL_MAP[label_name]
    vp = find_video_path(rel)
    if not vp: continue
    clips = extract_multiple_clips(vp)
    for i, clip in enumerate(clips):
        tensor = torch.from_numpy(clip).permute(3, 0, 1, 2).float() / 255.
        fname = f"{label}_{os.path.splitext(os.path.basename(rel))[0]}_{i}.pt"
        torch.save(tensor, os.path.join(CLIP_ROOT, fname))

# ======================= DATASET =======================
class VideoDataset(Dataset):
    def __init__(self, samples): self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, lbl = self.samples[idx]
        clip = torch.load(path)
        return clip, lbl

# ======================= MODEL =======================
def build_model():
    weights = R2Plus1D_18_Weights.DEFAULT
    model = r2plus1d_18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(DEVICE)

# ======================= EVALUATE =======================
def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    used = sorted(np.unique(np.concatenate([y_true, y_pred])))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred, labels=used))
    print("Classification Report:\n", classification_report(y_true, y_pred, labels=used, target_names=[ALL_CLASSES[i] for i in used]))

# ======================= TRAIN =======================
def train_kfold():
    files = [f for f in os.listdir(CLIP_ROOT) if f.endswith('.pt')]
    samples = [(os.path.join(CLIP_ROOT, f), int(f.split('_')[0])) for f in files]
    labels = [lbl for _, lbl in samples]

    class_weights = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    kfold = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(samples, labels)):
        print(f"\n========== Fold {fold + 1} ==========")
        train_s = [samples[i] for i in train_idx]
        val_s   = [samples[i] for i in val_idx]

        train_dl = DataLoader(VideoDataset(train_s), BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        val_dl   = DataLoader(VideoDataset(val_s),   BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

        model = build_model()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        for ep in range(1, EPOCHS + 1):
            model.train()
            total_loss = 0
            for x, y in train_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[Fold {fold+1} | Epoch {ep}] Loss: {total_loss/len(train_dl):.4f}")
            evaluate(model, val_dl)

        torch.save(model.state_dict(), f"/kaggle/working/r2plus1d_fold{fold+1}.pth")

# ======================= RUN =======================
if __name__ == "__main__":
    train_kfold()
