import os
import re
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label map (7 classes including normal)
decode_map = {
    'A': 0,  # Normal
    'B1': 1,  # Fighting
    'B2': 2,  # Shooting
    'B4': 3,  # Riot
    'B5': 4,  # Abuse
    'B6': 5,  # Car Accident
    'G':  6   # Explosion
}
num_classes = len(decode_map)

# Image transform
frame_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                         std=[0.22803, 0.22145, 0.216989])
])

# Read video frames
def read_frames(path, num_frames=16):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while cap.isOpened() and len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if not frames:
        return [np.zeros((112,112,3), dtype=np.uint8)] * num_frames
    if len(frames) > num_frames:
        idx = np.linspace(0, len(frames)-1, num_frames).astype(int)
        frames = [frames[i] for i in idx]
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames

# Parse labels as multi-hot vector
def parse_label_from_name(name):
    labels = torch.zeros(num_classes, dtype=torch.float32)
    if 'label_A' in name:
        labels[0] = 1  # Normal
    if '_B1' in name:
        labels[1] = 1
    if '_B2' in name:
        labels[2] = 1
    if '_B4' in name:
        labels[3] = 1
    if '_B5' in name:
        labels[4] = 1
    if '_B6' in name:
        labels[5] = 1
    if '_G' in name:
        labels[6] = 1
    return labels

# Dataset classes
class VideoDataset(Dataset):
    def __init__(self, root_dirs, num_frames=16, transform=frame_transform):
        self.paths = []
        for d in root_dirs:
            self.paths += sorted(Path(d).glob("*.mp4"))
        self.num_frames = num_frames
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        frames = read_frames(path, self.num_frames)
        clip = torch.stack([self.transform(f) for f in frames], dim=1)  # (C, T, H, W)
        label = parse_label_from_name(path.stem)
        return clip, label

# Model
class VideoCNN(nn.Module):
    def __init__(self, num_classes):
        super(VideoCNN, self).__init__()
        self.backbone = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

# Evaluation
def evaluate(model, loader, threshold=0.5):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for clips, labels in loader:
            clips, labels = clips.to(device), labels.to(device)
            logits = model(clips)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    print(classification_report(
        y_true, y_pred,
        target_names=list(decode_map.keys()),
        zero_division=0
    ))
    return f1_score(y_true, y_pred, average='macro')

# Main script
if __name__ == '__main__':
    train_dirs = [
        "/kaggle/input/xd-violence/1-1004",
        "/kaggle/input/xd-violence/1005-2004",
        "/kaggle/input/xd-violence/2005-2804"
    ]

    # Load dataset
    full_train_ds = VideoDataset(train_dirs)
    subset_size = int(0.2 * len(full_train_ds))
    subset_ds = Subset(full_train_ds, list(range(subset_size)))

    # Compute pos_weight
    label_list = [parse_label_from_name(Path(p).stem) for p in full_train_ds.paths[:subset_size]]
    label_matrix = torch.stack(label_list)
    counts = label_matrix.sum(dim=0)
    pos_weight = ((subset_size - counts) / (counts + 1e-6)).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_f1 = 0.0

    for fold, (tr_idx, val_idx) in enumerate(kf.split(subset_ds), 1):
        print(f"\n==== Fold {fold} ====")
        tr_loader = DataLoader(Subset(subset_ds, tr_idx), batch_size=4, shuffle=True, num_workers=0)
        vl_loader = DataLoader(Subset(subset_ds, val_idx), batch_size=4, shuffle=False, num_workers=0)

        model = VideoCNN(num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        for epoch in range(1, 6):  # 5 epochs for speed
            model.train()
            total_loss = 0.0
            for clips, labels in tr_loader:
                clips, labels = clips.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(clips)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            print(f"Epoch {epoch}/5 - Loss: {total_loss / len(tr_loader):.4f}")

        # Validation
        f1 = evaluate(model, vl_loader)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), 'best_model.pth')

    print("\n=== Final Test Evaluation ===")
    best_model = VideoCNN(num_classes).to(device)
    best_model.load_state_dict(torch.load('best_model.pth'))
    test_loader = DataLoader(subset_ds, batch_size=4, shuffle=False, num_workers=0)
    evaluate(best_model, test_loader)
