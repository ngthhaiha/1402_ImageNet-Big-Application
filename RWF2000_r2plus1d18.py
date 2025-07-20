# ========================== IMPORTS & SETUP ==========================
import os, cv2, time, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.models.video import r2plus1d_18
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_map = {'Fight': 1, 'NonFight': 0}

# ========================== DATASET ==========================
class RWF2000Dataset(Dataset):
    def __init__(self, video_dir, label_map, frames_per_clip=64, transform=None):
        self.paths, self.labels, self.transform = [], [], transform
        self.frames_per_clip = frames_per_clip
        for label in os.listdir(video_dir):
            class_dir = os.path.join(video_dir, label)
            for file in os.listdir(class_dir):
                if file.endswith(".avi"):
                    self.paths.append(os.path.join(class_dir, file))
                    self.labels.append(label_map[label])

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        path, label = self.paths[idx], self.labels[idx]
        cap = cv2.VideoCapture(path); frames = []; total = int(cap.get(7))
        for i in range(self.frames_per_clip):
            cap.set(1, int(i * total / self.frames_per_clip))
            ret, f = cap.read()
            if not ret: break
            f = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (112, 112))
            f = self.transform(f) if self.transform else f
            frames.append(f)
        cap.release()
        while len(frames) < self.frames_per_clip: frames.append(frames[-1])
        clip = torch.stack(frames).permute(1, 0, 2, 3)  # (C, T, H, W)
        return clip, label

# ========================== EVALUATION ==========================
def evaluate_model(model, data_loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in tqdm(data_loader, desc="Evaluating"):
            X, y = X.to(device), y.to(device)
            out = model(X).argmax(1)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(out.cpu().tolist())

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Classification Report
    target_names = ["NonFight", "Fight"]
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))

    # Metrics
    acc = report["accuracy"]
    macro_p = report["macro avg"]["precision"]
    macro_r = report["macro avg"]["recall"]
    macro_f1 = report["macro avg"]["f1-score"]
    weighted_f1 = report["weighted avg"]["f1-score"]

    return acc, macro_p, macro_r, macro_f1, weighted_f1

# ========================== SETUP DATA ==========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989])
])
data_path = "/kaggle/input/rwf2000/RWF-2000/train"
dataset = RWF2000Dataset(data_path, label_map, transform=transform)

# ========================== K-FOLD TRAINING ==========================
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []
start_time = time.time()

for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
    print(f"\n========== Fold {fold+1} ==========")

    train_sub = Subset(dataset, train_ids)
    val_sub = Subset(dataset, val_ids)
    train_loader = DataLoader(train_sub, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_sub, batch_size=4, shuffle=False, num_workers=2)

    model = r2plus1d_18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()

    EPOCHS = 5
    for ep in range(EPOCHS):
        model.train(); epoch_loss = 0
        for X, y in tqdm(train_loader, desc=f"Fold {fold+1} - Epoch {ep+1}"):
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            out = model(X)
            loss = loss_fn(out, y)
            loss.backward(); opt.step()
            epoch_loss += loss.item()
        scheduler.step()

    acc, macro_p, macro_r, macro_f1, weighted_f1 = evaluate_model(model, val_loader)
    fold_results.append({
        "fold": fold+1,
        "acc": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    })
    torch.save(model.state_dict(), f"r2plus1d_fold{fold+1}.pth")
    print(f"Fold {fold+1} Results - Acc: {acc:.4f}, Macro-F1: {macro_f1:.4f}, Weighted-F1: {weighted_f1:.4f}")

end_time = time.time()
print(f"\n⏱️ Tổng thời gian huấn luyện 5-fold: {(end_time - start_time)/60:.2f} phút")

# ========================== RESULTS SUMMARY ==========================
df = pd.DataFrame(fold_results)
print("\n===== 📊 Kết quả Cross-Validation (5-Fold) =====")
print(df.to_string(index=False))

avg_result = df.mean(numeric_only=True)
print("\n🎯 Trung bình toàn bộ:")
print(avg_result.round(4))
