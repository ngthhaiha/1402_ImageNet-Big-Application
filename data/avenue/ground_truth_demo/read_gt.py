import pickle

with open("gt_label.json", "rb") as f:
    gt = pickle.load(f)

frame_counts = {video_id: len(labels) for video_id, labels in gt.items()}
print(frame_counts)
