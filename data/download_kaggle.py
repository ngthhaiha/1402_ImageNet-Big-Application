import os
import shutil

# Phải set cache trước khi import kagglehub
HOME_DIR = "/home/grouphahieu"

os.environ["KAGGLEHUB_CACHE"] = f"{HOME_DIR}/kagglehub_cache"

import kagglehub

# Thư mục đích lưu dataset
DEST_DIR = f"{HOME_DIR}/datasets/optical-flow-ped2-ucsd"

# Download dataset từ Kaggle
path = kagglehub.dataset_download("lmnguynpht/optical-flow-ped2-ucsd")

print("Dataset downloaded to cache:", path)

# Copy dataset sang thư mục dễ quản lý
os.makedirs(os.path.dirname(DEST_DIR), exist_ok=True)
shutil.copytree(path, DEST_DIR, dirs_exist_ok=True)

print("Dataset copied to:", DEST_DIR)
