import os
import cv2
import numpy as np
from glob import glob

# ====== THAY ĐƯỜNG DẪN CHO ĐÚNG ======
CLEAN_DIR = r"data/clean"
DEGRADED_DIR = r"data/degraded"
# =====================================

os.makedirs(DEGRADED_DIR, exist_ok=True)

def make_darker(img, gamma=2.0):
    img = img.astype(np.float32) / 255.0
    img = np.power(img, gamma)
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)

def add_blur(img, ksize=9):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def degrade(img):
    darker = make_darker(img, gamma=2.0)   # chỉnh độ tối
    blurred = add_blur(darker, ksize=9)    # chỉnh độ mờ
    return blurred

img_paths = glob(os.path.join(CLEAN_DIR, "*.*"))

print("Tìm thấy", len(img_paths), "ảnh rõ.")

for p in img_paths:
    img = cv2.imread(p)
    if img is None:
        print("Không đọc được:", p)
        continue

    degraded = degrade(img)

    fname = os.path.basename(p)
    out_path = os.path.join(DEGRADED_DIR, fname)
    cv2.imwrite(out_path, degraded)

print("XONG! Ảnh mờ + tối được lưu vào:", DEGRADED_DIR)
