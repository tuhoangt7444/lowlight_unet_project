import os
import torch

# Thư mục gốc project (thường là thư mục hiện tại khi chạy train.py)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Đường dẫn dữ liệu
DATA_DIR = os.path.join(BASE_DIR, "data")
CLEAN_DIR = os.path.join(DATA_DIR, "clean")
DEGRADED_DIR = os.path.join(DATA_DIR, "degraded")

# Đường dẫn outputs
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
SAMPLES_DIR = os.path.join(OUTPUT_DIR, "samples")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

# Cấu hình train
IMAGE_SIZE = 256       # resize về 256x256
BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRINT_EVERY = 100      # in log mỗi 100 batch
VAL_SPLIT = 0.1        # 10% dữ liệu để validate
