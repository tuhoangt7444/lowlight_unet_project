import os
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class LowLightDataset(Dataset):
    """
    Dataset cho bài toán tăng cường ảnh low-light.

    - degraded_dir: thư mục chứa ảnh input (tối/mờ)
    - clean_dir:    thư mục chứa ảnh ground truth (rõ/sáng)
    - Các ảnh được ghép cặp theo tên file, ví dụ:
        data/degraded/0001.png <-> data/clean/0001.png
    - Ảnh sẽ được resize về (image_size, image_size) và chuyển thành tensor [0,1]
    """

    def __init__(self, degraded_dir: str, clean_dir: str, image_size: int = 256):
        super().__init__()

        self.degraded_dir = degraded_dir
        self.clean_dir = clean_dir
        self.image_size = image_size

        # Chỉ lấy các file ảnh hợp lệ
        valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}

        degraded_files = [
            f for f in os.listdir(degraded_dir)
            if os.path.splitext(f)[1].lower() in valid_ext
        ]

        self.pairs: List[Tuple[str, str]] = []

        for fname in sorted(degraded_files):
            degraded_path = os.path.join(degraded_dir, fname)
            clean_path = os.path.join(clean_dir, fname)

            if not os.path.isfile(degraded_path):
                continue

            if not os.path.isfile(clean_path):
                # Nếu không tìm thấy ảnh clean trùng tên thì bỏ qua
                # Bạn có thể print warning nếu muốn
                # print(f"⚠️ Không tìm thấy ground truth cho {degraded_path}")
                continue

            self.pairs.append((degraded_path, clean_path))

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"Không tìm được cặp ảnh nào trong:\n"
                f"  degraded_dir = {degraded_dir}\n"
                f"  clean_dir    = {clean_dir}\n"
                f"Kiểm tra lại đường dẫn và tên file (phải trùng nhau)."
            )

        # Transform chung: resize + ToTensor (0–1)
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        degraded_path, clean_path = self.pairs[idx]

        degraded_img = Image.open(degraded_path).convert("RGB")
        clean_img = Image.open(clean_path).convert("RGB")

        degraded_tensor = self.transform(degraded_img)
        clean_tensor = self.transform(clean_img)

        # train.py đang mong đợi: (inp, target)
        return degraded_tensor, clean_tensor
