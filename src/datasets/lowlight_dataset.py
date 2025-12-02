import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class LowLightDataset(Dataset):
    """
    Dataset cho bài toán:
    - input: ảnh degraded (tối + mờ)
    - target: ảnh clean (rõ)
    Yêu cầu: file trong clean/ và degraded/ cùng tên.
    """

    def __init__(self, degraded_dir, clean_dir, image_size=256):
        self.degraded_dir = degraded_dir
        self.clean_dir = clean_dir

        # Lấy danh sách file chung giữa 2 folder
        degraded_files = set(os.listdir(degraded_dir))
        clean_files = set(os.listdir(clean_dir))
        self.filenames = sorted(list(degraded_files & clean_files))

        if len(self.filenames) == 0:
            raise RuntimeError(f"Không tìm thấy ảnh chung giữa {degraded_dir} và {clean_dir}")

        # Transform: resize + ToTensor + normalize [0,1]
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),  # tự scale về [0,1]
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        degraded_path = os.path.join(self.degraded_dir, fname)
        clean_path = os.path.join(self.clean_dir, fname)

        degraded = Image.open(degraded_path).convert("RGB")
        clean = Image.open(clean_path).convert("RGB")

        degraded = self.transform(degraded)
        clean = self.transform(clean)

        return degraded, clean
