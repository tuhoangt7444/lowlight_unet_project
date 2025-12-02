import os
import torch
from PIL import Image
import torchvision.transforms as T

from config import DEVICE, CHECKPOINT_DIR
from models.unet_lite import UNetLite


def load_model():
    model = UNetLite(in_ch=3, out_ch=3).to(DEVICE)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def enhance_folder(input_dir, output_dir, image_size=256):
    os.makedirs(output_dir, exist_ok=True)

    model = load_model()

    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])

    to_pil = T.ToPILImage()

    for fname in os.listdir(input_dir):
        in_path = os.path.join(input_dir, fname)
        if not os.path.isfile(in_path):
            continue

        img = Image.open(in_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            y = model(x)[0].cpu()

        out_img = to_pil(y)
        out_img.save(os.path.join(output_dir, fname))

    print("Hoàn thành enhance folder:", input_dir)


if __name__ == "__main__":
    # Ví dụ: dùng lại data/degraded để xem model phục hồi thế nào
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = os.path.join(base_dir, "data", "degraded")
    output_dir = os.path.join(base_dir, "outputs", "samples", "test_restore")

    enhance_folder(input_dir, output_dir)
