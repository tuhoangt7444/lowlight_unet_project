import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from config import (
    CLEAN_DIR,
    DEGRADED_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    DEVICE,
    CHECKPOINT_DIR,
    VAL_SPLIT,
    PRINT_EVERY,
)
from datasets.lowlight_dataset import LowLightDataset
from models.unet_lite import UNetLite
from utils import seed_everything, psnr, save_checkpoint


def main():
    seed_everything(42)

    print("Thiết bị:", DEVICE)
    print("Clean dir:", CLEAN_DIR)
    print("Degraded dir:", DEGRADED_DIR)

    # Dataset
    dataset = LowLightDataset(
        degraded_dir=DEGRADED_DIR,
        clean_dir=CLEAN_DIR,
        image_size=IMAGE_SIZE,
    )
    total_len = len(dataset)
    val_len = int(total_len * VAL_SPLIT)
    train_len = total_len - val_len

    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    model = UNetLite(in_ch=3, out_ch=3).to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_psnr = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        # ---- TRAIN ----
        model.train()
        running_loss = 0.0

        for i, (inp, target) in enumerate(train_loader, 1):
            inp = inp.to(DEVICE)
            target = target.to(DEVICE)

            optimizer.zero_grad()
            out = model(inp)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % PRINT_EVERY == 0 or i == len(train_loader):
                avg_loss = running_loss / i
                print(f"[Epoch {epoch}/{NUM_EPOCHS}] [Batch {i}/{len(train_loader)}] Loss: {avg_loss:.4f}")

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        with torch.no_grad():
            for inp, target in val_loader:
                inp = inp.to(DEVICE)
                target = target.to(DEVICE)

                out = model(inp)
                loss = criterion(out, target)
                val_loss += loss.item()
                val_psnr += psnr(out, target).item()

        val_loss /= len(val_loader)
        val_psnr /= len(val_loader)

        print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}, Val PSNR = {val_psnr:.2f} dB")

        # Lưu best model
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            save_checkpoint(model, optimizer, epoch, ckpt_path)
            print(f"==> Lưu model tốt nhất (PSNR={best_val_psnr:.2f} dB) tại {ckpt_path}")


if __name__ == "__main__":
    main()
