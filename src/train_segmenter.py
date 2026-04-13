import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from datasets import CASIASegmentationDataset
from segformer_model import ForgerySegFormer
from utils import dice_score, iou_score


# =========================
# CONFIG
# =========================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-4
IMG_SIZE = 512
NUM_WORKERS = 0

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE_DIR, "data", "CASIA_SEG", "images")
MASK_DIR = os.path.join(BASE_DIR, "data", "CASIA_SEG", "masks")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SAVE_PATH = os.path.join(MODELS_DIR, "segformer_best.pth")

os.makedirs(MODELS_DIR, exist_ok=True)


# =========================
# REPRODUCIBILITY
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# =========================
# LOSS
# =========================
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        smooth = 1e-6

        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        dice = (2.0 * intersection + smooth) / (
            probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + smooth
        )
        dice_loss = 1.0 - dice.mean()

        return bce_loss + dice_loss


# =========================
# TRANSFORMS
# =========================
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(),
    ToTensorV2()
])


# =========================
# PATH DEBUG
# =========================
print("BASE_DIR:", BASE_DIR)
print("IMAGE_DIR:", IMAGE_DIR)
print("MASK_DIR:", MASK_DIR)
print("IMAGE_DIR exists:", os.path.exists(IMAGE_DIR))
print("MASK_DIR exists:", os.path.exists(MASK_DIR))

if os.path.exists(IMAGE_DIR):
    print("images folder items:", os.listdir(IMAGE_DIR)[:10])

if os.path.exists(MASK_DIR):
    print("masks folder items:", os.listdir(MASK_DIR)[:10])


# =========================
# DATASET
# =========================
full_dataset = CASIASegmentationDataset(
    IMAGE_DIR,
    MASK_DIR,
    transform=None,
    only_tampered=True
)

print("Total dataset size:", len(full_dataset))

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_subset, val_subset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

train_dataset = CASIASegmentationDataset(
    IMAGE_DIR,
    MASK_DIR,
    transform=train_transform,
    only_tampered=True
)

val_dataset = CASIASegmentationDataset(
    IMAGE_DIR,
    MASK_DIR,
    transform=val_transform,
    only_tampered=True
)

train_dataset.samples = [full_dataset.samples[i] for i in train_subset.indices]
val_dataset.samples = [full_dataset.samples[i] for i in val_subset.indices]

print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset))


# =========================
# DATALOADERS
# =========================
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE == "cuda")
)


# =========================
# MODEL / LOSS / OPTIMIZER
# =========================
model = ForgerySegFormer().to(DEVICE)
criterion = DiceBCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

best_val_iou = -1.0


# =========================
# TRAIN LOOP
# =========================
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

    for images, masks in train_bar:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        logits = model(images)
        logits = F.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    val_iou_total = 0.0
    val_dice_total = 0.0
    count = 0

    debug_done = False
    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")

    with torch.no_grad():
        for images, masks in val_bar:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)
            logits = F.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            loss = criterion(logits, masks)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.3).float()

            if not debug_done:
                print("\nValidation debug:")
                print("masks unique:", torch.unique(masks))
                print("preds unique:", torch.unique(preds))
                print("masks sum per image:", masks.view(masks.size(0), -1).sum(dim=1)[:8])
                print("preds sum per image:", preds.view(preds.size(0), -1).sum(dim=1)[:8])
                print("probs min/max:", probs.min().item(), probs.max().item())
                debug_done = True

            batch_iou = iou_score(probs, masks).item()
            batch_dice = dice_score(probs, masks).item()

            val_loss += loss.item()
            val_iou_total += batch_iou
            val_dice_total += batch_dice
            count += 1

            val_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                iou=f"{batch_iou:.4f}",
                dice=f"{batch_dice:.4f}"
            )

    avg_val_loss = val_loss / count
    avg_iou = val_iou_total / count
    avg_dice = val_dice_total / count

    print(
        f"\nEpoch {epoch+1}/{EPOCHS}: "
        f"Train Loss={avg_train_loss:.4f}, "
        f"Val Loss={avg_val_loss:.4f}, "
        f"Val IoU={avg_iou:.4f}, "
        f"Val Dice={avg_dice:.4f}"
    )

    if avg_iou > best_val_iou:
        best_val_iou = avg_iou
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"Saved best segformer model to: {SAVE_PATH}")

print(f"\nTraining completed. Best Val IoU: {best_val_iou:.4f}")