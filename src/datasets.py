import os
import cv2
import numpy as np
from torch.utils.data import Dataset


class SIDTDClassifierDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        base_dir = os.path.join(root_dir, "clips_cropped", "Images")

        label_folders = [
            ("real", 0),
            ("reals", 0),
            ("fake", 1),
            ("fakes", 1),
        ]

        added = set()

        for label_name, label in label_folders:
            folder = os.path.join(base_dir, label_name)
            if not os.path.exists(folder):
                continue

            for file_name in os.listdir(folder):
                if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                    file_path = os.path.join(folder, file_name)
                    if file_path not in added:
                        self.samples.append((file_path, label))
                        added.add(file_path)

        print(f"SIDTD samples found: {len(self.samples)}")

        if len(self.samples) == 0:
            raise ValueError(
                f"No SIDTD images found in: {base_dir}\n"
                f"Expected folders like real/reals and fake/fakes."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label


class CASIASegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, only_tampered=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.only_tampered = only_tampered
        self.samples = []

        valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

        image_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(valid_exts)
        ]

        mask_map = {
            os.path.splitext(f)[0]: os.path.join(mask_dir, f)
            for f in os.listdir(mask_dir)
            if f.lower().endswith(valid_exts)
        }

        total = 0
        removed = 0

        for img_name in image_files:
            base_name = os.path.splitext(img_name)[0]
            img_path = os.path.join(image_dir, img_name)

            if base_name not in mask_map:
                continue

            mask_path = mask_map[base_name]
            raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if raw_mask is None:
                continue

            total += 1
            bin_mask = (raw_mask > 0).astype(np.uint8)

            if self.only_tampered and bin_mask.sum() == 0:
                removed += 1
                continue

            self.samples.append((img_path, mask_path))

        print(f"Total pairs found: {total}")
        print(f"Removed empty masks: {removed}")
        print(f"Final usable samples: {len(self.samples)}")

        if len(self.samples) == 0:
            raise ValueError("No valid samples found after filtering.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if raw_mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")

        if image.shape[:2] != raw_mask.shape[:2]:
            print(
                f"Size mismatch fixed: {os.path.basename(img_path)} | "
                f"image={image.shape[:2]} mask={raw_mask.shape[:2]}"
            )
            raw_mask = cv2.resize(
                raw_mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        mask = (raw_mask > 0).astype(np.uint8)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.unsqueeze(0).float()