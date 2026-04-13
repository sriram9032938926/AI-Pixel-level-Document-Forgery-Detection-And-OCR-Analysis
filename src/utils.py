import cv2
import numpy as np
import torch


def mask_to_heatmap(image_rgb, pred_mask):
    pred_mask = np.clip(pred_mask, 0, 1)
    heat = (pred_mask * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_rgb, 0.65, heat, 0.35, 0)
    return heat, overlay


def calculate_tampered_percentage(pred_mask, threshold=0.5):
    binary = (pred_mask > threshold).astype(np.uint8)
    tampered_pixels = binary.sum()
    total_pixels = binary.size
    return (tampered_pixels / total_pixels) * 100.0


def dice_score(pred, target, eps=1e-6):
    pred = (pred > 0.3).float()
    target = target.float()

    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    pred_sum = pred.sum(dim=1)
    target_sum = target.sum(dim=1)
    union = pred_sum + target_sum

    valid = union > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device)

    dice = (2.0 * intersection[valid] + eps) / (union[valid] + eps)
    return dice.mean()


def iou_score(pred, target, eps=1e-6):
    pred = (pred > 0.3).float()
    target = target.float()

    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    pred_sum = pred.sum(dim=1)
    target_sum = target.sum(dim=1)
    union = pred_sum + target_sum - intersection

    valid = (pred_sum + target_sum) > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device)

    iou = (intersection[valid] + eps) / (union[valid] + eps)
    return iou.mean()