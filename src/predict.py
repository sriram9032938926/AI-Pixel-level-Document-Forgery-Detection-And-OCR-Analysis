import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.classifier_model import DocumentClassifier
from src.segformer_model import ForgerySegFormer
from src.utils import mask_to_heatmap, calculate_tampered_percentage


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

classifier_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
    ToTensorV2()
])

segment_transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(),
    ToTensorV2()
])


class ForgeryPredictor:
    def __init__(self, classifier_path, segmenter_path):
        self.classifier = DocumentClassifier().to(DEVICE)
        self.classifier.load_state_dict(
            torch.load(classifier_path, map_location=DEVICE)
        )
        self.classifier.eval()

        self.segmenter = ForgerySegFormer().to(DEVICE)
        self.segmenter.load_state_dict(
            torch.load(segmenter_path, map_location=DEVICE)
        )
        self.segmenter.eval()

    def predict(self, image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        cls_tensor = classifier_transform(image=image_rgb)["image"].unsqueeze(0).to(DEVICE)
        seg_tensor = segment_transform(image=image_rgb)["image"].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            cls_out = self.classifier(cls_tensor)
            cls_probs = torch.softmax(cls_out, dim=1).cpu().numpy()[0]
            cls_pred = int(np.argmax(cls_probs))
            cls_conf = float(cls_probs[cls_pred])

            seg_logits = self.segmenter(seg_tensor)
            seg_logits = F.interpolate(
                seg_logits,
                size=(image_rgb.shape[0], image_rgb.shape[1]),
                mode="bilinear",
                align_corners=False
            )
            seg_probs = torch.sigmoid(seg_logits).cpu().numpy()[0, 0]

        heatmap, overlay = mask_to_heatmap(image_rgb, seg_probs)
        tampered_percent = calculate_tampered_percentage(seg_probs)

        label_map = {0: "Real", 1: "Fake"}

        return {
            "label": label_map[cls_pred],
            "confidence": cls_conf,
            "mask": seg_probs,
            "heatmap": heatmap,
            "overlay": overlay,
            "tampered_percent": tampered_percent
        }


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    classifier_path = os.path.join(BASE_DIR, "models", "classifier_best.pth")
    segmenter_path = os.path.join(BASE_DIR, "models", "segformer_best.pth")

    image_path = input("Enter image path: ").strip()

    predictor = ForgeryPredictor(classifier_path, segmenter_path)

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    result = predictor.predict(image_bgr)

    print("Prediction:", result["label"])
    print("Confidence:", round(result["confidence"] * 100, 2), "%")
    print("Tampered Area:", round(result["tampered_percent"], 2), "%")

    cv2.imshow("Original", image_bgr)
    cv2.imshow("Predicted Mask", (result["mask"] * 255).astype(np.uint8))
    cv2.imshow("Heatmap Overlay", cv2.cvtColor(result["overlay"], cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()