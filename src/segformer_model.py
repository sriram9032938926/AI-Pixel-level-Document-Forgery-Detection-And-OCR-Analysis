import torch.nn as nn
from transformers import SegformerForSemanticSegmentation


class ForgerySegFormer(nn.Module):
    def __init__(self, model_name="nvidia/segformer-b0-finetuned-ade-512-512"):
        super().__init__()

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=1,
            ignore_mismatched_sizes=True
        )

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        return outputs.logits