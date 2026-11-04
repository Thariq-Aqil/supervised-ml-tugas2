
"""Model utilities for DiT-based document image classification."""

import torch
from transformers import BeitForImageClassification


MODEL_NAME = "microsoft/dit-base"
NUM_CLASSES = 16


def build_dit_model(num_labels=NUM_CLASSES, pretrained=True):
    """Create a pretrained DiT model for image classification."""
    if pretrained:
        model = BeitForImageClassification.from_pretrained(
            MODEL_NAME,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )
    else:
        model = BeitForImageClassification(
            BeitForImageClassification.config_class()
        )

    return model


def param_groups(model, lr, decay=0.7, wd=0.05):
    """Create optimizer groups with layer-wise learning rate decay."""
    n = model.config.num_hidden_layers
    groups = {}

    for name, parameter in model.named_parameters():
        if "encoder.layer." in name:
            layer_id = int(
                name.split("encoder.layer.")[1].split(".")[0]
            ) + 1
        elif "embeddings" in name:
            layer_id = 0
        else:
            layer_id = n + 1

        no_weight_decay = parameter.ndim == 1
        key = (layer_id, no_weight_decay)

        if key not in groups:
            groups[key] = {
                "params": [],
                "lr": lr * decay ** (n + 1 - layer_id),
                "weight_decay": 0.0 if no_weight_decay else wd,
            }

        groups[key]["params"].append(parameter)

    return list(groups.values())


@torch.no_grad()
def predict_dit(model, loader, device="cuda"):
    """Generate softmax probabilities for a DataLoader."""
    model.eval()
    outputs = []

    for batch in loader:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        images = images.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(pixel_values=images).logits

        outputs.append(logits.float().softmax(dim=-1).cpu())

    return torch.cat(outputs)
