import torch
import timm


def build_convnext_model(
    num_classes=16,
    pretrained=True,
    device="cuda"
):
    """
    Build ConvNeXt Tiny model for image classification.
    """
    model = timm.create_model(
        "convnext_tiny.fb_in22k_ft_in1k",
        pretrained=pretrained,
        num_classes=num_classes,
        drop_path_rate=0.1
    )

    model = model.to(device)

    if str(device).startswith("cuda"):
        model = model.to(memory_format=torch.channels_last)

    return model


@torch.no_grad()
def predict_cnn(model, loader, device="cuda"):
    """
    Generate class probabilities from a CNN model.
    """
    model.eval()
    predictions = []

    use_amp = str(device).startswith("cuda")

    for batch in loader:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch

        images = images.to(device, non_blocking=True)

        if use_amp:
            images = images.contiguous(
                memory_format=torch.channels_last
            )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp
        ):
            logits = model(images)

        probabilities = torch.softmax(logits.float(), dim=1)
        predictions.append(probabilities.cpu())

    return torch.cat(predictions, dim=0)
