import numpy as np
from sklearn.metrics import accuracy_score


def find_best_weight(
    dit_probs,
    cnn_probs,
    y_true,
    step=0.1
):
    """
    Find the best ensemble weight using validation accuracy.

    ensemble_probs = weight * dit_probs
                   + (1 - weight) * cnn_probs
    """
    best_weight = 0.0
    best_accuracy = 0.0

    weights = np.arange(0.0, 1.0 + step / 2, step)

    for weight in weights:
        ensemble_probs = (
            weight * dit_probs
            + (1.0 - weight) * cnn_probs
        )

        predictions = np.argmax(ensemble_probs, axis=1)
        accuracy = accuracy_score(y_true, predictions)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weight = weight

    return best_weight, best_accuracy


def blend_predictions(
    dit_probs,
    cnn_probs,
    dit_weight
):
    """
    Combine DiT and ConvNeXt class probabilities.
    """
    cnn_weight = 1.0 - dit_weight

    ensemble_probs = (
        dit_weight * dit_probs
        + cnn_weight * cnn_probs
    )

    return ensemble_probs
