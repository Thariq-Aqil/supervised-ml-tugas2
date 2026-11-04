import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def evaluate_predictions(y_true, probabilities):
    """
    Evaluate classification predictions.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    probabilities : array-like
        Predicted class probabilities.

    Returns
    -------
    dict
        Accuracy, classification report, and confusion matrix.
    """
    predictions = np.argmax(probabilities, axis=1)

    results = {
        "accuracy": accuracy_score(y_true, predictions),
        "classification_report": classification_report(
            y_true,
            predictions,
            zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            predictions
        )
    }

    return results
