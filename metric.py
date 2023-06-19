import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

def metrics(prediction, target, n_classes=None):
    """Compute and print metrics (accuracy, confusion matrix and F1 scores).

    Args:
        prediction: list of predicted labels
        target: list of target labels
        n_classes (optional): number of classes, max(target) by default
    Returns:
        accuracy, F1 score by class, confusion matrix
    """

    results = {}
    n_classes = np.max(target) + 1 if n_classes is None else n_classes

    #Compute confusion_matrix
    cm = confusion_matrix(target, prediction, labels=range(n_classes))
    results["Confusion matrix"] = cm

    # Compute OA
    total = np.sum(cm)
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    results["Accuracy"] = accuracy

    # Compute F1 score
    F1scores = np.zeros(len(cm))
    for i in range(len(cm)):
        try:
            F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except ZeroDivisionError:
            F1 = 0.
        F1scores[i] = F1
    results["F1 scores"] = F1scores

    # Compute kappa coefficient
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / \
        float(total * total)
    kappa = (pa - pe) / (1 - pe)
    results["Kappa"] = kappa * 100

    return results

def convert_to_color_(arr_2d, numClass):
    """Convert an array of labels to RGB color-encoded image.

    Args:
        arr_2d: int 2D array of labels
        palette: dict of colors used (label number -> RGB tuple)

    Returns:
        arr_3d: int 2D images of color-encoded labels in RGB format

    """
    palette = {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("hls", numClass - 1)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))

    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)


    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d