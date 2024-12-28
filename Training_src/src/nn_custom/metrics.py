import cupy as cp 

def accuracy(y_pred, y_true):
    """
    Compute the accuracy of predictions.

    :param y_pred: Predicted probabilities, shape = (batch_size, num_classes).
    :param y_true: True labels (one-hot encoded), shape = (batch_size, num_classes).
    :return: Accuracy as a float.
    """
    y_pred_labels = cp.argmax(y_pred, axis=1)
    y_true_labels = cp.argmax(y_true, axis=1)
    correct = cp.sum(y_pred_labels == y_true_labels)
    return correct / y_pred.shape[0]

def precision(y_pred, y_true):
    """
    Compute the precision of predictions.

    :param y_pred: Predicted probabilities, shape = (batch_size, num_classes).
    :param y_true: True labels (one-hot encoded), shape = (batch_size, num_classes).
    :return: Precision as a float.
    """
    y_pred_labels = cp.argmax(y_pred, axis=1)
    y_true_labels = cp.argmax(y_true, axis=1)
    tp = cp.sum(cp.logical_and(y_pred_labels == 1, y_true_labels == 1))
    fp = cp.sum(cp.logical_and(y_pred_labels == 1, y_true_labels == 0))
    return tp / (tp + fp)

def recall(y_pred, y_true):
    """
    Compute the recall of predictions.

    :param y_pred: Predicted probabilities, shape = (batch_size, num_classes).
    :param y_true: True labels (one-hot encoded), shape = (batch_size, num_classes).
    :return: Recall as a float.
    """
    y_pred_labels = cp.argmax(y_pred, axis=1)
    y_true_labels = cp.argmax(y_true, axis=1)
    tp = cp.sum(cp.logical_and(y_pred_labels == 1, y_true_labels == 1))
    fn = cp.sum(cp.logical_and(y_pred_labels == 0, y_true_labels == 1))
    return tp / (tp + fn)

def f1_score(y_pred, y_true):
    """
    Compute the F1 score of predictions.

    :param y_pred: Predicted probabilities, shape = (batch_size, num_classes).
    :param y_true: True labels (one-hot encoded), shape = (batch_size, num_classes).
    :return: F1 score as a float.
    """
    precision_score = precision(y_pred, y_true)
    recall_score = recall(y_pred, y_true)
    return 2 * (precision_score * recall_score) / (precision_score + recall_score)

def confusion_matrix(y_pred, y_true):
    """
    Compute the confusion matrix of predictions.

    :param y_pred: Predicted probabilities, shape = (batch_size, num_classes).
    :param y_true: True labels (one-hot encoded), shape = (batch_size, num_classes).
    :return: Confusion matrix as a numpy array.
    """
    y_pred_labels = cp.argmax(y_pred, axis=1)
    y_true_labels = cp.argmax(y_true, axis=1)
    num_classes = y_pred.shape[1]
    confusion_matrix = cp.zeros((num_classes, num_classes))
    for i in range(y_pred.shape[0]):
        confusion_matrix[y_true_labels[i], y_pred_labels[i]] += 1
    return confusion_matrix.astype(int)

f_metrics = {
    "accuracy": accuracy, 
    "precision": precision,
    "recall": recall,
    "f1_score": f1_score,
    "confusion_matrix": confusion_matrix
}