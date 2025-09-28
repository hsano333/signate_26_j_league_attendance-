from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


class BinaryAnalysis:
    def __init__(self):
        pass

    def transform_label(self, label):
        # print(f"{label.shape=}")
        # print(f"{len(label.shape)=}")
        # print(f"{label[0]=}")
        if len(label.shape) == 1:
            return label < 0.5
        return label[:, 0] < 0.5

    def accuracy(self, y_true, y_pred):
        """Calculate accuracy."""
        return accuracy_score(y_true, y_pred)

    def precision(self, y_true, y_pred):
        """Calculate precision."""
        return precision_score(y_true, y_pred)
        true_positive = ((y_true == 1) & (y_pred == 1)).sum()
        false_positive = ((y_true == 0) & (y_pred == 1)).sum()
        return (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive) > 0
            else 0
        )

    def recall(self, y_true, y_pred):
        """Calculate recall."""
        return recall_score(y_true, y_pred)
        true_positive = ((y_true == 1) & (y_pred == 1)).sum()
        false_negative = ((y_true == 1) & (y_pred == 0)).sum()
        return (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative) > 0
            else 0
        )

    def f1_score(self, y_true, y_pred):
        """Calculate F1 score."""
        return f1_score(y_true, y_pred)
        precision = self.precision(y_true, y_pred)
        recall = self.recall(y_true, y_pred)
        return (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

    def auc_score(self, y_true, y_pred):
        # Calculate AUC score.
        return roc_auc_score(y_true, y_pred)
