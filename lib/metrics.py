from abc import ABC
import numpy as np
from sklearn.metrics import f1_score


class ValidationMetric(ABC):
    def __init__(self, final_validation=False, sort_needed=False):
        self.final_validation = final_validation
        self.sort_needed = sort_needed

    def get_result(self):
        return self.result

    def __repr__(self):
        return self.__class__.__name__
    
class RPrecision(ValidationMetric):
    def __init__(self, final_validation=False):
        super().__init__(final_validation, False)

    def __call__(self, predictions, labels):
        count = 0
        for i, pred in enumerate(predictions):
            pred_sp = np.argsort(-pred)
            label_sp = np.argsort(-labels[i])
            size = int(np.sum(labels[i]))
            pred_sp = pred_sp[0:size]
            label_sp = label_sp[0:size]
            count += np.sum(np.in1d(pred_sp, label_sp))/size
        self.score = count/predictions.shape[0]
        return str(self)

    def is_better(self, score):
        return self.metric_score() > score

    def __str__(self):
        return 'R precision: %.4f' % self.score

class F1Score(ValidationMetric):
    def __init__(self, threshold=None, average='micro', final_validation=False):
        super().__init__(final_validation, False)
        self.forced_threshold = threshold
        self.threshold = threshold
        self.average = average

    def __call__(self, predictions, labels):
        if not self.forced_threshold:
            prop = np.sum(labels)/labels.size
            self.threshold = np.quantile(predictions, 1-prop)
        
        predictions = np.where(predictions >= self.threshold, 1, 0).astype(int)
        labels = labels.astype(int)
        self.score = f1_score(labels, predictions, average=self.average)
        return str(self)

    def is_better(self, score):
        return self.metric_score() > score

    def __str__(self):
        return '%s-F1Score (t=%.5f): %.4f' % (self.average, self.threshold, self.score)
