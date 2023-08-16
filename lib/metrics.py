import torch
from medpy import metric

def calculate_f1_score(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1

    pre = metric.binary.precision(pred, gt)
    recall = metric.binary.recall(pred, gt)

    f1 = 2*pre*recall/(pre+recall)

    return f1

class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += val.detach().cpu()
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n