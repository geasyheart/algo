# -*- coding: utf8 -*-
# https://www.zhihu.com/question/39840928
import numpy as np
from sklearn.metrics import auc, roc_auc_score, roc_curve


# ---自己按照公式实现
def auc_calculate(labels, preds, n_bins=100):
    postive_len = sum(labels)
    negative_len = len(labels) - postive_len
    total_case = postive_len * negative_len
    pos_histogram = [0 for _ in range(n_bins)]
    neg_histogram = [0 for _ in range(n_bins)]
    bin_width = 1.0 / n_bins
    for i in range(len(labels)):
        nth_bin = int(preds[i] / bin_width)
        if labels[i] == 1:
            pos_histogram[nth_bin] += 1
        else:
            neg_histogram[nth_bin] += 1
    accumulated_neg = 0
    satisfied_pair = 0
    for i in range(n_bins):
        satisfied_pair += (pos_histogram[i] * accumulated_neg +
                           pos_histogram[i] * neg_histogram[i] * 0.5)
        accumulated_neg += neg_histogram[i]

    return satisfied_pair / float(total_case)


def roc_auc_bruteforce(y_true, y_score):
    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]

    total = 0
    for ps in pos_scores:
        for ns in neg_scores:
            if ps > ns:
                total += 1
            elif ps == ns:
                total += 0.5

    return total / (len(pos_scores) * len(neg_scores))


if __name__ == '__main__':
    y = np.array([1, 0, 0, 0, 1, 0, 1, 0, ])
    pred = np.array([0.9, 0.8, 0.3, 0.1, 0.4, 0.9, 0.66, 0.7])

    fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
    print("-----sklearn:", auc(fpr, tpr),
          roc_auc_score(y_true=y, y_score=pred))
    print("-----py脚本:", auc_calculate(y, pred))

    print(roc_auc_bruteforce(y, pred))
