import pprint

import numpy as np
from torch import tensor


def compute_iou(box1, box2):
    """计算 IoU"""
    x1, y1, x2, y2 = box1
    x1_gt, y1_gt, x2_gt, y2_gt = box2
    xi1 = max(x1, x1_gt)
    yi1 = max(y1, y1_gt)
    xi2 = min(x2, x2_gt)
    yi2 = min(y2, y2_gt)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def compute_ap(precision, recall):
    """全点插值法计算 AP"""
    precision = np.concatenate(([0.0], precision, [0.0]))
    recall = np.concatenate(([0.0], recall, [1.0]))
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    ap = 0.0
    for i in range(1, len(recall)):
        ap += (recall[i] - recall[i - 1]) * precision[i]
    return ap


def compute_map_per_image(predictions, ground_truths, iou_threshold=0.75):
    """计算单张图片的 mAP"""
    classes = set([p[1] for p in predictions] + [gt[1] for gt in ground_truths])
    ap_scores = {}

    for cls in classes:
        cls_preds = sorted([p for p in predictions if p[1] == cls], key=lambda x: x[2], reverse=True)
        cls_gts = [gt for gt in ground_truths if gt[1] == cls]

        if not cls_gts:
            ap_scores[cls] = 0.0
            continue

        n_gt = len(cls_gts)
        tp = np.zeros(len(cls_preds))
        fp = np.zeros(len(cls_preds))
        used_gt = set()

        for i, (pred_bbox, _, score) in enumerate(cls_preds):
            max_iou = 0
            max_idx = -1
            for j, (gt_bbox, _) in enumerate(cls_gts):
                if j in used_gt:
                    continue
                iou = compute_iou(pred_bbox, gt_bbox)
                if iou > max_iou:
                    max_iou = iou
                    max_idx = j
            if max_iou >= iou_threshold:
                tp[i] = 1
                used_gt.add(max_idx)
            else:
                fp[i] = 1

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        recall = tp_cumsum / n_gt
        ap_scores[cls] = compute_ap(precision, recall)

    return ap_scores


# 数据

images = {
    "image1": {
        "predictions": [
            [[55, 55, 145, 145], 1, 0.9],
            [[210, 210, 290, 290], 2, 0.8],
            [[390, 390, 490, 490], 3, 0.6],
        ],
        "ground_truths": [
            [[50, 50, 150, 150], 1],
            [[200, 200, 300, 300], 2],
            [[400, 400, 500, 500], 3],
        ]
    },
    "image2": {
        "predictions": [],
        "ground_truths": []
    }
}
#
# 计算每张图片的 mAP
all_ap_scores = {}
for img_id, data in images.items():
    ap_scores = compute_map_per_image(data["predictions"], data["ground_truths"])
    all_ap_scores[img_id] = ap_scores
    print(f"{img_id} AP scores: {ap_scores}")

# 计算全局 mAP（按类别平均，再按图片平均）
class_ap = {}
for img_id, ap_scores in all_ap_scores.items():
    for cls, ap in ap_scores.items():
        if cls not in class_ap:
            class_ap[cls] = []
        class_ap[cls].append(ap)

mean_ap_per_class = {cls: np.mean(aps) for cls, aps in class_ap.items()}
global_map = np.mean(list(mean_ap_per_class.values()))

print(f"\n每类别平均 AP: {mean_ap_per_class}")
print(f"全局 mAP: {global_map:.4f}")
# ###############################################################################################

from torchmetrics.detection.mean_ap import MeanAveragePrecision

metric = MeanAveragePrecision(iou_type="bbox")
predictions = images['image1']['predictions'] + images['image2']['predictions']
ground_truths = images['image1']['ground_truths'] + images['image2']['ground_truths']
print()
preds = [
    dict(
        boxes=tensor([_[0] for _ in predictions]),
        scores=tensor([_[2] for _ in predictions]),
        labels=tensor([_[1] for _ in predictions]),
    )
]

target = [
    dict(
        boxes=tensor([_[0] for _ in ground_truths]),
        labels=tensor([_[1] for _ in ground_truths]),
    )
]

metric.update(preds, target)
pprint.pprint(metric.compute())
