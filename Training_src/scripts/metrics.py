import tensorflow as tf
import numpy as np

def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) between two sets of boxes.
    Args:
    - box1: shape (N, 4) [x1, y1, x2, y2] (for N boxes)
    - box2: shape (M, 4) [x1, y1, x2, y2] (for M boxes)
    Returns:
    - iou: shape (N, M) containing IoU values for each pair of boxes
    """
    x1 = np.maximum(box1[:, 0], box2[:, 0])
    y1 = np.maximum(box1[:, 1], box2[:, 1])
    x2 = np.minimum(box1[:, 2], box2[:, 2])
    y2 = np.minimum(box1[:, 3], box2[:, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = box1_area[:, None] + box2_area[None, :] - intersection

    iou = intersection / (union + 1e-7)
    return iou


def average_precision(recalls, precisions):
    """
    Compute the Average Precision (AP) from recall and precision values.
    Args:
    - recalls: array of recall values
    - precisions: array of precision values
    Returns:
    - ap: the average precision
    """
    recalls = np.concatenate(([0.], recalls, [1.]))
    precisions = np.concatenate(([0.], precisions, [0.]))
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return ap


def calculate_map(predictions, ground_truths, iou_threshold=0.5, num_classes=20):
    """
    Calculate the Mean Average Precision (mAP) for all classes.
    Args:
    - predictions: List of predicted bounding boxes, each element is [image_id, class_id, confidence, x1, y1, x2, y2]
    - ground_truths: List of ground-truth boxes, each element is [image_id, class_id, x1, y1, x2, y2]
    - iou_threshold: The IoU threshold to count as a match (default 0.5)
    - num_classes: Total number of classes (default 20 for Pascal VOC)
    Returns:
    - mAP: Mean Average Precision for all classes
    """
    average_precisions = []

    for c in range(num_classes):
        detections = [d for d in predictions if d[1] == c]
        ground_truth_class = [g for g in ground_truths if g[1] == c]
        
        image_ids = set([g[0] for g in ground_truth_class])
        gt_boxes = {image_id: [] for image_id in image_ids}
        for gt in ground_truth_class:
            gt_boxes[gt[0]].append(gt[2:])

        image_wise_detections = {image_id: [] for image_id in image_ids}
        for det in detections:
            image_wise_detections[det[0]].append(det)
        
        true_positives = []
        false_positives = []
        scores = []
        total_gt = sum([len(boxes) for boxes in gt_boxes.values()])

        for det in detections:
            scores.append(det[2])
            image_id, x1, y1, x2, y2 = det[0], det[3], det[4], det[5], det[6]
            if len(gt_boxes[image_id]) == 0:
                false_positives.append(1)
                true_positives.append(0)
                continue
            
            ious = compute_iou(np.array([[x1, y1, x2, y2]]), np.array(gt_boxes[image_id]))
            max_iou = np.max(ious)
            max_index = np.argmax(ious)

            if max_iou > iou_threshold:
                gt_boxes[image_id].pop(max_index)
                true_positives.append(1)
                false_positives.append(0)
            else:
                true_positives.append(0)
                false_positives.append(1)

        true_positives = np.cumsum(true_positives)
        false_positives = np.cumsum(false_positives)
        recalls = true_positives / (total_gt + 1e-7)
        precisions = true_positives / (true_positives + false_positives + 1e-7)

        ap = average_precision(recalls, precisions)
        average_precisions.append(ap)

    mAP = np.mean(average_precisions)
    return mAP
