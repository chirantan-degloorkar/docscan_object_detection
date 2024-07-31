import torch
import cv2
import os
from PIL import Image
import supervision as sv
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics import Precision, Recall, F1Score, Accuracy
from collections import defaultdict
from transformers import DetrForObjectDetection, DetrImageProcessor
from torchvision.ops import box_iou
from sklearn.metrics import confusion_matrix, classification_report
 
# Load the model 
CONFIDENCE_TRESHOLD = 0.5
IOU_TRESHOLD = 0.8
CHECKPOINT = 'facebook/detr-resnet-50'
# MODEL_PATH = 'models/DETR-run2'

# id2label = {
#     4: 'north-sign', 
#     2: 'color-stamp', 
#     1: 'bar-scale', 
#     3: 'detail-labels', 
#     5: None, 
#     6: None, 
#     7: None, 
#     8: None}
# id2label = {0: 'bar-scale', 1: 'color-stamp', 2: 'detail-labels', 3: 'north-sign'}


def loadModel(MODEL_PATH):
    model = DetrForObjectDetection.from_pretrained(MODEL_PATH)
    image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)
    
    return model, image_processor


def inference1(image_list, model, image_processor, image_folder, results_folder, ground_truth):
    model.eval()  # Set model to evaluation mode
    
    # Initialize metrics
    map_metric = MeanAveragePrecision()
    precision_metric = Precision(task="multiclass", num_classes=len(id2label), average='macro')
    recall_metric = Recall(task="multiclass", num_classes=len(id2label), average='macro')
    f1_metric = F1Score(task="multiclass", num_classes=len(id2label), average='macro')
    
    total_loss = 0
    all_preds = []
    all_targets = []

    for img in image_list:
        IMAGE_PATH = os.path.join(image_folder, img)
        print(f"Processing {IMAGE_PATH}")
        
        image = cv2.imread(IMAGE_PATH)
        inputs = image_processor(images=image, return_tensors='pt')
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Get ground truth for this image
        target = {k: torch.tensor(v).to(model.device) for k, v in ground_truth[img].items()}
        print(target)
        with torch.no_grad():
            outputs = model(**inputs, labels=target)
            
            # Calculate loss
            loss = outputs.loss
            total_loss += loss.item()

            # Post-process
            target_sizes = torch.tensor([image.shape[:2]]).to(model.device)
            results = image_processor.post_process_object_detection(
                outputs=outputs,
                threshold=CONFIDENCE_TRESHOLD,
                target_sizes=target_sizes
            )[0]

        # Prepare predictions and targets for metric calculation
        pred = {
            'boxes': results['boxes'].cpu(),
            'scores': results['scores'].cpu(),
            'labels': results['labels'].cpu()
        }
        all_preds.append(pred)
        all_targets.append({k: v.cpu() for k, v in target[0].items()})

        # Annotate image
        detections = sv.Detections.from_transformers(transformers_results=results).with_nms(threshold=IOU_TRESHOLD)
        labels = [f"{id2label[class_id]} {confidence:.2f}" for _, confidence, class_id, _ in detections]
        box_annotator = sv.BoxAnnotator()
        frame = box_annotator.annotate(scene=image, detections=detections, labels=labels)
        
        # Save annotated image
        output_path = os.path.join(results_folder, f"annotated_{img}")
        Image.fromarray(frame).save(output_path, format='PNG')

    # Calculate metrics
    map_metric.update(all_preds, all_targets)
    mAP = map_metric.compute()

    # Flatten predictions and targets for other metrics
    pred_classes = torch.cat([p['labels'] for p in all_preds])
    target_classes = torch.cat([t['labels'] for t in all_targets])

    precision = precision_metric(pred_classes, target_classes)
    recall = recall_metric(pred_classes, target_classes)
    f1_score = f1_metric(pred_classes, target_classes)

    # Calculate average loss
    avg_loss = total_loss / len(image_list)

    # Print results
    print(f"mAP: {mAP['map']:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    print(f"Validation Loss: {avg_loss:.4f}")

    return mAP, precision, recall, f1_score, avg_loss


def inference2(image_list, model, image_processor, image_folder, results_folder, ground_truth, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, id2label):
    model.eval()  # Set model to evaluation mode
    
    # Initialize metrics
    map_metric = MeanAveragePrecision()
    precision_metric = Precision(num_classes=len(id2label), average='macro', task='multiclass')
    recall_metric = Recall(num_classes=len(id2label), average='macro', task='multiclass')
    f1_metric = F1Score(num_classes=len(id2label), average='macro', task='multiclass')
    accuracy_metric = Accuracy(num_classes=len(id2label), average='macro', task='multiclass')

    
    total_loss = 0
    all_preds = []
    all_targets = []

    for img in image_list:
        IMAGE_PATH = os.path.join(image_folder, img)
        print(f"Processing {IMAGE_PATH}")
        
        image = cv2.imread(IMAGE_PATH)
        inputs = image_processor(images=image, return_tensors='pt')
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Get ground truth for this image
        target = ground_truth.get(img, {'boxes': torch.empty((0, 4)), 'labels': torch.empty((0,), dtype=torch.long)})
        target = {k: torch.tensor(v).to(model.device) for k, v in target.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Post-process
            target_sizes = torch.tensor([image.shape[:2]]).to(model.device)
            results = image_processor.post_process_object_detection(
                outputs=outputs,
                threshold=CONFIDENCE_THRESHOLD,
                target_sizes=target_sizes
            )[0]

        # Prepare predictions and targets for metric calculation
        pred = {
            'boxes': results['boxes'].cpu(),
            'scores': results['scores'].cpu(),
            'labels': results['labels'].cpu()
        }
        all_preds.append(pred)
        all_targets.append({k: v.cpu() for k, v in target.items()})

        # Annotate image
        detections = sv.Detections.from_transformers(transformers_results=results).with_nms(threshold=IOU_THRESHOLD)
        labels = [f"{id2label[class_id]} {confidence:.2f}" for _, confidence, class_id, _ in detections]

        # labels = [f"{id2label[class_id]} {confidence:.2f}" for class_id, confidence in zip(detections.class_ids, detections.scores)]
        box_annotator = sv.BoxAnnotator()
        frame = box_annotator.annotate(scene=image, detections=detections, labels=labels)
        
        # Save annotated image
        output_path = os.path.join(results_folder, f"annotated_{img}")
        Image.fromarray(frame).save(output_path, format='PNG')

    # Calculate metrics
    map_metric.update(all_preds, all_targets)
    mAP = map_metric.compute()

    # Flatten predictions and targets for other metrics
    pred_classes = torch.cat([p['labels'] for p in all_preds])
    target_classes = torch.cat([t['labels'] for t in all_targets])
    
    if pred_classes.shape != target_classes.shape:
        print(f"Shape mismatch: preds shape {pred_classes.shape}, targets shape {target_classes.shape}")
        common_length = min(pred_classes.shape[0], target_classes.shape[0])
        pred_classes = pred_classes[:common_length]
        target_classes = target_classes[:common_length]

    # print(pred_classes)
    # print(target_classes)
    # print(len(pred_classes))
    # print(len(target_classes))
    precision = precision_metric(pred_classes, target_classes)
    recall = recall_metric(pred_classes, target_classes)
    f1_score = f1_metric(pred_classes, target_classes)
    accuracy = accuracy_metric(pred_classes, target_classes)

    # Calculate average loss
    avg_loss = total_loss / len(image_list)

    # Print results
    print(f"mAP: {mAP['map']:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Validation Loss: {avg_loss:.4f}")

    return mAP, precision, recall, f1_score, accuracy, avg_loss
# metrics = inference(image_list, model, image_processor, image_folder, results_folder, ground_truth)

def inference3(image_list, model, image_processor, image_folder, results_folder):
    detection_stats = defaultdict(int)
    confidence_stats = defaultdict(list)

    for img in image_list:
        IMAGE_PATH = os.path.join(image_folder, img)
        print(f"Processing {IMAGE_PATH}")
        
        with torch.no_grad():
            # load image and predict
            image = cv2.imread(IMAGE_PATH)
            inputs = image_processor(images=image, return_tensors='pt')
            outputs = model(**inputs)

            # post-process
            target_sizes = torch.tensor([image.shape[:2]])
            results = image_processor.post_process_object_detection(
                outputs=outputs,
                threshold=CONFIDENCE_TRESHOLD,
                target_sizes=target_sizes
            )[0]

        # Collect statistics
        for label, score in zip(results['labels'], results['scores']):
            class_name = id2label[label.item()]
            detection_stats[class_name] += 1
            confidence_stats[class_name].append(score.item())

        # annotate
        detections = sv.Detections.from_transformers(transformers_results=results).with_nms(threshold=IOU_TRESHOLD)
        labels = [f"{id2label[class_id]} {confidence:.2f}" for _, confidence, class_id, _ in detections]
        box_annotator = sv.BoxAnnotator()
        frame = box_annotator.annotate(scene=image, detections=detections, labels=labels)
        
        # Save annotated image
        output_path = os.path.join(results_folder, f"annotated_{img}")
        Image.fromarray(frame).save(output_path, format='PNG')

    # Print statistics
    print("\nDetection Statistics:")
    for class_name, count in detection_stats.items():
        avg_confidence = sum(confidence_stats[class_name]) / count if count > 0 else 0
        print(f"{class_name}: {count} detections, Average confidence: {avg_confidence:.2f}")

    total_detections = sum(detection_stats.values())
    print(f"\nTotal detections across all images: {total_detections}")
    print(f"Average detections per image: {total_detections / len(image_list):.2f}")

    return detection_stats, confidence_stats



def inference4(image_list, model, image_processor, image_folder, results_folder, ground_truth, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, id2label):
    # model.eval()  # Set model to evaluation mode
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_pred_classes = 0
    total_actual_classes = 0

    for img in image_list:
        IMAGE_PATH = os.path.join(image_folder, img)
        print(f"Processing {IMAGE_PATH}")
        
        image = cv2.imread(IMAGE_PATH)
        inputs = image_processor(images=image, return_tensors='pt')
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Get ground truth for this image
        target = ground_truth.get(img, {'boxes': torch.empty((0, 4)), 'labels': torch.empty((0,), dtype=torch.long)})
        # target = {k: torch.tensor(v).to(model.device) for k, v in target.items()}
        target = {k: v.clone().detach().to(model.device) for k, v in target.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Post-process
            target_sizes = torch.tensor([image.shape[:2]]).to(model.device)
            results = image_processor.post_process_object_detection(
                outputs=outputs,
                threshold=CONFIDENCE_THRESHOLD,
                target_sizes=target_sizes
            )[0]

        pred_boxes = results['boxes'].cpu()
        pred_scores = results['scores'].cpu()
        pred_labels = results['labels'].cpu()

        true_boxes = target['boxes'].cpu()
        true_labels = target['labels'].cpu()

        iou_matrix = box_iou(pred_boxes, true_boxes)
        matches = iou_matrix > IOU_THRESHOLD
        # print(matches)

        tp = 0
        fp = 0
        fn = 0

        matched_true_boxes = set()

        for pred_idx in range(len(pred_boxes)):
            matched = matches[pred_idx]
            if matched.any():
                max_iou, max_iou_idx = iou_matrix[pred_idx].max(dim=0)
                if pred_labels[pred_idx] == true_labels[max_iou_idx] and max_iou > IOU_THRESHOLD:
                    tp += 1
                    matched_true_boxes.add(max_iou_idx.item())
                else:
                    fp += 1
            else:
                fp += 1

        fn = len(true_boxes) - len(matched_true_boxes)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_pred_classes += len(pred_labels)
        total_actual_classes += len(true_labels)
        
        # Annotate image
        detections = sv.Detections.from_transformers(transformers_results=results).with_nms(threshold=IOU_THRESHOLD)
        box_annotator = sv.BoxAnnotator()
        labels = [f"{id2label[class_id]} {confidence:.2f}" for _, confidence, class_id, _ in detections]
        frame = box_annotator.annotate(scene=image, detections=detections, labels=labels)
        
        # Save annotated image
        output_path = os.path.join(results_folder, f"annotated_{img}")
        Image.fromarray(frame).save(output_path, format='PNG')

    # Print results
    print(f"True Positives (TP): {total_tp}")
    print(f"False Positives (FP): {total_fp}")
    print(f"False Negatives (FN): {total_fn}")
    print(f"Total Predicted Classes: {total_pred_classes}")
    print(f"Total Actual Classes: {total_actual_classes}")
    # print(f"Classification Report: {classification_report(true_labels, pred_labels)}")

    return total_tp, total_fp, total_fn