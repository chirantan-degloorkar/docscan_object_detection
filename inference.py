# Imports 
import os
import supervision as sv
import torchvision
import torch
import pytorch_lightning
import transformers
from transformers import DetrForObjectDetection, DetrImageProcessor
import matplotlib.pyplot as plt
import cv2
import random
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image

# Load the model 
CONFIDENCE_TRESHOLD = 0.5
IOU_TRESHOLD = 0.8
CHECKPOINT = 'facebook/detr-resnet-50'
MODEL_PATH = 'models/DETR-run2'

id2label = {
    4: 'north-sign', 
    2: 'color-stamp', 
    1: 'bar-scale', 
    3: 'detail-labels', 
    5: None, 
    6: None, 
    7: None, 
    8: None}

def loadModel(MODEL_PATH):
    model = DetrForObjectDetection.from_pretrained(MODEL_PATH)
    image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)
    
    return model, image_processor

def inference(image_list, model, image_processor, image_folder, results_folder):
    for img in image_list:
        IMAGE_PATH = os.path.join(image_folder, img)
        print(IMAGE_PATH)
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

        # annotate
        detections = sv.Detections.from_transformers(transformers_results=results).with_nms(threshold=IOU_TRESHOLD)

        labels = [f"{id2label[class_id]} {confidence:.2f}" for _, confidence, class_id, _ in detections]

        box_annotator = sv.BoxAnnotator()
        frame = box_annotator.annotate(scene=image, detections=detections, labels=labels)
        Image.fromarray(frame).save(f"{results_folder}/annotated_{img}", format='PNG')

        
# img_folder = 'Images'
# res_folder = 'Temp'
# img_list = os.listdir(img_folder)

# inference(img_list, image_folder=img_folder, results_folder=res_folder)
