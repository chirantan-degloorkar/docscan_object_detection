import os 
import supervision as sv
from transformers import DetrForObjectDetection, DetrImageProcessor
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from inference2 import loadModel, inference4, inference2, inference3
from img2pdf import readPDF, savePDF
from createGroundTruth import create_ground_truth_dict
import time

from torchvision.ops import box_iou

# id2label = {
#     4: 'north-sign', 
#     2: 'color-stamp', 
#     1: 'bar-scale', 
#     3: 'detail-labels', 
#     }

# For DETR-run5
id2label = {0: 'bar-scale', 1: 'color-stamp', 2: 'detail-labels', 3: 'north-sign'}
MODEL_PATH = 'models/DETR-run7'

image_folder = 'Temp/images'
results_folder = 'Temp/results'

model, image_processor = loadModel(MODEL_PATH)
# pdf_path = 'Temp/test.pdf'
# readPDF(pdf_file=pdf_path, img_dir='Temp/images')

image_list = os.listdir('Temp/images')

ground_truth = create_ground_truth_dict('dataset2/test/result.json')
# detection_stats, confidence_stats = inference(image_list=image_list, model=model, image_processor=image_processor, image_folder='Temp/images', results_folder='Temp/results')
# inference(image_list=image_list, model=model, image_processor=image_processor, image_folder='Temp/images', results_folder='Temp/results')

# metrics = inference(image_list=image_list, model=model, image_processor=image_processor, image_folder='Temp/images', results_folder='Temp/results', ground_truth=ground_truth)
begin = time.time() 

# mAP, precision, recall, f1_score, accuracy, avg_loss = inference2(
#     image_list, model, image_processor, image_folder='Temp/images', results_folder='Temp/results', ground_truth=ground_truth, CONFIDENCE_THRESHOLD=0.5, IOU_THRESHOLD=0.7, id2label=id2label
# )
tp, fp, fn = inference4(image_list, model, image_processor, image_folder, results_folder, ground_truth, 0.8, 0.5, id2label)

print("Accuracy = ", tp/(tp+fp+fn))
print("Precision = ", tp/(tp+fp))
print("Recall = ", tp/(tp+fn))
# print("F1 Score = ", 2*tp(2*tp+fp+fn))
print("mAP = ", tp/(tp+fp+fn))
end = time.time() 
 
# total time taken 
print(f"Total runtime - {end - begin}") 
# detection_stats, confidence_stats = inference3(image_list=image_list, model=model, image_processor=image_processor, image_folder='Temp/images', results_folder='Temp/results')


savePDF(image_dir='Temp/results', pdf_path='Temp/output.pdf')