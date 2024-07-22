import os 
import supervision as sv
from transformers import DetrForObjectDetection, DetrImageProcessor
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from inference import loadModel, inference
from img2pdf import readPDF, savePDF

MODEL_PATH = 'models/DETR-run2'
model, image_processor = loadModel(MODEL_PATH)

pdf_path = 'Temp/test.pdf'
readPDF(pdf_file=pdf_path, img_dir='Temp/images')

image_list = os.listdir('Temp/images')

inference(image_list=image_list, model=model, image_processor=image_processor, image_folder='Temp/images', results_folder='Temp/results')

savePDF(image_dir='Temp/results', pdf_path='Temp/output.pdf')