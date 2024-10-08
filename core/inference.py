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
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
from collections import OrderedDict

from core import logger

# Load the model 
CONFIDENCE_TRESHOLD = 0.5
IOU_TRESHOLD = 0.8
CHECKPOINT = 'facebook/detr-resnet-50'
# MODEL_PATH = 'models/DETR-run2'

id2label = {0: 'bar-scale', 1: 'color-stamp', 2: 'detail-labels', 3: 'north-sign'}


async def inference_test(model, image_processor, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, images, results_folder):
    """ Predict the images in the provided list of images

    Args:
        model: The model to use for prediction
        image_processor: Image processor
        CONFIDENCE_THRESHOLD: Confidence threshold above which to consider a detection
        IOU_THRESHOLD: IOU threshold above which to consider two boxes as the same
        images (list): List of PIL.Image objects
        results_folder (str): Folder to save the results

    Returns:
        dict: Dictionary containing the results
    """
    results_dict = {}
    count = 0
    try:
        image_list = []
        for idx, img in enumerate(images):
            print(f"Processing image {idx + 1}")

            # Convert the PIL image to a format suitable for your model
            image_np = np.array(img)  # Convert to numpy array (assuming RGB format)
            image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR if necessary
            inputs = image_processor(images=image, return_tensors='pt')

            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                
                # Post-process
                target_sizes = torch.tensor([image.shape[:2]]).to(model.device)
                results = image_processor.post_process_object_detection(
                    outputs=outputs,
                    threshold=CONFIDENCE_THRESHOLD,
                    target_sizes=target_sizes
                )[0]
            
            detections = sv.Detections.from_transformers(transformers_results=results)
            labels = [f"{id2label[class_id]}" for _, confidence, class_id, _ in detections]
            
            print(set(detections.class_id)) 
            box_annotator = sv.BoxAnnotator()
            frame = box_annotator.annotate(scene=image, detections=detections, labels=labels)
            
            annotated_img = Image.fromarray(frame)
            image_path = f"{results_folder}/annotated_image_{idx + 1}.png"
            
            all_labels = {0, 1, 2, 3}
            missing_labels = all_labels - set(detections.class_id)
            image = add_missing_label(annotated_img, image_path, missing_labels)
            image_list.append(image)
            results_dict[f"image_{idx + 1}"] = results
            count += 1
            
        return results_dict, image_list
    except Exception as e:
        logger.log_message(message=f"ERROR: Inference - {e}", level=1)
        print(f"Error processing images: {e}")
        return None, None


def loadModel(MODEL_PATH, CHECKPOINT_PATH=None):
    """Load the model and image processor

    Args:
        MODEL_PATH (_type_): Path to the model.
        CHECKPOINT_PATH (_type_, optional): Path to the model checkpoint. Defaults to None.

    Returns:
        _type_: Model and Image Processor
    """
    try:
        model = DetrForObjectDetection.from_pretrained(MODEL_PATH)
        image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)
        
        # Load Checkpoint in Model
        if CHECKPOINT_PATH is not None:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
            state_dict = checkpoint['state_dict']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace("model.model.", "")
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=False)# type: ignore
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device) # type: ignore
        return model, image_processor
    except Exception as e:
        logger.log_message(message=f"ERROR: Load Model - {e}", level=1)
        print(f"Error loading model: {e}")

def add_missing_label(image, save_path, label):
    """Add missing label to the image

    Args:
        image (PIL.Image): Image to add label to
        save_path (str): Path where the image will be saved
        label (set): Labels to add to the image
    """
    try:
        id2label = {0: 'bar-scale', 1: 'color-stamp', 2: 'detail-labels', 3: 'north-sign'}
        draw = ImageDraw.Draw(image)
        
        # Load a TrueType or OpenType font file with a specific size
        font_size = 60  # Adjust this size as needed
        font = ImageFont.truetype("arial.ttf", font_size)  # You can use a different font file if needed
        
        if label:
            text = ""
            for i in label: 
                text += f"{id2label[i]} not detected\n"
        else:
            text = ""

        position = (10, 10)  # Position to start the text
        draw.text(position, text, fill="red", font=font)  # Draw the text with the specified font and size

        image.save(save_path)  # Save the image after adding the label
        return image
    except Exception as e:
        logger.log_message(message=f"ERROR: Add Missing Label - {e}", level=1)
        print(f"Error adding missing label: {e}")
        return None

def inference(model, image_processor, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, image_folder, results_folder):
    """ Predict the images in the image folder

    Args:
        model (_type_): model to use for prediction
        image_processor (_type_): image processor
        CONFIDENCE_THRESHOLD (_type_): confidence threshold above which to consider a detection
        IOU_THRESHOLD (_type_): IOU threshold above which to consider two boxes as the same
        image_folder (_type_): folder containing the images
        results_folder (_type_): folder to save the results

    Returns:
        _type_: results_dict (Dictionary containing the results)
    """
    results_dict = {}
    count = 0
    try:
        for img in os.listdir(image_folder):
            IMAGE_PATH = os.path.join(image_folder, img)
            print(f"Processing {IMAGE_PATH}")

            image = cv2.imread(IMAGE_PATH)
            # image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            inputs = image_processor(images=image, return_tensors='pt')

            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                
                # Post-process
                target_sizes = torch.tensor([image.shape[:2]]).to(model.device)
                results = image_processor.post_process_object_detection(
                    outputs=outputs,
                    threshold=CONFIDENCE_THRESHOLD,
                    target_sizes=target_sizes
                )[0]
            
            detections = sv.Detections.from_transformers(transformers_results=results)
            labels = [f"{id2label[class_id]}" for _, confidence, class_id, _ in detections]
            
            print(set(detections.class_id)) 
            box_annotator = sv.BoxAnnotator()
            frame = box_annotator.annotate(scene=image, detections=detections, labels=labels)
            
            image = Image.fromarray(frame)
            image_path = f"{results_folder}/annotated_{img}"
            
            all_labels = {0, 1, 2, 3}
            label = all_labels - set(detections.class_id)
            add_missing_label(image, image_path, label)
            
            results_dict[IMAGE_PATH.replace('Temp/', '')] = results
            # results_dict[count] = results
            count += 1
            
        return results_dict
    except Exception as e:
        logger.log_message(message=f"ERROR: Inference - {e}", level=1)
        print(f"Error processing images: {e}")

def multiInference(model, image_processor, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, image, results_folder):
    """same as inference but for multiprocessing

    Args:
        model (_type_): model to use for prediction
        image_processor (_type_): image processor
        CONFIDENCE_THRESHOLD (_type_): confidence threshold above which to consider a detection
        IOU_THRESHOLD (_type_): IOU threshold above which to consider two boxes as the same
        image_folder (_type_): folder containing the images
        results_folder (_type_): folder to save the results

    Returns:
        _type_: results_dict (Dictionary containing the results)
    """
    try:
            
        results_dict = {}
        
        # for img in os.listdir(image_folder):
        IMAGE_PATH = os.path.join(image)
        print(f"Processing {IMAGE_PATH}")

        image = cv2.imread(IMAGE_PATH)
        inputs = image_processor(images=image, return_tensors='pt')

        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Post-process
            target_sizes = torch.tensor([image.shape[:2]]).to(model.device)
            results = image_processor.post_process_object_detection(
                outputs=outputs,
                threshold=CONFIDENCE_THRESHOLD,
                target_sizes=target_sizes
            )[0]
        
        
        detections = sv.Detections.from_transformers(transformers_results=results)
        labels = [f"{id2label[class_id]} {confidence:.2f}" for _, confidence, class_id, _ in detections]
        print(set(detections.class_id)) 
        
        box_annotator = sv.BoxAnnotator()
        frame = box_annotator.annotate(scene=image, detections=detections, labels=labels)
        
        image = Image.fromarray(frame)
        image_path = f"{results_folder}/annotated_{image}"
        
        all_labels = {0, 1, 2, 3}
        label = all_labels - set(detections.class_id)
        add_missing_label(image, image_path, label)
        
        results_dict[IMAGE_PATH.replace('Temp/', '')] = results
        
        return results_dict
    except Exception as e:
        logger.log_message(message=f"ERROR: MultiInference - {e}", level=1)
        print(f"Error processing images: {e}")