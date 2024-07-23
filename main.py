import fitz  
import os
import shutil
import subprocess
from PIL import Image

def pdf_to_image_and_detect(pdf_path, output_image_base):
    output_dir = r"C:\Users\Eshita_Sapariya\Downloads\yolov4\input_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_prediction_dir = r"C:\Users\Eshita_Sapariya\Downloads\yolov4\output_images"
    if not os.path.exists(output_prediction_dir):
        os.makedirs(output_prediction_dir)

    pdf_document = fitz.open(pdf_path)

    try:
        labeled_images = []

        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()

            image_name = f"{output_image_base}_{page_num + 1}.png"
            image_path = os.path.join(output_dir, image_name)
            pix.save(image_path)

            print(f"Page {page_num + 1} saved as {image_path}")

            darknet_command = fr"darknet.exe detector test data\images.data C:\Users\Eshita_Sapariya\Downloads\yolov4\yolov4-custom.cfg C:\Users\Eshita_Sapariya\Downloads\yolov4\backup\yolov4_custom_last_2.weights {image_path} -thresh 0.4"
            
            try:
                subprocess.run(darknet_command, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error executing Darknet command: {e}")
                continue

            # Move predictions.jpg to output_prediction_dir
            predictions_src = os.path.join(r"C:\Users\Eshita_Sapariya\Downloads\yolov4\darknet-master", "predictions.jpg")
            print(predictions_src)
            predictions_dst = os.path.join(output_prediction_dir, f"{output_image_base}_{page_num + 1}_predictions.jpg")

            if os.path.exists(predictions_src):
                shutil.move(predictions_src, predictions_dst)
                labeled_images.append(predictions_dst)
                print(f"Moved {predictions_src} to {predictions_dst}")
            else:
                print(f"Predictions file not found: {predictions_src}")

        pdf_document.close()

        if labeled_images:
            pdf_output_path = os.path.join(output_dir, "labeled_images.pdf")
            convert_images_to_pdf(labeled_images, pdf_output_path)
            print(f"All labeled images saved as {pdf_output_path}")

    except Exception as e:
        print(f"Error processing PDF: {e}")

def convert_images_to_pdf(image_paths, output_pdf_path):
    images = []
    try:
        for image_path in image_paths:
            img = Image.open(image_path)
            images.append(img)

        if images:
            images[0].save(output_pdf_path, save_all=True, append_images=images[1:])
            print(f"PDF created: {output_pdf_path}")
    except Exception as e:
        print(f"Error converting images to PDF: {e}")

# Example usage:
pdf_file = r"C:\Users\Eshita_Sapariya\Downloads\yolov4\Object_detection_5_pages.pdf"
output_image_base = "input_file"
pdf_to_image_and_detect(pdf_file, output_image_base)