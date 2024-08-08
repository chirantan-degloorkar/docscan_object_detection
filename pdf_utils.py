import fitz
import os
import shutil
import subprocess
from image_utils import annotate_image, convert_images_to_pdf
from darknet_utils import process_darknet_output

def pdf_to_image_and_detect(pdf_path, output_image_base):
    '''
    Input - Path of pdf, input base
    Output - Labelled PDF
    Function - Converts input PDF to images, predicts the output, annotates it and converts it to output PDF
    '''
    try:
        labeled_images = []
        pdf_document = fitz.open(pdf_path)

        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()

            image_name = f"{output_image_base}_{page_num + 1}.png"
            image_path = os.path.join("input_images/", image_name)
            pix.save(image_path)

            print(f"Page {page_num + 1} saved as {image_path}")

            darknet_command = fr"darknet.exe detector test data\images.data yolov4-custom.cfg weights\yolov4-custom_4000.weights {image_path} -thresh 0.4 -ext_output -dont_show -out output.json"
            
            try:
                subprocess.run(darknet_command, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error executing Darknet command: {e}")
                continue

            missing_objects = process_darknet_output("output.json")

            predicted_image_src = "predictions.jpg"
            predicted_image_dst = os.path.join("output_images/", f"{output_image_base}_{page_num + 1}_predictions.jpg")

            if os.path.exists(predicted_image_src):
                annotate_image(predicted_image_src, missing_objects)
                shutil.move(predicted_image_src, predicted_image_dst)
                labeled_images.append(predicted_image_dst)
                print(f"Moved and annotated {predicted_image_src} to {predicted_image_dst}")
            else:
                print(f"Predicted image not found: {predicted_image_src}")

        pdf_document.close()

        if labeled_images:
            pdf_output_path = os.path.join("input_images/", "labeled_images.pdf")
            convert_images_to_pdf(labeled_images, pdf_output_path)
            return pdf_output_path  
        else:
            return None

    except Exception as e:
        print(f"Error processing PDF: {e}")
        return None
