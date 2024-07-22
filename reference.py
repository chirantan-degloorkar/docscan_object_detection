from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import fitz  
import os
import shutil
import subprocess
from PIL import Image
 
app = FastAPI()
 
#Paths for input and output directories
output_dir = r"C:\Users\Eshita_Sapariya\Downloads\Engineering-Drawing\input_images"
output_prediction_dir = r"C:\Users\Eshita_Sapariya\Downloads\Engineering-Drawing\output_images"
 
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_prediction_dir, exist_ok=True)
 
#Function to process PDF, detect objects, and convert to labeled PDF
def pdf_to_image_and_detect(pdf_path, output_image_base):
    try:
        labeled_images = []
        pdf_document = fitz.open(pdf_path)
 
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()
 
            #Save page as image
            image_name = f"{output_image_base}_{page_num + 1}.png"
            image_path = os.path.join(output_dir, image_name)
            pix.save(image_path)
 
            print(f"Page {page_num + 1} saved as {image_path}")
 
           
            darknet_command = fr"darknet.exe detector test C:\Users\Eshita_Sapariya\Downloads\Engineering-Drawing\data\images.data C:\Users\Eshita_Sapariya\Downloads\Engineering-Drawing\cfg\yolov4-custom.cfg C:\Users\Eshita_Sapariya\Downloads\Engineering-Drawing\trained_weights\yolov4-custom_best_2.weights {image_path} -thresh 0.4"
           
            try:
                subprocess.run(darknet_command, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error executing Darknet command: {e}")
                continue
 
            #Move predictions.jpg to output_prediction_dir
            predictions_src = os.path.join(r"C:\Users\Eshita_Sapariya\Downloads\Engineering-Drawing\darknet-master", "predictions.jpg")
            predictions_dst = os.path.join(output_prediction_dir, f"{output_image_base}_{page_num + 1}_predictions.jpg")
 
            if os.path.exists(predictions_src):
                shutil.move(predictions_src, predictions_dst)
                labeled_images.append(predictions_dst)
                print(f"Moved {predictions_src} to {predictions_dst}")
            else:
                print(f"Predictions file not found: {predictions_src}")
 
        pdf_document.close()
 
        #Convert labeled images to PDF
        if labeled_images:
            pdf_output_path = os.path.join(output_dir, "labeled_images.pdf")
            convert_images_to_pdf(labeled_images, pdf_output_path)
            return pdf_output_path  
        else:
            return None
 
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return None
 
#Function to convert images to PDF
def convert_images_to_pdf(image_paths, output_pdf_path):
    images = []
    try:
        for image_path in image_paths:
            img = Image.open(image_path)
            images.append(img)
 
        if images:
            images[0].save(output_pdf_path, save_all=True, append_images=images[1:])
            print(f"PDF created: {output_pdf_path}")
            return True
        else:
            return False
    except Exception as e:
        print(f"Error converting images to PDF: {e}")
        return False
 
#Request body model for PDF upload
class PDFUpload(BaseModel):
    file: UploadFile
 
#Route for uploading PDF and processing
@app.post("/upload-pdf/")
async def upload_pdf(pdf_file: UploadFile = File(...)):
    try:
        #Save uploaded PDF temporarily
        pdf_path = os.path.join(output_dir, pdf_file.filename)
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(pdf_file.file, buffer)
 
        #Process PDF
        labeled_pdf_path = pdf_to_image_and_detect(pdf_path, "input_file")
 
        if labeled_pdf_path:
            return {"message": "Processing completed successfully", "pdf_path": labeled_pdf_path}
        else:
            raise HTTPException(status_code=500, detail="Error processing PDF")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading or processing PDF: {e}")
 
#Route to download labeled PDF
@app.get("/download-labeled-pdf/")
async def download_labeled_pdf():
    try:
        pdf_path = os.path.join(output_dir, "labeled_images.pdf")
        if os.path.exists(pdf_path):
            return FileResponse(pdf_path, filename="labeled_images.pdf", media_type="application/pdf")
        else:
            raise HTTPException(status_code=404, detail="Labeled PDF not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading PDF: {e}")
 
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)