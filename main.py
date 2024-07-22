from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import fitz  
import os
import shutil
import subprocess
from PIL import Image
from inference import loadModel, inference
from img2pdf import readPDF, savePDF

app = FastAPI()
MODEL_PATH = 'models/DETR-run2'
model, image_processor = loadModel(MODEL_PATH)
output_dir = 'Output'
image_dir = 'Output/images'
result_dir = 'Output/results'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

class PDFUpload(BaseModel):
    file: UploadFile
    
@app.post("/upload-pdf/")
async def upload_pdf(pdf_file: UploadFile = File(...)):
    try:
        #Save uploaded PDF temporarily
        pdf_path = os.path.join(output_dir, pdf_file.filename)
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(pdf_file.file, buffer)
        readPDF(pdf_file=pdf_path, img_dir=image_dir)
        image_list = os.listdir(image_dir)
        inference(image_list=image_list, model=model, image_processor=image_processor, image_folder=image_dir, results_folder=result_dir)
        savePDF(image_dir=result_dir, pdf_path='Output/output.pdf')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading or processing PDF: {e}")
 
@app.get("/download-labeled-pdf/")
async def download_labeled_pdf():
    try:
        pdf_path = os.path.join(output_dir, "output.pdf")
        if os.path.exists(pdf_path):
            return FileResponse(pdf_path, filename="labeled_images.pdf", media_type="application/pdf")
        else:
            raise HTTPException(status_code=404, detail="Labeled PDF not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading PDF: {e}")
    
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)