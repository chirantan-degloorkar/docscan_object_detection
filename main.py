import time
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import fitz  
from datetime import datetime
import os
import shutil
import subprocess
from PIL import Image
from core.inference import loadModel, inference, multiInference, inference_test
from core.img2pdf import readPDF, savePDF, zip_dir, delete_directory_contents, readPDF_test
from core.generateSummary import summarize
from torch.multiprocessing import Pool
from pathlib import Path
from core import logger
from fastapi import BackgroundTasks

app = FastAPI()
# Directories to store and load images
output_dir = 'Output'
image_dir = 'Output/images'
result_dir = 'Output/results'
result_dir2 = 'results'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

dir_list = [image_dir, result_dir, result_dir2]

CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.6
# MODEL_PATH = 'models/DETR-run7'
MODEL_PATH = 'models/DETR-resnet34-1'
CHECKPOINT_PATH = "models/DETR-run7/ModelCheckpoints2/detr-epoch=47-val_loss=0.53.ckpt"

# Load Model and the Image Processor
model, image_processor = loadModel(MODEL_PATH) # type: ignore

class PDFUpload(BaseModel):
    file: UploadFile
    

@app.post("/detr/")
async def upload_pdf(background_tasks: BackgroundTasks, pdf_file: UploadFile = File(...)):
    """ Upload the PDF file and process it asynchronously """
    
    logger.log_message(message=f"================================", level=0)
    
    pdf_path = os.path.join(output_dir, pdf_file.filename) # type: ignore
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(pdf_file.file, buffer)
    logger.log_message(message=f"EXEC: {pdf_file.filename} received", level=0)
    
    # Extract images from PDF as a list
    images = readPDF_test(pdf_path)
    logger.log_message(message=f"EXEC: Images Extracted", level=0)
    
    if not images:
        raise HTTPException(status_code=500, detail="Error reading PDF or no images found")
    
    # Add the inference task to the background
    
    path = await process_pdf_images(images, pdf_file.filename)
    # background_tasks.add_task(process_pdf_images, images, pdf_file.filename)
    
    # return {"message": "PDF is being processed"}
    return path

async def process_pdf_images(images, filename):
    """ Function to perform inference on the list of images """
    start = time.time()
    
    results_dict, image_list = await inference_test(
        model=model, 
        image_processor=image_processor, 
        images=images,  # Pass the list of images here
        results_folder=result_dir,
        CONFIDENCE_THRESHOLD=CONFIDENCE_THRESHOLD,
        IOU_THRESHOLD=IOU_THRESHOLD
    )
    
    filename = filename.replace('.pdf', '')
    os.makedirs(f'results/{filename}', exist_ok=True)
    summarize(results_dict, f'results/{filename}/summary_{datetime.now().date()}.csv')
    logger.log_message(message=f"EXEC: Summary Generated", level=0)

    savePDF(image_list=image_list, pdf_path=f'results/{filename}/output_{datetime.now().date()}.pdf')
    logger.log_message(message=f"EXEC: PDF saved", level=0)
    # Create a zip file
    current_dir = Path.cwd() / 'results'
    tozip_dir = current_dir / f"{filename}"
    zip_path = tozip_dir / f'{filename}_results_{datetime.now().date()}.zip'
    zip_dir(tozip_dir, zip_path)
    end = time.time()
    logger.log_message(message=f"EXEC: {filename} Generated in {end - start} seconds", level=0)
    
    # delete_directory_contents(dir_list)
    return tozip_dir

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
    
# Run the FastAPI server using the command:
# uvicorn alt:app --reload / --workers 4