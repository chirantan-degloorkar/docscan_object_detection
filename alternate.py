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
from core.inference import loadModel, inference, multiInference
from core.img2pdf import readPDF, savePDF, zip_dir, delete_directory_contents
from core.generateSummary import summarize
from torch.multiprocessing import Pool
from pathlib import Path
from core import logger

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
async def upload_pdf(pdf_file: UploadFile = File(...)):
    """ Upload the PDF file and process it

    Args:
        pdf_file (UploadFile, optional): PDF file to upload. Defaults to File(...).

    Raises:
        HTTPException: Error uploading or processing PDF
    """
    try:
        delete_directory_contents(dir_list)
        start = time.time()
        #Save uploaded PDF temporarily
        pdf_path = os.path.join(output_dir, pdf_file.filename) # type: ignore
        
        print(pdf_file)
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(pdf_file.file, buffer)
        readPDF(pdf_file=pdf_path, img_dir=image_dir)
        logger.log_message(message=f"EXEC: Extracting Images from {pdf_file.filename}", level=0)
        # image_list = os.listdir(image_dir)
        # image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
        
        results_dict = inference(
            model=model, 
            image_processor=image_processor, 
            # image_list=image_list,
            image_folder=image_dir, 
            results_folder=result_dir,
            CONFIDENCE_THRESHOLD=CONFIDENCE_THRESHOLD,
            IOU_THRESHOLD=IOU_THRESHOLD
        )
        
        logger.log_message(message=f"EXEC: Images Annotated", level=0)
        summarize(results_dict, f'results/summary_{datetime.now().date()}.csv')
        logger.log_message(message=f"EXEC: Summary Generated", level=0)
        
        # MultiProcessing
        # num_processes = 4  # Adjust based on your CPU cores
        # # Prepare arguments for each process
        # args_list = [
        #     (model, image_processor, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, image_path, result_dir)
        #     for image_path in image_paths
        # ]
        # with Pool(num_processes) as pool:
        #     results = pool.starmap(multiInference, args_list)
        
        savePDF(image_dir=result_dir, pdf_path=f'results/output_{datetime.now().date()}.pdf')
        
        ## Create a zip file
        current_dir = Path.cwd()  
        tozip_dir = current_dir / "results"
        zip_path = current_dir / f'{pdf_file.filename}_results_{datetime.now().date()}.zip'
        zip_dir(tozip_dir, zip_path)
        end = time.time()
        
        
        print(f"Time taken: {end-start} seconds")
        logger.log_message(message=f"EXEC: results.zip Generated in {end-start} seconds", level=0)
        
        delete_directory_contents(dir_list)
        
        return zip_path
        
    except Exception as e:
        logger.log_message(message=f"ERROR: {e}", level=1)
        raise HTTPException(status_code=500, detail=f"Error uploading or processing PDF: {e}")
 

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, workers=4)