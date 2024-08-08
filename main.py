from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import fitz  
import os
import shutil
import subprocess
from PIL import Image
from core.inference import loadModel, inference, multiInference
from core.img2pdf import readPDF, savePDF, zipFiles
from core.generateSummary import summarize
from torch.multiprocessing import Pool


app = FastAPI()
# Directories to store and load images
output_dir = 'Output'
image_dir = 'Output/images'
result_dir = 'Output/results'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.6
# MODEL_PATH = 'models/DETR-run7'
MODEL_PATH = 'models/DETR-run5'
CHECKPOINT_PATH = "models/DETR-run7/ModelCheckpoints2/detr-epoch=47-val_loss=0.53.ckpt"

# Load Model and the Image Processor
model, image_processor = loadModel(MODEL_PATH)

class PDFUpload(BaseModel):
    file: UploadFile
    
@app.post("/upload-pdf/")
async def upload_pdf(pdf_file: UploadFile = File(...)):
    """ Upload the PDF file and process it

    Args:
        pdf_file (UploadFile, optional): PDF file to upload. Defaults to File(...).

    Raises:
        HTTPException: Error uploading or processing PDF
    """
    try:
        #Save uploaded PDF temporarily
        pdf_path = os.path.join(output_dir, pdf_file.filename) # type: ignore
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(pdf_file.file, buffer)
        readPDF(pdf_file=pdf_path, img_dir=image_dir)
        # image_list = os.listdir(image_dir)
        
        image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
        
        results_dict = inference(
            model=model, 
            image_processor=image_processor, 
            image_folder=image_dir, 
            results_folder=result_dir,
            CONFIDENCE_THRESHOLD=CONFIDENCE_THRESHOLD,
            IOU_THRESHOLD=IOU_THRESHOLD
        )
        summarize(results_dict, 'results/summary.csv')
        
        # MultiProcessing
        # num_processes = 4  # Adjust based on your CPU cores
        # # Prepare arguments for each process
        # args_list = [
        #     (model, image_processor, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, image_path, result_dir)
        #     for image_path in image_paths
        # ]
        # with Pool(num_processes) as pool:
        #     results = pool.starmap(multiInference, args_list)
        
        savePDF(image_dir=result_dir, pdf_path='results/output.pdf')
        # zip_path = 'output.zip'
        # try:
        #     zipFiles('results', zip_path)
        # except Exception as e:
        #     print("Error zipping files: ", e)
            
        # if os.path.exists(zip_path):
        #     return FileResponse(zip_path, filename="labeled_images.zip", media_type="application/zip")
        return FileResponse('results/output.pdf', filename="labeled_images.pdf", media_type="application/pdf") 
        # return [FileResponse('results/output.pdf', filename="labeled_images.pdf", media_type="application/pdf"), 
        #         FileResponse('results/summary.csv', filename="summary.csv", media_type="text/csv")]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading or processing PDF: {e}")
 
# @app.get("/download-labeled-pdf/")
# async def download_labeled_pdf():
#     """ Download the labeled PDF
    
#     """
#     try:
#         pdf_path = os.path.join(output_dir, "output.pdf")
#         if os.path.exists(pdf_path):
#             return FileResponse(pdf_path, filename="labeled_images.pdf", media_type="application/pdf")
#         else:
#             raise HTTPException(status_code=404, detail="Labeled PDF not found")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error downloading PDF: {e}")
    
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)