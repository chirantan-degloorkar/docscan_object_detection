from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pdf_utils import pdf_to_image_and_detect
from pydantic import BaseModel
import os
import shutil

app = FastAPI()

#Path for input and output directories
output_dir = "input_images/"
output_prediction_dir = "output_images"

#Ensure directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_prediction_dir, exist_ok=True)

class PDFUpload(BaseModel):
    file: UploadFile

@app.post("/upload-pdf/")
async def upload_pdf(pdf_file: UploadFile = File(...)):
    try:
        pdf_path = os.path.join(output_dir, pdf_file.filename)
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(pdf_file.file, buffer)

        labeled_pdf_path = pdf_to_image_and_detect(pdf_path, "input_file")

        if labeled_pdf_path:
            return {"message": "Processing completed successfully", "pdf_path": labeled_pdf_path}
        else:
            raise HTTPException(status_code=500, detail="Error processing PDF")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading or processing PDF: {e}")

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





