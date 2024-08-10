# from PyMuPDF import fitz
import fitz
from PIL import Image
import os
import zipfile
import shutil
import zipfile
from pathlib import Path

from core import logger

pdf_file = "test.pdf"
def readPDF(pdf_file, img_dir):
    """Reads a PDF file and saves each page as an image in the specified directory

    Args:
        pdf_file (_type_): pdf file to read
        img_dir (_type_): directory to save the images
    """
    try:
        doc = fitz.open(pdf_file)
        page_count = doc.page_count
        for i in range (page_count):
            page = doc.load_page(i)
            pix = page.get_pixmap() # type: ignore
            output = f"outfile{i+1}.png"
            pix.save(img_dir+'/'+output)
    except Exception as e:
        logger.log_message(message=f"ERROR: Read PDF - {e}", level=1)

        print(f"Error reading PDF: {e}")

def savePDF(image_dir, pdf_path):
    """Saves images in the image directory as a PDF file

    Args:
        image_dir (_type_): directory containing the images
        pdf_path (_type_): path to save the PDF file
    """
    try:
        imagelist = []
        im = None
        for path in os.listdir(image_dir):
            im = Image.open(os.path.join(image_dir, path)).convert('RGB')
            imagelist.append(im)
            
        im.save(pdf_path,save_all=True, append_images=imagelist) # type: ignore
    except Exception as e:
        logger.log_message(message=f"ERROR: Save PDF - {e}", level=1)

        print(f"Error saving PDF: {e}")
                

def zip_dir(path: Path, zip_file_path: Path):
    """ Zip the contents of a directory

    Args:
        path (Path): directory to zip
        zip_file_path (Path): path to save the zipped file
    """
    try:
        files_to_zip = [
            file for file in path.glob('*') if file.is_file()]
        with zipfile.ZipFile(
            zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zip_f:
            for file in files_to_zip:
                print(f"Processed - {file.name}")
                zip_f.write(file, file.name)
            print("Zip File Created")
    except Exception as e:
        logger.log_message(message=f"ERROR: Zip File - {e}", level=1)

        print(f"Error zipping directory: {e}")
