# from PyMuPDF import fitz
import fitz
from PIL import Image
import os
import zipfile
import shutil
import zipfile
from pathlib import Path
import numpy as np

from core import logger

pdf_file = "test.pdf"
def readPDF(pdf_file, img_dir):
    """Reads a PDF file and saves each page as an image in the specified directory

    Args:
        pdf_file (_type_): pdf file to read
        img_dir (_type_): directory to save the images
    """
    img_list = []
    try:
        doc = fitz.open(pdf_file)
        page_count = doc.page_count
        for i in range (page_count):
            page = doc.load_page(i)
            mat = fitz.Matrix(5.0,5.0)
            pix = page.get_pixmap(matrix=mat) # type: ignore
            img = Image.frombytes("RGB",[pix.width,pix.height],pix.samples) # type: ignore
            # img_array = np.array(img)
            # pix = page.get_pixmap() # type: ignore
            output = f"outfile{i+1}.png"
            img.save(img_dir+'/'+output)
            # img_list.append(img_array)
            # pix.save(img_dir+'/'+output)
        # return img_list
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


def delete_directory_contents(dir_list):
    """Deletes the contents of two directories.

    Args:
        dir1 (str): Path to the first directory.
        dir2 (str): Path to the second directory.
    """
    try:
        for directory in dir_list:
            if os.path.exists(directory):
                for filename in os.listdir(directory):
                    file_path = os.path.join(directory, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f"Failed to delete {file_path}. Reason: {e}")
            else:
                print(f"Directory {directory} does not exist.")
    except Exception as e:
        print(f"Error processing directories: {e}")
