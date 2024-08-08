# from PyMuPDF import fitz
import fitz
from PIL import Image
import os
import zipfile
import shutil

pdf_file = "test.pdf"
def readPDF(pdf_file, img_dir):
    """Reads a PDF file and saves each page as an image in the specified directory

    Args:
        pdf_file (_type_): pdf file to read
        img_dir (_type_): directory to save the images
    """
    doc = fitz.open(pdf_file)
    page_count = doc.page_count
    for i in range (page_count):
        page = doc.load_page(i)
        pix = page.get_pixmap() # type: ignore
        output = f"outfile{i+1}.png"
        pix.save(img_dir+'/'+output)

def savePDF(image_dir, pdf_path):
    """Saves images in the image directory as a PDF file

    Args:
        image_dir (_type_): directory containing the images
        pdf_path (_type_): path to save the PDF file
    """
    imagelist = []
    im = None
    for path in os.listdir(image_dir):
        im = Image.open(os.path.join(image_dir, path)).convert('RGB')
        imagelist.append(im)
        
    im.save(pdf_path,save_all=True, append_images=imagelist) # type: ignore
    
def zipFiles(output_dir, zip_path):
    """Zips the files in the list to the specified zip file

    Args:
        file_paths (_type_): list of file paths to zip
        zip_path (_type_): path to save the zip file
    """
    # zip_path = os.path.join(output_dir, "output.zip")        
    # # Create a zip file of the output directory
    # shutil.make_archive(zip_path.replace('.zip', ''), 'zip', output_dir)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=output_dir)
                zipf.write(file_path, arcname)