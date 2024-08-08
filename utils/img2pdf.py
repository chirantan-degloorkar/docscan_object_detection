# from PyMuPDF import fitz
import fitz
from PIL import Image
import os

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
