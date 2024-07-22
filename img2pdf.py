# from PyMuPDF import fitz
import fitz
from PIL import Image
import os

pdf_file = "test.pdf"
def readPDF(pdf_file, img_dir):
    doc = fitz.open(pdf_file)
    page_count = doc.page_count
    for i in range (page_count):
        page = doc.load_page(i)
        pix = page.get_pixmap()
        output = f"outfile{i+1}.png"
        pix.save(img_dir+'/'+output)

def savePDF(image_dir, pdf_path):
    imagelist = []
    im = None
    for path in os.listdir(image_dir):
        im = Image.open(os.path.join(image_dir, path)).convert('RGB')
        imagelist.append(im)
    # im3 = image3.convert('RGB')

    im.save(pdf_path,save_all=True, append_images=imagelist)

# savePDF(image_dir=res_folder)