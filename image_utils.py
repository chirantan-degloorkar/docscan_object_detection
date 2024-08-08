from PIL import Image, ImageDraw, ImageFont

def annotate_image(image_path, missing_objects):
    '''
    Input - Path of predicted image, missing objects
    Output - Saves the annotated image with missing objects
    Function - Annotates the image with missing objects and saves it
    '''
    try:
        with Image.open(image_path) as img:
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()
            
            if missing_objects:
                text = "Missing: " + ", ".join(missing_objects)
            else:
                text = "All objects present"
            
            position = (10, 10)
            draw.text(position, text, fill="red", font=font)
            img.save(image_path)
            print(f"Annotated image saved as {image_path}")
    except Exception as e:
        print(f"Error annotating image: {e}")

def convert_images_to_pdf(image_paths, output_pdf_path):
    '''
    Input - Path of images, output PDF path
    Output - Output PDF
    Function - Converts the images into a PDF
    '''
    images = []
    try:
        for image_path in image_paths:
            img = Image.open(image_path)
            images.append(img)

        if images:
            images[0].save(output_pdf_path, save_all=True, append_images=images[1:])
            print(f"PDF created: {output_pdf_path}")
            return True
        else:
            return False
    except Exception as e:
        print(f"Error converting images to PDF: {e}")
        return False
