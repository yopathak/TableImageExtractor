import pytesseract

def ocr_image(cell_img):
    # Convert the cell image to string using pytesseract
    text = pytesseract.image_to_string(cell_img, config='--psm 6').strip()  # PSM 6 assumes a uniform block of text
    return text
