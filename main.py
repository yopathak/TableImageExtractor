
import cv2
import pandas as pd
from rotation import correct_rotation
from cell_extraction import extract_table_cells
from ocr import ocr_image
from concurrent.futures import ThreadPoolExecutor

def extract_to_dataframe(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # Correct rotation
    rotated_image = correct_rotation(image)
    
    # Extract cells
    sliced_cells = extract_table_cells(rotated_image)
    
    # Extract text from each cell using ThreadPoolExecutor for parallel processing
    table_data = []
    with ThreadPoolExecutor() as executor:
        for row in sliced_cells:
            row_texts = list(executor.map(ocr_image, row))
            table_data.append(row_texts)

    # Convert the table data to a pandas DataFrame
    df = pd.DataFrame(table_data)
    return df
