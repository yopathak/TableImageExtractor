import cv2

# Constants
ksize = 50
min_contour_area = 500
row_threshold = 20

def extract_table_cells(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Binarize the image using OTSU's thresholding
    _, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Define vertical and horizontal kernels
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ksize))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, 1))
    
    # Apply morphology operations
    vertical_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    horizontal_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    # Combine vertical and horizontal lines to get the table structure
    table_structure = cv2.addWeighted(vertical_lines, 1, horizontal_lines, 1, 0)
    
    # Find contours on the table structure
    contours, _ = cv2.findContours(table_structure, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Remove the largest contour (boundary of the entire table)
    largest_contour = max(contours, key=cv2.contourArea)
    contours.remove(largest_contour)
    
    # Filter out smaller contours
    contours_filtered = [ctr for ctr in contours if cv2.contourArea(ctr) > min_contour_area]
    
    # Group contours by their vertical position to determine rows
    grouped_contours = []
    current_row = []
    for ctr in sorted(contours_filtered, key=lambda x: cv2.boundingRect(x)[1]):  # Sort by y-coordinate
        if not current_row:
            current_row.append(ctr)
        else:
            if abs(cv2.boundingRect(current_row[0])[1] - cv2.boundingRect(ctr)[1]) <= row_threshold:
                current_row.append(ctr)
            else:
                grouped_contours.append(sorted(current_row, key=lambda x: cv2.boundingRect(x)[0]))  # Sort by x-coordinate
                current_row = [ctr]
    # Add the last row
    grouped_contours.append(sorted(current_row, key=lambda x: cv2.boundingRect(x)[0]))
    
    # Extract individual cells
    sliced_cells = []
    for row_contours in grouped_contours:
        row_cells = []
        for ctr in row_contours:
            x, y, w, h = cv2.boundingRect(ctr)
            cell_img = image[y:y+h, x:x+w]
            row_cells.append(cell_img)
        sliced_cells.append(row_cells)
    
    return sliced_cells
