import cv2

def correct_rotation(image):
    # Convert to grayscale and binarize
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the largest contour (assuming it's the table boundary)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the rotated rectangle around the largest contour
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    
    # Calculate the angle of rotation
    angle = rect[-1]
    if angle < -45:
        angle += 90
    
    # Get the image center and rotate the image
    center = (image.shape[1] // 2, image.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated_image

