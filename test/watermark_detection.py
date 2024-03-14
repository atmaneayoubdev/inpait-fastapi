import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image


def detect_watermark(image_url):
    # Download the image from the URL
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    img = np.array(img)  # Convert image to numpy array

    # Define region of interest (ROI) for watermark detection (e.g., lower right corner)
    roi_height, roi_width = 100, 200
    roi = img[-roi_height:, -roi_width:]

    # Convert ROI to grayscale
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Perform Canny edge detection on the ROI
    edges = cv2.Canny(roi_gray, 50, 150)

    # Find contours of potential watermarks in the ROI
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around contours found in the ROI
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Watermark Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    image_url = "https://images.bayut.com/thumbnails/520556304-800x600.webp"
    detect_watermark(image_url)
