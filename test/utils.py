import gc
from io import BytesIO
import os
import numpy as np
import requests
import cv2
import torch


def get_bounding_box_coordinates(image_url):
    # URL of the API endpoint
    api_url = "https://property.restb.ai/v1/multianalyze?client_key=8c57199bd6239b90d070bd146d25fc24c1a63b2c2a99cbaa8941f52ce2c2f01b"

    # JSON body data to be sent in the POST request
    data = {
        "image_urls": [image_url],
        "solutions": {
            "compliance": 3
        }
    }

    try:
        # Make the POST request
        response = requests.post(api_url, json=data)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Extract bounding box coordinates from the JSON response
            json_data = response.json()
            bounding_box = json_data["response"]["solutions"]["compliance"][
                "results"][0]["values"]["detections"][0]["bounding_box"]

            # Assign bounding box coordinates to variables
            top_left_x = bounding_box["top_left_x"]
            top_left_y = bounding_box["top_left_y"]
            bottom_right_x = bounding_box["bottom_right_x"]
            bottom_right_y = bounding_box["bottom_right_y"]
            print(bounding_box)

            # Return the bounding box coordinates
            return top_left_x, top_left_y, bottom_right_x, bottom_right_y
        else:
            print("POST request failed")
            print("Status code:", response.status_code)
            return None
    except Exception as e:
        print("An error occurred:", e)
        return None


def draw_bounding_box(image_url, top_left_x, top_left_y, bottom_right_x, bottom_right_y):
    # Download the image from the URL
    response = requests.get(image_url)
    if response.status_code == 200:
        # Create a stream-like object from the image content
        image_content = BytesIO(response.content)
        # Decode the image content with OpenCV
        image = cv2.imdecode(np.frombuffer(
            image_content.read(), np.uint8), cv2.IMREAD_COLOR)

        # Calculate bounding box coordinates in pixel values
        height, width, _ = image.shape
        x1 = int(top_left_x * width)
        y1 = int(top_left_y * height)
        x2 = int(bottom_right_x * width)
        y2 = int(bottom_right_y * height)

        # Draw bounding box on the image
        color = (0, 255, 0)  # Green color
        thickness = 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Save the mask image
        create_mask(image.shape, top_left_x, top_left_y,
                    bottom_right_x, bottom_right_y, "Images/restb_mask.jpg")

        # Display the image with bounding box
        cv2.imshow("Image with Bounding Box", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to download image from the URL")


def create_mask(image_shape, top_left_x, top_left_y, bottom_right_x, bottom_right_y, save_path):
    # Create a black image mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Calculate the coordinates of the bounding box
    top_left = (
        int(top_left_x * image_shape[1]), int(top_left_y * image_shape[0]))
    bottom_right = (
        int(bottom_right_x * image_shape[1]), int(bottom_right_y * image_shape[0]))

    # Draw the bounding box on the mask
    # Fill the rectangle with white color
    cv2.rectangle(mask, top_left, bottom_right, (255, 255, 255), -1)

    # Save the mask image
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, mask)


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()
