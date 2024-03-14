import os
import requests
import base64

# Define the endpoint URL
endpoint_url = "http://localhost:8001/api/v1/inpaint"


# Read image and mask files as binary data
image_path = os.path.join("images", "image_003.jpg")
mask_path = os.path.join("images", "image_003_mask.jpg")

with open(image_path, "rb") as image_file:
    image_data = base64.b64encode(image_file.read()).decode("utf-8")

with open(mask_path, "rb") as mask_file:
    mask_data = base64.b64encode(mask_file.read()).decode("utf-8")

# Define the request payload
payload = {
    "image": image_data,
    "mask": mask_data,
    # "sd_seed": 123  # Example seed value
}

# Define the headers
headers = {
    "Content-Type": "application/json"
}

# Ensure the outputs folder exists, or create it if it doesn't
output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)

# Send POST request to the API endpoint with headers
response = requests.post(endpoint_url, json=payload, headers=headers)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Save the inpainted image
    output_filename = os.path.splitext(os.path.basename(image_path))[
        0] + "_output.png"
    output_path = os.path.join(output_folder, output_filename)
    with open(output_path, "wb") as output_file:
        output_file.write(response.content)
    print("Inpainted image saved successfully.")
else:
    print(f"Request failed with status code: {response.content}")
