import os
import requests

base_url = "http://localhost:8000"

# Define the endpoint URL
endpoint_url = f"{base_url}/agency-inpaint"
# Define the image URL and mask name
# Update with your image URL
image_url = "https://images.bayut.com/thumbnails/625732376-800x600.webp"
# Update with your mask name
mask_name = "AXCAPITALRealEstate.jpg"

# Define the request payload
payload = {
    "image_url": image_url,
    "mask_name": mask_name,
}

# Send POST request to the API endpoint
response = requests.post(endpoint_url, json=payload)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Save the response content as the inpainted image
    output_folder = "outputs"
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(
        output_folder, mask_name.replace(".jpg", "_inpainted.jpg"))
    with open(output_path, "wb") as output_file:
        output_file.write(response.content)
    print("Inpainted image saved successfully.")
else:
    print(f"Request failed with status code: {response.status_code}")
