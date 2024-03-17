import requests

# Define the URL of the FastAPI application
# Update with the actual URL of your FastAPI application
base_url = "http://localhost:8000"

# Define the endpoint URL
endpoint_url = f"{base_url}/check-agency-watermark"

# Define the request data
request_data = {
    "name": "AZCO-LeaseJVC.jpg"  # Update with the name of the image you want to check
}

# Send a POST request to the endpoint
response = requests.post(endpoint_url, json=request_data)

# Check the response status code
if response.status_code == 200:
    data = response.json()
    print("Image exists in the GCS bucket:", data['exists'])
elif response.status_code == 400:
    print("Bad request:", response.json()['error'])
else:
    print("Internal server error occurred.")
