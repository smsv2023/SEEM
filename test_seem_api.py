# created to test the seem_api.py

import requests
from PIL import Image
import io
import base64

# Load an image and encode it as a base64 string:
def load_image_as_base64(filepath):
    with open(filepath, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

image_base64 = load_image_as_base64("/home/ec2-user/SAM/Segment-Everything-Everywhere-All-At-Once/demo_code/examples/river2.png")

# Send a request to the API with the image and task data

url = "http://localhost:5000/api/segment"

data = {
    "task": "Panoptic"
}

files = {
    "image": ("image.png", base64.b64decode(image_base64), "image/png")
}

response = requests.post(url, data=data, files=files)

response_json = response.json()

segmentation_image_base64 = response_json["segmentation_image"]
segmentation_image_data = base64.b64decode(segmentation_image_base64)

segmentation_image = Image.open(io.BytesIO(segmentation_image_data))
segmentation_image.save("segmentation_result.png")
