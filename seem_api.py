# 5/8/23 created to enable web services
from flask import Flask, request, jsonify, send_file
from PIL import Image
import io
import base64
from app import inference

# enable logging
#import logging
#logging.basicConfig(filename='flask.log', level=logging.DEBUG)

#Initialize the Flask app and set up the route for the API:
app = Flask(__name__)

@app.route('/api/segment', methods=['GET'])
def segment_get():
    response = {
        'error': 'Invalid request method. Please use a POST request with both an image and a task.'
    }
    return jsonify(response), 405

@app.route('/api/segment', methods=['POST'])
def segment():
    # Your code to handle the request and perform segmentation here
    # extract the necessary data from the request and convert it to the appropriate format
    image_data = request.files.get('image')
    task = request.form.get('task')

    if image_data is None or task is None:
        # Return an error message if the request is not in the expected format
        response = {
            'error': 'Invalid request format. Please provide both an image and a task.'
        }
        return jsonify(response), 400

    # Read the image data and convert it to an Image object
    image_data = image_data.read()
    image = Image.open(io.BytesIO(image_data))

    # Perform segmentation using the 'inference' function
    # Call the inference function with the necessary parameters and capture the output:
    result_image, result_video = inference({"image":image, "mask":None}, task)

    # Convert the result image and video into a format that can be sent back in the response. 
    # For example, you can convert the image to a PNG and encode it as a base64 string:
    img_bytes = io.BytesIO()
    result_image.save(img_bytes, format='PNG')
    img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')

    # Return the response as a JSON object:
    response = {
        'segmentation_image': img_base64
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
