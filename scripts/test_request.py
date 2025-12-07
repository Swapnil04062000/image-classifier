import requests
import numpy as np
import os
from preprocess import preprocess_image

def main():
    """
    Test script to send an image to a TensorFlow Serving model and fetch predictions.
    """
    # Configurable variables
    image_path = "data/sample_image2.jpg"  # Replace with your test image
    url = "http://localhost:8501/v1/models/image_classifier:predict"  # Model server URL

    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return

    # Preprocess the image
    image_data = preprocess_image(image_path)
    if image_data is None:
        print("Error during image preprocessing. Exiting.")
        return

    # Make the prediction request
    try:
        print("Sending request to the model server...")
        response = requests.post(url, json={"instances": image_data.tolist()})
        response.raise_for_status()  # Raise an HTTPError for bad responses
        prediction = response.json()
        print("Prediction response:", prediction)
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    main()



