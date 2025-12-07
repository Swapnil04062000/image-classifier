import gradio as gr
import numpy as np
import requests
from preprocess import preprocess_image

# Define the classification function
def classify_image(image):
    """
    Preprocesses the input image, sends it to the model server, and returns predictions.

    Args:
        image: PIL Image object provided by Gradio.

    Returns:
        dict: A dictionary mapping class names to prediction probabilities.
    """
    try:
        # Preprocess the image
        image_data = preprocess_image(image)
        # Send the image to the model server
        response = requests.post(
            "http://localhost:8501/v1/models/image_classifier:predict",
            json={"instances": image_data.tolist()}
        )
        response.raise_for_status()  # Raise an HTTP error for bad responses

        # Parse predictions
        predictions = response.json().get("predictions", [[]])[0]
        classes = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]
        return {classes[i]: predictions[i] for i in range(len(classes))}
    except Exception as e:
        return {"Error": str(e)}

# Define the Gradio interface
gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),  # Expect a PIL Image
    outputs=gr.Label(),
    title="CIFAR-10 Image Classifier",
    description="Upload an image, and the model will classify it into one of the CIFAR-10 categories."
).launch()
