import requests
import numpy as np
from PIL import Image
import json
from preprocess import preprocess_image

# Define the endpoint for the model
url = "http://localhost:8501/v1/models/image_classifier:predict"

# Load class labels from a file (e.g., 'labels.txt')
def load_labels(label_file_path):
    with open(label_file_path, "r") as file:
        labels = file.readlines()
    return [label.strip() for label in labels]

# Load your actual class labels
class_labels = load_labels("labels.txt")  # Replace with the correct path to your label file

# Load and preprocess the image
# def preprocess_image(image_path):
#     img = Image.open(image_path)
#     img = img.resize((224, 224))  # Assuming your model expects 224x224 input size
#     img = np.array(img) / 255.0  # Normalize the image
#     img = np.expand_dims(img, axis=0)  # Add batch dimension
#     return img.tolist()  # Convert to list for JSON serialization


# Make a prediction
def get_prediction(image_path):
    image_data =  preprocess_image(image_path)
    
    # Send the request to the TensorFlow Serving model
    # response = requests.post(url, data=data, headers={"content-type": "application/json"})
    response = requests.post(url, json={"instances": image_data.tolist()})
    response.raise_for_status()  # Raise an HTTPError for bad responses
   
    # Get the predicted probabilities from the response
    predictions = response.json()['predictions'][0]
    print(predictions)
    # print(predictions['predictions'][0])
    
    # Find the index of the class with the highest probability
    predicted_class_index = np.argmax(predictions)

    predicted_class = class_labels[predicted_class_index]
    predicted_probability = predictions[predicted_class_index]
    
    # Return the predicted class and probability
    return predicted_class, predicted_probability

# Example usage
image_path = "data/sample_image5.jpg"  # Replace with your image path
predicted_class, predicted_probability = get_prediction(image_path)

print(f"Predicted class: {predicted_class}")
print(f"Prediction probability: {predicted_probability}")
