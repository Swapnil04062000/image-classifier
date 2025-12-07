from PIL import Image
import numpy as np

def preprocess_image(image_path, target_size=(32, 32)):
    """
    Preprocesses the input image for prediction.

    Args:
        image_path (str): Path to the input image file.
        target_size (tuple): Desired image size (width, height) for resizing.

    Returns:
        np.ndarray: Preprocessed image array, normalized and ready for model inference.
    """
    try:
        img = Image.open(image_path).convert('RGB').resize(target_size)
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
        return img_array
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None
