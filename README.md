Image Classification Model with Docker Deployment

Here, we have trained and deployed an image classification model using TensorFlow Serving and Docker. Below is the summary of key steps and scripts used:

Model Training
    We trained an image classification model using a Simple Temporal Convolutional Network (TCN).
   
    preprocess.py:
        Handles image preprocessing (e.g., resizing, normalization) to prepare input data for the model.
    train.py:
        Loads the CIFAR-10 dataset.
    Defines and trains the TCN model.
        Saves the trained model in the saved_models/image_classifier directory in TensorFlow's SavedModel format.


Docker Setup for Model Deployment
    We created a Docker container to deploy the model using TensorFlow Serving.
    Dockerfile:
        Uses TensorFlow Serving as the base image.
        Copies the trained model from saved_models/image_classifier to /models/image_classifier inside the container.
        Sets the model name as image_classifier.
        
    Runs TensorFlow Serving on port 8501.
        Commands:
        
        Build the Docker Image:
            docker build -t image_classifier_server .
        Run the Docker Container:
            docker run -p 8501:8501 --name=tf_serving image_classifier_server


Testing the Deployment

    Test Input Preparation:
    Loaded a sample image and preprocessed it (resized to 32x32, normalized pixel values).
    Saved the preprocessed input as test_data.json.
    Send Predictions to TensorFlow Serving:
    Used curl to send the test input to the model's endpoint:
        curl -X POST http://localhost:8501/v1/models/image_classifier:predict -d @test_data.json
    Received predictions as raw probabilities for each class.


Custom API with FastAPI
To enhance the deployment, we wrapped TensorFlow Serving with a lightweight web server (FastAPI) to provide:

    Custom Endpoints: Such as /predict-image for easier API interaction.
    Preprocessing: Automates tasks like resizing and normalizing input images before sending them to TensorFlow Serving.
    Post-processing: Converts raw model predictions into human-readable class names for better usability.
    Why it’s needed:
    Wrapping TensorFlow Serving with FastAPI makes the deployment more user-friendly and flexible, allowing end-users to interact with the model using intuitive APIs and ensuring preprocessing and post-processing are standardized.