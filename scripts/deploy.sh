#!/bin/bash

# Check if Docker is installed
if ! command -v docker &> /dev/null
then
    echo "Docker is not installed. Please install Docker and try again."
    exit 1
fi

# Check if a container with the same name already exists
if [ "$(docker ps -aq -f name=tf_serving)" ]; then
    echo "Stopping and removing existing container 'tf_serving'..."
    docker stop tf_serving > /dev/null
    docker rm tf_serving > /dev/null
fi

# Run TensorFlow Serving with the saved model
echo "Starting TensorFlow Serving..."
docker run -p 8501:8501 --name=tf_serving \
    --mount type=bind,source=$(pwd)/saved_models/image_classifier,target=/models/image_classifier \
    -e MODEL_NAME=image_classifier \
    -t tensorflow/serving

# Check if the container started successfully
if [ $? -eq 0 ]; then
    echo "TensorFlow Serving started successfully."
    echo "Access the model at http://localhost:8501/v1/models/image_classifier:predict"
else
    echo "Failed to start TensorFlow Serving. Check the logs for details."
    exit 1
fi
