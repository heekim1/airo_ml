#!/bin/bash

# Set the image name and tag
IMAGE_NAME="heekim1/airo_ml"
ECR_REPO_URI="public.ecr.aws/f3z9z6m7/airosolution/airo_ml"
TAG="latest"

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# Step 1: Build the Docker image for the specified platform
echo "Building Docker image..."
docker build --platform linux/amd64 -t $IMAGE_NAME .

# Step 2: Tag the image for the ECR repository
echo "Tagging image..."
docker tag $IMAGE_NAME:latest $ECR_REPO_URI:$TAG

# Step 3: Push the Docker image to the ECR repository
echo "Pushing image to ECR repository..."
docker push $ECR_REPO_URI:$TAG

# Confirmation message
echo "Docker image has been successfully built and pushed to $ECR_REPO_URI:$TAG"
