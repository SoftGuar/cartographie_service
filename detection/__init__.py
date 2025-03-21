import os
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Roboflow client using environment variables
CLIENT = InferenceHTTPClient(
    api_url=os.getenv("ROBOFLOW_API_URL"),
    api_key=os.getenv("ROBOFLOW_API_KEY")
)

# Configuration for different models
text_config = InferenceConfiguration(confidence_threshold=0.45)
walls_doors_config = InferenceConfiguration(confidence_threshold=0.4)
furniture_config = InferenceConfiguration(confidence_threshold=0.1)