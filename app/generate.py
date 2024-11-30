import pickle
import os
import logging
import torch
from typing import Optional
from google.cloud import storage
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the GCS bucket and model info
BUCKET_NAME = 'pre-trained-model-iykra'
MODEL_BLOB_NAME = 'nike_product_generator.pkl'

def load_model():
    """
    Load the pre-trained model from Google Cloud Storage bucket.
    
    Returns:
        model: The loaded model object
    """
    try:
        # Set the device to CPU if CUDA is not available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Create a temporary file to store the downloaded model
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # Initialize GCS client and download the model
            storage_client = storage.Client()
            bucket = storage_client.bucket(BUCKET_NAME)
            blob = bucket.blob(MODEL_BLOB_NAME)
            
            logger.info(f"Downloading model from GCS bucket: {BUCKET_NAME}")
            blob.download_to_filename(temp_file.name)
            
            # Load the model from the temporary file
            model = torch.load(temp_file.name)
            
        # Clean up the temporary file
        os.unlink(temp_file.name)
        
        # Move model to appropriate device
        model = model.to(device)
        logger.info("Model loaded successfully from GCS")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def generate_description(input_text: str, max_length: Optional[int] = 100) -> str:
    """
    Generate a product description based on user input.
    
    Args:
        input_text (str): The input text to base the generation on
        max_length (int, optional): Maximum length of generated text. Defaults to 100.
    
    Returns:
        str: Generated product description
    """
    try:
        model = load_model()
        # Generate text using the model
        # Note: The exact generation code might need to be adjusted based on your specific model
        generated_text = model.generate(input_text, max_length=max_length)
        
        logger.info("Text generated successfully")
        return generated_text
    
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise