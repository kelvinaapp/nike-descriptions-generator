import pickle
import os
import logging
from google.cloud import storage
import tempfile
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import gzip
import torch
import torch.quantization

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the GCS bucket and model info
BUCKET_NAME = 'pre-trained-model-iykra'
MODEL_BLOB_NAME = 'nike_product_generator_quantization.pkl'

def load_model():
    """
    Load the pre-trained quantized model from Google Cloud Storage bucket.
    
    Returns:
        dict: Dictionary containing the model state dict
    """
    try:
        # Create a temporary file to store the downloaded model
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # Initialize GCS client and download the model
            storage_client = storage.Client()
            bucket = storage_client.bucket(BUCKET_NAME)
            blob = bucket.blob(MODEL_BLOB_NAME)
            
            logger.info(f"Downloading model from GCS bucket: {BUCKET_NAME}")
            blob.download_to_filename(temp_file.name)
            
            # Load and decompress the quantized model
            with gzip.open(temp_file.name, 'rb') as f:
                buffer = f.read()
            quantized_dict = pickle.loads(buffer)
            
        # Clean up the temporary file
        os.unlink(temp_file.name)
        
        # Reconstruct the state dict
        state_dict = {}
        for key, tensor_data in quantized_dict.items():
            if 'min' in tensor_data:  # This was a quantized float tensor
                # Dequantize
                quantized = torch.from_numpy(tensor_data['data'])
                qscale = (tensor_data['max'] - tensor_data['min']) / 255.0
                tensor = quantized.float() * qscale + tensor_data['min']
                tensor = tensor.reshape(tensor_data['shape'])
            else:
                # Reconstruct non-quantized tensor
                tensor = torch.from_numpy(tensor_data['data'])
                if tensor_data['shape']:
                    tensor = tensor.reshape(tensor_data['shape'])
            
            state_dict[key] = tensor
        
        logger.info("Model loaded and dequantized successfully from GCS")
        return state_dict
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def generate_description(prompt, max_length=150, num_return_sequences=1):
    """
    Generate a product description based on user input.
    
    Args:
        prompt (str): The input text to base the generation on
        max_length (int, optional): Maximum length of generated text. Defaults to 150.
        num_return_sequences (int, optional): Number of sequences to generate. Defaults to 1.
    
    Returns:
        list: List of generated product descriptions
    """
    # Load the model dictionary
    state_dict = load_model()
    
    # Initialize model and tokenizer
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    model.load_state_dict(state_dict)

    # Configure tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # Set model to evaluation mode
    model.eval()

    # Encode input with attention mask
    encoded_input = tokenizer(
        prompt,
        return_tensors='pt',
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True
    )

    # Generate text
    outputs = model.generate(
        input_ids=encoded_input['input_ids'],
        attention_mask=encoded_input['attention_mask'],
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # Decode the generated text
    generated_texts = [
        tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for output in outputs
    ]
    
    return generated_texts