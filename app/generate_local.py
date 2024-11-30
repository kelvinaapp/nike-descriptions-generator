import pickle
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from pathlib import Path

def load_model():
    # Load the saved model and tokenizer
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(current_dir, 'model/nike_product_generator_cpu.pkl')
    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)

    return model_dict


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
    model_dict = load_model()
    
    # Initialize model and tokenizer
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.load_state_dict(model_dict['model'])
    
    tokenizer = model_dict['tokenizer']

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
