# models/model_loader.py

import os
from groq import Groq
from dotenv import load_dotenv
from logger import get_logger

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

# Cache for loaded models
_model_cache = {}

def load_model(model_choice):
    """
    Loads and caches the Groq Llama Vision model.
    """
    global _model_cache

    if model_choice in _model_cache:
        logger.info(f"Model '{model_choice}' loaded from cache.")
        return _model_cache[model_choice]

    if model_choice == 'groq-llama-vision':
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")
        client = Groq(api_key=api_key)
        _model_cache[model_choice] = client
        logger.info("Groq Llama Vision model loaded and cached.")
        return _model_cache[model_choice]

    else:
        logger.error(f"Invalid model choice: {model_choice}")
        raise ValueError("Invalid model choice.")
