# models/responder.py

from models.model_loader import load_model
from logger import get_logger
import os
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_response(images, query, session_id, model_choice='groq-llama-vision'):
    """
    Generates a response using the Groq Llama Vision model based on the query and images.
    Returns: (response_text, used_images)
    """
    logger = get_logger(__name__)

    try:
        logger.info(f"Generating response using model '{model_choice}'.")

        # Ensure images are full paths
        full_image_paths = [os.path.join('static', img) if not img.startswith('static') else img for img in images]
        
        # Check if any valid images exist
        valid_images = [img for img in full_image_paths if os.path.exists(img)]

        if not valid_images:
            logger.warning("No valid images found for analysis.")
            return "No images could be loaded for analysis.", []

        client = load_model(model_choice)

        content = [{"type": "text", "text": query}]

        # Use only the first image for simplicity
        if valid_images:
            img_path = valid_images[0]
            base64_image = encode_image(img_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })

        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                model="llama-3.2-90b-vision-preview",
            )
            generated_text = chat_completion.choices[0].message.content
            logger.info("Response generated using Groq Llama Vision model.")
            return generated_text, valid_images
        except Exception as e:
            logger.error(f"Error in Groq Llama Vision processing: {str(e)}", exc_info=True)
            return f"An error occurred while processing the image: {str(e)}", []

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"An error occurred while generating the response: {str(e)}", []

