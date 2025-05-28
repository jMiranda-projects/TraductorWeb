# ai_logic.py
"""
Handles all Artificial Intelligence operations for the application.

This module is responsible for:
- Loading a fine-tuned MarianMT translation model and its tokenizer.
- Providing a function to translate text from English to Spanish,
  specifically optimized for logistics terminology.
- Managing the AI model and tokenizer as global, lazily-loaded resources
  to ensure efficiency and performance.
- Configuring logging for AI-specific operations.
- Managing device selection (GPU/CPU) for model inference.
"""

import torch
from transformers import MarianMTModel, MarianTokenizer
import os
import logging
import sys # Imported to be used by the streamHandler

# --- Logger Configuration ---
# Sets up a logger for this module to provide insights into its operations.
# This is useful for debugging and monitoring the AI component.
logger = logging.getLogger(__name__) # Creates a logger named after the module.
if not logger.handlers: # Avoids adding multiple handlers if the module is reloaded.
    logger.setLevel(logging.INFO) # Sets the minimum logging level to INFO.
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ) # Defines the log message format.
    streamHandler = logging.StreamHandler(sys.stdout) # Configures logs to output to the console.
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)

# --- Model Path Configuration ---
# CRITICAL: This variable must be updated to the exact name of the folder
# containing the fine-tuned Helsinki-NLP model. This folder should be
# located within the 'ai_model_assets/' directory.
# Example: 'my_translator_model_v1' or 'helsinki-nlp-en-es-logistics'
FINETUNED_MODEL_FOLDER_NAME = 'Helsinki'

# Construct the absolute paths to the model assets.
# os.path.dirname(__file__) gets the directory of the current script (ai_logic.py).
# This makes the path relative to the script's location, enhancing portability.
BASE_ASSETS_PATH = os.path.join(os.path.dirname(__file__), 'ai_model_assets')
FINETUNED_TRANSLATION_MODEL_PATH = os.path.join(BASE_ASSETS_PATH, FINETUNED_MODEL_FOLDER_NAME)

# --- Device Configuration ---
# Automatically selects CUDA (NVIDIA GPU) if available, otherwise defaults to CPU.
# Using a GPU significantly speeds up model inference.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Inference Parameters ---
# Defines the maximum sequence length for the translation model.
# MarianMT models typically have a limit of 512 tokens. Texts longer
# than this will be truncated.
MAX_TRANSLATION_LENGTH = 512

# --- Global Model and Tokenizer Cache ---
# These global variables will store the loaded model and tokenizer.
# This lazy-loading approach ensures that the potentially time-consuming
# model loading process happens only once, improving application responsiveness.
globalTranslationModel = None
globalTranslationTokenizer = None

def loadGlobalTranslationModel():
    """
    Loads the fine-tuned translation model and its tokenizer from the specified path.

    This function implements a singleton pattern for model loading:
    - It checks if the model and tokenizer are already loaded. If so, it returns early.
    - If not loaded, it attempts to load them from `FINETUNED_TRANSLATION_MODEL_PATH`.
    - The loaded model is moved to the configured `DEVICE` (GPU or CPU).
    - Sets the model to evaluation mode (`model.eval()`), which is crucial for inference.
    - Handles potential errors during loading, such as `FileNotFoundError` or other exceptions.

    Raises:
        FileNotFoundError: If the model directory specified by
                           `FINETUNED_TRANSLATION_MODEL_PATH` does not exist.
        Exception: Any other exception encountered during model/tokenizer loading
                   from Hugging Face's `from_pretrained` method.
    """
    global globalTranslationModel, globalTranslationTokenizer # Allows modification of global variables.

    # Optimization: If model and tokenizer are already loaded, do nothing.
    if globalTranslationModel is not None and globalTranslationTokenizer is not None:
        logger.info("AI Logic: Translation model and tokenizer are already loaded.")
        return

    logger.info(f"AI Logic: Initiating load of translation model from: {FINETUNED_TRANSLATION_MODEL_PATH}")
    logger.info(f"AI Logic: Target device for model: {DEVICE} (cuda=GPU, cpu=CPU)")

    # Validate that the model path exists before attempting to load.
    if not os.path.isdir(FINETUNED_TRANSLATION_MODEL_PATH):
        errorMsg = (f"Model directory not found at: {FINETUNED_TRANSLATION_MODEL_PATH}. "
                    f"Please verify 'FINETUNED_MODEL_FOLDER_NAME' in 'ai_logic.py' "
                    f"and ensure the folder '{FINETUNED_MODEL_FOLDER_NAME}' exists within 'ai_model_assets/'.")
        logger.error(f"AI Logic: {errorMsg}")
        raise FileNotFoundError(errorMsg)

    try:
        # Load the pre-trained model using Hugging Face's `from_pretrained`.
        # This method intelligently loads the model configuration and weights
        # from the specified directory.
        globalTranslationModel = MarianMTModel.from_pretrained(FINETUNED_TRANSLATION_MODEL_PATH)
        globalTranslationModel.to(DEVICE) # Move the model to the selected device (GPU/CPU).

        # Load the corresponding tokenizer for the model.
        globalTranslationTokenizer = MarianTokenizer.from_pretrained(FINETUNED_TRANSLATION_MODEL_PATH)

        # Some MarianMT models might not have `pad_token_id` set by default.
        # Setting it (often to `eos_token_id`) can be important for robust padding,
        # although for single sentence translation with truncation, its direct impact might be minimal.
        if globalTranslationTokenizer.pad_token_id is None and globalTranslationTokenizer.eos_token_id is not None:
            globalTranslationTokenizer.pad_token_id = globalTranslationTokenizer.eos_token_id
            logger.info(f"AI Logic: Tokenizer 'pad_token_id' was not set. Assigned 'eos_token_id' ({globalTranslationTokenizer.eos_token_id}).")

        # Set the model to evaluation mode. This disables layers like Dropout,
        # which are used during training but not for inference, ensuring deterministic outputs.
        globalTranslationModel.eval()
        logger.info(f"AI Logic: Translation model and tokenizer successfully loaded and configured on {DEVICE}.")

    except Exception as e:
        # Catch-all for any other errors during the loading process.
        logger.error(f"AI Logic: Critical error during model/tokenizer loading from '{FINETUNED_TRANSLATION_MODEL_PATH}': {e}", exc_info=True)
        # Re-raise the exception to halt execution if the model cannot be loaded,
        # as it's a critical component of the application.
        raise e

def translateSingleTextDirect(originalText: str) -> str:
    """
    Translates a single string of text from English to Spanish using the loaded model.

    Args:
        originalText (str): The English text to be translated.

    Returns:
        str: The translated Spanish text. Returns an error message string if the model
             is not loaded or if an internal translation error occurs. Returns an
             empty string if the input `originalText` is empty or invalid.
    """
    # Ensure the model and tokenizer are loaded before attempting translation.
    if globalTranslationModel is None or globalTranslationTokenizer is None:
        logger.error("AI Logic (Translation): Attempted to translate, but model/tokenizer are not loaded.")
        return "[ERROR: MODELO NO CARGADO]" # User-facing error message (Spanish)

    # Handle empty or invalid input text.
    if not originalText or not isinstance(originalText, str) or not originalText.strip():
        logger.debug("AI Logic (Translation): Input text is empty or invalid. Returning empty string.")
        return ""

    logger.debug(f"AI Logic: Translating on {DEVICE}: '{originalText[:50]}...'") # Log snippet of text.
    try:
        # Tokenize the input text.
        # `return_tensors="pt"`: Returns PyTorch tensors.
        # `padding=False`: Padding is typically handled per batch; for single text, not strictly needed here if truncation is active.
        # `truncation=True`: Truncates text longer than `max_length`.
        # `max_length`: Limits input length to `MAX_TRANSLATION_LENGTH`.
        inputs = globalTranslationTokenizer(
            [originalText], # Tokenizer expects a list of sentences.
            return_tensors="pt",
            padding=False, # Set to True or 'longest' if batching multiple varied-length sentences without truncation.
            truncation=True,
            max_length=MAX_TRANSLATION_LENGTH
        )

        # Move tokenized inputs to the same device as the model.
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

        # Generation parameters for the model. These can be tuned for translation quality vs. speed.
        # - `max_length`: Maximum length of the generated translated text.
        # - `num_beams`: Number of beams for beam search. Higher can improve quality but is slower.
        # - `early_stopping`: If True, generation finishes when all beam hypotheses reach EOS.
        # - `no_repeat_ngram_size`: Prevents the model from repeating n-grams (sequences of n words).
        generationKwargs = {
            "max_length": MAX_TRANSLATION_LENGTH,
            "num_beams": 4,
            "early_stopping": True,
            "no_repeat_ngram_size": 2
        }

        # Perform inference within `torch.no_grad()` context.
        # This disables gradient calculations, reducing memory usage and speeding up computations,
        # as gradients are only needed for training.
        with torch.no_grad():
            # Generate translation token IDs.
            outputTokens = globalTranslationModel.generate(**inputs, **generationKwargs)

        # Decode the generated token IDs back into a human-readable string.
        # `skip_special_tokens=True` removes tokens like <s>, </s>, <pad> from the output.
        translation = globalTranslationTokenizer.decode(outputTokens[0], skip_special_tokens=True)
        logger.debug(f"AI Logic: Translation result: '{translation[:50]}...'") # Log snippet of translation.
        return translation

    except Exception as e:
        logger.error(f"AI Logic: Error during translation of text '{originalText[:30]}...': {e}", exc_info=True)
        return "[ERROR INTERNO DE TRADUCCIÓN]" # User-facing error message (Spanish)

# --- Module Usage Notes ---
# The `loadGlobalTranslationModel()` function is intended to be called once at application startup
# (e.g., from app.py when the Flask app initializes).
# The `translateSingleTextDirect()` function is then called as needed to perform translations.