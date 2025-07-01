# TraductorWeb
Logistics CSV translation demo (English to Spanish) with a custom-trained AI model (Python, Flask, Transformers).
This project is a web application designed to translate CSV files, with a special focus on **English-to-Spanish** translation for **logistics terminology**. It utilizes a custom fine-tuned AI model based on the MarianMT architecture from Hugging Face Transformers to provide accurate and context-aware translations for the logistics industry.

The application allows users to upload a CSV file, specify the column containing English text to be translated, define the CSV delimiter, and then download the processed file with an added column containing the Spanish translations. The translation process is handled asynchronously in the background, with real-time progress updates displayed to the user.

This project serves as a demonstration of building a full-stack web application with an integrated, custom-trained machine learning model.

## Key Features

* **Specialized Translation:** AI model fine-tuned on logistics-specific vocabulary for improved accuracy in English-to-Spanish translation.
* **CSV File Processing:** Accepts CSV files as input and generates a new CSV with translations.
* **User-Friendly Interface:** Simple web interface for file upload, column selection (by name or index), and delimiter specification.
* **Asynchronous Task Processing:** Translations are handled in the background using threading, allowing the UI to remain responsive.
* **Real-time Progress Updates:** Users can see the progress of their translation task.
* **Direct Download:** Translated files are available for immediate download upon completion.

## Technology Stack

* **Backend:** Python, Flask
* **AI/Machine Learning:** PyTorch, Hugging Face Transformers (MarianMT)
* **Frontend:** HTML5, CSS3, JavaScript (ES6+)
* **Key Python Libraries:** `torch`, `transformers`, `flask`
* **Key Frontend Concepts:** AJAX (Fetch API) for dynamic updates, DOM manipulation.

---

## Project Setup & Usage

1.  **Clone the repository:**
    ```bash
    git clone <tu-url-de-repositorio-aqui>
    cd <nombre-de-tu-repositorio>
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Asegúrate de crear un archivo `requirements.txt` con Flask, Torch, Transformers, etc.)*

4.  **Place your trained model:**
    * Ensure your fine-tuned MarianMT model files are located in a subfolder within the `ai_model_assets/` directory (e.g., `ai_model_assets/Helsinki/`).
    * Update the `FINETUNED_MODEL_FOLDER_NAME` variable in `ai_logic.py` to match your model's folder name (e.g., `FINETUNED_MODEL_FOLDER_NAME = 'Helsinki'`).

5.  **Run the application:**
    ```bash
    python app.py
    ```
    The application should then be accessible at `http://127.0.0.1:5000/` or `http://0.0.0.0:5000/`.

## Project Structure
