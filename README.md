# Complaint Management System

This project is a multi-modal complaint management system designed for citizens to submit grievances. It leverages various AI and Large Language Model (LLM) technologies to streamline the complaint submission process through voice, text, and document uploads.

The system automatically transcribes audio, performs OCR on documents, extracts key information (name, phone, address, etc.), categorizes the complaint, and helps assign it to the appropriate department.

## Key Features

*   **Multi-Modal Input**:
    *   **Voice Complaints**: Record audio directly, upload a WAV file, or type/paste text. Supported Language right now: English, Hindi, and Telugu.
    *   **Document Complaints**: Upload images (JPG, PNG) of handwritten or printed letters.
*   **AI-Powered Automation**:
    *   **Speech-to-Text**: Utilizes SarvamAI and Groq for accurate, multi-lingual voice transcription.
    *   **Optical Character Recognition (OCR)**: Uses Google's Gemini 2.5 Pro to extract and translate text from documents in multiple Indian languages.
    *   **Smart Information Extraction**: Leverages Groq's Llama 3.1 to automatically parse text from voice or documents and fill in the person's details (Name, Phone, Aadhaar, Address).
    *   **AI-Based Categorization**: Intelligently classifies the complaint into predefined categories using a hybrid keyword and LLM approach.
*   **Automated Workflow**:
    *   **Assignment**: Automatically assigns the complaint to the relevant department based on a predefined matrix.
    *   **SLA Management**: Calculates and displays the Service Level Agreement (SLA) date for resolution.
    *   **Tracking**: Stores all complaints in a CSV file and allows users to check the status of their complaint using their phone number, Aadhaar, or complaint number.
*   **User-Friendly Interface**: Built with Gradio for an intuitive, web-based user experience with separate tabs for different functionalities.

## Technology Stack

*   **Backend**: Python
*   **UI**: Gradio
*   **AI & LLM Services**:
    *   **Information Extraction & Categorization**: Groq 
    *   **Speech-to-Text**: SarvamAI, SpeechRecognition 
    *   **OCR & Translation**: Google Generative AI 

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    Create a `requirements.txt` file with the following content:
    ```
    gradio
    pandas
    numpy
    SpeechRecognition
    Pillow
    python-dotenv
    groq
    google-generativeai
    sarvamai
    pyaudio
    ```
    Then run:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure API Keys:**
    Create a `.env` file in the root directory and add your API keys:
    ```.env
    GROQ_API_KEY="your-groq-api-key"
    GEMINI_API_KEY="your-gemini-api-key"
    SARVAM_API_KEY="your-sarvam-api-key"
    ```

5.  **Prepare Data Files:**
    Ensure the `complaint_assignment_matrix_levels.csv` file is present in the root directory. This file maps complaint types to the responsible assignee.

## How to Run

1.  Execute the main script from your terminal:
    ```bash
    python complaint_management_system.py
    ```

2.  The application will start and provide a local URL (usually `http://127.0.0.1:8080`). Open this URL in your web browser to access the system.
