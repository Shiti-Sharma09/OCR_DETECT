# OCRDETECT

OCR & Translation Tool

## 📌 Overview

This is a Streamlit-based OCR (Optical Character Recognition) and Translation tool that extracts text from images while preserving structure, including tables and paragraphs. It also provides translation capabilities to multiple languages using the Google Translator API.

## 🚀 Features

Extracts text from uploaded images using EasyOCR.

Maintains text structure and detects table elements.

Provides translation into multiple languages.

Displays structured output by overlaying extracted text on a blank image.

Supports various languages including English, Hindi, Punjabi, French, Spanish, and more.

## 📂 Installation

Follow these steps to set up the project locally:

### 1️⃣ Clone the Repository

 ```sh
  git clone https://github.com/yourusername/ocr-translation-tool.git
cd ocr-translation-tool
  ```

### 2️⃣ Create a Virtual Environment (Optional but Recommended)

 ```sh
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows
  ```

### 3️⃣ Install Dependencies

 ```sh
pip install -r requirements.txt
  ```

## 🛠️ Usage

Run the application with the following command:

 ```sh
streamlit run app.py
 ```

## 📤 Upload Image

Click on the "Choose an image" button to upload a JPG, PNG, or JPEG file.

Select OCR languages and a translation language.

Click "Process Image" to extract and translate text.

## 📊 Output

Extracted structured text will be displayed.

Translation of the extracted text in the selected language.

Table data (if detected) will be displayed in a structured format.

A reconstructed image maintaining the original text layout.

📜 #Dependencies

All required dependencies are listed in requirements.txt.
