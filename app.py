import streamlit as st
import easyocr
import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from deep_translator import GoogleTranslator
from sklearn.cluster import KMeans
import re

# Function to initialize the OCR reader with selected languages
def initialize_ocr(languages):
    return easyocr.Reader(languages)

# Function to detect table lines in an image using morphological operations
def detect_table_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert image to grayscale
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Detect horizontal and vertical lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detected_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detected_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    return cv2.add(detected_horizontal, detected_vertical)

# Function to clean extracted text and remove unwanted symbols
def clean_text(text):
    text = re.sub(r'[_]+', ' ', text)  # Replace underscores with spaces
    text = re.sub(r'\s*,\s*', ', ', text)  # Fix comma spacing
    return text.strip()

# Function to cluster text lines based on Y-coordinates for structured output
def cluster_paragraphs(text_data):
    if len(text_data) < 2:
        return text_data  # No clustering needed for single line
    
    y_values = np.array([y for _, y, _ in text_data]).reshape(-1, 1)
    num_clusters = min(len(y_values), 5)  # Limit number of clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(y_values)
    
    clustered_text = {i: [] for i in range(num_clusters)}
    for (x, y, text), cluster in zip(text_data, kmeans.labels_):
        clustered_text[cluster].append((x, y, text))
    
    return [sorted(clustered_text[c], key=lambda t: t[0]) for c in sorted(clustered_text)]

# Function to extract text from an image while preserving structure
def extract_text_with_structure(image, reader):
    image_np = np.array(image)  # Convert PIL image to numpy array
    table_mask = detect_table_lines(image_np)  # Detect table structures
    
    results = reader.readtext(image_np)  # Perform OCR on image
    if not results:
        return "No text detected.", []
    
    # Process extracted text
    text_data = [(int(box[0][0]), int(box[0][1]), clean_text(text)) for box, text, _ in results]
    clustered_text = cluster_paragraphs(text_data)
    
    extracted_text = []
    table_data = []
    
    # Organize extracted text and identify table content
    for cluster in clustered_text:
        extracted_text.append(" ".join([text for _, _, text in cluster]))
        extracted_text.append("\n\n")
        
        for x, y, text in cluster:
            if table_mask[y, x] > 0:
                table_data.append((x, y, text))
    
    table_df = None
    if table_data:
        table_df = pd.DataFrame(table_data, columns=["X", "Y", "Text"])
        table_df = table_df.sort_values(by=["Y", "X"]).reset_index(drop=True)
    
    return "".join(extracted_text).strip(), text_data, table_df

# Function to translate extracted text to the selected language
def translate_text(text, dest_language):
    try:
        return GoogleTranslator(source="auto", target=dest_language).translate(text)
    except Exception as e:
        return f"Translation Error: {str(e)}"

# Function to overlay extracted text on a blank image to maintain structure
def overlay_text_on_blank_image(image, structured_data):
    width, height = image.size
    blank_image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(blank_image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 18)  # Load font
    except:
        font = ImageFont.load_default()
    
    # Draw text at detected locations
    for x, y, text in structured_data:
        draw.text((x, y), text, fill="black", font=font)
    
    return blank_image

# Main function to run the Streamlit app
def main():
    st.title("📄SnapScribe(OCR & Translation Tool)")
    st.write("Upload an image to extract text from it and translate it into different languages.")

    # File uploader for image input
    uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "png", "jpeg"])

    # Language options for OCR and translation
    lang_options = {"English": "en", "Hindi": "hi", "Punjabi": "pa", "French": "fr",
        "Spanish": "es", "German": "de", "Chinese (Simplified)": "zh-cn", "Japanese": "ja", 
        "Korean": "ko", "Russian": "ru", "Arabic": "ar", "Italian": "it", "Portuguese": "pt"}

    # Language selection for OCR
    ocr_languages = st.multiselect("🌍 Select OCR languages", list(lang_options.keys()), default=["English"])
    selected_ocr_languages = [lang_options[lang] for lang in ocr_languages]
    dest_language = st.selectbox("🌐 Select translation language", list(lang_options.keys()))

    if uploaded_file:
        image = Image.open(uploaded_file)  # Open uploaded image
        st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

        if st.button("🔍 Process Image"):
            reader = initialize_ocr(selected_ocr_languages)  # Initialize OCR reader
            extracted_text, structured_data, table_df = extract_text_with_structure(image, reader)

            st.subheader("📝 Extracted Text (Structured):")
            st.text_area("Recognized Text", extracted_text, height=200)

            if extracted_text and extracted_text != "No text detected.":
                translated_text = translate_text(extracted_text, lang_options[dest_language])
                st.subheader(f"🌎 Translated Text ({dest_language}):")
                st.text_area("Translated Output", translated_text, height=200)
                
if __name__ == "__main__":
    main()
