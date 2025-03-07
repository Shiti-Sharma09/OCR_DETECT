import streamlit as st
import easyocr
import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from deep_translator import GoogleTranslator
from sklearn.cluster import KMeans
import re

def initialize_ocr(languages):
    """Initialize EasyOCR reader with selected languages."""
    return easyocr.Reader(languages)

def detect_table_lines(image):
    """Detect horizontal and vertical lines in an image to identify tables."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detected_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detected_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    return cv2.add(detected_horizontal, detected_vertical)

def clean_text(text):
    """Remove unwanted symbols and fix text formatting."""
    text = re.sub(r'[_]+', ' ', text)  # Replace underscores with spaces
    text = re.sub(r'\s*,\s*', ', ', text)  # Fix comma spacing
    return text.strip()

def cluster_paragraphs(text_data):
    """Cluster text lines based on Y-coordinates to preserve paragraph structure."""
    if len(text_data) < 2:
        return text_data  # No clustering needed for single line
    
    y_values = np.array([y for _, y, _ in text_data]).reshape(-1, 1)
    num_clusters = min(len(y_values), 5)  # Prevent excessive clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(y_values)
    
    clustered_text = {i: [] for i in range(num_clusters)}
    for (x, y, text), cluster in zip(text_data, kmeans.labels_):
        clustered_text[cluster].append((x, y, text))
    
    return [sorted(clustered_text[c], key=lambda t: t[0]) for c in sorted(clustered_text)]

def extract_text_with_structure(image, reader):
    """Extract text while preserving structure, including paragraphs and tables."""
    image_np = np.array(image)
    table_mask = detect_table_lines(image_np)
    
    results = reader.readtext(image_np)
    if not results:
        return "No text detected.", []
    
    text_data = [(int(box[0][0]), int(box[0][1]), clean_text(text)) for box, text, _ in results]
    clustered_text = cluster_paragraphs(text_data)
    
    extracted_text = []
    table_data = []
    
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

def translate_text(text, dest_language):
    try:
        return GoogleTranslator(source="auto", target=dest_language).translate(text)
    except Exception as e:
        return f"Translation Error: {str(e)}"

def overlay_text_on_blank_image(image, structured_data):
    """Render extracted text on a blank canvas to maintain original structure."""
    width, height = image.size
    blank_image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(blank_image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()
    
    for x, y, text in structured_data:
        draw.text((x, y), text, fill="black", font=font)
    
    return blank_image

def main():
    st.title("📄 OCR & Translation Tool (Structured Output)")
    st.write("Upload an image to extract and translate text while preserving structure.")

    uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "png", "jpeg"])

    lang_options = {"English": "en", "Hindi": "hi", "Punjabi": "pa", "French": "fr",
        "Spanish": "es", "German": "de", "Chinese (Simplified)": "zh-cn", "Japanese": "ja", 
        "Korean": "ko", "Russian": "ru", "Arabic": "ar", "Italian": "it", "Portuguese": "pt"}

    ocr_languages = st.multiselect("🌍 Select OCR languages", list(lang_options.keys()), default=["English"])
    selected_ocr_languages = [lang_options[lang] for lang in ocr_languages]
    dest_language = st.selectbox("🌐 Select translation language", list(lang_options.keys()))

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

        if st.button("🔍 Process Image"):
            reader = initialize_ocr(selected_ocr_languages)
            extracted_text, structured_data, table_df = extract_text_with_structure(image, reader)

            st.subheader("📝 Extracted Text (Structured):")
            st.text_area("Recognized Text", extracted_text, height=200)

            if extracted_text and extracted_text != "No text detected.":
                translated_text = translate_text(extracted_text, lang_options[dest_language])
                st.subheader(f"🌎 Translated Text ({dest_language}):")
                st.text_area("Translated Output", translated_text, height=200)
                structured_output_image = overlay_text_on_blank_image(image, structured_data)
                st.image(structured_output_image, caption="Text Layout Mimicking Original", use_column_width=True)
                
                if table_df is not None:
                    st.subheader("📊 Extracted Table Data:")
                    st.dataframe(table_df.drop(columns=["X", "Y"]))

if __name__ == "__main__":
    main()
