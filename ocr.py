import streamlit as st
from paddleocr import PaddleOCR, draw_ocr
import cv2
import numpy as np
from PIL import Image

# Supported languages (add more languages as needed)
languages = {
    "English": "en",
    "Arabic": "ar",
    "Hindi": "hi",
    "French":"fr"
    
}

# Default language
selected_lang = st.sidebar.selectbox("Select Language", list(languages.keys()))
lang_code = languages[selected_lang]

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang=lang_code, use_gpu=False)

# Streamlit app
st.title('OCR Image Processing with PaddleOCR')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    
    # Display the uploaded image
    st.image(img, channels="BGR")

    # Save the uploaded image to a temporary file
    img_path = 'temp_img.jpg'
    cv2.imwrite(img_path, img)
    
    # Perform OCR
    result = ocr.ocr(img_path, cls=True)

    # Display OCR results with confidence scores
    st.subheader("OCR Text with Confidence:")
    for line in result:
        # Initialize a list to store formatted text
        formatted_text = []
        
        # Iterate through each detection in the line
        for detection in line:
            # Extract text and confidence score from the detection
            text = detection[1][0]
            confidence = detection[1][1]
            
            # Format the text for display
            formatted_text.append(f"<span style='font-weight: bold;'>~ {text}</span> - : {confidence:.2f}")
        
        # Combine all formatted text in the line
        combined_text = "<br>".join(formatted_text)
        
        # Display the combined text with HTML rendering
        st.write(f"{combined_text}", unsafe_allow_html=True)
        
        # Add an empty line to separate each line of text
        st.write("")  # Add an empty line between each line of text
