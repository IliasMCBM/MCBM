import streamlit as st
from paddleocr import PaddleOCR
import numpy as np
from PIL import Image
from rag import embedding_processing, user_chat, generate_embeddings_for_vault_content, load_embeddings

# Load embeddings only once when the app starts
embeddings = load_embeddings()  # Load embeddings from file
if embeddings is None:
    print('Embedding not foun')
    embeddings = generate_embeddings_for_vault_content()  # Generate embeddings if not found
else:
    print('Loading embedding')

# Supported languages for OCR
languages = {
    "English": "en",
    "Arabic": "ar",
    "Hindi": "hi",
    "French": "fr"
}

# Default language selection
selected_lang = st.sidebar.selectbox("Select Language", list(languages.keys()))
lang_code = languages[selected_lang]

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang=lang_code, use_gpu=False)

# Title for the app
st.title("Text Input or OCR from Image")
st.markdown("""**You are a helpful assistant that is an expert at extracting the most useful information from a given text.** 
**System Prompt:**

    You are a nutrition expert with in-depth knowledge of healthy eating and food science. 
    Your goal is to provide users with personalized, evidence-based food recommendations based 
    on the food item they mention. You prioritize health, nutrient balance, and dietary guidelines 
    in all your responses. For each food item, suggest healthy alternatives, preparation methods, or 
    complementary foods to improve the nutritional value of the user's diet. Be concise, accurate, 
    and clear in your recommendations.

**Instructions:**

    When a user inputs a specific food item (e.g., "pizza"), suggest healthier variations or alternatives, 
    along with nutrient information and possible health benefits.
    Provide simple preparation methods if relevant (e.g., suggest how to make a healthier version of a dish).
    Consider common dietary preferences (e.g., vegetarian, vegan, low-carb, gluten-free) and adapt 
    suggestions accordingly if the user specifies.
    If the food item is healthy, explain why and suggest complementary foods that could 
    enhance the meal's nutritional profile""")

# User input interface
input_option = st.radio("Input Type", ("Text", "Image"))

# If the user selects Text input
if input_option == "Text":
    # Input box for user text
    user_text = st.text_input("Enter your text")

    # Display the entered text when a button is pressed
    if st.button("Look for Query"):
        if user_text:
            response = user_chat(user_text, embeddings)  # Pass embeddings to the chat function
            st.write(f"Generated Output: {response}")
        else:
            st.error("Please enter some text")

# If the user selects Image input
elif input_option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Convert the uploaded file to an image
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        # Perform OCR on the uploaded image
        ocr_result = ocr.ocr(img_array, cls=True)
        extracted_text = " ".join(line[1][0].strip() for line in ocr_result[0]).replace('\n', ' ')  # Extract text without confidence scores

        if extracted_text:
            response = user_chat(extracted_text, embeddings)  # Pass embeddings to the chat function
            st.write(f"Generated Output: {response}")
        else:
            st.error("OCR could not extract text")

