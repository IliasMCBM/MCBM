import pytesseract
from PIL import Image

def extract_text_from_image(image_path):
    # Open the image file
    image = Image.open(image_path)

    # Extract text from the image in English
    text = pytesseract.image_to_string(image, lang='eng')

    return text

# Example usage
image_path = r'C:\Users\devcloud\PycharmProjects\pythonProject\MCBM\Lays_ingredients.jpg'  # Raw string path
text = extract_text_from_image(image_path)
print("Extracted text:", text)
