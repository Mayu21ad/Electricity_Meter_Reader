import streamlit as st
from PIL import Image, ImageOps, ImageDraw
import google.generativeai as genai
import cv2
import numpy as np
import pytesseract
import io


with open('api.txt', 'r') as file:
    GOOGLE_API_KEY = file.read().strip()

genai.configure(api_key=GOOGLE_API_KEY)

MODEL_CONFIG = {
  "temperature": 0.2,
  "top_p": 1,
  "top_k": 32,
  "max_output_tokens": 4096,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  }
]

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash", 
    generation_config=MODEL_CONFIG,
    safety_settings=safety_settings
)

def preprocess_image(image):
    
    gray_image = ImageOps.grayscale(image)
  
    image_np = np.array(gray_image)
    
    blurred_image = cv2.GaussianBlur(image_np, (5, 5), 0)
    
    _, thresh_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded_image = cv2.erode(thresh_image, kernel, iterations=1)
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)
    
    processed_image_pil = Image.fromarray(dilated_image)
    
    return processed_image_pil


def draw_bounding_boxes(image):
    
    image_np = np.array(image)
  
    d = pytesseract.image_to_data(image_np, output_type=pytesseract.Output.DICT)
    
    draw = ImageDraw.Draw(image)
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
    
    return image


def image_format(image):
    
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    image_parts = [
        {
            "mime_type": "image/jpeg",
            "data": img_bytes.getvalue()
        }
    ]
    return image_parts


def gemini_output(image, system_prompt, user_prompt):
    image_info = image_format(image)
    input_prompt = [system_prompt, image_info[0], user_prompt]
    response = model.generate_content(input_prompt)
    return response.text

system_prompt = """
You are a specialist in comprehending electricity meter readings.
Input images in the form of electricity meter readings will be provided to you,
and your task is to respond to questions based on the content of the input image.
"""

st.title("OCR Electricity Meter Reader Application")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Processing...")

    preprocessed_image = preprocess_image(image)
  
    boxed_image = draw_bounding_boxes(preprocessed_image)
    
    st.image(boxed_image, caption='Preprocessed Image with Bounding Boxes.', use_column_width=True)

    user_prompt = "What is the electricity meter reading?"   
    meter_number_output = gemini_output(preprocessed_image, system_prompt, user_prompt)
    st.write("Meter Number: ", meter_number_output)
