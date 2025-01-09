import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import necessary modules
try:
    from inference_sdk import InferenceHTTPClient
except ImportError as e:
    print("Inference SDK module is not installed. Please install it using 'pip install inference_sdk'")
    raise e

import json
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import easyocr
import numpy as np
import pandas as pd

# Initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="Y10vh3ZwlwXaBpUEcDJ3"
)

# Infer on a local image
image_path = "PREM KUMAR-Java Developer-Labri Edge Technologies Pvt Ltd_page-0001.jpg"
try:
    result = CLIENT.infer(image_path, model_id="resume-parse/13")
    print(json.dumps(result, indent=4))  # Print the result to understand its structure
except Exception as e:
    print(f"An error occurred during inference: {e}")
    raise e

# Load the image
image = Image.open(image_path)

# Create a drawing context
draw = ImageDraw.Draw(image)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# List to store extracted texts
extracted_texts = []

# Assume the result is a dictionary containing 'predictions'
for idx, prediction in enumerate(result.get('predictions', [])):
    x = prediction['x']
    y = prediction['y']
    width = prediction['width']
    height = prediction['height']
    label = prediction.get('class', 'N/A')
    confidence = prediction.get('confidence', 1.0)

    # Calculate the bounding box
    left = x - width / 2
    top = y - height / 2
    right = x + width / 2
    bottom = y + height / 2

    # Draw the bounding box (optional, remove if not needed)
    draw.rectangle([left, top, right, bottom], outline="red", width=2)

    # Extract the region of interest
    roi = image.crop((left, top, right, bottom))

    # Use easyocr to extract text from the ROI
    text = reader.readtext(np.array(roi), detail=0)

    # Store the extracted text in the list
    extracted_texts.append({
        "label": label,
        "confidence": confidence,
        "text": ' '.join(text).strip()
    })

# Display the image with bounding boxes (optional)
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('off')
plt.show()

# Print the extracted texts
print(json.dumps(extracted_texts, indent=4))

# Create a DataFrame from the extracted texts
df = pd.DataFrame(extracted_texts)

# Save the DataFrame to an Excel file
output_excel_path = "extracted_texts.xlsx"
df.to_excel(output_excel_path, index=False)
print(f"Extracted texts saved to {output_excel_path}")
