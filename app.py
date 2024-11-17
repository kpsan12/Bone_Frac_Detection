import os
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# Configuration for upload and output folders
UPLOAD_FOLDER = 'uploads/'
OUTPUT_FOLDER = 'output/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
st.set_page_config(layout="wide")

# Load YOLO models
MODEL_PATHS = ["models/model1.pt", "models/model2.pt"] #, "models/model3.pt"]
models = [YOLO(path) for path in MODEL_PATHS]

# Set confidence threshold for all models
for model in models:
    model.conf = 0.4

# Streamlit UI Setup
st.title("Bone Fracture Detection with Deep Learning")

# Add a small description below the title
st.markdown(
    """
    Hi..., this is Kiruthik Pranav, this app leverages deep learning models to detect bone fractures in images. 
    You can upload your own image or test the models with some sample images to see how it works.
    Feel free to experiment with the sample images on the left.
    """
)

st.sidebar.header("Options")
test_mode = st.sidebar.radio("Choose Input Type:", ["Upload Image", "Use Sample Image"])

# Sample images for testing
sample_images = {
    "Sample 1": r"sample_images/sample1.jpg",
    "Sample 2": r"sample_images/sample2.jpg",
    "Sample 3": r"sample_images/sample3.jpg"
}

def resize_image(image, max_width=400, max_height=400):
    """
    Resize the given image to fit within the specified dimensions.
    
    Args:
        image (PIL.Image): Input image to be resized.
        max_width (int): Maximum allowed width.
        max_height (int): Maximum allowed height.

    Returns:
        PIL.Image: Resized image.
    """
    width, height = image.size
    if width > max_width:
        new_width = max_width
        new_height = int((max_width / width) * height)
        image = image.resize((new_width, new_height))
    if height > max_height:
        new_height = max_height
        new_width = int((max_height / height) * width)
        image = image.resize((new_width, new_height))
    return image

def handle_file_upload(uploaded_file):
    """
    Handle file upload and return the image path.
    
    Args:
        uploaded_file (str or UploadedFile): Image file uploaded by user.

    Returns:
        str: Path of the uploaded image.
    """
    try:
        if isinstance(uploaded_file, str):  # If it's a sample image
            return uploaded_file
        else:  # If a file is uploaded
            image_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            return image_path
    except Exception as e:
        st.error(f"Error in file upload: {e}")
        raise

def detect_fracture(image_path):
    """
    Detect bone fractures in an image using the pre-trained YOLO models.
    
    Args:
        image_path (str): Path to the image for detection.

    Returns:
        bool: True if a fracture is detected, False otherwise.
        PIL.Image: Annotated image with bounding boxes.
    """
    try:
        results = [model(image_path) for model in models]
        fracture_detected = any(len(result) > 0 and len(result[0].boxes) > 0 for result in results)

        if fracture_detected:
            best_result = max(
                results, 
                key=lambda x: max(box.conf.item() for box in x[0].boxes) if len(x) > 0 and len(x[0].boxes) > 0 else 0
            )
            boxes = best_result[0].boxes.xyxy.cpu().numpy()
            confidences = best_result[0].boxes.conf.cpu().numpy()

            # Convert image to OpenCV format (BGR)
            image_cv = cv2.imread(image_path)

            # Draw rectangles on the image for each detected box
            for box, confidence in zip(boxes, confidences):
                x1, y1, x2, y2 = map(int, box)  # Get the coordinates
                cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw red rectangle

            # Convert the annotated image back to PIL format for display
            annotated_image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
            return True, annotated_image
        else:
            return False, None

    except Exception as e:
        st.error(f"Error during detection: {e}")
        return False, None

def main():
    """
    Main function to run the Streamlit app for bone fracture detection.
    """
    # Image selection logic
    if test_mode == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    else:
        selected_sample = st.sidebar.selectbox("Choose a sample image", list(sample_images.keys()))
        uploaded_file = sample_images[selected_sample]

    if uploaded_file:
        try:
            # Handle file upload
            image_path = handle_file_upload(uploaded_file)

            # Display the uploaded image
            input_image = Image.open(image_path)
            resized_image = resize_image(input_image)

            col1, col2 = st.columns(2)
            with col1:
                st.image(resized_image, caption="Uploaded Image")

            # Button to trigger fracture detection
            if st.button("Detect Fracture"):
                fracture_detected, annotated_image = detect_fracture(image_path)

                if fracture_detected:
                    st.success("Fracture detected")

                    # Resize annotated image and display
                    resized_annotated_image = resize_image(annotated_image)
                    with col2:
                        st.image(resized_annotated_image, caption="Detected Image")
                else:
                    st.warning("No fractures detected")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
