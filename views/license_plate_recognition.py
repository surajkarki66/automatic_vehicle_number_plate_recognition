import os

import streamlit as st

import cv2
import numpy as np
import tempfile

from ultralytics import YOLO

from sort.sort import Sort
from utils import get_car, read_license_plate, write_csv

import torch
# Temporary fix for an error
torch.classes.__path__ = []

# Get the current working directory
BASE_PATH = os.getcwd()

# Define absolute paths for the models inside the 'assets' folder
COCO_MODEL_PATH = os.path.join(BASE_PATH, 'assets', 'yolov8n.pt')
LICENSE_PLATE_MODEL_PATH = os.path.join(BASE_PATH, 'assets', 'best.pt')

# Cache model loading for efficiency
@st.cache_resource
def load_models():
    mot_tracker = Sort()
    coco_model = YOLO(COCO_MODEL_PATH)  # Pre-trained COCO model
    license_plate_detector = YOLO(LICENSE_PLATE_MODEL_PATH)  # Custom license plate detector
    return coco_model, license_plate_detector, mot_tracker

# Load models once
coco_model, license_plate_detector, mot_tracker = load_models()
# Streamlit UI
st.title("🚗 License Plate Recognition System")
st.write("Upload a video to detect vehicles and recognize license plates.")

st.markdown(
    '[📥 Download Sample Video](https://drive.google.com/file/d/1pxgfMnACjBdkQ4JWm8pj8wE2pimEChCG/view?usp=sharing)', 
    unsafe_allow_html=True
)

# File uploader
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file:
    if st.button("Run Recognition"):
        # Initialize progress bar
        progress_bar = st.progress(0)
        progress_status = st.empty()

        progress_status.text("🔄 Loading video...")

        # Save uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total frame count
        processed_frames = 0  # Initialize processed frames counter

        results = {}
        vehicles = [2, 3, 5, 7]  # Car, Motorcycle, Bus, Truck classes in COCO dataset

        frame_nmr = 0
        ret = True

        progress_status.text("⏳ Processing video and extracting license plate information...")

        while ret:
            ret, frame = cap.read()
            if not ret:
                break

            results[frame_nmr] = {}

            # Detect vehicles
            detections = coco_model(frame)[0]
            detections_ = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicles:
                    detections_.append([x1, y1, x2, y2, score])

            # Track vehicles
            track_ids = mot_tracker.update(np.asarray(detections_))

            # Detect license plates
            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                # Assign license plate to a vehicle
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                if car_id != -1:
                    # Crop license plate
                    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                    # Convert to grayscale & apply thresholding
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                    # Read license plate number
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                    if license_plate_text is not None:
                        results[frame_nmr][car_id] = {
                            'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                            'license_plate': {
                                'bbox': [x1, y1, x2, y2],
                                'text': license_plate_text,
                                'bbox_score': score,
                                'text_score': license_plate_text_score
                            }
                        }

            # Update progress bar
            processed_frames += 1
            progress_percent = int((processed_frames / total_frames) * 100)
            progress_bar.progress(progress_percent)
            progress_status.text(f"⏳ Processing video: {progress_percent}% complete...")

            frame_nmr += 1

        cap.release()

        # Save results to CSV
        csv_filename = "license_plate_results.csv"
        write_csv(results, csv_filename)

        # Ensure progress bar reaches 100%
        progress_bar.progress(100)
        progress_status.text("✅ Processing complete! Download your results below.")

        # Allow user to download CSV file
        st.download_button(
            label="📥 Download CSV",
            data=open(csv_filename, "rb").read(),
            file_name=csv_filename,
            mime="text/csv"
        )
