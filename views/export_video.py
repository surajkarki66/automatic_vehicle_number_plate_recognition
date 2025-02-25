import streamlit as st
import csv
import pandas as pd
import tempfile

from utils import interpolate_bounding_boxes, post_processing_video

def process_video(csv_path, video_path, output_path):
    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        data = list(reader)
    
    interpolated_data = interpolate_bounding_boxes(data)
    fieldnames = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
    temp_csv_path = tempfile.mktemp(suffix=".csv")

    with open(temp_csv_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(interpolated_data)

    results = pd.read_csv(temp_csv_path)
    new_output_path = post_processing_video(video_path, results, output_path)
    return new_output_path

# Streamlit UI
st.title("License Plate Detection & Video Processing")
st.caption("Upload the license plate CSV obtained from the License Plate Recognition page and the original video you want to analyze.")  # Caption added

uploaded_csv = st.file_uploader("Upload CSV File", type=["csv"])
uploaded_video = st.file_uploader("Upload Video File", type=["mp4"])

if uploaded_csv and uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as csv_tmp:
        csv_tmp.write(uploaded_csv.getvalue())
        csv_path = csv_tmp.name
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as video_tmp:
        video_tmp.write(uploaded_video.getvalue())
        video_path = video_tmp.name
    
    output_path = tempfile.mktemp(suffix=".mp4")
    
    # Show loading spinner while processing
    with st.spinner("Processing video... This may take a while ⏳"):
        new_output = process_video(csv_path, video_path, output_path)
    
    st.success("Processing complete! 🎉")
    st.video(new_output)
