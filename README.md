# 🚗 License Plate Recognition System

This project is a License Plate Recognition System that detects vehicles and extracts license plate information from uploaded videos using deep learning models. The system utilizes YOLOv8 for vehicle detection and a custom YOLO model for license plate detection. The license plate format used in this project follows the UK style.

## 📌 Features
- Detects vehicles (cars, motorcycles, buses, and trucks) in video footage
- Identifies and extracts license plate numbers in UK format
- Tracks vehicles across frames
- Outputs detected license plate information as a CSV file
- User-friendly web interface built with Streamlit

## 🛠️ Tech Stack
- **Python**
- **Streamlit** (for UI)
- **OpenCV** (for image processing)
- **YOLOv8** (for vehicle detection)
- **Custom YOLO Model** (for license plate detection)
- **SORT Algorithm** (for tracking vehicles)
- **Torch & Ultralytics YOLO** (for deep learning)

## 📦 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/surajkarki66/automatic_vehicle_number_plate_recognition
cd automatic_vehicle_number_plate_recognition
```

### 2️⃣ Install Dependencies (Using `uv` Instead of `pip`)
We use `uv` for faster dependency management.


#### Install required packages:
```bash
uv sync
```

## 🚀 Running the Application
Start the Streamlit app:
```bash
uv run streamlit run streamlit_app.py 
```

## 📄 Usage Instructions
1. Open the web app using the provided Streamlit link.
2. **Download a sample video** if needed:
   [📥 Download Sample Video](https://www.pexels.com/video/cars-on-the-road-3158823/)
3. Upload a video file (`.mp4`, `.avi`, `.mov`).
4. Click the **Run Recognition** button to start processing.
5. Wait for detection and tracking to complete.
6. Download the extracted license plate information as a CSV file.

## ✨ Contributing
Feel free to submit pull requests, report issues, or suggest improvements!

---
🚀 Happy coding and enjoy building intelligent systems! 🚗

