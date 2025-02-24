import os
import streamlit as st

# --- MAIN TITLE ---
st.title("Automatic Vehicle License Plate Recognition", anchor=False)

# --- HERO SECTION ---
col1, col2 = st.columns(2, gap="large", vertical_alignment="center")

with col1:
    img_path = os.path.join(os.getcwd(), "assets", "anpr.png")
    st.image(img_path, use_container_width=True)

with col2:
    st.write("### AI-Powered License Plate Recognition 🚗📸")
    st.write(
        """
        Welcome to our cutting-edge **Automatic Vehicle License Plate Recognition (ANPR) system**.  
        Our AI-driven technology enables **real-time detection, recognition, and logging** of vehicle license plates with high accuracy.  
        
        🚀 **Key Features:**
        - **Real-time vehicle detection** using deep learning  
        - **High-accuracy OCR** for license plate recognition  
        - **Fast and scalable** system for traffic monitoring and security  
        - **Seamless integration** with existing surveillance and access control systems  
        
        Our ANPR system is ideal for **traffic management, law enforcement, toll collection, and smart city applications**.
        """
    )