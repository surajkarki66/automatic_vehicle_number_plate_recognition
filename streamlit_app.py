import streamlit as st


# --- PAGE SETUP ---
home_page = st.Page(
    "views/home.py",
    title="Home",
    icon=":material/home:",
    default=True,
)

license_plate_recognition_page = st.Page(
    "views/license_plate_recognition.py",
    title="License Plate Recognition",
    icon=":material/qr_code_scanner:",
)


# --- NAVIGATION SETUP [WITH SECTIONS]---
pg = st.navigation(
    {
        "Info": [home_page],
        "Analysis": [license_plate_recognition_page],
    }
)

# --- SHARED ON ALL PAGES ---
st.logo("assets/anpr.png", size="large")
st.sidebar.image("assets/anpr.png")
st.sidebar.markdown("Made with ❤️ by Suraj Karki")


# --- RUN NAVIGATION ---
pg.run()

