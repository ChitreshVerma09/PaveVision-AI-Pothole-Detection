# 🛣️ PaveVision AI - Pothole Detection System

**PaveVision AI** is an automated road infrastructure inspection and pothole detection web application built using **Python, YOLOv8, OpenCV, and Streamlit**. It processes road photographs and dashcam videos to detect road potholes, filter out moving traffic, track unique defects across video frames, classify damage severity, and generate downloadable **PDF Engineering Reports** and **CSV Logs**.

---

## ✨ Key Features

- **Dual Media Support:** Analyze both static road images (`.jpg`, `.jpeg`, `.png`) and continuous dashcam/mobile videos (`.mp4`, `.avi`, `.mov`).
- **Smart Vehicle Auto-Filter:** Automatically masks moving cars, bikes, buses, and trucks so vehicles are never miscounted as potholes.
- **Duplicate-Free Video Counting:** Uses centroid-based multi-frame tracking to assign a **Unique ID** to every pothole, preventing duplicate counts across consecutive frames.
- **Color-Coded Severity Grading:** Classifies every detected pothole based on surface area ratio:
  - 🔴 **High / Deep Severity (`>= 3.8%` area):** Critical structural damage requiring immediate patching.
  - 🟡 **Medium Severity (`1.2% – 3.8%` area):** Moderate depression scheduled for routine maintenance.
  - 🟢 **Low / Little Severity (`< 1.2%` area):** Minor surface wear for routine monitoring.
- **Interactive Analytics Dashboard:** Features live frame-by-frame video scanning, KPI summary metrics, estimated patching area (m²), color-coded Altair bar charts, and severity filters.
- **One-Click Engineering Exports:** Download a formatted **PDF Inspection Report**, structured **CSV Data Log**, and **Annotated Video** without resetting the dashboard session.

---

## 🛠️ Tech Stack

- **Programming Language:** Python 3.10+
- **Deep Learning & Computer Vision:** Ultralytics YOLOv8, OpenCV (`opencv-python-headless`), NumPy
- **Frontend & Visualization:** Streamlit, Altair, Pillow
- **Automated Reporting:** FPDF2, Pandas

---

## 📂 Project Structure

```text
PaveVision-AI-Pothole-Detection/
├── app.py                  # Main Streamlit web dashboard & session state controller
├── detector.py             # YOLOv8 + OpenCV detection, vehicle filter & centroid tracker
├── report_generator.py     # Automated PDF engineering report & CSV log generator
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 🚀 Installation & How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ChitreshVerma09/PaveVision-AI-Pothole-Detection.git
   cd PaveVision-AI-Pothole-Detection
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application:**
   ```bash
   python -m streamlit run app.py
   ```
   *(Note: Base `yolov8n.pt` weights download automatically on first run if custom weights are not provided.)*
