import pandas as pd
from fpdf import FPDF
from datetime import datetime
import os

class ReportGenerator:
    @staticmethod
    def generate_csv(detections, csv_path):
        df = pd.DataFrame(detections)
        if df.empty:
            df = pd.DataFrame(columns=["id", "severity", "confidence", "area_px", "est_area_sqm", "frame"])
        df.to_csv(csv_path, index=False)
        return csv_path

    @staticmethod
    def generate_pdf(detections, road_name, media_name, pdf_path, preview_image_path=None):
        pdf = FPDF()
        pdf.add_page()

        # Header
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_text_color(20, 50, 100)
        pdf.cell(0, 12, "PaveVision AI - Road Inspection Report", ln=True, align="C")
        pdf.ln(4)

        # Metadata
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(40, 40, 40)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pdf.cell(0, 7, f"Inspection Date & Time : {timestamp}", ln=True)
        pdf.cell(0, 7, f"Road / Location Name   : {road_name}", ln=True)
        pdf.cell(0, 7, f"Analyzed Media File    : {media_name}", ln=True)
        pdf.ln(5)

        # Summary Metrics
        total = len(detections)
        high = sum(1 for d in detections if d["severity"] == "High")
        med = sum(1 for d in detections if d["severity"] == "Medium")
        low = sum(1 for d in detections if d["severity"] == "Low")
        total_area = round(sum(d["est_area_sqm"] for d in detections), 2)

        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 9, "1. Executive Summary", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 7, f"- Total Unique Potholes Detected : {total}", ln=True)
        pdf.cell(0, 7, f"- High Severity (Critical)       : {high}", ln=True)
        pdf.cell(0, 7, f"- Medium Severity (Moderate)     : {med}", ln=True)
        pdf.cell(0, 7, f"- Low Severity (Minor)           : {low}", ln=True)
        pdf.cell(0, 7, f"- Est. Total Patching Area       : {total_area} sq. meters", ln=True)
        pdf.ln(5)

        # Annotated Image Preview
        if preview_image_path and os.path.exists(preview_image_path):
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 9, "2. Visual Detection Snapshot", ln=True)
            pdf.image(preview_image_path, w=170)
            pdf.ln(5)

        # Detailed Log Table
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 9, "3. Detailed Pothole Log", ln=True)
        pdf.set_font("Helvetica", "B", 10)
        headers = ["Pothole ID", "Severity", "Confidence", "Est. Area (sqm)", "Frame #"]
        col_widths = [30, 35, 35, 45, 30]

        for h, w in zip(headers, col_widths):
            pdf.cell(w, 8, h, border=1, align="C")
        pdf.ln()

        pdf.set_font("Helvetica", "", 10)
        if not detections:
            pdf.cell(sum(col_widths), 8, "No potholes detected on this road segment.", border=1, align="C")
            pdf.ln()
        else:
            for d in detections:
                pdf.cell(col_widths[0], 8, str(d["id"]), border=1, align="C")
                pdf.cell(col_widths[1], 8, d["severity"], border=1, align="C")
                pdf.cell(col_widths[2], 8, f"{d['confidence']}%", border=1, align="C")
                pdf.cell(col_widths[3], 8, str(d["est_area_sqm"]), border=1, align="C")
                pdf.cell(col_widths[4], 8, str(d["frame"]), border=1, align="C")
                pdf.ln()

        pdf.output(pdf_path)
        return pdf_path