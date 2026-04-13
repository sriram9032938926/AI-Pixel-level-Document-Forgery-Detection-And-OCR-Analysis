import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image

from src.pdf_utils import pdf_bytes_to_images
from src.premium_predict import PremiumForgerySystem

st.set_page_config(page_title="Premium Document Forgery Detection", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_system():
    return PremiumForgerySystem(
        classifier_path=os.path.join(BASE_DIR, "models", "classifier_best.pth"),
        segmenter_path=os.path.join(BASE_DIR, "models", "segformer_best.pth")
    )


system = load_system()

st.title("AI-Based Document Forgery Detection and OCR Analysis")
st.write("Upload a document image or PDF to detect visual tampering and text-level inconsistencies.")

uploaded_file = st.file_uploader(
    "Upload document",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp", "pdf"]
)

if uploaded_file is not None:
    file_ext = uploaded_file.name.split(".")[-1].lower()

    # =========================================================
    # IMAGE INPUT
    # =========================================================
    if file_ext != "pdf":
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        with st.spinner("Analyzing image..."):
            result = system.analyze(image_bgr)

        visual = result["visual"]
        display_label = "Fake-like" if visual["label"] == "Fake" else "Real-like"

        st.subheader("Summary")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Document Similarity", display_label)
        c2.metric("Confidence", f"{visual['confidence'] * 100:.2f}%")
        c3.metric("Tampered Area", f"{visual['tampered_percent']:.2f}%")
        c4.metric("Overall Forgery Risk", result["final_risk"])
        c5.metric("Fraud Score", f"{result['final_score']:.2f}")

        if result["final_risk"] == "Low":
            st.success("Overall system assessment: low forgery risk.")
        elif result["final_risk"] == "Medium":
            st.warning("Overall system assessment: medium forgery risk. Manual review recommended.")
        else:
            st.error("Overall system assessment: high forgery risk detected.")

        st.subheader("Visual Forgery Localization")
        v1, v2, v3 = st.columns(3)
        v1.image(image_np, caption="Original", use_container_width=True)
        v2.image(
            (visual["mask"] * 255).astype(np.uint8),
            caption="Predicted Mask",
            use_container_width=True
        )
        v3.image(
            visual["overlay"],
            caption="Heatmap Overlay",
            use_container_width=True
        )

        st.subheader("OCR Output")
        st.text_area("Extracted Text", result["ocr"]["full_text"], height=220)

        st.subheader("Extracted Fields")
        st.json(result["fields"])

        st.subheader("Text-Level Findings")
        if result["text_analysis"]["issues"]:
            for issue in result["text_analysis"]["issues"]:
                st.warning(issue)
        else:
            st.success("No strong text-level inconsistencies found.")

        st.metric("Text Risk Score", result["text_analysis"]["risk_score"])

        st.subheader("Final Explanation")
        st.write(result["explanation"])

    # =========================================================
    # PDF INPUT
    # =========================================================
    else:
        with st.spinner("Converting PDF pages to images..."):
            pdf_bytes = uploaded_file.read()
            pages_bgr = pdf_bytes_to_images(pdf_bytes, dpi=200)

        st.subheader(f"PDF contains {len(pages_bgr)} page(s)")

        page_summaries = []

        for idx, page_bgr in enumerate(pages_bgr, start=1):
            with st.spinner(f"Analyzing page {idx}..."):
                result = system.analyze(page_bgr)

            visual = result["visual"]
            display_label = "Fake-like" if visual["label"] == "Fake" else "Real-like"
            page_rgb = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2RGB)

            st.markdown(f"## Page {idx}")

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Document Similarity", display_label)
            c2.metric("Confidence", f"{visual['confidence'] * 100:.2f}%")
            c3.metric("Tampered Area", f"{visual['tampered_percent']:.2f}%")
            c4.metric("Overall Forgery Risk", result["final_risk"])
            c5.metric("Fraud Score", f"{result['final_score']:.2f}")

            v1, v2, v3 = st.columns(3)
            v1.image(page_rgb, caption=f"Original - Page {idx}", use_container_width=True)
            v2.image(
                (visual["mask"] * 255).astype(np.uint8),
                caption=f"Predicted Mask - Page {idx}",
                use_container_width=True
            )
            v3.image(
                visual["overlay"],
                caption=f"Heatmap Overlay - Page {idx}",
                use_container_width=True
            )

            with st.expander(f"Page {idx} OCR + Analysis"):
                st.text_area(
                    f"Extracted Text - Page {idx}",
                    result["ocr"]["full_text"],
                    height=180,
                    key=f"ocr_{idx}"
                )

                st.subheader("Extracted Fields")
                st.json(result["fields"])

                st.subheader("Text-Level Findings")
                if result["text_analysis"]["issues"]:
                    for issue in result["text_analysis"]["issues"]:
                        st.warning(issue)
                else:
                    st.success("No strong text-level inconsistencies found.")

                st.metric("Text Risk Score", result["text_analysis"]["risk_score"])

                st.subheader("Final Explanation")
                st.write(result["explanation"])

            page_summaries.append({
                "page": idx,
                "document_similarity": display_label,
                "confidence": round(visual["confidence"] * 100, 2),
                "tampered_percent": round(visual["tampered_percent"], 2),
                "overall_forgery_risk": result["final_risk"],
                "fraud_score": round(result["final_score"], 2)
            })

        st.subheader("PDF Summary")
        st.json(page_summaries)