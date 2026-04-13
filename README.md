# AI-Pixel-level-Document-Forgery-Detection-And-OCR-Analysis

An end-to-end AI system that performs **pixel-level forgery detection**, **document classification**, and **text-level fraud analysis** on document images and PDFs.

This project combines **Computer Vision + Deep Learning + OCR + Rule-Based Reasoning** to build a real-world **document verification pipeline** used in domains like KYC, banking, and government validation systems.

---

## 🧠 What Makes This Project Special?

Unlike traditional systems, this project:

✔ Detects **exact tampered regions at pixel-level**  
✔ Combines **visual + textual fraud signals**  
✔ Provides **human-readable explanations**  
✔ Works on **real-world noisy documents**  
✔ Is designed as a **decision-support system**, not just a model  

---

## 🎯 Primary Use Cases

- **Banking & KYC Verification** 🏦  
- **Government Document Validation** 🏛️  
- **Certificate Verification** 🎓  
- **Insurance Fraud Detection** 🧾  
- **Digital Onboarding Systems** 📑  
---
## 🏗️ AI System Architecture

<img width="700" height="800" alt="AI document forgery detection workflow" src="https://github.com/user-attachments/assets/78183dc1-a2d1-4b71-9275-02e4d62e2e80" />

---
## ⚙️ How It Works

1. Upload document (Image or PDF)
2. Convert PDF → images
3. Classify document (Real-like / Fake-like)
4. Detect tampered regions (pixel-level)
5. Extract text using OCR
6. Extract structured fields
7. Apply fraud detection rules
8. Generate final fraud score and explanation
---
## 🚀 Key Features
- Real-time document analysis ⚡
- Parallel AI module execution 🧩
- Pixel-level forgery detection 🔍
- Reduced false positives via multi-modal fusion 🎯
- Explainable outputs (heatmaps, masks, scores) 📊
- Multi-page PDF support 📄
- Modular architecture 🛠️
----
## 🧰 Tech Stack
- **Programming Language:** Python  
- **Computer Vision:** OpenCV  
- **Deep Learning:** PyTorch  
- **Classification:** ResNet18  
- **Segmentation:** SegFormer  
- **OCR:** EasyOCR  
- **UI:** Streamlit  

---
## 📊 Sample Output

<img width="400" height="400" alt="Screenshot 2026-04-13 113232" src="https://github.com/user-attachments/assets/7a5e8943-dd9e-44a1-b8e8-c4ef852c890a" />



<img width="400" height="400" alt="Screenshot 2026-04-13 113304" src="https://github.com/user-attachments/assets/397e7836-de91-476f-9222-acec493bf12a" />

---
## ▶️ Run the Project

```bash
git clone https://github.com/your-username/repo-name.git
cd repo-name
pip install -r requirements.txt
streamlit run app.py
---
