# 🧠 AI Brain Disorder Detection & 3D MRI Visualization System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green)
![Plotly](https://img.shields.io/badge/Plotly-3D%20Visualization-purple)
![SQLite](https://img.shields.io/badge/SQLite-Database-blue)
![License](https://img.shields.io/badge/License-MIT-green)

### AI-Powered Brain MRI Analysis using Deep Learning with Interactive 2D & 3D Visualization

Developed by **Lokesh M M**

</div>

---

# 📌 Project Overview

The **AI Brain Disorder Detection & 3D MRI Visualization System** is a medical imaging application developed to analyze Brain MRI scans using Artificial Intelligence.

The system accepts MRI images or volumetric MRI (.nii) files, preprocesses them, performs AI-based classification, generates confidence scores, visualizes MRI slices in **2D**, reconstructs the complete brain in **3D**, and produces an automated medical report.

This project combines:

- Artificial Intelligence
- Medical Image Processing
- Deep Learning
- Computer Vision
- Flask Web Development
- 3D Medical Visualization

to provide an intelligent diagnosis support system.

---

# 🎯 Objectives

The major objectives of this project are:

- Detect brain abnormalities from MRI scans
- Differentiate between healthy and abnormal brains
- Predict multiple neurological disorders
- Visualize MRI slices in 2D
- Render complete MRI volume in 3D
- Generate confidence score
- Produce downloadable medical reports
- Assist radiologists with AI-powered diagnosis

---

# 🚀 Features

## 🧠 AI Disease Prediction

The model predicts:

- ✅ Normal Brain
- ✅ Brain Tumor
- ✅ Alzheimer's Disease
- ✅ Parkinson's Disease
- ✅ Stroke

---

## 📂 MRI Upload

Supports:

- JPG
- PNG
- JPEG
- NIfTI (.nii)
- NIfTI (.nii.gz)

---

## 🖼 2D MRI Viewer

Features:

- Slice visualization
- Zoom
- Brightness adjustment
- Contrast enhancement
- Tumor highlighting
- Image preprocessing preview

---

## 🌍 3D Brain Reconstruction

Interactive 3D visualization using MRI volume.

Features include:

- Brain rotation
- Zoom
- Slice navigation
- Volume rendering
- Surface rendering
- Tumor localization
- Interactive controls

---

## 📈 AI Prediction Dashboard

Displays

- Disease Name
- Confidence Score
- Risk Level
- Prediction Probability
- Model Accuracy
- Processing Time

---

## 📊 Probability Graph

Visualizes probability for every class.

Example:

Normal          ██████████ 95%

Tumor           ██ 2%

Stroke          █ 1%

Alzheimer       █ 1%

Parkinson       █ 1%

---

## 📑 Automated Medical Report

Generated automatically after prediction.

Contains:

- Patient Details
- MRI Information
- AI Prediction
- Confidence Score
- Risk Assessment
- Clinical Recommendation
- Date & Time

Downloadable as PDF.

---

# 🏗 System Architecture

```
          MRI Scan
              │
              ▼
      Image Upload Module
              │
              ▼
      Image Preprocessing
              │
              ▼
      Deep Learning Model
              │
      ┌───────┴────────┐
      ▼                ▼
2D Visualization   3D Visualization
      │                │
      └───────┬────────┘
              ▼
      AI Prediction Engine
              │
              ▼
 Confidence + Disease Class
              │
              ▼
     Medical Report Generator
```

---

# 🔍 Workflow

### Step 1

Upload MRI scan

↓

### Step 2

Preprocessing

- Resize
- Normalize
- Denoising
- Contrast Enhancement

↓

### Step 3

Deep Learning Prediction

↓

### Step 4

Disease Classification

↓

### Step 5

Generate Confidence Score

↓

### Step 6

Display 2D MRI

↓

### Step 7

Generate 3D Brain Model

↓

### Step 8

Generate Medical Report

---

# 🧠 Deep Learning Pipeline

Input MRI

↓

Preprocessing

↓

CNN / 3D CNN Model

↓

Feature Extraction

↓

Classification Layer

↓

Prediction

↓

Confidence Score

↓

Visualization

↓

Report Generation

---

# 🛠 Technologies Used

## Programming

- Python

## Backend

- Flask

## Frontend

- HTML
- CSS
- Bootstrap
- JavaScript

## AI Framework

- PyTorch

## Image Processing

- OpenCV
- Pillow
- NumPy

## Medical Imaging

- Nibabel

## Visualization

- Plotly
- Matplotlib

## Database

- SQLite

## PDF Generation

- ReportLab

---

# 📂 Project Structure

```
Brain_Disorder_Detection/

│
├── app.py
├── config.py
├── requirements.txt
│
├── models/
│      brain_model.pth
│
├── templates/
│      index.html
│      upload.html
│      result.html
│      dashboard.html
│
├── static/
│      css/
│      js/
│      images/
│
├── uploads/
│
├── reports/
│
├── database/
│      patients.db
│
├── utils/
│      preprocess.py
│      prediction.py
│      visualization.py
│      report.py
│
└── README.md
```

---

# 📊 Model Information

Deep Learning Framework

- PyTorch

Input Size

- 128 × 128 (2D)
- MRI Volume (.nii)

Output Classes

- Normal
- Brain Tumor
- Alzheimer's
- Parkinson's
- Stroke

Prediction

Softmax Classification

---

# 📈 Dataset

The project can be trained using Brain MRI datasets from Kaggle.

Supported MRI formats:

- JPG
- PNG
- JPEG
- NIfTI (.nii)
- NIfTI (.nii.gz)

---

# 💻 Installation

Clone Repository

```bash
git clone https://github.com/yourusername/Brain-Disorder-Detection.git
```

Move into folder

```bash
cd Brain-Disorder-Detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run application

```bash
python app.py
```

Open browser

```
http://127.0.0.1:5000
```

---

# 📸 Screens

- Home Page
- MRI Upload
- AI Prediction
- 2D Viewer
- 3D Brain Viewer
- Dashboard
- PDF Report

---

# 🔮 Future Enhancements

- Real-time MRI streaming
- Grad-CAM heatmaps
- Tumor segmentation
- Tumor volume calculation
- Multi-hospital integration
- Cloud deployment
- PACS integration
- Doctor Login
- Patient Portal
- Email Notification
- Mobile Application
- AI Chat Assistant
- Multi-language Support

---

# 🎓 Applications

- Hospitals
- Medical Colleges
- Research Centers
- Diagnostic Laboratories
- Healthcare AI
- Radiology Departments

---

# 📚 Research Areas

- Artificial Intelligence
- Medical Imaging
- Computer Vision
- Deep Learning
- MRI Analysis
- Healthcare Analytics

---

# 👨‍💻 Developed By

**Lokesh M M**

Malnad College of Engineering

Department of Computer Science & Engineering

AI & Machine Learning Enthusiast

---

# ⭐ Key Highlights

✅ AI-Based Brain Disease Detection

✅ 2D MRI Visualization

✅ Interactive 3D Brain Rendering

✅ Multi-Class Disease Prediction

✅ Confidence Score

✅ Automated Medical Report

✅ Flask Web Application

✅ PyTorch Deep Learning Model

✅ NIfTI MRI Support

✅ Real-Time Analysis Ready

---

# 📜 License

This project is developed for educational, research, and healthcare innovation purposes.

---

## 🌟 If you found this project useful, don't forget to ⭐ star the repository!
