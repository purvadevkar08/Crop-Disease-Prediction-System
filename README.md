# 🌿 Potato Leaf Disease Detection and Pesticide Recommendation System

An AI-powered application that analyzes potato leaf images, predicts possible diseases using a pretrained CLIP model, and provides pesticide suggestions and crop management advice.

## 📌 Project Overview

This project combines computer vision and natural language generation to support potato leaf disease identification.

Users upload a potato leaf image through a Streamlit web application. The system predicts one of four classes and displays prediction probabilities, rule-based pesticide suggestions, and AI-generated management advice.

## 🚀 Key Features

- Upload potato leaf images in JPG, JPEG, or PNG format.
- Predict one of four potato leaf classes.
- Display prediction probabilities for each class.
- Visualize prediction scores with a bar chart.
- Provide rule-based pesticide suggestions and care tips.
- Generate additional management advice using FLAN-T5.
- Use pretrained AI models without requiring local checkpoint files.

## 🧠 Models Used

### 1. CLIP
- Model: `openai/clip-vit-base-patch32`
- Purpose: Compare the uploaded leaf image with text descriptions of the supported classes.

### 2. FLAN-T5
- Model: `google/flan-t5-small`
- Purpose: Generate text-based crop management advice.

**Note:** The application currently uses pretrained Hugging Face models. Its predictions and recommendations should be validated before practical agricultural use.

## 🛠️ Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- CLIP
- FLAN-T5
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Pillow
- Jupyter Notebook

## 🌱 Supported Classes

- Potato Early Blight
- Potato Late Blight
- Potato Healthy
- Potato Bacterial Spot

## 📂 Project Structure

```text
Crop-Disease-Prediction-System/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── CropDisease PredictionAnd pesticidesRecommendedSystem.ipynb
└── mapping/
    └── pesticide_map.csv