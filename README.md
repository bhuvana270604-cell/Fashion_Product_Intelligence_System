# 👕 Fashion Product Intelligence System

## 📌 Project Overview

The Fashion Product Intelligence System is a deep learning based application that identifies the category of a fashion product from an uploaded image.

The system uses MobileNetV2 Transfer Learning for image classification and provides the predicted product category along with the confidence score through a Streamlit web application.

## 🎯 Objective

To develop an image classification system that can automatically identify fashion product categories from product images.

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- NumPy
- Pandas
- Pillow
- Streamlit
- Git & GitHub

## 🤖 Machine Learning Model

The project uses **MobileNetV2 Transfer Learning**.

The pretrained MobileNetV2 base is used for feature extraction, with additional classification layers for fashion product classification.

## 📂 Product Categories

The classification model predicts five categories:

1. Casual Shoes
2. Shirts
3. Socks
4. Sports Shoes
5. Tshirts

## 📊 Model Performance

- Training Accuracy: **97.40%**
- Validation Accuracy: **95.89%**

## 🚀 Streamlit Application

The trained model is integrated into a Streamlit web application.

Users can upload a JPG, JPEG, or PNG fashion product image, and the application displays:

- Uploaded product image
- Predicted product category
- Prediction confidence

## 🌐 Live Demo

[Open Fashion Product Intelligence System](https://fashionappuctintelligencesystem-sqwwxsi8jebfyos5sgk5csv.streamlit.app/)

## 📁 Project Structure

```text
Fashion_Product_Intelligence_System/
│
├── app.py
├── train_classifier.py
├── test_classifier.py
├── prepare_classification_dataset.py
├── split_dataset.py
├── image_feature_extraction.py
├── image_similarity.py
├── similarity_recommendation.py
├── fashion_classifier_mobilenetv2.keras
├── classification_dataset.csv
├── image_feature_metadata.csv
├── image_features.npy
├── requirements.txt
└── README.md