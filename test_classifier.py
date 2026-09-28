import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "fashion_classifier_mobilenetv2.keras"

IMAGE_SIZE = (224, 224)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Class names — same order used during training
class_names = [
    "Casual Shoes",
    "Shirts",
    "Socks",
    "Sports Shoes",
    "Tshirts"
]

# Ask for image path
image_path = input("Enter test image path: ")

# Load image
image = tf.keras.utils.load_img(
    image_path,
    target_size=IMAGE_SIZE
)

# Convert image to array
image_array = tf.keras.utils.img_to_array(image)

# Add batch dimension
image_array = np.expand_dims(image_array, axis=0)

# MobileNetV2 preprocessing
image_array = preprocess_input(image_array)

# Prediction
predictions = model.predict(image_array, verbose=0)

predicted_index = np.argmax(predictions[0])
predicted_class = class_names[predicted_index]
confidence = predictions[0][predicted_index] * 100

print("\n==============================")
print("IMAGE CLASSIFICATION RESULT")
print("==============================")
print("Predicted Product:", predicted_class)
print(f"Confidence: {confidence:.2f}%")
print("==============================")