import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==============================
# SETTINGS
# ==============================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 5

TRAIN_DIR = "dataset/train"
VALIDATION_DIR = "dataset/validation"

# ==============================
# LOAD DATASET
# ==============================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_dataset.class_names

print("\nClasses:")
print(class_names)

# ==============================
# PREPROCESSING
# ==============================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.map(
    lambda x, y: (preprocess_input(x), y)
).prefetch(AUTOTUNE)

validation_dataset = validation_dataset.map(
    lambda x, y: (preprocess_input(x), y)
).prefetch(AUTOTUNE)

# ==============================
# MOBILENETV2
# ==============================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers
base_model.trainable = False

# ==============================
# CLASSIFICATION MODEL
# ==============================

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(len(class_names), activation="softmax")
])

# ==============================
# COMPILE
# ==============================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==============================
# TRAIN
# ==============================

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)

# ==============================
# SAVE MODEL
# ==============================

model.save("fashion_classifier_mobilenetv2.keras")

print("\n==============================")
print("CLASSIFICATION TRAINING DONE")
print("==============================")
print("Classes:", class_names)
print("Model saved:")
print("fashion_classifier_mobilenetv2.keras")