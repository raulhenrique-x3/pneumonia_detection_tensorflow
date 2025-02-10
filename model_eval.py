import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

test_data_paths = {
    "test_normal": "./data/archive/chest_xray/test/NORMAL",
    "test_pneumonia": "./data/archive/chest_xray/test/PNEUMONIA"
}

IMG_SIZE = (224, 224)

def load_images_from_directory(directory, label):
    images = []
    labels = []
    for file in os.listdir(directory):
        img_path = os.path.join(directory, file)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img, IMG_SIZE)
            images.append(img)
            labels.append(label)
    return images, labels

test_images, test_labels = [], []

test_normal, labels_test_normal = load_images_from_directory(test_data_paths["test_normal"], label=0)
test_pneumonia, labels_test_pneumonia = load_images_from_directory(test_data_paths["test_pneumonia"], label=1)

test_images.extend(test_normal + test_pneumonia)
test_labels.extend(labels_test_normal + labels_test_pneumonia)

test_images = np.array(test_images, dtype="float32") / 255.0
test_labels = np.array(test_labels, dtype="int")
test_labels = to_categorical(test_labels, num_classes=2)

model_dir = "./models"

for model_file in os.listdir(model_dir):
    if model_file.endswith(".keras"):
        model_path = os.path.join(model_dir, model_file)
        print(f"\n🔹 Carregando modelo: {model_file}")
        model = load_model(model_path)
        
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
        print(f"✅ Acurácia do modelo {model_file}: {test_acc * 100:.2f}%")
