import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50

DATA_PATHS = {
    "train_normal": "./data/archive/chest_xray/train/NORMAL",
    "train_pneumonia": "./data/archive/chest_xray/train/PNEUMONIA",
    "test_normal": "./data/archive/chest_xray/test/NORMAL",
    "test_pneumonia": "./data/archive/chest_xray/test/PNEUMONIA",
}

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0005 

data_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

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

train_images, train_labels = [], []
test_images, test_labels = [], []

train_normal, labels_normal = load_images_from_directory(DATA_PATHS["train_normal"], label=0)
train_pneumonia, labels_pneumonia = load_images_from_directory(DATA_PATHS["train_pneumonia"], label=1)

train_images.extend(train_normal + train_pneumonia)
train_labels.extend(labels_normal + labels_pneumonia)

test_normal, labels_test_normal = load_images_from_directory(DATA_PATHS["test_normal"], label=0)
test_pneumonia, labels_test_pneumonia = load_images_from_directory(DATA_PATHS["test_pneumonia"], label=1)

test_images.extend(test_normal + test_pneumonia)
test_labels.extend(labels_test_normal + labels_test_pneumonia)

train_images = np.array(train_images, dtype="float32") / 255.0
test_images = np.array(test_images, dtype="float32") / 255.0
train_labels = np.array(train_labels, dtype="int")
test_labels = np.array(test_labels, dtype="int")

train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42, stratify=train_labels
)

train_labels = to_categorical(train_labels, num_classes=2)
val_labels = to_categorical(val_labels, num_classes=2)
test_labels = to_categorical(test_labels, num_classes=2)

def train_and_evaluate(model, model_name):
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=LEARNING_RATE),
        metrics=['accuracy']
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    history = model.fit(
        data_gen.flow(train_images, train_labels, batch_size=BATCH_SIZE),
        validation_data=(val_images, val_labels),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, reduce_lr]
    )

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"🔹 Acurácia do modelo {model_name} no conjunto de teste: {test_acc * 100:.2f}%")

    model.save(f"{model_name}.keras")
    print(f"✅ Modelo salvo como {model_name}.keras")

def build_custom_cnn():
    model = Sequential([
        Conv2D(64, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(256, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(2, activation='softmax')
    ])
    return model

def build_vgg16():
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    for layer in base_model.layers[-4:]:
        layer.trainable = True
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])
    return model

def build_resnet50():
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    for layer in base_model.layers[-4:]:
        layer.trainable = True
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])
    return model

train_and_evaluate(build_custom_cnn(), "custom_cnn")
train_and_evaluate(build_vgg16(), "vgg16")
train_and_evaluate(build_resnet50(), "resnet50")