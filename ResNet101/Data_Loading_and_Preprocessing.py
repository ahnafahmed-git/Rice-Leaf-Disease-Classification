import os, zipfile
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet import preprocess_input

# Mount drive and unzip dataset
from google.colab import drive
drive.mount('/content/drive')

file_path = '/content/drive/MyDrive/content/rice leaf diseases dataset-2.zip'
extract_path = '/content/rice_disease_data'
with zipfile.ZipFile(file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

data_dir = os.path.join(extract_path, 'RiceLeafsDisease')
train_dir = os.path.join(data_dir, 'train')
val_dir   = os.path.join(data_dir, 'validation')

print("Extracted folders:", os.listdir(extract_path))

# Label setup
class_names = sorted(os.listdir(train_dir))
label_to_index = {name: idx for idx, name in enumerate(class_names)}

print("Sample count per class:")
for label in class_names:
    class_dir = os.path.join(train_dir, label)
    print(f"{label}: {len(os.listdir(class_dir))} images")

# Collect training paths
train_image_paths, train_labels = [], []
for label in class_names:
    class_dir = os.path.join(train_dir, label)
    for fname in os.listdir(class_dir):
        train_image_paths.append(os.path.join(class_dir, fname))
        train_labels.append(label_to_index[label])

train_image_paths = np.array(train_image_paths)
train_labels = np.array(train_labels)

# Split train/test
train_paths, test_paths, train_labels, test_labels = train_test_split(
    train_image_paths, train_labels, test_size=0.2, stratify=train_labels, random_state=42
)

# Validation set
val_image_paths, val_labels = [], []
for label in class_names:
    class_dir = os.path.join(val_dir, label)
    for fname in os.listdir(class_dir):
        val_image_paths.append(os.path.join(class_dir, fname))
        val_labels.append(label_to_index[label])

val_paths = np.array(val_image_paths)
val_labels = np.array(val_labels)

# Image preprocessing
IMG_SIZE = (224, 224)

def process_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = preprocess_input(image)  # ResNet101 preprocessing
    return image, tf.one_hot(label, depth=len(class_names))

def build_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(buffer_size=len(paths))
    ds = ds.batch(128).prefetch(tf.data.AUTOTUNE)
    return ds

# Build datasets
train_ds = build_dataset(train_paths, train_labels, training=True)
val_ds   = build_dataset(val_paths, val_labels)
test_ds  = build_dataset(test_paths, test_labels)

print("\nDataset Split Summary:")
print(f"Train samples: {len(train_paths)}")
print(f"Validation samples: {len(val_paths)}")
print(f"Test samples: {len(test_paths)}")
