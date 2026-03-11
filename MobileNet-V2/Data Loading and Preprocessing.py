# Dataset Link: https://www.kaggle.com/datasets/dedeikhsandwisaputra/rice-leafs-disease-dataset
# Mount Drive to access the ZIP
drive.mount('/content/drive')

# Path to ZIP file in Drive
file_path = '/content/drive/MyDrive/Rice Disease Dataset.zip' # Corrected file path

# Check if the file exists before attempting to extract
if not os.path.exists(file_path):
    print(f"Error: The file was not found at {file_path}")
else:
    # Extract to local Colab VM (not Drive)
    extract_path = '/content/rice_disease_data'
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    data_dir = os.path.join(extract_path, 'RiceLeafsDisease')
    train_dir = os.path.join(data_dir, 'train')
    val_dir   = os.path.join(data_dir, 'validation')

    print("Extracted folders:", os.listdir(extract_path))
    print('\n')

    class_names = sorted(os.listdir(train_dir))

    # Build label-to-index mapping from class names
    label_to_index = {name: idx for idx, name in enumerate(class_names)}

    # Count samples per class in the train set
    print("Sample count per class:")
    for label in class_names:
        class_dir = os.path.join(train_dir, label)
        count = len(os.listdir(class_dir))
        print(f"{label}: {count} images")

    # Loading train image paths and labels
    train_image_paths = []
    train_labels = []

    for label in class_names:
        class_dir = os.path.join(train_dir, label)
        for fname in os.listdir(class_dir):
            fpath = os.path.join(class_dir, fname)
            train_image_paths.append(fpath)
            train_labels.append(label_to_index[label])

    # Convert to NumPy arrays
    train_image_paths = np.array(train_image_paths)
    train_labels = np.array(train_labels)
    print(f"\nTotal training samples: {len(train_image_paths)} (before split)")

    # Splitting 80% train 20% test
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        train_image_paths, train_labels, test_size=0.2, stratify=train_labels, random_state=42
    )

    # Loading validation set
    val_image_paths, val_labels = [], []

    for label in class_names:
        class_path = os.path.join(val_dir, label)
        for fname in os.listdir(class_path):
            fpath = os.path.join(class_path, fname)
            val_image_paths.append(fpath)
            val_labels.append(label_to_index[label])

    # Convert to NumPy arrays
    val_paths = np.array(val_image_paths)
    val_labels = np.array(val_labels)

    IMG_SIZE = (224, 224)

    # Preprocessing
    def process_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, IMG_SIZE) # Resizing each sample to 224x224
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image) # Preprocess image for MobileNetV2
        return image, tf.one_hot(label, depth=len(class_names))

    def build_dataset(paths, labels):
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(128).prefetch(tf.data.AUTOTUNE)
        return ds

    # Building dataset (train, test and validation)
    train_ds = build_dataset(train_paths, train_labels)
    val_ds   = build_dataset(val_paths, val_labels)
    test_ds  = build_dataset(test_paths, test_labels)

    print("\nDataset Split Summary:")
    print(f"Train samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    print(f"Test samples: {len(test_paths)}")
    print('\n')