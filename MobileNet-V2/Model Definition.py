# Device Setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    device = '/GPU:0'
    print("GPU is available. Using GPU.")
else:
    device = '/CPU:0'
    print("GPU not found. Using CPU.")

print(f"Using device: {device}\n")
tf.debugging.set_log_device_placement(True)  # Logs ops placement

# Build Model with MobileNetV2
with tf.device(device):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze convolutional base

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),  # Preferred over Flatten
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(class_names), activation='softmax')  # Dynamic class count
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
