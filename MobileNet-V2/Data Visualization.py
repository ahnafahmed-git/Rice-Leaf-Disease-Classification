# Reverse MobileNetV2 preprocessing: [-1, 1] → [0, 255]
def restore_image(img):
    img = (img + 1.0) * 127.5
    return tf.cast(tf.clip_by_value(img, 0, 255), tf.uint8)

# Peek into a batch
for images, labels in train_ds.take(1):
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
print('\n')

# Map class indices back to names
index_to_label = {v: k for k, v in label_to_index.items()}

# Collect 1 image per class
samples_per_class = {label: [] for label in range(len(class_names))}

for image, label in train_ds.unbatch():
    label_idx = tf.argmax(label).numpy()
    if len(samples_per_class[label_idx]) < 1:
        samples_per_class[label_idx].append(image)
    if all(len(v) == 1 for v in samples_per_class.values()):
        break  # Stop once we have 1 per class

# Flatten into a single list of (image, label)
sample_images = []
sample_labels = []
for label_idx, imgs in samples_per_class.items():
    sample_images.append(imgs[0])
    sample_labels.append(index_to_label[label_idx])

# Plot 3×2 grid with titles
plt.figure(figsize=(9, 6))
plt.suptitle("Data Visualization – 1 Sample per Class", fontsize=16, fontweight='bold')

for i in range(len(sample_images)):
    plt.subplot(2, 3, i + 1)
    restored_img = restore_image(sample_images[i])
    plt.imshow(restored_img)
    plt.title(sample_labels[i], fontsize=10)
    plt.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()
print('\n')
