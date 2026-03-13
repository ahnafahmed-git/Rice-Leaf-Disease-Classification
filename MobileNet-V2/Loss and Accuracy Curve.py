# Create subplots: 1 row, 2 columns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Loss Curve
ax1.plot(history.history["loss"], label="Train Loss", color='blue')
ax1.plot(history.history["val_loss"], label="Validation Loss", color='orange')
ax1.set_title("Loss Curve")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True)

# Accuracy Curve
ax2.plot(history.history["accuracy"], label="Train Accuracy", color='green')
ax2.plot(history.history["val_accuracy"], label="Validation Accuracy", color='red')
ax2.set_title("Accuracy Curve")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()