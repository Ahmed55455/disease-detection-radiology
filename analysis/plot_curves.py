import matplotlib.pyplot as plt

epochs = list(range(1, 9))
train_loss = [0.406, 0.206, 0.165, 0.132, 0.119, 0.098, 0.088, 0.077]
val_loss   = [0.172, 0.158, 0.170, 0.123, 0.135, 0.133, 0.126, 0.120]
train_acc  = [0.842, 0.911, 0.927, 0.940, 0.948, 0.956, 0.960, 0.964]
val_acc    = [0.929, 0.926, 0.942, 0.946, 0.949, 0.954, 0.951, 0.953]

fig, ax = plt.subplots(1, 2, figsize=(11, 4))

ax[0].plot(epochs, train_loss, marker="o", label="Train")
ax[0].plot(epochs, val_loss, marker="o", label="Validation")
ax[0].set_title("Loss (EfficientNet-B0)")
ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss"); ax[0].legend()

ax[1].plot(epochs, train_acc, marker="o", label="Train")
ax[1].plot(epochs, val_acc, marker="o", label="Validation")
ax[1].set_title("Accuracy (EfficientNet-B0)")
ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Accuracy"); ax[1].legend()

plt.tight_layout()
plt.savefig("analysis/training_curves_efficientnet_b0.png", dpi=150)
plt.show()
