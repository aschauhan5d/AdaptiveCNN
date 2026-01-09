
import tensorflow as tf
from model.adaptive_model import AdaptiveCNN

# Load dataset
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.cifar10.load_data()
x_train, x_val = x_train / 255.0, x_val / 255.0

# Initialize adaptive model
adaptive = AdaptiveCNN((32, 32, 3), 10)
adaptive.add_conv_block()

best_loss = float('inf')
threshold = 0.01
max_steps = 5

for step in range(max_steps):
    print(f"\nTraining Step {step+1}")
    model = adaptive.build_model()

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=3,
        batch_size=64,
        verbose=1
    )

    val_loss = history.history['val_loss'][-1]
    print(f"Validation Loss: {val_loss:.4f}")

    if val_loss < best_loss - threshold:
        best_loss = val_loss
        print("Improvement detected.")
    else:
        print("No improvement → Adding new Conv layer.")
        adaptive.add_conv_block()
