import os
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import KFold
from tensorflow.keras import layers, models, optimizers

# -----------------------------
# 1. Config
# -----------------------------
NUM_FOLDS = 5
EPOCHS = 5
BATCH_SIZE = 32
LOG_BASE_DIR = "logs/main3-5btensorboard+frac1to5"

# For reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# -----------------------------
# 2. Example dataset
# (replace with your dataset)
# -----------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)  # (N, 28, 28, 1)
x_test = np.expand_dims(x_test, -1)

# -----------------------------
# 3. Build model
# -----------------------------
def build_model(trainable_frac=0.05):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(96, 96, 3),
        include_top=False,
        weights="imagenet"
    )
    
    # Freeze most layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Unfreeze last N% of layers
    n_trainable = int(len(base_model.layers) * trainable_frac)
    for layer in base_model.layers[-n_trainable:]:
        layer.trainable = True
    
    model = models.Sequential([
        tf.keras.layers.Resizing(96, 96),
        tf.keras.layers.Conv2D(3, (3, 3), padding="same"),  # Ensure RGB input
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax")
    ])
    
    model.compile(
        optimizer=optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# -----------------------------
# 4. TensorBoard callback per fold
# -----------------------------
def get_tb_callback(fold):
    log_dir = os.path.join(LOG_BASE_DIR, f"fold_{fold}", datetime.now().strftime("%Y%m%d-%H%M%S"))
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# -----------------------------
# 5. Train with KFold
# -----------------------------
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
fold_results = {}

for fold, (train_idx, val_idx) in enumerate(kf.split(x_train, y_train), 1):
    print(f"\n===== Fold {fold} / {NUM_FOLDS} =====")
    start = time.time()
    
    model = build_model(trainable_frac=0.05)  # Fine-tune ~5%
    tb_callback = get_tb_callback(fold)
    
    history = model.fit(
        x_train[train_idx], y_train[train_idx],
        validation_data=(x_train[val_idx], y_train[val_idx]),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[tb_callback],
        verbose=2
    )
    
    # Evaluate on held-out test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    
    fold_results[fold] = {
        "train_loss": history.history["loss"][-1],
        "val_loss": history.history["val_loss"][-1],
        "train_acc": history.history["accuracy"][-1],
        "val_acc": history.history["val_accuracy"][-1],
        "test_loss": test_loss,
        "test_acc": test_acc,
        "time": time.time() - start
    }
    print(f"Fold {fold} done. Test acc: {test_acc:.4f}")

# -----------------------------
# 6. Report results
# -----------------------------
print("\n===== Final Results =====")
for fold, res in fold_results.items():
    print(f"Fold {fold}: "
          f"TrainAcc={res['train_acc']:.4f}, ValAcc={res['val_acc']:.4f}, "
          f"TestAcc={res['test_acc']:.4f}, Time={res['time']:.1f}s")

