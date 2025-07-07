import os
import numpy as np
import pandas as pd
import matplotlib
# Use a non-interactive backend so the script also works on headless servers
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping, LearningRateScheduler,
                                        Callback)
from tensorflow.keras import regularizers

# =====================================================================================
# CONFIGURATION
# =====================================================================================
K_FOLDS    = 5
BATCH_SIZE = 8
IMG_SIZE   = (224, 224)
NUM_CLASSES = 5
SEED       = 42
OUTPUT_DIR = os.getcwd()
BASE_DIR   = os.path.join(os.getcwd(), 'train_samples')
MODEL_OUT_TEMPLATE = os.path.join(OUTPUT_DIR, 'cats_fold{fold}.keras')
FINAL_MODEL_PATH   = os.path.join(OUTPUT_DIR, 'cats.keras')

# Ensure reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Enable mixed precision for faster training on GPU
try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
except Exception:
    pass

# =====================================================================================
# CALLBACKS
# =====================================================================================
class RealTimePlot(Callback):
    """Realtime plot of accuracy/loss during training and validation."""
    def __init__(self, fold=None):
        super().__init__()
        self.fold = fold
        self.history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 4))
        plt.tight_layout()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k in self.history.keys():
            self.history[k].append(logs.get(k))

        # Loss subplot
        self.ax1.clear()
        self.ax1.plot(self.history['loss'], label='train_loss')
        self.ax1.plot(self.history['val_loss'], label='val_loss')
        self.ax1.set_title('Loss')
        self.ax1.legend()

        # Accuracy subplot
        self.ax2.clear()
        self.ax2.plot(self.history['accuracy'], label='train_acc')
        self.ax2.plot(self.history['val_accuracy'], label='val_acc')
        self.ax2.set_title('Accuracy')
        self.ax2.legend()

        # Render and save current figure
        self.fig.canvas.draw()
        plot_name = f'training_plot_fold_{self.fold if self.fold is not None else "full"}.png'
        self.fig.savefig(plot_name)

class PerClassAccuracy(Callback):
    """Print per-class validation accuracy at the end of each epoch."""
    def __init__(self, val_generator, class_indices):
        super().__init__()
        self.val_gen = val_generator
        # reverse lookup list
        self.class_names = [None] * len(class_indices)
        for name, idx in class_indices.items():
            self.class_names[idx] = name

    def on_epoch_end(self, epoch, logs=None):
        y_true, y_pred = [], []
        for i in range(len(self.val_gen)):
            x_batch, y_batch = self.val_gen[i]
            preds = self.model.predict(x_batch, verbose=0)
            y_true.extend(np.argmax(y_batch, axis=1))
            y_pred.extend(np.argmax(preds, axis=1))
        y_true, y_pred = np.array(y_true), np.array(y_pred)

        print(f"\n— Fold-epoch {epoch+1} per-class val accuracy:")
        for idx, name in enumerate(self.class_names):
            mask = y_true == idx
            acc = np.nan if mask.sum() == 0 else accuracy_score(y_true[mask], y_pred[mask])
            print(f"    {name:10s}: {acc*100:5.2f}%")

# =====================================================================================
# DATA PREPARATION – BUILD A SINGLE DATAFRAME TO AVOID LEAKAGE
# =====================================================================================
filepaths, labels = [], []
for breed in sorted(os.listdir(BASE_DIR)):
    breed_dir = os.path.join(BASE_DIR, breed)
    if not os.path.isdir(breed_dir):
        continue
    for fname in sorted(os.listdir(breed_dir)):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            filepaths.append(os.path.join(breed, fname))  # keep relative path from BASE_DIR
            labels.append(breed)

df = pd.DataFrame({
    'filename': filepaths,
    'class': labels
})

print("Total samples:", len(df))
print(df['class'].value_counts())

# =====================================================================================
# IMAGE DATA GENERATORS
# =====================================================================================
train_idg = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.02,
    zoom_range=0.1,
    brightness_range=(0.9, 1.1),
    fill_mode='nearest'
)
val_idg = ImageDataGenerator(rescale=1./255)

# =====================================================================================
# MODEL FACTORY
# =====================================================================================

def build_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    predictions = Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

# =====================================================================================
# LEARNING RATE SCHEDULE
# =====================================================================================

def lr_schedule(epoch, lr):
    if epoch < 10:
        return lr
    elif epoch < 20:
        return lr * 0.5
    elif epoch < 30:
        return lr * 0.1
    else:
        return lr * 0.01

lr_scheduler = LearningRateScheduler(lr_schedule, verbose=0)

# =====================================================================================
# TRAINING WITH STRATIFIED K-FOLD
# =====================================================================================
fold_accuracies = []
skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df['filename'], df['class'])):
    print("\n" + "="*80)
    print(f"STARTING FOLD {fold_idx+1}/{K_FOLDS}")
    print("="*80)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)

    train_gen = train_idg.flow_from_dataframe(
        train_df,
        directory=BASE_DIR,
        x_col='filename',
        y_col='class',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=SEED
    )

    val_gen = val_idg.flow_from_dataframe(
        val_df,
        directory=BASE_DIR,
        x_col='filename',
        y_col='class',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    # Compute class weights for the training split
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weights = dict(enumerate(class_weights))

    model, base_model = build_model()

    # Callbacks specific to this fold
    checkpoint_path = MODEL_OUT_TEMPLATE.format(fold=fold_idx+1)
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True,
                                 verbose=1)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    plot_cb   = RealTimePlot(fold=fold_idx+1)
    per_class_cb = PerClassAccuracy(val_generator=val_gen, class_indices=train_gen.class_indices)

    # ============= Stage 1: Head training =============
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy'])

    model.fit(train_gen,
              epochs=15,
              validation_data=val_gen,
              callbacks=[checkpoint, early_stop, plot_cb, per_class_cb],
              class_weight=class_weights,
              verbose=2)

    # ============= Stage 2: Partial fine-tuning =============
    for layer in base_model.layers[-10:]:
        layer.trainable = True

    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy'])

    model.fit(train_gen,
              epochs=20,
              validation_data=val_gen,
              callbacks=[checkpoint, early_stop, lr_scheduler, plot_cb, per_class_cb],
              class_weight=class_weights,
              verbose=2)

    # ============= Stage 3: Global fine-tuning =============
    for layer in base_model.layers:
        layer.trainable = True
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=5e-6),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy'])

    history = model.fit(train_gen,
                        epochs=20,
                        validation_data=val_gen,
                        callbacks=[checkpoint, early_stop, plot_cb, per_class_cb],
                        class_weight=class_weights,
                        verbose=2)

    best_val_acc = max(history.history.get('val_accuracy', [0]))
    fold_accuracies.append(best_val_acc)

    print(f"Finished Fold {fold_idx+1}. Best Val Accuracy: {best_val_acc:.4f}")

print("\n" + "#"*80)
print("CROSS-VALIDATION COMPLETE")
print("Validation accuracies per fold:", [f"{acc:.4f}" for acc in fold_accuracies])
print(f"Mean val accuracy: {np.mean(fold_accuracies):.4f}  |  Std: {np.std(fold_accuracies):.4f}")
print("#"*80)

# =====================================================================================
# OPTIONAL: TRAIN FINAL MODEL ON FULL DATASET USING BEST SETTINGS AND SAVE TO cats.keras
# =====================================================================================
print("\nTraining final model on the entire dataset…")
train_gen_full = train_idg.flow_from_dataframe(
    df,
    directory=BASE_DIR,
    x_col='filename',
    y_col='class',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=SEED
)

class_weights_full = class_weight.compute_class_weight('balanced', np.unique(train_gen_full.classes), train_gen_full.classes)
class_weights_full = dict(enumerate(class_weights_full))

model_final, base_model_final = build_model()
for layer in base_model_final.layers:
    layer.trainable = False
model_final.compile(optimizer=Adam(learning_rate=1e-3),
                   loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                   metrics=['accuracy'])
model_final.fit(train_gen_full, epochs=25, class_weight=class_weights_full, verbose=2)

model_final.save(FINAL_MODEL_PATH)
print(f"Final model saved to {FINAL_MODEL_PATH}")