import os
import numpy as np
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
# --- 引入回调函数 ---
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score


# 数据目录和参数
base_dir    = os.path.join(os.getcwd(), 'train_samples')
batch_size  = 16
# 建议使用 ResNet 的标准输入尺寸
img_size    = (224, 224) 
num_classes = 5

class PerClassAccuracy(tf.keras.callbacks.Callback):                                                    #打印每个类别的准确率
    def __init__(self, val_generator, class_indices, verbose=1):
        super().__init__()
        self.val_gen = val_generator
        # class_indices: dict, e.g. {'Pallas':0, ...}
        # invert to list of class names by index
        self.class_names = [None] * len(class_indices)
        for name, idx in class_indices.items():
            self.class_names[idx] = name
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        # Gather all ground-truth and preds
        y_true = []
        y_pred = []
        for i in range(len(self.val_gen)):
            x_batch, y_batch = self.val_gen[i]
            preds = self.model.predict(x_batch, verbose=0)
            y_true.extend(np.argmax(y_batch, axis=1))
            y_pred.extend(np.argmax(preds, axis=1))
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Compute per-class accuracy
        report = {}
        for idx, name in enumerate(self.class_names):
            mask = (y_true == idx)
            if np.sum(mask) == 0:
                acc = np.nan
            else:
                acc = accuracy_score(y_true[mask], y_pred[mask])
            report[name] = acc

        # Print
        if self.verbose:
            print(f"\n— epoch {epoch+1} per-class val accuracy:")
            for name, acc in report.items():
                print(f"    {name:10s}: {acc*100:5.2f}%")
        # Optionally, log into logs so it's visible in History
        for name, acc in report.items():
            logs[f"val_acc_{name}"] = acc


# 构建训练/验证数据生成器
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    # shear_range=0.2,                  # 剪切变换                      
    # zoom_range=0.2,
    # brightness_range=(0.8, 1.2),
    # channel_shift_range=0.1,
    # fill_mode='nearest'
)
train_gen = datagen.flow_from_directory(
    base_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)
val_gen = datagen.flow_from_directory(
    base_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# 查看类别与索引的对应关系：
print("Class indices:", train_gen.class_indices)
# e.g. {'Pallas':0, 'Persian':1, 'Ragdoll':2, 'Singapura':3, 'Sphynx':4}

class_indices = train_gen.class_indices  # e.g. {'Pallas':0,...}

perclass_cb = PerClassAccuracy(val_generator=val_gen,
                               class_indices=class_indices,
                               verbose=1)

# 计算类别权重
# train_gen.classes 会给你所有训练样本的类别标签（0,0,1,2,3,4...）
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
# 将其转换为Keras需要的字典格式
class_weights_dict = dict(enumerate(class_weights))

print("Calculated Class Weights:", class_weights_dict)
# 输出可能像这样: {0: 0.8, 1: 0.8, 2: 0.8, 3: 2.4, 4: 0.8}
# 其中key是类别索引，value是权重。新加坡猫的权重会高很多。

# 加载预训练主体网络
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(*img_size, 3)
)

# 构建微调模型
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# --- 定义回调函数 ---
output_path = os.path.join(os.getcwd(), 'cats_best.keras')

# 1. ModelCheckpoint: 只保存在验证集上性能最好的模型
checkpoint = ModelCheckpoint(
    filepath=output_path,
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# 2. EarlyStopping: 如果验证集损失在10个周期内没有改善，则停止训练
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True # 训练结束后，模型权重将恢复为最佳状态
)

# 3. ReduceLROnPlateau: 如果验证集损失在5个周期内没有改善，则降低学习率
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2, # 学习率将乘以0.2
    patience=5,
    min_lr=1e-6 # 学习率不会低于这个值
)

# 将所有回调函数放入一个列表
callbacks_list = [checkpoint, early_stopping, reduce_lr]
callbacks_list.append(perclass_cb)

# === 阶段一：冻结主体，只训练头部 ===
print("--- Starting Stage 1: Training the head ---")
for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_gen,
    epochs=25, # <-- 增加周期
    validation_data=val_gen,
    callbacks=callbacks_list, # <-- 使用回调函数
    class_weight=class_weights_dict # <-- 加入这一行！
)

# === 阶段二：解冻部分层，进行微调 ===
print("--- Starting Stage 2: Fine-tuning ---")
for layer in base_model.layers[-50:]: # 解冻更多层也可以尝试，如 -50
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5), # 微调时使用更低的学习率
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 注意：EarlyStopping 会继续工作，如果模型性能饱和，会自动停止
history_fine_tune = model.fit(
    train_gen,
    epochs=100, # <-- 大幅增加周期上限
    validation_data=val_gen,
    callbacks=callbacks_list, # <-- 继续使用回调函数
    class_weight=class_weights_dict # <-- 加入这一行！
)

print(f"Fine-tuning complete. Best model saved to: {output_path}")