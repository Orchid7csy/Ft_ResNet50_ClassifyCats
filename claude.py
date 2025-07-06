import os
import numpy as np
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input  # 添加这行
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Conv2D, multiply, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import tf_keras
import cv2

# 设置随机种子确保可重复性
tf.random.set_seed(42)
np.random.seed(42)

# 启用混合精度训练
tf.keras.mixed_precision.set_global_policy('mixed_float16')

print("开始诊断并优化猫分类模型训练...")

# 数据目录和参数 - 针对低准确率问题优化
base_dir = os.path.join(os.getcwd(), 'train_samples')
batch_size = 8  # 进一步减小batch size提高训练稳定性
img_size = (224, 224)
num_classes = 5

print(f"数据目录: {base_dir}")
print(f"批次大小: {batch_size}")
print(f"图像尺寸: {img_size}")

# 自定义预处理函数 - 使用letterboxing
def letterbox_preprocess(image, target_size=(224, 224)):
    """
    使用letterboxing保持长宽比，然后应用ResNet50预处理
    """
    if len(image.shape) == 3:
        h, w, _ = image.shape
    else:
        h, w = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    target_h, target_w = target_size
    
    # 计算缩放比例
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # 保持长宽比缩放
    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 创建灰色画布
    canvas = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
    
    # 居中粘贴
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized_img
    
    return canvas

class LetterboxDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, directory, target_size, batch_size, class_mode='categorical', 
                 subset=None, shuffle=True, seed=None, validation_split=0.2):
        self.directory = directory
        self.target_size = target_size
        self.batch_size = batch_size
        self.class_mode = class_mode
        self.subset = subset
        self.shuffle = shuffle
        self.seed = seed
        self.validation_split = validation_split
        
        # 获取所有图片路径和标签
        self.class_indices = {}
        self.samples = []  # 保持为列表，存储文件路径
        self.classes = []  # 保持为列表，存储类别标签
        
        class_dirs = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
        
        for i, class_name in enumerate(class_dirs):
            self.class_indices[class_name] = i
            class_path = os.path.join(directory, class_name)
            
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(class_path, img_file)
                    self.samples.append(img_path)
                    self.classes.append(i)
        
        # 转换为numpy数组
        self.classes = np.array(self.classes)
        
        # 分割训练/验证数据
        if subset:
            if seed is not None:
                np.random.seed(seed)
            indices = np.random.permutation(len(self.samples))
            split_idx = int(len(self.samples) * (1 - validation_split))
            
            if subset == 'training':
                indices = indices[:split_idx]
            else:  # validation
                indices = indices[split_idx:]
            
            # 更新样本和类别
            self.samples = [self.samples[i] for i in indices]
            self.classes = self.classes[indices]
        
        # 存储样本数量
        self.num_samples = len(self.samples)
        
        print(f"数据生成器初始化完成: {self.num_samples} 个样本")
        
        # 初始化索引
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))
    
    def __getitem__(self, index):
        # 获取当前批次的索引
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.num_samples)
        batch_indices = self.indices[start_idx:end_idx]
        
        # 生成批次数据
        batch_size_actual = len(batch_indices)
        X = np.zeros((batch_size_actual, *self.target_size, 3), dtype=np.float32)
        y = np.zeros((batch_size_actual, len(self.class_indices)), dtype=np.float32)
        
        for i, idx in enumerate(batch_indices):
            try:
                # 读取图片
                img = cv2.imread(self.samples[idx])
                if img is None:
                    print(f"警告: 无法读取图片 {self.samples[idx]}")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 应用letterboxing
                img = letterbox_preprocess(img, self.target_size)
                
                # 转换为numpy数组并应用ResNet50预处理
                img_array = np.array(img, dtype=np.float32)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)
                
                X[i] = img_array[0]
                
                # 创建one-hot编码
                y[i, self.classes[idx]] = 1.0
                
            except Exception as e:
                print(f"处理图片时出错 {self.samples[idx]}: {e}")
                continue
        
        return X, y
    
    def on_epoch_end(self):
        """在每个epoch结束时调用"""
        self.indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)
    

# 数据集检查函数
def check_dataset_structure(base_dir):
    """检查数据集结构和分布"""
    print("\n=== 数据集结构检查 ===")
    if not os.path.exists(base_dir):
        print(f"❌ 错误：数据目录 {base_dir} 不存在！")
        return False
    
    class_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    print(f"发现类别数: {len(class_dirs)}")
    print(f"类别名称: {class_dirs}")
    
    total_images = 0
    for class_name in class_dirs:
        class_path = os.path.join(base_dir, class_name)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"  {class_name}: {len(image_files)} 张图片")
        total_images += len(image_files)
    
    print(f"总图片数: {total_images}")
    
    if len(class_dirs) != num_classes:
        print(f"⚠️ 警告：期望{num_classes}类，实际发现{len(class_dirs)}类")
    
    return total_images > 0

# 检查数据集
if not check_dataset_structure(base_dir):
    print("数据集检查失败，请确认数据目录结构正确！")
    exit(1)

# 使用自定义数据生成器
print("\n=== 创建自定义数据生成器 ===")
train_gen = LetterboxDataGenerator(
    directory=base_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42,
    validation_split=0.2
)

val_gen = LetterboxDataGenerator(
    directory=base_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42,
    validation_split=0.2
)

print("Class indices:", train_gen.class_indices)
print(f"训练样本数: {train_gen.samples}")
print(f"验证样本数: {val_gen.samples}")

# 检查数据生成器
print("\n=== 检查数据生成器 ===")
try:
    sample_batch = train_gen[0]
    print(f"样本批次形状: {sample_batch[0].shape}")
    print(f"标签批次形状: {sample_batch[1].shape}")
    print(f"图片数值范围: [{sample_batch[0].min():.3f}, {sample_batch[0].max():.3f}]")
    print(f"标签示例: {sample_batch[1][0]}")
except Exception as e:
    print(f"❌ 数据生成器错误: {e}")
    exit(1)

# 计算类别权重
print("\n=== 计算类别权重 ===")
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights_dict = dict(enumerate(class_weights))
print("类别权重:", class_weights_dict)

# 简化的网络结构 - 避免过度复杂化
print("\n=== 构建简化的网络结构 ===")
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(*img_size, 3)
)

# 构建模型
x = base_model.output
x = GlobalAveragePooling2D()(x)

# 简化的全连接层
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

predictions = Dense(num_classes, activation='softmax', dtype='float32')(x)
model = Model(inputs=base_model.input, outputs=predictions)

print("模型结构:")
model.summary()

# 学习率调度函数
def lr_schedule(epoch, lr):
    """自定义学习率调度"""
    if epoch < 10:
        return lr
    elif epoch < 20:
        return lr * 0.5
    elif epoch < 30:
        return lr * 0.1
    else:
        return lr * 0.01

# 回调函数
output_path = os.path.join(os.getcwd(), 'cats.keras')

checkpoint = ModelCheckpoint(
    filepath=output_path,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1,
    save_weights_only=False
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)

# 计算steps
steps_per_epoch = max(1, len(train_gen))
validation_steps = max(1, len(val_gen))

print(f"每个epoch训练步数: {steps_per_epoch}")
print(f"每个epoch验证步数: {validation_steps}")

# 阶段一：只训练分类头，使用较大学习率
print("\n=== 阶段一：训练分类头 ===")
for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("开始阶段一训练...")
history1 = model.fit(
    train_gen,
    epochs=15,
    validation_data=val_gen,
    callbacks=[checkpoint, early_stopping],
    class_weight=class_weights_dict,
    verbose=1
)

print(f"阶段一最佳验证准确率: {max(history1.history['val_accuracy']):.4f}")

# 阶段二：微调最后几层
print("\n=== 阶段二：微调最后几层 ===")
for layer in base_model.layers[-10:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("开始阶段二训练...")
history2 = model.fit(
    train_gen,
    epochs=25,
    validation_data=val_gen,
    callbacks=[checkpoint, early_stopping, lr_scheduler],
    class_weight=class_weights_dict,
    verbose=1
)

print(f"阶段二最佳验证准确率: {max(history2.history['val_accuracy']):.4f}")

# 阶段三：全局微调
print("\n=== 阶段三：全局微调 ===")
for layer in base_model.layers:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=5e-6),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("开始阶段三训练...")
history3 = model.fit(
    train_gen,
    epochs=20,
    validation_data=val_gen,
    callbacks=[checkpoint, early_stopping],
    class_weight=class_weights_dict,
    verbose=1
)

print(f"阶段三最佳验证准确率: {max(history3.history['val_accuracy']):.4f}")

# 加载最佳模型进行评估
print("\n=== 加载最佳模型进行详细评估 ===")
model.tf_keras.load_model(output_path)

# 在验证集上评估
val_loss, val_accuracy = model.evaluate(val_gen, verbose=0)
print(f"最终验证损失: {val_loss:.4f}")
print(f"最终验证准确率: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")

# 保存最终模型
final_model_path = os.path.join(os.getcwd(), 'cats_final.keras')
model.save(final_model_path)
print(f"\n最终模型保存至: {final_model_path}")

# 打印训练历史
all_val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy'] + history3.history['val_accuracy']
best_acc = max(all_val_acc)

print(f"\n=== 最终结果 ===")
print(f"最佳验证准确率: {best_acc:.4f} ({best_acc*100:.2f}%)")

print("\n训练完成！现在训练和推理使用相同的预处理方法。")