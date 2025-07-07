import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import seaborn as sns
from collections import defaultdict
import cv2
from PIL import Image
import shutil
import glob

# 设置随机种子确保可重复性
tf.random.set_seed(42)
np.random.seed(42)

# 启用混合精度训练
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# 参数设置
BASE_DIR = 'train_samples'
CLEANED_DIR = 'train_samples_cleaned'
BATCH_SIZE = 16  # 增加batch size以提高训练稳定性
IMG_SIZE = (224, 224)
NUM_CLASSES = 5
K_FOLDS = 5

class DataCleaner:
    """数据清理类，解决数据标注错误问题"""
    
    def __init__(self, base_dir, cleaned_dir):
        self.base_dir = base_dir
        self.cleaned_dir = cleaned_dir
        self.class_names = ['Pallas', 'Persian', 'Ragdoll', 'Singapura', 'Sphynx']
    
    def extract_true_label_from_filename(self, filename):
        """从文件名提取真实标签"""
        filename_lower = filename.lower()
        for class_name in self.class_names:
            if class_name.lower() in filename_lower:
                return class_name
        return None
    
    def clean_and_reorganize_data(self):
        """清理并重新组织数据"""
        print("开始清理数据...")
        
        # 创建清理后的目录
        if os.path.exists(self.cleaned_dir):
            shutil.rmtree(self.cleaned_dir)
        
        for class_name in self.class_names:
            os.makedirs(os.path.join(self.cleaned_dir, class_name), exist_ok=True)
        
        moved_files = defaultdict(int)
        error_files = []
        
        # 遍历所有原始文件
        for class_dir in os.listdir(self.base_dir):
            class_path = os.path.join(self.base_dir, class_dir)
            if not os.path.isdir(class_path):
                continue
                
            print(f"处理目录: {class_dir}")
            files = glob.glob(os.path.join(class_path, '*.*'))
            
            for file_path in files:
                filename = os.path.basename(file_path)
                true_label = self.extract_true_label_from_filename(filename)
                
                if true_label:
                    # 移动到正确的目录
                    new_path = os.path.join(self.cleaned_dir, true_label, filename)
                    if not os.path.exists(new_path):  # 避免重复文件
                        shutil.copy2(file_path, new_path)
                        moved_files[true_label] += 1
                else:
                    error_files.append(file_path)
        
        # 打印清理结果
        print("\n数据清理完成！")
        print("每个类别的文件数量:")
        for class_name in self.class_names:
            count = moved_files[class_name]
            print(f"  {class_name}: {count}")
        
        if error_files:
            print(f"\n无法识别标签的文件数量: {len(error_files)}")
            print("前10个无法识别的文件:")
            for f in error_files[:10]:
                print(f"  {f}")
        
        return self.cleaned_dir

class RealTimePlottingCallback(Callback):
    """实时绘图回调函数"""
    
    def __init__(self, save_path='training_plots'):
        super().__init__()
        self.save_path = save_path
        self.epoch_logs = []
        
        # 设置matplotlib后端
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Training Progress - Real Time', fontsize=16)
        
        # 初始化子图
        self.axes[0, 0].set_title('Training Loss')
        self.axes[0, 0].set_xlabel('Epoch')
        self.axes[0, 0].set_ylabel('Loss')
        
        self.axes[0, 1].set_title('Training Accuracy')
        self.axes[0, 1].set_xlabel('Epoch')
        self.axes[0, 1].set_ylabel('Accuracy')
        
        self.axes[1, 0].set_title('Validation Loss')
        self.axes[1, 0].set_xlabel('Epoch')
        self.axes[1, 0].set_ylabel('Loss')
        
        self.axes[1, 1].set_title('Validation Accuracy')
        self.axes[1, 1].set_xlabel('Epoch')
        self.axes[1, 1].set_ylabel('Accuracy')
        
        plt.tight_layout()
        
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        
        # 保存日志
        epoch_log = {'epoch': epoch + 1}
        epoch_log.update(logs)
        self.epoch_logs.append(epoch_log)
        
        # 清除之前的图像
        for ax in self.axes.flat:
            ax.clear()
        
        # 提取数据
        epochs = [log['epoch'] for log in self.epoch_logs]
        train_loss = [log.get('loss', 0) for log in self.epoch_logs]
        train_acc = [log.get('accuracy', 0) for log in self.epoch_logs]
        val_loss = [log.get('val_loss', 0) for log in self.epoch_logs]
        val_acc = [log.get('val_accuracy', 0) for log in self.epoch_logs]
        
        # 绘制训练损失
        self.axes[0, 0].plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
        self.axes[0, 0].set_title('Training Loss')
        self.axes[0, 0].set_xlabel('Epoch')
        self.axes[0, 0].set_ylabel('Loss')
        self.axes[0, 0].grid(True, alpha=0.3)
        self.axes[0, 0].legend()
        
        # 绘制训练准确率
        self.axes[0, 1].plot(epochs, train_acc, 'g-', label='Train Accuracy', linewidth=2)
        self.axes[0, 1].set_title('Training Accuracy')
        self.axes[0, 1].set_xlabel('Epoch')
        self.axes[0, 1].set_ylabel('Accuracy')
        self.axes[0, 1].grid(True, alpha=0.3)
        self.axes[0, 1].legend()
        
        # 绘制验证损失
        self.axes[1, 0].plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
        self.axes[1, 0].set_title('Validation Loss')
        self.axes[1, 0].set_xlabel('Epoch')
        self.axes[1, 0].set_ylabel('Loss')
        self.axes[1, 0].grid(True, alpha=0.3)
        self.axes[1, 0].legend()
        
        # 绘制验证准确率
        self.axes[1, 1].plot(epochs, val_acc, 'orange', label='Val Accuracy', linewidth=2)
        self.axes[1, 1].set_title('Validation Accuracy')
        self.axes[1, 1].set_xlabel('Epoch')
        self.axes[1, 1].set_ylabel('Accuracy')
        self.axes[1, 1].grid(True, alpha=0.3)
        self.axes[1, 1].legend()
        
        # 更新显示
        self.fig.suptitle(f'Training Progress - Epoch {epoch + 1}', fontsize=16)
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
        
        # 保存图像
        os.makedirs(self.save_path, exist_ok=True)
        self.fig.savefig(os.path.join(self.save_path, f'training_progress_epoch_{epoch+1:03d}.png'), 
                        dpi=150, bbox_inches='tight')
        
        # 打印当前epoch的结果
        print(f"\nEpoch {epoch + 1}: "
              f"loss: {train_loss[-1]:.4f}, "
              f"acc: {train_acc[-1]:.4f}, "
              f"val_loss: {val_loss[-1]:.4f}, "
              f"val_acc: {val_acc[-1]:.4f}")

class PerClassAccuracy(Callback):
    """每个类别的准确率回调"""
    
    def __init__(self, val_generator, class_names, verbose=1):
        super().__init__()
        self.val_gen = val_generator
        self.class_names = class_names
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        # 收集所有验证数据的真实标签和预测
        y_true = []
        y_pred = []
        
        for i in range(len(self.val_gen)):
            x_batch, y_batch = self.val_gen[i]
            preds = self.model.predict(x_batch, verbose=0)
            y_true.extend(np.argmax(y_batch, axis=1))
            y_pred.extend(np.argmax(preds, axis=1))
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # 计算每个类别的准确率
        if self.verbose:
            print(f"\n— Epoch {epoch+1} 各类别验证准确率:")
            for idx, name in enumerate(self.class_names):
                mask = (y_true == idx)
                if np.sum(mask) == 0:
                    acc = 0.0
                else:
                    acc = accuracy_score(y_true[mask], y_pred[mask])
                print(f"    {name:10s}: {acc*100:5.2f}%")

class StratifiedDataGenerator:
    """分层数据生成器，确保训练/验证分割的平衡性"""
    
    def __init__(self, data_dir, img_size, batch_size, val_split=0.2):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.val_split = val_split
        self.class_names = ['Pallas', 'Persian', 'Ragdoll', 'Singapura', 'Sphynx']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
    def load_dataset_info(self):
        """加载数据集信息"""
        file_paths = []
        labels = []
        
        for class_name in self.class_names:
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.exists(class_dir):
                files = glob.glob(os.path.join(class_dir, '*.*'))
                file_paths.extend(files)
                labels.extend([self.class_to_idx[class_name]] * len(files))
        
        return np.array(file_paths), np.array(labels)
    
    def create_stratified_split(self, file_paths, labels):
        """创建分层分割"""
        from sklearn.model_selection import train_test_split
        
        # 使用分层抽样确保每个类别在训练/验证集中的比例一致
        train_files, val_files, train_labels, val_labels = train_test_split(
            file_paths, labels, 
            test_size=self.val_split, 
            stratify=labels, 
            random_state=42
        )
        
        return train_files, val_files, train_labels, val_labels

def create_improved_model(img_size, num_classes, dropout_rate=0.5):
    """创建改进的模型架构，加强防过拟合"""
    
    # 加载预训练模型
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(*img_size, 3)
    )
    
    # 构建自定义顶层
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # 第一个全连接层
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # 第二个全连接层
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate * 0.7)(x)
    
    # 第三个全连接层（可选）
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate * 0.5)(x)
    
    # 输出层
    predictions = Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

def create_advanced_data_generators(train_files, val_files, train_labels, val_labels, 
                                   img_size, batch_size, class_names):
    """创建高级数据生成器"""
    
    # 更激进的数据增强用于训练
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=20,  # 增加旋转范围
        width_shift_range=0.1,  # 增加位移范围
        height_shift_range=0.1,
        shear_range=0.05,  # 增加剪切变换
        zoom_range=0.1,  # 增加缩放范围
        brightness_range=(0.8, 1.2),  # 增加亮度变化
        channel_shift_range=0.1,  # 添加通道偏移
        fill_mode='nearest'
    )
    
    # 验证集只进行归一化
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # 创建数据框
    train_df = pd.DataFrame({
        'filename': train_files,
        'class': [class_names[label] for label in train_labels]
    })
    
    val_df = pd.DataFrame({
        'filename': val_files,
        'class': [class_names[label] for label in val_labels]
    })
    
    # 生成器
    train_gen = train_datagen.flow_from_dataframe(
        train_df,
        x_col='filename',
        y_col='class',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    val_gen = val_datagen.flow_from_dataframe(
        val_df,
        x_col='filename',
        y_col='class',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        seed=42
    )
    
    return train_gen, val_gen

def train_with_k_fold_cv(cleaned_data_dir, k_folds=5):
    """使用k折交叉验证进行训练"""
    
    print(f"开始{k_folds}折交叉验证训练...")
    
    # 数据生成器
    data_gen = StratifiedDataGenerator(cleaned_data_dir, IMG_SIZE, BATCH_SIZE, val_split=0.2)
    file_paths, labels = data_gen.load_dataset_info()
    
    print(f"总共加载 {len(file_paths)} 个文件")
    
    # K折交叉验证
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(file_paths, labels)):
        print(f"\n{'='*50}")
        print(f"开始第 {fold + 1}/{k_folds} 折训练")
        print(f"{'='*50}")
        
        # 分割数据
        train_files = file_paths[train_idx]
        val_files = file_paths[val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        
        print(f"训练集大小: {len(train_files)}")
        print(f"验证集大小: {len(val_files)}")
        
        # 创建数据生成器
        train_gen, val_gen = create_advanced_data_generators(
            train_files, val_files, train_labels, val_labels,
            IMG_SIZE, BATCH_SIZE, data_gen.class_names
        )
        
        # 计算类别权重
        unique_labels, counts = np.unique(train_labels, return_counts=True)
        class_weights = class_weight.compute_class_weight(
            'balanced', classes=unique_labels, y=train_labels
        )
        class_weights_dict = dict(zip(unique_labels, class_weights))
        
        print(f"类别权重: {class_weights_dict}")
        
        # 创建模型
        model, base_model = create_improved_model(IMG_SIZE, NUM_CLASSES, dropout_rate=0.6)
        
        # 回调函数
        callbacks = [
            ModelCheckpoint(
                f'best_model_fold_{fold+1}.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=12,  # 增加patience
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            RealTimePlottingCallback(save_path=f'plots_fold_{fold+1}'),
            PerClassAccuracy(val_gen, data_gen.class_names)
        ]
        
        # 训练策略：三阶段训练
        fold_history = train_three_stage(model, base_model, train_gen, val_gen, 
                                       callbacks, class_weights_dict, fold)
        
        # 评估当前折
        val_loss, val_acc = model.evaluate(val_gen, verbose=0)
        fold_scores.append(val_acc)
        
        print(f"第 {fold + 1} 折验证准确率: {val_acc:.4f}")
        
        # 清理GPU内存
        tf.keras.backend.clear_session()
    
    # 输出交叉验证结果
    print(f"\n{'='*60}")
    print("K折交叉验证结果汇总:")
    print(f"{'='*60}")
    for i, score in enumerate(fold_scores):
        print(f"第 {i+1} 折: {score:.4f}")
    print(f"平均准确率: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"{'='*60}")
    
    return fold_scores

def train_three_stage(model, base_model, train_gen, val_gen, callbacks, class_weights_dict, fold):
    """三阶段训练策略"""
    
    print(f"\n--- 第 {fold+1} 折 - 阶段1: 训练分类头部 ---")
    
    # 阶段1: 冻结基础模型，只训练头部
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    stage1_callbacks = [cb for cb in callbacks if not isinstance(cb, RealTimePlottingCallback)]
    stage1_callbacks.append(RealTimePlottingCallback(save_path=f'plots_fold_{fold+1}_stage1'))
    
    history1 = model.fit(
        train_gen,
        epochs=10,  # 减少epochs避免过拟合
        validation_data=val_gen,
        callbacks=stage1_callbacks,
        class_weight=class_weights_dict,
        verbose=1
    )
    
    print(f"\n--- 第 {fold+1} 折 - 阶段2: 微调顶层 ---")
    
    # 阶段2: 解冻顶层进行微调
    for layer in base_model.layers[-15:]:  # 解冻更多层
        layer.trainable = True
        # 保持BN层冻结
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    
    model.compile(
        optimizer=Adam(learning_rate=5e-5),  # 更低的学习率
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    stage2_callbacks = [cb for cb in callbacks if not isinstance(cb, RealTimePlottingCallback)]
    stage2_callbacks.append(RealTimePlottingCallback(save_path=f'plots_fold_{fold+1}_stage2'))
    
    history2 = model.fit(
        train_gen,
        epochs=15,  # 减少epochs
        validation_data=val_gen,
        callbacks=stage2_callbacks,
        class_weight=class_weights_dict,
        verbose=1
    )
    
    print(f"\n--- 第 {fold+1} 折 - 阶段3: 全局微调 ---")
    
    # 阶段3: 全局微调（可选）
    for layer in base_model.layers:
        layer.trainable = True
        # 保持BN层冻结
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    
    model.compile(
        optimizer=Adam(learning_rate=1e-6),  # 极低的学习率
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    stage3_callbacks = [cb for cb in callbacks if not isinstance(cb, RealTimePlottingCallback)]
    stage3_callbacks.append(RealTimePlottingCallback(save_path=f'plots_fold_{fold+1}_stage3'))
    
    history3 = model.fit(
        train_gen,
        epochs=10,  # 少量epochs
        validation_data=val_gen,
        callbacks=stage3_callbacks,
        class_weight=class_weights_dict,
        verbose=1
    )
    
    return [history1, history2, history3]

def main():
    """主训练函数"""
    print("开始改进的猫品种分类训练流程...")
    
    # 1. 数据清理
    cleaner = DataCleaner(BASE_DIR, CLEANED_DIR)
    cleaned_dir = cleaner.clean_and_reorganize_data()
    
    # 2. K折交叉验证训练
    fold_scores = train_with_k_fold_cv(cleaned_dir, K_FOLDS)
    
    # 3. 训练最终模型（使用全部数据）
    print(f"\n{'='*60}")
    print("训练最终模型（使用全部清理后的数据）...")
    print(f"{'='*60}")
    
    data_gen = StratifiedDataGenerator(cleaned_dir, IMG_SIZE, BATCH_SIZE, val_split=0.15)
    file_paths, labels = data_gen.load_dataset_info()
    train_files, val_files, train_labels, val_labels = data_gen.create_stratified_split(file_paths, labels)
    
    train_gen, val_gen = create_advanced_data_generators(
        train_files, val_files, train_labels, val_labels,
        IMG_SIZE, BATCH_SIZE, data_gen.class_names
    )
    
    # 计算类别权重
    unique_labels, counts = np.unique(train_labels, return_counts=True)
    class_weights = class_weight.compute_class_weight(
        'balanced', classes=unique_labels, y=train_labels
    )
    class_weights_dict = dict(zip(unique_labels, class_weights))
    
    # 创建最终模型
    final_model, base_model = create_improved_model(IMG_SIZE, NUM_CLASSES, dropout_rate=0.5)
    
    # 最终模型的回调函数
    final_callbacks = [
        ModelCheckpoint(
            'cats_improved_final.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        RealTimePlottingCallback(save_path='final_training_plots'),
        PerClassAccuracy(val_gen, data_gen.class_names)
    ]
    
    # 训练最终模型
    final_history = train_three_stage(final_model, base_model, train_gen, val_gen, 
                                    final_callbacks, class_weights_dict, "final")
    
    print("训练完成！")
    print(f"最终模型保存为: cats_improved_final.keras")
    print(f"K折交叉验证平均准确率: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

if __name__ == "__main__":
    main()