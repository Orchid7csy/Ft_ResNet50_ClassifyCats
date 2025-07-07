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
BATCH_SIZE = 16
IMG_SIZE = (224, 224)
NUM_CLASSES = 5
K_FOLDS = 5

class SafeMicrotuningStrategy:
    """安全的微调策略类"""
    
    @staticmethod
    def get_layer_groups(base_model):
        """将ResNet50分为不同层组"""
        total_layers = len(base_model.layers)
        
        # ResNet50层分组策略
        groups = {
            'early_layers': base_model.layers[:50],      # 早期特征层 (conv1, conv2_x)
            'middle_layers': base_model.layers[50:100],  # 中期特征层 (conv3_x)  
            'late_layers': base_model.layers[100:140],   # 后期特征层 (conv4_x)
            'top_layers': base_model.layers[140:]        # 顶层特征层 (conv5_x)
        }
        
        print(f"📊 ResNet50层分组:")
        for group_name, layers in groups.items():
            print(f"  {group_name}: {len(layers)} 层 (索引 {layers[0].name if layers else 'None'} ~ {layers[-1].name if layers else 'None'})")
        
        return groups
    
    @staticmethod
    def freeze_bn_layers(model):
        """冻结所有BatchNormalization层"""
        bn_count = 0
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
                bn_count += 1
        print(f"🔒 冻结了 {bn_count} 个BatchNormalization层")

def create_improved_model_safe(img_size, num_classes, dropout_rate=0.5):
    """创建改进的模型架构（安全版本）"""
    
    # 加载预训练模型
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(*img_size, 3)
    )
    
    # 构建自定义顶层（更保守的设计）
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # 简化的分类头部（减少过拟合风险）
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate * 0.7)(x)
    
    # 输出层
    predictions = Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

def train_three_stage_safe(model, base_model, train_gen, val_gen, callbacks, class_weights_dict, fold):
    """安全的三阶段训练策略"""
    
    # 获取层分组
    layer_groups = SafeMicrotuningStrategy.get_layer_groups(base_model)
    
    print(f"\n--- 第 {fold+1} 折 - 阶段1: 训练分类头部 ---")
    
    # 阶段1: 冻结整个基础模型，只训练头部
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
        epochs=12,  # 增加epochs让头部训练更充分
        validation_data=val_gen,
        callbacks=stage1_callbacks,
        class_weight=class_weights_dict,
        verbose=1
    )
    
    print(f"\n--- 第 {fold+1} 折 - 阶段2: 安全的顶层微调 ---")
    
    # 阶段2: 只解冻顶层和后期层（保护早期特征）
    for layer in base_model.layers:
        layer.trainable = False
    
    # 只解冻后期层和顶层
    for layer in layer_groups['late_layers'] + layer_groups['top_layers']:
        layer.trainable = True
    
    # 冻结所有BN层以保持稳定性
    SafeMicrotuningStrategy.freeze_bn_layers(model)
    
    trainable_params = sum([np.prod(v.get_shape()) for v in model.trainable_variables])
    total_params = sum([np.prod(v.get_shape()) for v in model.variables])
    print(f"📊 可训练参数: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    model.compile(
        optimizer=Adam(learning_rate=3e-5),  # 更保守的学习率
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    stage2_callbacks = [cb for cb in callbacks if not isinstance(cb, RealTimePlottingCallback)]
    stage2_callbacks.append(RealTimePlottingCallback(save_path=f'plots_fold_{fold+1}_stage2'))
    
    history2 = model.fit(
        train_gen,
        epochs=18,  # 适当增加epochs
        validation_data=val_gen,
        callbacks=stage2_callbacks,
        class_weight=class_weights_dict,
        verbose=1
    )
    
    print(f"\n--- 第 {fold+1} 折 - 阶段3: 可选的中层微调 ---")
    
    # 检查是否应该进行第三阶段
    val_acc_stage2 = max(history2.history['val_accuracy'])
    
    if val_acc_stage2 > 0.85:  # 只有在性能足够好时才进行第三阶段
        print(f"✅ 第2阶段验证准确率 {val_acc_stage2:.3f} > 0.85, 进行保守的第3阶段微调")
        
        # 阶段3: 非常保守的中层微调（保护早期层）
        # 只额外解冻中期层，早期层始终冻结
        for layer in layer_groups['middle_layers']:
            layer.trainable = True
        
        # 继续冻结所有BN层
        SafeMicrotuningStrategy.freeze_bn_layers(model)
        
        trainable_params = sum([np.prod(v.get_shape()) for v in model.trainable_variables])
        print(f"📊 第3阶段可训练参数: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")
        
        model.compile(
            optimizer=Adam(learning_rate=1e-6),  # 极低的学习率
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        stage3_callbacks = [cb for cb in callbacks if not isinstance(cb, RealTimePlottingCallback)]
        stage3_callbacks.append(RealTimePlottingCallback(save_path=f'plots_fold_{fold+1}_stage3'))
        
        # 添加额外的早停保护
        early_stop_protective = EarlyStopping(
            monitor='val_accuracy',
            patience=3,  # 更早停止
            restore_best_weights=True,
            mode='max',
            verbose=1
        )
        stage3_callbacks.append(early_stop_protective)
        
        history3 = model.fit(
            train_gen,
            epochs=8,  # 少量epochs
            validation_data=val_gen,
            callbacks=stage3_callbacks,
            class_weight=class_weights_dict,
            verbose=1
        )
        
        # 检查第三阶段是否有效
        val_acc_stage3 = max(history3.history['val_accuracy'])
        if val_acc_stage3 < val_acc_stage2 - 0.01:
            print(f"⚠️  第3阶段性能下降，从 {val_acc_stage2:.3f} 到 {val_acc_stage3:.3f}")
            print("🔄 建议重新加载第2阶段的最佳模型")
        
        return [history1, history2, history3]
    
    else:
        print(f"⚠️  第2阶段验证准确率 {val_acc_stage2:.3f} < 0.85, 跳过第3阶段避免过拟合")
        print("✅ 使用第2阶段的结果作为最终模型")
        return [history1, history2]

# 其他类和函数保持不变...
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

def main():
    """主训练函数"""
    print("🛡️  安全的猫品种分类训练流程...")
    print("⚠️  采用保守的微调策略，保护预训练特征")
    
    # 数据清理
    cleaner = DataCleaner(BASE_DIR, CLEANED_DIR)
    cleaned_dir = cleaner.clean_and_reorganize_data()
    
    print(f"\n🔒 安全微调策略说明:")
    print(f"  ✅ 阶段1: 只训练分类头部")
    print(f"  ✅ 阶段2: 只解冻ResNet50后期层 + 顶层")
    print(f"  ⚠️  阶段3: 条件解冻中期层（性能>85%时）")
    print(f"  🛡️  始终保护: 早期层 + 所有BN层")
    
    # 演示安全微调（简化版，实际使用时请使用完整版本）
    print(f"\n📊 预期效果:")
    print(f"  🎯 更稳定的训练过程")
    print(f"  🛡️  保护预训练的基础特征") 
    print(f"  📈 可能略低但更稳定的性能")
    print(f"  ⚡ 更快的收敛速度")

if __name__ == "__main__":
    main()