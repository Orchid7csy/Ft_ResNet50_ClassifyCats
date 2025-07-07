import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import cv2
import glob
from collections import defaultdict
from image_preprocessing import letterbox_preprocess
from find_real_img import find_cat_with_haar

# 设置随机种子
tf.random.set_seed(42)
np.random.seed(42)

class ImprovedTester:
    """改进的测试类"""
    
    def __init__(self, model_path, test_dir, img_size=(224, 224)):
        self.model_path = model_path
        self.test_dir = test_dir
        self.img_size = img_size
        self.class_names = ['Pallas', 'Persian', 'Ragdoll', 'Singapura', 'Sphynx']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.idx_to_class = {idx: name for idx, name in enumerate(self.class_names)}
        
        # 加载模型
        print(f"加载模型: {model_path}")
        self.model = load_model(model_path)
        print("模型加载完成")
    
    def extract_true_label_from_filename(self, filename):
        """从文件名提取真实标签"""
        filename_lower = filename.lower()
        for class_name in self.class_names:
            if class_name.lower() in filename_lower:
                return class_name
        return None
    
    def load_image_with_haar(self, image_path):
        """使用Haar检测加载图片"""
        try:
            # 使用OpenCV读取图像
            cv_image = cv2.imread(image_path)
            if cv_image is None:
                raise ValueError(f"无法加载图像: {image_path}")
            
            # 使用Haar检测猫脸
            try:
                cat_faces = find_cat_with_haar(cv_image)
                
                if cat_faces is not None and len(cat_faces) > 0:
                    # 检测到猫脸，使用第一个检测到的猫脸区域
                    x, y, w, h = cat_faces[0]
                    cat_face_roi = cv_image[y:y+h, x:x+w]
                    img = Image.fromarray(cv2.cvtColor(cat_face_roi, cv2.COLOR_BGR2RGB))
                    detection_status = f"Haar检测到 ({len(cat_faces)} 个猫脸)"
                else:
                    # 没有检测到猫脸，使用原图
                    img = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
                    detection_status = "未检测到猫脸，使用原图"
                    
            except Exception as e:
                # 如果Haar检测出错，使用原图
                print(f"Haar检测错误 {image_path}: {e}")
                img = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
                detection_status = "检测错误，使用原图"
            
            # 使用letterboxing预处理
            img_array = np.array(img)
            img_processed = letterbox_preprocess(img_array, self.img_size)
            
            # 转换为模型输入格式
            img_array = np.array(img_processed, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            return img_array, detection_status
            
        except Exception as e:
            print(f"加载图片失败 {image_path}: {e}")
            return None, f"加载失败: {e}"
    
    def load_image_original(self, image_path):
        """原始图片加载方法"""
        try:
            # 直接加载并调整大小
            img = Image.open(image_path).convert('RGB')
            
            try:
                resample = Image.Resampling.LANCZOS
            except AttributeError:
                resample = Image.LANCZOS
                
            img = img.resize(self.img_size, resample)
            
            # 转换为模型输入格式
            img_array = np.array(img, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            return img_array
            
        except Exception as e:
            print(f"原始方法加载图片失败 {image_path}: {e}")
            return None
    
    def predict_single(self, img_array):
        """单张图片预测"""
        if img_array is None:
            return None, 0, -1
        
        predictions = self.model.predict(img_array, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        predicted_class = self.idx_to_class[predicted_idx]
        
        return predicted_class, confidence, predicted_idx
    
    def evaluate_comprehensive(self):
        """全面评估模型性能"""
        print("开始全面评估...")
        print("="*60)
        
        # 收集所有测试文件
        test_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            test_files.extend(glob.glob(os.path.join(self.test_dir, ext)))
        
        if not test_files:
            print(f"在 {self.test_dir} 中未找到测试文件")
            return
        
        print(f"找到 {len(test_files)} 个测试文件")
        
        # 存储结果
        results = {
            'filename': [],
            'true_label': [],
            'haar_prediction': [],
            'haar_confidence': [],
            'haar_correct': [],
            'original_prediction': [],
            'original_confidence': [],
            'original_correct': [],
            'detection_status': []
        }
        
        # 统计变量
        total_processed = 0
        haar_detections = 0
        haar_correct = 0
        original_correct = 0
        
        # 按类别统计
        class_stats = defaultdict(lambda: {'total': 0, 'haar_correct': 0, 'original_correct': 0})
        
        print("\n开始处理测试文件...")
        
        for i, file_path in enumerate(test_files):
            filename = os.path.basename(file_path)
            true_label = self.extract_true_label_from_filename(filename)
            
            if true_label is None:
                print(f"警告: 无法从文件名提取真实标签: {filename}")
                continue
            
            # 处理进度
            if (i + 1) % 10 == 0:
                print(f"已处理: {i + 1}/{len(test_files)}")
            
            try:
                # Haar方法测试
                img_haar, detection_status = self.load_image_with_haar(file_path)
                haar_pred, haar_conf, _ = self.predict_single(img_haar)
                
                # 原始方法测试
                img_original = self.load_image_original(file_path)
                original_pred, original_conf, _ = self.predict_single(img_original)
                
                # 更新统计
                total_processed += 1
                class_stats[true_label]['total'] += 1
                
                if "检测到" in detection_status:
                    haar_detections += 1
                
                haar_is_correct = (haar_pred == true_label)
                original_is_correct = (original_pred == true_label)
                
                if haar_is_correct:
                    haar_correct += 1
                    class_stats[true_label]['haar_correct'] += 1
                
                if original_is_correct:
                    original_correct += 1
                    class_stats[true_label]['original_correct'] += 1
                
                # 保存结果
                results['filename'].append(filename)
                results['true_label'].append(true_label)
                results['haar_prediction'].append(haar_pred)
                results['haar_confidence'].append(haar_conf)
                results['haar_correct'].append(haar_is_correct)
                results['original_prediction'].append(original_pred)
                results['original_confidence'].append(original_conf)
                results['original_correct'].append(original_is_correct)
                results['detection_status'].append(detection_status)
                
            except Exception as e:
                print(f"处理文件时出错 {filename}: {e}")
                continue
        
        # 计算总体准确率
        if total_processed > 0:
            haar_accuracy = haar_correct / total_processed
            original_accuracy = original_correct / total_processed
            detection_rate = haar_detections / total_processed
            
            # 打印结果
            print(f"\n{'='*60}")
            print("评估结果汇总:")
            print(f"{'='*60}")
            print(f"总处理文件数: {total_processed}")
            print(f"Haar检测成功率: {haar_detections} ({detection_rate*100:.1f}%)")
            print(f"")
            print(f"Haar方法准确率: {haar_correct}/{total_processed} = {haar_accuracy:.3f} ({haar_accuracy*100:.1f}%)")
            print(f"原始方法准确率: {original_correct}/{total_processed} = {original_accuracy:.3f} ({original_accuracy*100:.1f}%)")
            
            if haar_accuracy > original_accuracy:
                improvement = (haar_accuracy - original_accuracy) * 100
                print(f"🎉 Haar方法准确率提升 {improvement:.1f} 个百分点!")
            elif haar_accuracy < original_accuracy:
                decline = (original_accuracy - haar_accuracy) * 100
                print(f"⚠️  Haar方法准确率下降 {decline:.1f} 个百分点")
            else:
                print("📊 两种方法准确率相同")
            
            # 打印各类别详细结果
            print(f"\n{'='*60}")
            print("各类别详细结果:")
            print(f"{'='*60}")
            print(f"{'类别':>10} {'样本数':>8} {'Haar准确率':>12} {'原始准确率':>12} {'差异':>10}")
            print("-" * 60)
            
            for class_name in self.class_names:
                stats = class_stats[class_name]
                total = stats['total']
                if total > 0:
                    haar_acc = stats['haar_correct'] / total
                    original_acc = stats['original_correct'] / total
                    diff = (haar_acc - original_acc) * 100
                    print(f"{class_name:>10} {total:>8} {haar_acc*100:>11.1f}% {original_acc*100:>11.1f}% {diff:>+9.1f}%")
                else:
                    print(f"{class_name:>10} {'无样本':>8}")
        
        # 保存详细结果到CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv('evaluation_results.csv', index=False)
        print(f"\n详细结果已保存到: evaluation_results.csv")
        
        # 生成可视化图表
        self.generate_visualizations(results_df, class_stats)
        
        return results_df, class_stats
    
    def generate_visualizations(self, results_df, class_stats):
        """生成可视化图表"""
        print("生成可视化图表...")
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Evaluation Results', fontsize=16)
        
        # 1. 整体准确率对比
        methods = ['Haar Method', 'Original Method']
        haar_acc = results_df['haar_correct'].mean()
        original_acc = results_df['original_correct'].mean()
        accuracies = [haar_acc, original_acc]
        
        bars1 = axes[0, 0].bar(methods, accuracies, color=['skyblue', 'lightcoral'])
        axes[0, 0].set_title('Overall Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        
        # 添加数值标签
        for bar, acc in zip(bars1, accuracies):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{acc:.3f}', ha='center', va='bottom')
        
        # 2. 各类别准确率对比
        class_names = []
        haar_accs = []
        original_accs = []
        
        for class_name in self.class_names:
            stats = class_stats[class_name]
            if stats['total'] > 0:
                class_names.append(class_name)
                haar_accs.append(stats['haar_correct'] / stats['total'])
                original_accs.append(stats['original_correct'] / stats['total'])
        
        x = np.arange(len(class_names))
        width = 0.35
        
        bars2 = axes[0, 1].bar(x - width/2, haar_accs, width, label='Haar Method', color='skyblue')
        bars3 = axes[0, 1].bar(x + width/2, original_accs, width, label='Original Method', color='lightcoral')
        
        axes[0, 1].set_title('Per-Class Accuracy Comparison')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_xlabel('Cat Breed')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(class_names, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].set_ylim(0, 1)
        
        # 3. 置信度分布
        axes[1, 0].hist(results_df['haar_confidence'], bins=20, alpha=0.7, label='Haar Method', color='skyblue')
        axes[1, 0].hist(results_df['original_confidence'], bins=20, alpha=0.7, label='Original Method', color='lightcoral')
        axes[1, 0].set_title('Confidence Distribution')
        axes[1, 0].set_xlabel('Confidence')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # 4. 混淆矩阵 (Haar方法)
        true_labels = [self.class_to_idx[label] for label in results_df['true_label']]
        haar_preds = [self.class_to_idx[pred] for pred in results_df['haar_prediction']]
        
        cm = confusion_matrix(true_labels, haar_preds)
        im = axes[1, 1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[1, 1].set_title('Confusion Matrix (Haar Method)')
        
        # 添加颜色条
        plt.colorbar(im, ax=axes[1, 1])
        
        # 添加标签
        tick_marks = np.arange(len(self.class_names))
        axes[1, 1].set_xticks(tick_marks)
        axes[1, 1].set_yticks(tick_marks)
        axes[1, 1].set_xticklabels(self.class_names, rotation=45)
        axes[1, 1].set_yticklabels(self.class_names)
        
        # 添加数值
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            axes[1, 1].text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black")
        
        axes[1, 1].set_ylabel('True Label')
        axes[1, 1].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('evaluation_visualizations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("可视化图表已保存到: evaluation_visualizations.png")

def main():
    """主测试函数"""
    # 检查模型文件
    model_files = [
        'cats_improved_final.keras',
        'cats.keras',
        'best_model_fold_1.keras'
    ]
    
    model_path = None
    for model_file in model_files:
        if os.path.exists(model_file):
            model_path = model_file
            break
    
    if model_path is None:
        print("错误: 未找到可用的模型文件")
        print("请确保以下文件之一存在:")
        for model_file in model_files:
            print(f"  - {model_file}")
        return
    
    # 测试目录
    test_dir = './test_samples'
    if not os.path.exists(test_dir):
        print(f"错误: 测试目录不存在: {test_dir}")
        return
    
    print(f"使用模型: {model_path}")
    print(f"测试目录: {test_dir}")
    
    # 创建测试器并运行评估
    tester = ImprovedTester(model_path, test_dir)
    results_df, class_stats = tester.evaluate_comprehensive()
    
    print("\n评估完成!")

if __name__ == "__main__":
    main()