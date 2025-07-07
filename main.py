from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
import numpy as np
from PIL import Image
from os import listdir
from os.path import join
import tf_keras
from tensorflow.keras.applications.resnet50 import preprocess_input
import re
import cv2
import os
from image_preprocessing import letterbox_preprocess

# 导入find_real_img模块中的Haar检测函数
from find_real_img import find_cat_with_haar

MODEL_NAME='cats.keras'

# Our samples directory
SAMPLE_PATH = './test_samples'

dict={0:'Pallas', 1:'Persian', 2:'Ragdoll', 3:'Singapura', 4:'Sphynx'}

# Create reverse dictionary for label to index mapping
label_to_index = {label: index for index, label in dict.items()}

# Function to extract true label from filename
def extract_true_label(filename):
    """
    Extract cat breed from filename
    Returns the breed name if found, None otherwise
    """
    # Convert filename to lowercase for case-insensitive matching
    filename_lower = filename.lower()
    
    # Check each breed name in the filename
    for breed in dict.values():
        if breed.lower() in filename_lower:
            return breed
    
    return None

# Takes in a loaded model, an image in numpy matrix format,
# And a label dictionary
def classify(model, image):
    result = model.predict(image)
    themax = np.argmax(result)
    return (dict[themax], result[0][themax], themax)

# Load image with Haar detection
def load_image_with_haar(image_fname):
    """
    Load image and apply Haar cascade detection for cat faces.
    If cat face is detected, use the detected region; otherwise use original image.
    """
    # 1) 使用OpenCV读取图像
    cv_image = cv2.imread(image_fname)
    if cv_image is None:
        raise ValueError(f"Could not load image: {image_fname}")
    
    # 2) 使用Haar检测猫脸
    try:
        cat_faces = find_cat_with_haar(cv_image)
        
        # 检查返回值是否有效
        if cat_faces is not None and len(cat_faces) > 0:
            # 检测到猫脸，使用第一个检测到的猫脸区域
            x, y, w, h = cat_faces[0]
            cat_face_roi = cv_image[y:y+h, x:x+w]
            # 转换为PIL Image
            img = Image.fromarray(cv2.cvtColor(cat_face_roi, cv2.COLOR_BGR2RGB))
            detection_status = f"Haar detected ({len(cat_faces)} face(s))"
        else:
            # 没有检测到猫脸，使用原图
            img = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            detection_status = "No detection, using original"
            
    except Exception as e:
        # 如果Haar检测出错，也使用原图
        print(f"Haar detection error for {image_fname}: {e}")
        img = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        detection_status = "Detection error, using original"
    
    # 3) 使用letterboxing调整到目标尺寸（与训练时一致）
    # 先转换PIL图片到numpy数组（RGB格式）
    img_array = np.array(img)

    # 应用letterboxing预处理
    img_processed = letterbox_preprocess(img_array, (224, 224))

    # 4) 转为numpy数组并扩展维度
    img_array = np.array(img_processed, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # 形状 (1,224,224,3)

    # 5) 调用 ResNet50 的预处理
    img_array = preprocess_input(img_array)

    return img_array, detection_status

# Original load image function for comparison
def load_image_original(image_fname):
    """
    Original image loading function without Haar detection
    """
    # 1) 打开并强制 RGB
    img = Image.open(image_fname).convert('RGB')

    # 2) 调整大小到 224x224（模型输入要求）
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        # Pillow < 9.1.0
        resample = Image.LANCZOS
        
    img = img.resize((224, 224), resample)
 
    # 3) 转为 numpy 数组并扩展维度
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # 形状 (1,224,224,3)
    # 4) 调用 ResNet50 的预处理
    img_array = preprocess_input(img_array)
    return img_array

# Test main
def main():
    print("Loading model from ", MODEL_NAME)
    model = tf_keras.models.load_model(MODEL_NAME)
    print("Done")

    print("Now classifying files in ", SAMPLE_PATH)
    print("Testing with Haar cascade cat face detection...")
    print("=" * 80)

    sample_files = listdir(SAMPLE_PATH)
    
    # Initialize counters for accuracy calculation
    total_predictions = 0
    correct_predictions = 0
    correct_predictions_original = 0
    haar_detections = 0
    
    # Track predictions per class for detailed statistics
    class_stats = {breed: {'total': 0, 'correct': 0, 'correct_original': 0} for breed in dict.values()}

    for filename in sample_files:
        full_path = join(SAMPLE_PATH, filename)
        
        # Extract true label from filename
        true_label = extract_true_label(filename)
        
        if true_label is None:
            print(f"Warning: Could not extract breed label from filename: {filename}")
            continue
        
        try:
            # Test with Haar detection
            img_haar, detection_status = load_image_with_haar(full_path)
            predicted_label_haar, prob_haar, _ = classify(model, img_haar)
            
            # Test with original method for comparison
            img_original = load_image_original(full_path)
            predicted_label_original, prob_original, _ = classify(model, img_original)
            
            # Update statistics
            total_predictions += 1
            class_stats[true_label]['total'] += 1
            
            if "Haar detected" in detection_status:
                haar_detections += 1
            
            # Check Haar method accuracy
            if predicted_label_haar == true_label:
                correct_predictions += 1
                class_stats[true_label]['correct'] += 1
                status_haar = "✓ CORRECT"
            else:
                status_haar = "✗ WRONG"
            
            # Check original method accuracy
            if predicted_label_original == true_label:
                correct_predictions_original += 1
                class_stats[true_label]['correct_original'] += 1
                status_original = "✓ CORRECT"
            else:
                status_original = "✗ WRONG"
            
            print(f"\nFile: {filename}")
            print(f"  True label: {true_label}")
            print(f"  Detection: {detection_status}")
            print(f"  Haar method: {predicted_label_haar} (certainty {prob_haar:.3f}) {status_haar}")
            print(f"  Original method: {predicted_label_original} (certainty {prob_original:.3f}) {status_original}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    # Calculate and display results
    if total_predictions > 0:
        haar_accuracy = correct_predictions / total_predictions
        original_accuracy = correct_predictions_original / total_predictions
        detection_rate = haar_detections / total_predictions
        
        print("\n" + "="*80)
        print("COMPARISON RESULTS:")
        print("="*80)
        print(f"Total images processed: {total_predictions}")
        print(f"Haar detections: {haar_detections} ({detection_rate*100:.1f}%)")
        print(f"")
        print(f"HAAR METHOD ACCURACY: {correct_predictions}/{total_predictions} = {haar_accuracy:.3f} ({haar_accuracy*100:.1f}%)")
        print(f"ORIGINAL METHOD ACCURACY: {correct_predictions_original}/{total_predictions} = {original_accuracy:.3f} ({original_accuracy*100:.1f}%)")
        print(f"")
        if haar_accuracy > original_accuracy:
            improvement = (haar_accuracy - original_accuracy) * 100
            print(f"🎉 HAAR METHOD IMPROVED ACCURACY BY {improvement:.1f} percentage points!")
        elif haar_accuracy < original_accuracy:
            decline = (original_accuracy - haar_accuracy) * 100
            print(f"⚠️  HAAR METHOD DECREASED ACCURACY BY {decline:.1f} percentage points")
        else:
            print("📊 BOTH METHODS ACHIEVED SAME ACCURACY")
        
        print("\n" + "="*80)
        print("PER-CLASS COMPARISON:")
        print("="*80)
        print(f"{'Breed':>10} {'Samples':>8} {'Haar Acc':>10} {'Original Acc':>12} {'Difference':>12}")
        print("-" * 80)
        for breed in dict.values():
            total = class_stats[breed]['total']
            correct_haar = class_stats[breed]['correct']
            correct_original = class_stats[breed]['correct_original']
            if total > 0:
                acc_haar = correct_haar / total
                acc_original = correct_original / total
                diff = (acc_haar - acc_original) * 100
                print(f"{breed:>10} {total:>8} {acc_haar*100:>9.1f}% {acc_original*100:>11.1f}% {diff:>+11.1f}%")
            else:
                print(f"{breed:>10} {'No samples':>8}")
    else:
        print("\nNo valid predictions were made.")

if __name__ == "__main__":
    main()