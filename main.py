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


MODEL_NAME='debug_cats_best.h5'

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

# Load image
def load_image(image_fname):
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

    sample_files = listdir(SAMPLE_PATH)
    
    # Initialize counters for accuracy calculation
    total_predictions = 0
    correct_predictions = 0
    
    # Track predictions per class for detailed statistics
    class_stats = {breed: {'total': 0, 'correct': 0} for breed in dict.values()}

    for filename in sample_files:
        full_path = join(SAMPLE_PATH, filename)
        
        # Extract true label from filename
        true_label = extract_true_label(filename)
        
        if true_label is None:
            print(f"Warning: Could not extract breed label from filename: {filename}")
            continue
            
        img = load_image(full_path)
        predicted_label, prob, _ = classify(model, img)
        
        # Update statistics
        total_predictions += 1
        class_stats[true_label]['total'] += 1
        
        if predicted_label == true_label:
            correct_predictions += 1
            class_stats[true_label]['correct'] += 1
            status = "✓ CORRECT"
        else:
            status = "✗ WRONG"
        
        print(f"image {filename}: true={true_label}, predicted={predicted_label} (certainty {prob:.2f}) {status}")

    # Calculate and display overall accuracy
    if total_predictions > 0:
        overall_accuracy = correct_predictions / total_predictions
        print("\n" + "="*60)
        print(f"OVERALL ACCURACY: {correct_predictions}/{total_predictions} = {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")
        print("="*60)
        
        # Display per-class accuracy
        print("\nPER-CLASS ACCURACY:")
        print("-" * 40)
        for breed in dict.values():
            total = class_stats[breed]['total']
            correct = class_stats[breed]['correct']
            if total > 0:
                accuracy = correct / total
                print(f"{breed:>10}: {correct:>2}/{total:<2} = {accuracy:.3f} ({accuracy*100:.1f}%)")
            else:
                print(f"{breed:>10}: No samples")
    else:
        print("\nNo valid predictions were made.")

if __name__ == "__main__":
    main()