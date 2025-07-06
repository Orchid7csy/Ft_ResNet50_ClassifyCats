from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
import numpy as np
from PIL import Image
from os import listdir
from os.path import join
import tf_keras
from tensorflow.keras.applications.resnet50 import preprocess_input


MODEL_NAME='cats.keras'

# Our samples directory
SAMPLE_PATH = './test_samples'

dict={0:'Pallas', 1:'Persian', 2:'Ragdoll', 3:'Singapura', 4:'Sphynx'}

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
    # img_array = img_array / 255.0  # 归一化到[0,1]范围
    return img_array



# Test main
def main():
    print("Loading model from ", MODEL_NAME)
    model = tf_keras.models.load_model(MODEL_NAME)
    print("Done")

    print("Now classifying files in ", SAMPLE_PATH)

    sample_files = listdir(SAMPLE_PATH)

    for filename in sample_files:
        filename = join(SAMPLE_PATH, filename)
        img = load_image(filename)
        label,prob,_ = classify(model, img)
        print("image %s is %s. with certainty %3.2f"%(filename, label, prob))

if __name__ == "__main__":
    main()