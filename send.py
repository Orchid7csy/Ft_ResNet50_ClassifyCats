import paho.mqtt.client as mqtt
import numpy as np
import json
from PIL import Image
from os import listdir
from os.path import join, isfile
from time import sleep

PATH = "./samples"

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected successfully with result code " + str(rc))
        client.subscribe("Group2/IMAGE/predict")

    else:
        print("Connection failed with result code " + str(rc))

def on_message(client, userdata, msg):
    print("Received message on topic:", msg.topic)
    resp_dict = json.loads(msg.payload)
    print("Prediction: %s, Score: %3.4f" % (resp_dict['prediction'], float(resp_dict['score'])))

def setup(hostname):
    client = mqtt.Client()
    client.username_pw_set("csy", "1")  # Set username and password
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(hostname)
    client.loop_start()  # Start the loop to process network traffic and callbacks
    return client

def load_image(filename):
    img = Image.open(filename)
    img = img.resize((249, 249))  # Resize to the expected input size
    imgarray = np.array(img)/255.0
    final = np.expand_dims(imgarray, axis=0)  # Add batch dimension
    return final

def send_image(client, filename):
    img = load_image(filename)
    img_list = img.tolist()  # Convert to list for JSON serialization
    send_dict = {
        "filename": filename,
        "data": img_list
    }
    client.publish("Group2/IMAGE/classify", json.dumps(send_dict))

def main():
    client = setup("127.0.0.1")
    sleep(1)

    # Get all files from the directory [cite: 343]
    files = [f for f in listdir(PATH) if isfile(join(PATH, f))]

    for file in files:
        # Ignore system files like .DS_Store
        if file.startswith('.'):
            continue

        filepath = join(PATH, file)
        print(f"Sending data for {filepath}...")
        send_image(client, filepath)
        sleep(1) # Wait a moment before sending the next file

    print("All files sent. Waiting for final results...")
    # The program will continue to run and receive messages in the background
    # due to client.loop_start()
    sleep(10) # Wait for 10 seconds to receive all messages
    print("Done.")
    
    while True:
        pass

if __name__ == "__main__":
    main()
