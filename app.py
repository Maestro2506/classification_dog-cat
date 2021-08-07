from flask import Flask, request, abort, jsonify
import numpy as np
import cv2
import base64
from tensorflow.keras.models import load_model
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=bool, default=False, help='set to True if want to use your gpu')
args = parser.parse_args()

print(args.gpu)
if args.gpu:
    device = tf.device("/GPU:0")
else:
    device = tf.device("/CPU:0")

def base64_to_arr(element): #converts base64 image to numpy array
    img_code = base64.b64decode(element['img_code'])
    np_arr = np.frombuffer(img_code, np.uint8)
    return np_arr

def image_preprocessing(np_arr): #array resize, normalization and dimension expansion
    img_arr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img_arr = cv2.resize(img_arr, (64, 64))
    img_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB).astype(np.float64)
    img_rgb = img_rgb/255
    final_img = np.expand_dims(img_rgb, axis=0)
    return final_img

app = Flask(__name__)

classifier = load_model('dogcat_model_bak.h5')

@app.route('/predict', methods=['POST'])
def predict():
    if not request.json:
        abort(400)
    predictions = []
    for element in request.json["photos"]:
        img = image_preprocessing(base64_to_arr(element))
        with device:
            prediction = classifier.predict(img, batch_size=None, steps=1)
        predictions.append({'ID': element['ID'],
                            'cat_prob': str(1 - prediction[0,0]),
                            'dog_prob': str(prediction[0,0])})
    return jsonify({'results': predictions}), 200


if __name__ == '__main__':
    app.run()
