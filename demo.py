from flask import Flask, request, jsonify
from Latex.Latex import Latex
import numpy as np
import tensorflow as tf
import cv2
from skimage import io
from flask_cors import CORS,cross_origin
import tensorflow.contrib.legacy_seq2seq as seq2seq

app = Flask(__name__)
CORS(app, support_credentials=True)
app.config['CORS_HEADERS'] = 'Content-Type'

# load the learner
mean_train = np.load("train_images_mean.npy")
std_train = np.load("train_images_std.npy")

tf.reset_default_graph()
model = Latex("model", mean_train, std_train, plotting=False)

def predict_single(img_file):
    'function to take image and return prediction'
    formula = io.imread(img_file)
    formula = cv2.cvtColor(formula, cv2.COLOR_BGR2GRAY)
    latex = model.predict(formula)
    return {'equation': latex['formula']}
#     prediction = learn.predict(open_image(img_file))
#     probs_list = prediction[2].numpy()
#     return {
#         'category': classes[prediction[1].item()],
#         'probs': {c: round(float(probs_list[i]), 5) for (i, c) in enumerate(classes)}


# route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    return jsonify(predict_single(request.files['image']))


@app.route('/')
def index():
    return "<h1>Welcome to our server !!</h1>"

if __name__ == '__main__':
    app.run()
