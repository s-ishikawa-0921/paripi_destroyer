# -*- coding: utf-8 -*-
from flask import Flask, jsonify, abort, make_response, request, render_template
from flask_cors import CORS

import cv2
import base64
import uuid
import argparse
import os
import numpy as np
import tensorflow as tf
import json
import werkzeug
from datetime import datetime

import object_detection_util


# ---------------------------------------------------------------------------------
# PRIVATE METHOD
# ---------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
UPLOAD_DIR = 'temp/'


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_paripi/', methods=['POST'])
def check_paripi():

    file_path = save_file()

    result = object_detection_util.check(
        file_path,
        "../paripi_object_detection/paripi_check/retrained_graph.pb",
        "../paripi_object_detection/paripi_check/retrained_labels.txt"
    )
    remove_file(file_path)

    return make_response(jsonify(result))
    # Unicodeにしたくない場合は↓
    # return make_response(json.dumps(result, ensure_ascii=False))

@app.route('/destroy_paripi/', methods=['POST'])
def destroy_paripi():

    image_path = save_file()

    result_image = object_detection_util.mosaic(
        image_path,
        "../paripi_object_detection/paripi_mosaic/exported_graphs/frozen_inference_graph.pb",
        "../models/research/object_detection/data/mscoco_label_map.pbtxt",
        "../paripi_object_detection/paripi_mosaic/karintou.png"
    )

    cv2.imwrite(image_path, result_image)
    b64 = base64.encodestring(open(image_path, 'rb').read())

    remove_file(image_path)

    result = {
        "binary": 'data:image/png;base64,' + b64.decode('utf8')
    }
    return make_response(jsonify(result))


def save_file():
    if 'uploadFile' not in request.files:
        print('uploadFile is required.')
        make_response(jsonify({'result':'uploadFile is required.'}))

    file = request.files['uploadFile']
    fileName = file.filename

    if '' == fileName:
        make_response(jsonify({'result':'filename must not empty.'}))

    saveFileName = datetime.now().strftime("%Y%m%d_%H%M%S_") \
        + werkzeug.utils.secure_filename(fileName)

    file.save(os.path.join(UPLOAD_DIR, saveFileName))

    return UPLOAD_DIR+saveFileName
def remove_file(file_path):
    os.remove(file_path)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
