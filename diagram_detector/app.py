from flask import Flask, request, jsonify, abort

import base64
import numpy as np
from PIL import Image
import io
from pathlib import Path

from hello import detect_diagram_objects


dir_project = Path().resolve()
PATH_DIAGRAM_GURU_PROJECT = str(dir_project)
PATH_DIAGRAM_UPLOADS = PATH_DIAGRAM_GURU_PROJECT + '/uploads/'


app = Flask(__name__)


# @app.route('/api/v1/detect', methods=['POST'])
@app.route('/api/v1/detect', methods=['POST'])
def detect():

    # if not request.json or 'image' not in request.json:
    #     abort(400)

    # get the base64 encoded string
    im_b64 = request.json['image']

    # convert it into bytes
    img_bytes = base64.b64decode(im_b64.encode('utf-8'))

    # convert bytes data to PIL Image object
    img = Image.open(io.BytesIO(img_bytes))

    # PIL image object to numpy array
    img_arr = np.asarray(img)
    print('img shape', img_arr.shape)

    diagram_name = request.json['diagram_name']

    save_image_to_local(img, diagram_name)

    dict_objects = detect_diagram_objects(diagram_name, img_arr)

    return jsonify(dict_objects)

    # result_dict = {'output': 'output_key'}
    # return result_dict


def save_image_to_local(img_bytes, diagram_name):
    path_diagram = PATH_DIAGRAM_UPLOADS + diagram_name
    img_bytes.save(path_diagram)


# @app.route("/")
# def index():
#     return "This is API for detecting diagram objects."


if __name__ == "__main__":
    # app.run()
    app.run(host="0.0.0.0", port=5000, threaded=False)

    # app.run(host="0.0.0.0", port=5000, debug=True)
    # app.run(debug=True)
    # app.run()
