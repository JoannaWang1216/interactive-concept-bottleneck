from http import HTTPStatus

import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

from concept_model.inference import (
    IMAGE_ATTRIBUTES,
    AttributesToClassModel,
    ImageToAttributesModel,
)

app = Flask(__name__)
CORS(app)

IMAGE_TO_ATTRIBUTES_MODEL = ImageToAttributesModel(
    "independent_image_to_attributes.pth"
)

ATTRIBUTES_TO_CLASS_MODEL = AttributesToClassModel(
    "independent_attributes_to_class.pth"
)


@app.route("/predict", methods=["POST"])
def predict_user_input():
    stream = request.files["image"].stream.read()

    (
        recognized_concepts,
        final_prediction_input_concepts,
    ) = IMAGE_TO_ATTRIBUTES_MODEL.predict(stream)

    final_prediction = ATTRIBUTES_TO_CLASS_MODEL.predict(
        final_prediction_input_concepts
    )

    response = jsonify(
        {
            "recognized_concepts": recognized_concepts,
            "final_prediction": final_prediction,
        }
    )
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@app.route("/rerun", methods=["POST"])
def rerun():
    if request.json is None:
        return jsonify({"error": "No JSON provided"}), HTTPStatus.BAD_REQUEST

    updated_concepts = request.json["updated_concepts"]

    updated_concepts_prob_dict = {}
    for page in updated_concepts:
        for concept in page:
            idx = np.asarray(IMAGE_ATTRIBUTES == concept[0]).nonzero()[0][0]
            if concept[1] > 0.5:
                updated_concepts_prob_dict[idx] = 1
            else:
                updated_concepts_prob_dict[idx] = 0

    updated_final_prediction_input_concepts = []
    for i in range(312):
        updated_final_prediction_input_concepts.append(updated_concepts_prob_dict[i])

    updated_final_prediction_input_concepts = torch.tensor(
        updated_final_prediction_input_concepts
    )

    final_prediction = ATTRIBUTES_TO_CLASS_MODEL.predict(
        updated_final_prediction_input_concepts, rerun=True
    )

    response = jsonify(
        {
            "final_prediction": final_prediction,
        }
    )
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response
