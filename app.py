from http import HTTPStatus
from io import BytesIO

import numpy as np
import numpy.typing as npt
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from torchvision import transforms
from torchvision.ops import MLP

from concept_model.dataset import ROOT

app = Flask(__name__)
CORS(app)

IMAGE_ATTRIBUTES: npt.NDArray[np.str_] = np.genfromtxt(
    ROOT / "attributes.txt", usecols=(1,), dtype=np.str_
).flatten()

CLASSES: npt.NDArray[np.str_] = np.genfromtxt(
    ROOT / "CUB_200_2011" / "classes.txt", usecols=(1,), dtype=np.str_
).flatten()


@app.route("/predict", methods=["POST"])
def predict_user_input():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stream = request.files["image"].stream.read()
    img = Image.open(BytesIO(stream))

    preprocess = transforms.Compose(
        [
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = preprocess(img)
    img.to(device)  # type: ignore

    image_to_attributes_model = torch.hub.load(
        "pytorch/vision:v0.10.0", "inception_v3", pretrained=False, num_classes=312
    )
    image_to_attributes_model.load_state_dict(torch.load("image_to_attributes.pth"))
    image_to_attributes_model.to(device)
    image_to_attributes_model.eval()

    with torch.no_grad():
        attributes = image_to_attributes_model(img.unsqueeze(0).to(device))  # type: ignore
        concepts_prob = torch.nn.Sigmoid()(attributes)
        final_prediction_input_concepts = (concepts_prob > 0.5).to(torch.int64)

    recognized_concepts: dict[str, float] = {}
    for i, concept_prob in enumerate(concepts_prob[0]):
        recognized_concepts[IMAGE_ATTRIBUTES[i]] = concept_prob.item()

    attributes_to_class_model = MLP(
        in_channels=312,
        hidden_channels=[200],
        dropout=0.2,
    )
    attributes_to_class_model.load_state_dict(torch.load("attributes_to_class.pth"))
    attributes_to_class_model.to(device)
    attributes_to_class_model.eval()

    final_prediction_input_concepts.to(device)
    with torch.no_grad():
        class_prediction = attributes_to_class_model(
            final_prediction_input_concepts.to(torch.float)
        )
        species_probs = torch.nn.Softmax(dim=1)(class_prediction)[0]

    final_prediction: dict[str, float] = {}
    for i, species_prob in enumerate(species_probs):
        final_prediction[CLASSES[i]] = species_prob.item()

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
    print(updated_concepts, type(updated_concepts))
    return "", HTTPStatus.NO_CONTENT
