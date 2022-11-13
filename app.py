from http import HTTPStatus

from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route("/predict", methods=["POST"])
def predict_user_input():
    print(request.files)
    print(request.files["image"].stream.read())
    return "", HTTPStatus.NO_CONTENT
