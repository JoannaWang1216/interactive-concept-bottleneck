from io import BytesIO

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from torchvision.ops import MLP

from concept_model.dataset import DEFAULT_IMAGE_TRANSFORM, ROOT

IMAGE_ATTRIBUTES: npt.NDArray[np.str_] = np.genfromtxt(
    ROOT / "attributes.txt", usecols=(1,), dtype=np.str_
).flatten()

CLASSES: npt.NDArray[np.str_] = np.genfromtxt(
    ROOT / "CUB_200_2011" / "classes.txt", usecols=(1,), dtype=np.str_
).flatten()


class ImageToAttributesModel:
    def __init__(self, model_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0", "inception_v3", weights=None, num_classes=312
        )
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, img_stream: bytes) -> tuple[dict[np.str_, float], torch.Tensor]:
        img = Image.open(BytesIO(img_stream))
        img = DEFAULT_IMAGE_TRANSFORM(img)
        img.to(self.device)  # type: ignore
        with torch.no_grad():
            attributes = self.model(img.unsqueeze(0).to(self.device))  # type: ignore
            concepts_prob = torch.nn.Sigmoid()(attributes)
            final_prediction_input_concepts = (concepts_prob > 0.5).to(torch.int64)

        recognized_concepts: dict[np.str_, float] = {}
        for i, concept_prob in enumerate(concepts_prob[0]):
            recognized_concepts[IMAGE_ATTRIBUTES[i]] = concept_prob.item()

        return recognized_concepts, final_prediction_input_concepts


class AttributesToClassModel:
    def __init__(self, model_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = MLP(
            in_channels=312,
            hidden_channels=[200],
            dropout=0.2,
        )
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self, final_prediction_input_concepts: torch.Tensor
    ) -> dict[np.str_, float]:
        final_prediction_input_concepts.to(self.device)
        with torch.no_grad():
            class_prediction = self.model(
                final_prediction_input_concepts.to(torch.float)
            )
            species_probs = torch.nn.Softmax(dim=1)(class_prediction)[0]

        final_prediction: dict[np.str_, float] = {}
        for i, species_prob in enumerate(species_probs):
            final_prediction[CLASSES[i]] = species_prob.item()

        return final_prediction
