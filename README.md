# Interactive Concept Bottleneck Model

CS593HAI Human-AI Interaction programming assignment 2 @Purdue.

## Getting Started

This project only supports Ubuntu 20.04 or later.

Install Python 3.10.x.

Install Poetry **1.2 or later**. See
[Poetry's documentation](https://python-poetry.org/docs/) for details.

> Poetry earlier than 1.2 will not work with this project.

Install the project's dependencies:

```sh
poetry install --no-root
```

Activate the virtual environment:

```sh
poetry shell
```

Start the application:

```sh
flask run
```

> Make sure you are in the project's root directory and your have activated the
> virtual environment.

Open the `index.html` file in the `ui` folder in your browser.

## Model Architecture

The accuracy of the pipeline on the test set of the bird dataset can be found in `test_models.ipynb`. Also, you can find the inference script in `concept_model/inference.py`.

### Independent Model

Model Accuracy: 25.73%

#### [Image to Concepts Model](./independent_image_to_attributes.ipynb)

- A pre-trained Inception_v3 model (use finetuning)
- Loss function: BCEWithLogitsLoss
- Optimizer: SGD
- Learning rate: 0.01
- Momentum: 0.9
- Epochs: 100
- Accuracy: 91.9% (per attribute)

#### [Concepts to Class Model](./independent_attributes_to_class.ipynb)

- A 2-layer MLP
- Loss function: CrossEntropyLoss
- Optimizer: SGD
- Learning rate: 0.001
- Momentum: 0.9
- Epochs: 200
- Accuracy: 49.24%

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.
