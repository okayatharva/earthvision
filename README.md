# EarthVision — Satellite Image Classification

EarthVision is a computer vision project that classifies RGB satellite images into land-use and land-cover categories using a convolutional neural network (CNN) built with PyTorch.

## Features

- EuroSAT RGB satellite imagery
- Image preprocessing and normalization with Torchvision
- Training, validation, and test data splits
- CNN model implemented in PyTorch
- Evaluation using accuracy, precision, recall, F1 score, and a confusion matrix
- Top-three image predictions
- Interactive Streamlit web application
- Automated model tests with pytest

## Technology Stack

Python, PyTorch, Torchvision, NumPy, scikit-learn, Matplotlib, Pillow, Streamlit, pytest.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Dataset and Training

The project uses the EuroSAT RGB dataset. The data pipeline downloads the dataset when required.

```bash
python -m src.train
```

The trained checkpoint is saved to `models/earthvision_cnn.pth`.

## Evaluation

```bash
python -m src.evaluate
```

Evaluation outputs are saved in the `reports/` directory when generated.

## Run the Application

```bash
streamlit run app.py
```

Upload a JPG, JPEG, PNG, or WebP satellite image to view the top-three predicted categories.

## Tests

```bash
python -m pytest -v
```

## Limitations

The model is trained on EuroSAT RGB optical satellite imagery. It is not a SAR or radar image classifier. Predictions on images that differ from the training dataset may be less reliable. Model scores are estimates and should not be treated as verified real-world land-use probabilities.

## Future Improvements

- Improve model performance through experiments
- Add sample images and detailed evaluation results
- Explore transfer learning with pretrained computer vision models
- Expand the application with model evaluation visualizations
