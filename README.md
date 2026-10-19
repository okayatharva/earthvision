# EarthVision — Satellite Image Classification

EarthVision is a computer vision project that classifies RGB satellite images into land-use and land-cover categories using a Convolutional Neural Network (CNN) built with PyTorch.

## Features

- EuroSAT RGB satellite imagery
- Image preprocessing and normalization with Torchvision
- Training, validation, and test data splits
- CNN model implemented using PyTorch
- Model evaluation using accuracy, precision, recall, F1 score, and a confusion matrix
- Top-three image predictions
- Interactive Streamlit web application
- Automated model tests using pytest

## Technology Stack

- **Language:** Python
- **Deep Learning:** PyTorch, Torchvision
- **Data Processing:** NumPy, scikit-learn
- **Visualization:** Matplotlib
- **Image Processing:** Pillow
- **Web Application:** Streamlit
- **Testing:** pytest

## Project Setup

### 1. Clone the repository

```bash
git clone https://github.com/okayatharva/earthvision.git
cd earthvision
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Dataset and Training

EarthVision uses the EuroSAT RGB satellite image dataset for land-use and land-cover classification. The data pipeline downloads the dataset when required.

To train the model, run:

```bash
python -m src.train
```

The trained model checkpoint is saved to:

```text
models/earthvision_cnn.pth
```

## Model Evaluation

To evaluate the trained model, run:

```bash
python -m src.evaluate
```

The evaluation process uses metrics such as accuracy, precision, recall, and F1 score. Evaluation outputs are saved in the `reports/` directory when generated.

## Run the Streamlit Application

Start the web application locally with:

```bash
streamlit run app.py
```

Upload a JPG, JPEG, PNG, or WebP satellite image to view the model's top-three predicted land-cover categories.

## Live Demo

Try the deployed application:

**https://earthvision-a.streamlit.app/**

## Tests

Run the automated tests using:

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

## Author

**Atharva Vichare**

GitHub: https://github.com/okayatharva
