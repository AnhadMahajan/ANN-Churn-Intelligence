# ANN Churn Intelligence

A compact, production-ready overview for the ANN Churn Intelligence repository. This project contains training artifacts, notebooks, and a saved Keras model for predicting customer churn.

## Project Overview

- Purpose: Predict customer churn using an Artificial Neural Network (ANN).
- Inputs: tabular customer data (CSV) in the `Data/` folder.
- Outputs: a trained model (`model.h5`) and encoder artifacts stored under `artifacts/encoders/`.

## Quick Start

Prerequisites:

- Python 3.8+ (recommend 3.9)
- Create and activate a virtual environment

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the primary entry (example):

```bash
python app.py
```

Notes:

- `app.py` is provided as an inference or demo entrypoint — inspect it to see how the model is loaded and invoked.
- Notebooks `Experiments.ipynb` and `Prediction.ipynb` contain exploratory work and example inference steps.

## Directory Layout

- `app.py` — example inference/demo entrypoint
- `model.h5` — trained Keras model
- `artifacts/encoders/` — preprocessing encoders (pickles/serialised objects)
- `Data/Churn_Modelling.csv` — original dataset used for experiments
- `logs/` — TensorBoard logs and training metrics
- `Experiments.ipynb` — training experiments and notes
- `Prediction.ipynb` — inference examples

## Usage

1. Prepare environment and install dependencies (see Quick Start).
2. Confirm the required artifacts exist: `model.h5` and encoder files in `artifacts/encoders/`.
3. Run the demo or integrate model loading into your service.

Example: load model and run a single prediction (adapt encoder and preprocessing to your artifacts):

```python
from tensorflow.keras.models import load_model
import pickle
import pandas as pd

# load model
model = load_model('model.h5')

# load encoders (adjust filenames to match what's in artifacts/encoders/)
with open('artifacts/encoders/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# example input as DataFrame
raw = pd.DataFrame([{
    'CreditScore': 600,
    'Geography': 'France',
    'Gender': 'Male',
    'Age': 40,
    # ... fill remaining features as expected by the pipeline
}])

# transform and predict
X = preprocessor.transform(raw)
pred = model.predict(X)
churn_prob = float(pred[0, 0])
churn = bool(churn_prob > 0.5)
print({'churn_prob': churn_prob, 'churn': churn})
```

Adjust thresholds, preprocessing and model input shape as required by your training configuration.

## Training

- Reproduce experiments with `Experiments.ipynb` or reimplement a training script using `Data/Churn_Modelling.csv`.
- Ensure you persist any preprocessing steps (scalers, encoders) into `artifacts/encoders/` to guarantee consistent inference.

Training tips:

- Use TensorBoard logs in `logs/` to inspect training/validation metrics.
- Save model checkpoints and retain the best model by validation metric.

## Configuration & Logging

- Logging: training logs and TensorBoard events are under `logs/`.
- If you add environment-specific configuration, prefer environment variables or a small `config.yml` rather than committing secrets.

## Testing & Validation

- The repository currently contains notebooks demonstrating the workflow — convert key checks into unit/integration tests for CI.
- Validate model performance on a hold-out set and verify data schema before deploying.

## Deployment

- Containerize with a minimal `Dockerfile` and run the inference entrypoint behind a WSGI server (e.g., Gunicorn) if exposing an HTTP API.
- Serve the model in a scalable way (TF Serving, TorchServe, or a simple Flask/Gunicorn app) depending on latency and throughput requirements.

## Troubleshooting

- Missing encoders: confirm files in `artifacts/encoders/` and update paths used in the code.
- Shape mismatches: verify the input order and column names match the preprocessing pipeline used at training time.

## Contributing

- Follow conventional commits and add clear PR descriptions.
- Add tests for new code and update notebooks/examples.

## License

Specify the license for your project here (e.g., MIT). Add a `LICENSE` file to the repository.

## Contact

For questions about the project, open an issue or contact the maintainer listed in the repository metadata.
