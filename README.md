# Crop Recommendation System (Phase 1)

A college-demo web application that recommends a suitable crop based on 7
soil and weather inputs using a trained **DecisionTreeClassifier**
(scikit-learn). All predictions come from the trained model — there are no
hardcoded predictions or if/else crop rules.

## Flow

```
User enters 7 soil/weather values
        ↓
Flask Backend (app.py)
        ↓
Trained Decision Tree (model/crop_model.pkl)
        ↓
Crop Prediction + Confidence + Top-3
        ↓
Result shown on the webpage
```

## Project Structure

```
crop-recommendation-system/
├── app.py                     # Flask backend + API
├── train_model.py             # Dataset inspection + Decision Tree training
├── requirements.txt
├── README.md
├── data/
│   └── crop_recommendation.csv   # 2,200 samples, 7 features + label (22 crops)
├── model/
│   ├── crop_model.pkl         # Trained model bundle (joblib)
│   └── metadata.json          # Real metrics, importances, dataset stats
├── templates/
│   └── index.html
└── static/
    ├── style.css
    └── script.js
```

## Setup & Run

```bash
pip install -r requirements.txt

# 1. Train the model (also inspects the dataset and prints metrics)
python train_model.py

# 2. Start the Flask server
python app.py

# 3. Open http://127.0.0.1:5000
```

## Inputs (7 features)

| Feature        | Meaning                  | Typical dataset range |
|----------------|--------------------------|-----------------------|
| N              | Nitrogen ratio in soil   | 0 – 140               |
| P              | Phosphorus ratio in soil | 5 – 145               |
| K              | Potassium ratio in soil  | 5 – 205               |
| temperature    | Temperature (°C)         | ~8.8 – 43.7           |
| humidity       | Relative humidity (%)    | ~14 – 100             |
| ph             | Soil pH                  | ~3.5 – 10             |
| rainfall       | Rainfall (mm)            | ~20 – 299             |

Exact accepted ranges are derived from the dataset by `train_model.py`
(stored in `model/metadata.json`) and shown on the page.

## Dataset

Standard public Crop Recommendation dataset: 2,200 rows, 7 numeric features
(N, P, K, temperature, humidity, ph, rainfall) and a `label` target with 22
balanced crop classes (rice, maize, chickpea, kidneybeans, pigeonpeas,
mothbeans, mungbean, blackgram, lentil, pomegranate, banana, mango, grapes,
watermelon, muskmelon, apple, orange, papaya, coconut, cotton, jute, coffee).
`train_model.py` verifies shape, dtypes, missing values, duplicates, class
balance and the target column before training.

## Model

- `DecisionTreeClassifier(random_state=42, min_samples_leaf=8, ccp_alpha=0.003)`
  (regularization so `predict_proba()` keeps real class proportions in
  crop-overlap leaves instead of always splitting to a single class; tuned so
  ~1.6x more nearby inputs receive genuine multi-class probabilities while
  test accuracy stays at 97.50%)
- Stratified 80/20 train/test split (1,760 / 440 samples)
- Real computed metrics on the test set: Accuracy, weighted Precision,
  weighted Recall, weighted F1 + full per-class report
- Real feature importances from the trained tree
- Saved with joblib; a save/load round-trip check runs during training

## API Routes

| Route              | Method | Purpose                                    |
|--------------------|--------|--------------------------------------------|
| `/`                | GET    | Web page                                   |
| `/api/predict`     | POST   | JSON `{N, P, K, temperature, humidity, ph, rainfall}` → crop, confidence, top-3, input summary, crop info, explanation |
| `/api/metrics`     | GET    | Real evaluation metrics + feature importance |
| `/api/dataset`     | GET    | Dataset info, valid ranges, per-crop means |
| `/api/health`      | GET    | Health/model-loaded check                  |

## Features

- 7 validated inputs (ranges from the actual dataset)
- 5 demo presets (Rice / Maize / Coffee / Tropical / Fruit) — presets only
  fill values; the prediction always comes from the Decision Tree
- Recommended crop + confidence %, top-3 model probabilities
- Input summary, short crop information, honest explanation, feature importance
- Model performance and dataset info sections
- Prediction history in browser localStorage (view / restore / clear)
- Reset button for the current form/result

## Notes / Limitations

- Educational demo: not agronomic advice. The explanation uses feature
  importances and dataset comparisons — a Decision Tree cannot explain
  itself in human language, and none is invented.
- History is browser-local only (no database, per project scope).
