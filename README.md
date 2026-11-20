# 🛡️ NetGuard AI

**Network Intrusion Detection System using Autoencoder, Random Forest, and XGBoost**

NetGuard AI is a machine learning project designed to analyze network traffic and identify potentially malicious activities. It combines Autoencoder-based feature extraction with two classification models: Random Forest and XGBoost.

The project also includes a FastAPI backend and a Streamlit dashboard for interacting with the trained models and analyzing network traffic.

---

## 📌 Project Overview

Network intrusion detection plays an important role in protecting computer systems and networks.

NetGuard AI explores how deep learning and traditional machine learning can work together to detect suspicious network traffic.

The project uses the **UNSW-NB15 dataset** for model development and evaluation.

### Main Objectives

- Preprocess network traffic data.
- Learn useful representations using an Autoencoder.
- Extract latent features and reconstruction errors.
- Train and compare Random Forest and XGBoost classifiers.
- Evaluate model performance using classification metrics.
- Provide an API and an interactive dashboard for predictions.

---

## 🧠 System Architecture

The detection pipeline follows these steps:

1. **Data Preprocessing**
   - Handle missing and infinite values.
   - Encode categorical features.
   - Scale numerical features.
   - Prepare training and testing data.

2. **Autoencoder**
   - Train the Autoencoder using normal network traffic.
   - Learn a compact representation of the input features.
   - Extract latent features and reconstruction error.

3. **Classification**
   - Train a Random Forest classifier.
   - Train an XGBoost classifier.
   - Use the same extracted Autoencoder features for both models.

4. **Evaluation**
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - ROC-AUC
   - Confusion Matrix

5. **Deployment**
   - FastAPI for prediction endpoints.
   - Streamlit for the interactive dashboard.

---

## 🛠️ Technologies

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Deep Learning | TensorFlow / Keras |
| Machine Learning | Scikit-learn, XGBoost |
| API | FastAPI |
| Frontend | Streamlit |
| Visualization | Matplotlib, Seaborn |
| Environment Management | uv |

---

## 📂 Project Structure

```text
NetGuard-AI/
│
├── backend/
│   ├── main.py
│   ├── predictor.py
│   └── schemas.py
│
├── frontend/
│   └── app.py
│
├── training/
│   ├── preprocessing.py
│   ├── train_autoencoder_rf.py
│   ├── train_autoencoder_xgboost.py
│   └── evaluate_models.py
│
├── models/
│   ├── autoencoder.keras
│   ├── encoder.keras
│   ├── random_forest.pkl
│   └── xgboost_autoencoder.json
│
├── results/
│   ├── ae_rf_metrics.json
│   ├── ae_xgb_metrics.json
│   └── model_comparison.csv
│
├── src/
│   └── netguardai/
│
├── dataset/                 # Local dataset, not included in Git
│
├── .gitignore
├── .gitattributes
├── pyproject.toml
├── uv.lock
└── README.md
```

*Note: The structure above describes the main project components. Some generated files may differ depending on the current development stage.*

---

## 📊 Dataset

This project uses the **UNSW-NB15 dataset**, a network intrusion detection dataset containing normal traffic and different types of attacks.

The dataset is not included in this repository.

Download the dataset separately and place the training and testing CSV files in the local `dataset/` directory:

```text
dataset/
├── UNSW_NB15_training-set.csv
└── UNSW_NB15_testing-set.csv
```

The dataset directory is excluded from Git to avoid uploading large data files.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/naibimohmed/NetGuard-AI.git

cd NetGuard-AI
```

### 2. Install uv

If you do not already have uv installed, follow its official installation instructions:

https://docs.astral.sh/uv/

### 3. Install project dependencies

```bash
uv sync
```

The project requires Python 3.12 or later, according to `pyproject.toml`.

---

## 🚀 Running the Project

### 1. Start the FastAPI backend

From the project root:

```bash
uv run uvicorn backend.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### 2. Start the Streamlit dashboard

Open another terminal:

```bash
uv run streamlit run frontend/app.py
```

The dashboard will be available at:

```text
http://localhost:8501
```

### 3. Analyze network traffic

1. Open the Streamlit dashboard.
2. Select a detection model.
3. Navigate to Traffic Analysis.
4. Upload a network traffic CSV file.
5. Start the analysis.
6. Review the predictions and download the results.

The CSV analysis feature requires the `/predict-batch` endpoint to be available in the backend.

---

## 📈 Model Evaluation

The project compares two classification approaches using features extracted by the same Autoencoder:

| Model | Feature Extraction | Classifier |
|---|---|---|
| Model 1 | Autoencoder | Random Forest |
| Model 2 | Autoencoder | XGBoost |

The evaluation focuses on:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

Actual evaluation results are stored in the `results/` directory.

---

## 🔮 Future Improvements

- Improve detection performance and generalization.
- Add more detailed attack analysis.
- Improve dashboard visualizations.
- Add prediction history and database storage.
- Support larger CSV files through batch processing.
- Add automated testing and deployment.

---

## 👨‍💻 Author

**Mohamed Naibi**

- GitHub: [@naibimohmed](https://github.com/naibimohmed)
- LinkedIn: [Mohamed Naibi](https://www.linkedin.com/in/mohamed-naibi-979664352/)

---

## 📄 License

This project is developed for educational and research purposes.

A formal open-source license can be added to the repository if the project is intended for public reuse.