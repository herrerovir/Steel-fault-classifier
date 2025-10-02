# ⚙️🤖 Industrial Steel Defect Detection Using XGBoost
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B2B?style=flat&logo=streamlit&logoColor=white)

This project implements a machine learning solution for automated defect detection in steel manufacturing, with a focus on class imbalance and model interpretability. It walks through the full pipeline, from data cleaning and feature engineering to model training, tuning, and testing on synthetic data.

The final **XGBoost model**, enhanced with **SMOTE** resampling and **SHAP** explainability, delivers strong performance across all defect classes, including the rare ones. Designed with industrial applications in mind, this solution offers both predictive accuracy and transparent decision-making, making it a strong candidate for real-time quality control in production environments.

The model is deployed as a real-time Streamlit app, which allows users to interactively input features and predict defect types. 

👉 [**Try the Live Demo**](https://steel-fault-classifier.streamlit.app/)

📌 Originally built for a technical screening and later adapted into a portfolio project.

## 🎯 Goal

Build a robust machine learning model to accurately classify defects in industrial steel plates. This includes data cleaning, feature exploration, model comparison, hyperparameter tuning, and testing model behavior on synthetic data to ensure generalization.

## 🏭 Real-World Relevance

In steel manufacturing, early and accurate defect detection is critical to reduce production downtime, material waste, and customer returns. 

- **False negatives** or missed defects may let faulty materials reach customers which is costly and reputation-damaging.
- **False positives** or flagging non-defective plates increase manual inspection time and slow production.

This model can serve as a foundational component in an automated inspection pipeline to minimize both failure risks and operational inefficiencies.

## 🗂️ Project Structure

```plaintext
Steel-fault-classifier/
│
├── data/
│   ├── raw/                               # Original dataset
│   ├── processed/                         # Cleaned dataset
│   └── feature-engineering/               # Data after feature engineering
│
├── figures/                               # Visualizations
│   ├── fault-type-distribution.png        
│   └── xgboost-model-confusion-matrix.png                 
│
├── notebooks/                             # Jupyter notebooks
│   └──steel-fault-classifier.ipynb        # End-to-end project
│
├── models/                                # Trained model
│   ├── xgboost_model.json                 # Final model (JSON)
│   └── xgboost_model.pkl                  # Final model (Pickle)
│
├── results/                               # Model output
│   └── metrics                            # Model metrics
│       └── xgboost-model-metrics.txt
│
├── requirements.txt                       # Dependencies
└── README.md                              # Project documentation  
```

## 🧐 Project Overview

1. **Introduction** – Problem overview and motivation
2. **Dataset** – Overview of the data used
3. **Data Cleaning** – Prepare the data for analysis
4. **Exploratory Data Analysis** – Understand distributions and correlations
5. **Feature Engineering** – Create new features to improve modeling
6. **Preprocessing** – Standardize and scale the features
7. **Modeling** – Train and evaluate multiple classification models
8. **XGBoost Optimization** – Hyperparameter tuning and SMOTE resampling
9. **SHAP Explainability** – Analyzing model predictions with SHAP to understand the most impactful features and decision-making process
10. **Robustness Testing** – Testing the model's stability and predictive consistency with synthetic data to simulate new, unseen examples
11. **Results** – Model performance metrics and analysis
12. **Cloud Deployment** – Deployment of the model as a Streamlit app for interactive defect prediction in real-time

## 💻 Installation

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## ▶️ Running the Project

1. Clone the repo

   ```bash
   git clone https://github.com/herrerovir/Steel-fault-classifier.git
   ```
2. Move into the directory

   ```bash
   cd Steel-fault-classifier
   ```
3. Open the notebook

   ```bash
   jupyter notebook
   ```

## 📊 Dataset

The dataset contains 1941 steel plate samples with 27 features and 7 binary labels indicating different defect types. It is sourced from Kaggle’s [Faulty Steel Plates Dataset](https://www.kaggle.com/datasets/uciml/faulty-steel-plates).

## 🧠 Modeling

Multiple classification algorithms were tested, including:

- Decision Tree
- Random Forest
- XGBoost
- Support Vector Machine
- Multilayer Perceptron

XGBoost was selected as the best-performing model. To address class imbalance, **SMOTE** was applied to synthetically augment minority class samples and improve detection of less frequent defect types. Hyperparameter tuning via grid search further optimized model performance. The final XGBoost model demonstrates strong precision and recall across all defect classes.

**Final Model Performance (XGBoost + SMOTE)**

- Overall Accuracy: 0.80
- Macro F1-Score: 0.80
- Weighted F1-Score: 0.80
- ROC AUC Score: 0.97

**Key Takeaways**

- Strong performance on major defect types like classes 1, 2, and 3, each achieving F1-scores above 0.90.
- Good generalization even for smaller classes like class 4 (only 8 samples), thanks to SMOTE.
- High ROC AUC (0.97) shows strong class separation across the board.

## 🔮 Robustness Testing with Synthetic Data

To evaluate model robustness beyond the original dataset, synthetic samples were generated by sampling from the statistical distributions of the training features. This approach simulates new, unseen examples to test model stability and predictive consistency. Results showed the model maintained expected class distribution patterns, especially for frequent defect classes like class 6 (the most common defect type), confirming reliable generalization.

## 🧾 SHAP Explainability

Model predictions were further analyzed using SHAP (SHapley Additive exPlanations) to understand which features influence decisions across and within classes. This helped identify the most impactful features, both globally and locally.

- Global analysis highlighted features like Steel Plate Thickness, Length of Conveyer, and Luminosity Index as key drivers of predictions across classes.
- Local analysis showed how different feature values (e.g., high brightness, thin plates, specific steel types) influenced individual predictions and contributed to both correct classifications and errors.

SHAP was used on the resampled training data (after SMOTE) to stay consistent with what the model actually learned. Visualizations and class-specific summaries provided insights into how the model reasons, where it’s strong, and where it can be sensitive—especially when feature values fall outside typical ranges.

These insights help build trust in the model and offer guidance on where to focus future improvements.

## 🥇 Results

The fine-tuned XGBoost model achieved strong performance with 80% accuracy, 80% macro F1-score, and 97% ROC AUC on the test set. It reliably detects various defects in industrial steel plates, including both common and rare types. The model's predictions are consistent across real and synthetic data, making it suitable for real-time quality control applications in steel manufacturing.

## 🌐 Cloud Deployment

The model has been deployed as a **Streamlit web application** on **Streamlit Cloud**. This deployment allows users to input steel defect parameters and receive real-time predictions directly through a user-friendly web interface.

You can access the deployed app here:

👉 [**Try the Live Demo**](https://steel-fault-classifier.streamlit.app/)
