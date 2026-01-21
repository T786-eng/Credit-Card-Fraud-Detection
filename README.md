# Fraud Detection System

A machine learning project that detects fraudulent credit card transactions using a Random Forest Classifier. This system generates synthetic transaction data, preprocesses it, trains a model, and visualizes the performance using Confusion Matrices and ROC Curves.

## 📌 Project Overview

This project implements a complete pipeline for fraud detection:
1.  **Data Generation:** Creates a synthetic dataset with a realistic class imbalance (approx. 5% fraud).
2.  **Preprocessing:** Scales features using `StandardScaler` and handles train-test splitting.
3.  **Modeling:** Trains a **Random Forest Classifier** (with class weighting to handle imbalance).
4.  **Evaluation:** Outputs classification reports, accuracy scores, and visualizes results.

## 🚀 Key Features

* **Synthetic Data Engine:** Generates realistic transaction data including features like `amount`, `time`, and anonymized vectors (`v1`-`v4`).
* **Imbalance Handling:** Uses `class_weight='balanced'` to ensure the model learns to identify the minority class (fraud) effectively.
* **Visualization:** Automatically generates and saves a `fraud_detection_results.png` showing the Confusion Matrix and ROC Curve.
* **Real-time Prediction:** Includes a method to predict individual transactions.

## 🛠️ Technologies Used

* **Python 3.x**
* **Scikit-Learn:** For model building, preprocessing, and metrics.
* **Pandas & NumPy:** For data manipulation and synthetic data generation.
* **Matplotlib & Seaborn:** For plotting and visualization.

## 📦 Installation

To run this project, simply install the required dependencies:

```bash
pip install -r requirements.txt



💻 Usage
Run the main script to generate data, train the model, and view results:

```bash
python fraud_detector.py


Output:
    The script will output model performance metrics in the console and save a visualization image:

    Console: Prints the Confusion Matrix, Classification Report, and ROC-AUC Score.

    Image: Saves fraud_detection_results.png (Confusion Matrix + ROC Curve).

📊 Results:
     The model achieves high accuracy on the synthetic dataset. Below is an example of the performance metrics visualized:

     ROC-AUC Score: ~1.00 (Excellent separation between classes)

     Precision/Recall: High detection rate for fraudulent transactions with minimal false positives.

📝 License
This project is open-source and available under the MIT License.