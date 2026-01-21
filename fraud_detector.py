import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

class FraudDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        
    def generate_synthetic_data(self, n_samples=10000):
        """Generate synthetic credit card transaction data"""
        print("Generating synthetic transaction data...")
        
        # Normal transactions (95%)
        n_normal = int(n_samples * 0.95)
        normal_transactions = {
            'amount': np.random.gamma(2, 50, n_normal),
            'time': np.random.randint(0, 86400, n_normal),
            'v1': np.random.normal(0, 1, n_normal),
            'v2': np.random.normal(0, 1, n_normal),
            'v3': np.random.normal(0, 1, n_normal),
            'v4': np.random.normal(0, 1, n_normal),
            'is_fraud': np.zeros(n_normal)
        }
        
        # Fraudulent transactions (5%)
        n_fraud = n_samples - n_normal
        fraud_transactions = {
            'amount': np.random.exponential(200, n_fraud),
            'time': np.random.choice([0, 3600, 7200, 21600], n_fraud),
            'v1': np.random.normal(2, 1.5, n_fraud),
            'v2': np.random.normal(-2, 1.5, n_fraud),
            'v3': np.random.normal(1, 1.5, n_fraud),
            'v4': np.random.normal(-1, 1.5, n_fraud),
            'is_fraud': np.ones(n_fraud)
        }
        
        # Combine datasets
        df_normal = pd.DataFrame(normal_transactions)
        df_fraud = pd.DataFrame(fraud_transactions)
        df = pd.concat([df_normal, df_fraud], ignore_index=True)
        
        # Shuffle the dataset
        df = df.sample(frac=1).reset_index(drop=True)
        
        print(f"Dataset created: {len(df)} transactions")
        print(f"Fraud cases: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess and split data"""
        X = df.drop('is_fraud', axis=1)
        y = df['is_fraud']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train, model_type='random_forest'):
        """Train the fraud detection model"""
        print(f"\nTraining {model_type} model...")
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        else:
            self.model = LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
        
        self.model.fit(X_train, y_train)
        print("Model training completed!")
        
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Normal', 'Fraud']))
        
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"\nROC-AUC Score: {roc_auc:.4f}")
        
        return y_pred, y_pred_proba, cm
    
    def plot_results(self, y_test, y_pred_proba, cm):
        """Visualize results"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_ylabel('Actual')
        axes[0].set_xlabel('Predicted')
        axes[0].set_xticklabels(['Normal', 'Fraud'])
        axes[0].set_yticklabels(['Normal', 'Fraud'])
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend(loc="lower right")
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('fraud_detection_results.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'fraud_detection_results.png'")
        plt.show()
    
    def predict_transaction(self, transaction_data):
        """Predict if a single transaction is fraudulent"""
        transaction_scaled = self.scaler.transform([transaction_data])
        prediction = self.model.predict(transaction_scaled)[0]
        probability = self.model.predict_proba(transaction_scaled)[0][1]
        
        return prediction, probability


def main():
    # Initialize detector
    detector = FraudDetector()
    
    # Generate data
    df = detector.generate_synthetic_data(n_samples=10000)
    
    # Preprocess
    X_train, X_test, y_train, y_test = detector.preprocess_data(df)
    
    # Train model
    detector.train_model(X_train, y_train, model_type='random_forest')
    
    # Evaluate
    y_pred, y_pred_proba, cm = detector.evaluate_model(X_test, y_test)
    
    # Plot results
    detector.plot_results(y_test, y_pred_proba, cm)
    
    # Example prediction
    print("\n" + "="*50)
    print("EXAMPLE PREDICTIONS")
    print("="*50)
    
    # Normal transaction
    normal_trans = [75.0, 43200, 0.2, -0.1, 0.3, 0.1]
    pred, prob = detector.predict_transaction(normal_trans)
    print(f"\nNormal Transaction: {normal_trans}")
    print(f"Prediction: {'FRAUD' if pred == 1 else 'NORMAL'}")
    print(f"Fraud Probability: {prob:.2%}")
    
    # Suspicious transaction
    suspicious_trans = [450.0, 2000, 2.5, -2.3, 1.8, -1.5]
    pred, prob = detector.predict_transaction(suspicious_trans)
    print(f"\nSuspicious Transaction: {suspicious_trans}")
    print(f"Prediction: {'FRAUD' if pred == 1 else 'NORMAL'}")
    print(f"Fraud Probability: {prob:.2%}")


if __name__ == "__main__":
    main()