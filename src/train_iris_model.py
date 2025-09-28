import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
import os

def train_iris_model(model_type='random_forest'):

    df = pd.read_csv('data/processed/iris_processed.csv')

    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    X = df[feature_cols]
    Y = df['species']

    X_train, X_test, Y_train_,  Y_test = train_test_split(
        X,Y,test_size=0.2, random_state=42, stratify=y
    )

    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    else:
        model = LogisticRegression(
            random_state=42,
            max_iter=1000
        )
    
    model.fit(X_train, Y_train_)

    Y_pred = model.predict(X_test)
    Y_pred_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(Y_test, Y_pred)

    os.makedirs('models', exist_ok=True)
    model_path = f'models/{model_type}_model.pkl'
    joblib.dump(model, model_path)

    metrics = {
        'model_type': model_type,
        'accuracy': float(accuracy),
        'n_features': len(feature_cols),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test)
    }

    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    with open('models/features.json', 'w') as f:
        json.dump(feature_cols, f, indent=2)
    
    print(f"✅ Model eğitildi: {model_type}")
    print(f"📊 Accuracy: {accuracy:.4f}")
    print(f"💾 Model kaydedildi: {model_path}")
    
    print("\n📈 Classification Report:")
    print(classification_report(Y_test, Y_pred))
    
    return model, metrics


if __name__ == "__main__":
    model, metrics = train_iris_model('random_forest')

    model_lr, metrics_lr = train_iris_model('logistic_regression')
    print("Modeller eğitildi.")