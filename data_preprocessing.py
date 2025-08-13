import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'cough_duration_days': np.random.randint(0, 30, n_samples),
        'fever': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        'weight_loss': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        'night_sweats': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'tb_positive': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    df = pd.DataFrame(data)
    df.to_csv('tb_synthetic_data.csv', index=False)
    return df

def preprocess_data(data_path='tb_synthetic_data.csv'):
    if not os.path.exists(data_path):
        df = generate_synthetic_data()
    else:
        df = pd.read_csv(data_path)
    
    X = df[['age', 'cough_duration_days', 'fever', 'weight_loss', 'night_sweats']]
    y = df['tb_positive']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = preprocess_data()
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")