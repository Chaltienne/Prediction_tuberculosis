import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from data_preprocessing import preprocess_data

# Load preprocessed data
X_train, X_test, y_train, y_test, scaler = preprocess_data()

# Train model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, 'tb_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and scaler saved as tb_model.pkl and scaler.pkl")

# Evaluate model
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# Save metrics for dashboard
metrics = {
    'confusion_matrix': cm.tolist(),
    'classification_report': report,
    'feature_importance': model.coef_[0].tolist(),
    'features': ['Age', 'Cough Duration', 'Fever', 'Weight Loss', 'Night Sweats']
}
pd.to_pickle(metrics, 'metrics.pkl')
print('Confusion Matrix')
print(cm)
print('Classification Report')
print(classification_report(y_test, y_pred))