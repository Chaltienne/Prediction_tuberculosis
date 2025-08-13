from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import joblib

app = Flask(__name__)

# Load model, scaler, and metrics
model = joblib.load('tb_model.pkl')
scaler = joblib.load('scaler.pkl')
metrics = pd.read_pickle('metrics.pkl')

@app.route('/')
def index():
    # Feature importance plot
    fig1 = go.Figure(data=[
        go.Bar(
            x=metrics['features'],
            y=metrics['feature_importance'],
            marker_color='#007bff'
        )
    ])
    fig1.update_layout(
        title='Feature Importance',
        xaxis_title='Features',
        yaxis_title='Coefficient',
        template='plotly_white'
    )
    plot1 = pio.to_html(fig1, full_html=False)

    # Confusion matrix heatmap
    cm = metrics['confusion_matrix']
    fig2 = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Normal', 'Tuberculosis'],
        y=['Normal', 'Tuberculosis'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        showscale=True
    ))
    fig2.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        template='plotly_white'
    )
    plot2 = pio.to_html(fig2, full_html=False)

    # Classification report metrics
    report = metrics['classification_report']
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']

    return render_template('index.html', plot1=plot1, plot2=plot2, 
                          precision=precision, recall=recall, f1_score=f1_score)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        age = float(data['age'])
        cough_duration = float(data['cough_duration'])
        fever = float(data['fever'])
        weight_loss = float(data['weight_loss'])
        night_sweats = float(data['night_sweats'])
        
        input_data = np.array([[age, cough_duration, fever, weight_loss, night_sweats]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][prediction]
        
        result = "Tuberculosis" if prediction == 1 else "Normal"
        return jsonify({'prediction': result, 'confidence': f"{probability:.2%}"})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)