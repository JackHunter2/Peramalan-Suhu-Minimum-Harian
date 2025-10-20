from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64
from datetime import timedelta

app = Flask(__name__)

# Paths (sesuaikan)
MODEL_PATH = 'models/final_lstm_model.keras'  # atau 'models/final_lstm_model.h5' jika pakai h5
SCALER_PATH = 'outputs/temp_scaler.joblib'
CSV_PATH = 'daily-minimum-temperatures.csv'  # default dataset

# Load model & scaler (lazy load to allow running even jika model belum ada)
model = None
scaler = None
SEQ_LEN = 30

def load_model_and_scaler():
    global model, scaler
    if model is None:
        try:
            model = load_model(MODEL_PATH)
        except Exception as e:
            print("Model load error:", e)
            model = None
    if scaler is None:
        try:
            scaler = joblib.load(SCALER_PATH)
        except Exception as e:
            print("Scaler load error:", e)
            scaler = None

def plot_to_base64(df, title="Time Series", target_col=None, figsize=(10,3)):
    fig, ax = plt.subplots(figsize=figsize)
    if target_col:
        ax.plot(df[target_col], linewidth=1)
        ax.set_ylabel(target_col)
    else:
        ax.plot(df, linewidth=1)
    ax.set_title(title)
    ax.grid(True)
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png', dpi=120)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    plt.close(fig)
    return img_b64

@app.route('/', methods=['GET'])
def index():
    # Load CSV (if exists)
    try:
        df = pd.read_csv(CSV_PATH)
        df.columns = [c.strip() for c in df.columns]
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], errors='coerce')
        df = df.set_index(df.columns[0]).sort_index()
        # convert target to numeric safely
        target_col = df.columns[0] if len(df.columns)==1 else df.columns[1]
        df[target_col] = pd.to_numeric(df[target_col].astype(str).str.replace('°C','', regex=False), errors='coerce')
        df = df.dropna(subset=[target_col])
        hist_plot = plot_to_base64(df, title="Historical Data", target_col=target_col)
    except Exception as e:
        print("Error reading default CSV:", e)
        df = None
        hist_plot = None
        target_col = None

    return render_template('index.html', historical_plot=hist_plot, prediction_plot=None)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file:
        return redirect(url_for('index'))

    try:
        df = pd.read_csv(file, on_bad_lines='skip')
    except Exception as e:
        return f"Failed to read uploaded file: {e}", 400

    df.columns = [c.strip() for c in df.columns]
    time_col = df.columns[0]
    target_col = df.columns[1] if len(df.columns)>1 else df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.set_index(time_col).sort_index()
    df[target_col] = pd.to_numeric(df[target_col].astype(str).str.replace('°C','', regex=False), errors='coerce')
    df = df.dropna(subset=[target_col])

    hist_plot = plot_to_base64(df, title=f"Data Historis ({target_col})", target_col=target_col)
    # Store uploaded file temporarily for future prediction if user wants
    df.to_csv(CSV_PATH, index=True)
    return render_template('index.html', historical_plot=hist_plot, prediction_plot=None)

@app.route('/predict', methods=['GET','POST'])
def predict():
    load_model_and_scaler()
    # Load the stored CSV data
    try:
        df = pd.read_csv(CSV_PATH)
        df.columns = [c.strip() for c in df.columns]
        time_col = df.columns[0]
        target_col = df.columns[1] if len(df.columns)>1 else df.columns[0]
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.set_index(time_col).sort_index()
        df[target_col] = pd.to_numeric(df[target_col].astype(str).str.replace('°C','', regex=False), errors='coerce')
        df = df.dropna(subset=[target_col])
    except Exception as e:
        return f"Cannot load data for prediction: {e}", 500

    # default N days
    if request.method == 'POST':
        try:
            N = int(request.form.get('n_days', 7))
        except:
            N = 7
    else:
        N = int(request.args.get('n', 7))

    if model is None or scaler is None:
        # If model/scaler not available, show message
        msg = "Model or scaler not found. Please train model first (run notebook)."
        return render_template('predict.html', prediction_plot=None, message=msg)

    # Build iterative forecast using last SEQ_LEN points
    recent = df[target_col].values[-SEQ_LEN:].reshape(-1,1)
    recent_scaled = scaler.transform(recent)
    seq = recent_scaled.copy()
    preds_scaled = []
    for i in range(N):
        x_input = seq[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)
        yhat = model.predict(x_input)
        preds_scaled.append(yhat[0,0])
        seq = np.vstack([seq, yhat.reshape(1,1)])

    preds_inv = scaler.inverse_transform(np.array(preds_scaled).reshape(-1,1)).flatten()
    last_date = df.index[-1]
    dates = [last_date + timedelta(days=i+1) for i in range(N)]

    # Plot recent history + forecast
    # plot last 200 days of history if available
    recent_history = df[target_col].values[-200:]
    recent_dates = df.index[-200:]
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(recent_dates, recent_history, label='History (last 200)')
    ax.plot(dates, preds_inv, marker='o', linestyle='--', color='tab:red', label='Forecast')
    ax.set_title(f'Forecast next {N} days')
    ax.legend()
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png', dpi=120)
    buf.seek(0)
    pred_img = base64.b64encode(buf.getvalue()).decode('ascii')
    plt.close(fig)

    results = list(zip([d.strftime('%Y-%m-%d') for d in dates], [float(round(p,3)) for p in preds_inv]))
    return render_template('predict.html', prediction_plot=pred_img, results=results, message=None)

# =====================
# API Endpoints (JSON)
# =====================

@app.route('/api/history', methods=['GET'])
def api_history():
    # Load CSV and return JSON time series
    try:
        df = pd.read_csv(CSV_PATH)
        df.columns = [c.strip() for c in df.columns]
        time_col = df.columns[0]
        target_col = df.columns[1] if len(df.columns)>1 else df.columns[0]
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.set_index(time_col).sort_index()
        df[target_col] = pd.to_numeric(df[target_col].astype(str).str.replace('°C','', regex=False), errors='coerce')
        df = df.dropna(subset=[target_col])
    except Exception as e:
        return jsonify({"error": f"Cannot load history: {e}"}), 500

    # Limit for performance if very long
    max_points = int(request.args.get('max_points', 2000))
    if len(df) > max_points:
        df = df.iloc[-max_points:]

    return jsonify({
        "target": target_col,
        "dates": [idx.strftime('%Y-%m-%d') for idx in df.index],
        "values": [None if pd.isna(v) else float(v) for v in df[target_col].tolist()]
    })


@app.route('/api/forecast', methods=['GET'])
def api_forecast():
    load_model_and_scaler()
    try:
        df = pd.read_csv(CSV_PATH)
        df.columns = [c.strip() for c in df.columns]
        time_col = df.columns[0]
        target_col = df.columns[1] if len(df.columns)>1 else df.columns[0]
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.set_index(time_col).sort_index()
        df[target_col] = pd.to_numeric(df[target_col].astype(str).str.replace('°C','', regex=False), errors='coerce')
        df = df.dropna(subset=[target_col])
    except Exception as e:
        return jsonify({"error": f"Cannot load data for prediction: {e}"}), 500

    try:
        N = int(request.args.get('n', 7))
    except Exception:
        N = 7

    if model is None or scaler is None:
        return jsonify({"error": "Model or scaler not found. Please train model first (run notebook)."}), 503

    if len(df) < SEQ_LEN:
        return jsonify({"error": f"Not enough data points. Need at least {SEQ_LEN}."}), 400

    recent = df[target_col].values[-SEQ_LEN:].reshape(-1,1)
    recent_scaled = scaler.transform(recent)
    seq = recent_scaled.copy()
    preds_scaled = []
    for _ in range(N):
        x_input = seq[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)
        yhat = model.predict(x_input, verbose=0)
        preds_scaled.append(yhat[0,0])
        seq = np.vstack([seq, yhat.reshape(1,1)])

    preds_inv = scaler.inverse_transform(np.array(preds_scaled).reshape(-1,1)).flatten()
    last_date = df.index[-1]
    dates = [last_date + timedelta(days=i+1) for i in range(N)]

    return jsonify({
        "target": target_col,
        "dates": [d.strftime('%Y-%m-%d') for d in dates],
        "predictions": [float(round(p,3)) for p in preds_inv]
    })

if __name__ == '__main__':
    app.run(debug=True)
