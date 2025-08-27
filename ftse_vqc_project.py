# ftse_vqc_full.py
"""
FTSE 100 Up/Down classification with classical baselines and a VQC-style QNN (EstimatorQNN).
Windows-friendly, uses current Qiskit primitives where possible.

Outputs:
 - outputs/metrics.csv
 - outputs/confusion_matrix_<model>.png
 - outputs/roc_curve_<model>.png

Requirements (install first):
 pip install yfinance pandas numpy scikit-learn matplotlib
 pip install qiskit-aer qiskit qiskit-machine-learning
 pip install torch

If you have problems installing qiskit versions on Windows, try
 pip install qiskit
 pip install qiskit-aer
 pip install qiskit-machine-learning
 (or use conda environments)
"""

import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Outputs folder
OUTDIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTDIR, exist_ok=True)

# ------------------ Data download ------------------
print("Downloading FTSE 100 data from 2013-01-01 to today ...")
try:
    import yfinance as yf
except Exception as e:
    raise SystemExit("Please install yfinance: pip install yfinance") from e

START = "2013-01-01"
END = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
df = yf.download("^FTSE", start=START, end=END, progress=False)
if df.empty:
    raise SystemExit("No data returned. Check internet or ticker symbol.")

# Flatten MultiIndex columns (yfinance sometimes returns MultiIndex)
if isinstance(df.columns, pd.MultiIndex):
    try:
        df.columns = df.columns.droplevel(1)
    except Exception:
        # fallback: rename columns manually
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

df.columns = [c.lower() for c in df.columns]  # normalize column names to lowercase
print("Columns after flattening:", df.columns.tolist())

# ------------------ Feature engineering ------------------
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def sma(series, window):
    return series.rolling(window).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = pd.Series(gain, index=series.index).rolling(period).mean()
    loss = pd.Series(loss, index=series.index).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

# create features (using lowercase column names from yfinance)
price_col = "close"
df["return_1d"] = df[price_col].pct_change()
df["return_5d"] = df[price_col].pct_change(5)
df["volatility_10d"] = df["return_1d"].rolling(10).std()
df["sma_10"] = sma(df[price_col], 10)
df["sma_20"] = sma(df[price_col], 20)
df["ema_12"] = ema(df[price_col], 12)
df["ema_26"] = ema(df[price_col], 26)
df["rsi_14"] = rsi(df[price_col], 14)
df["macd"], df["macd_signal"], df["macd_hist"] = macd(df[price_col])
bb_u = sma(df[price_col], 20) + 2 * df[price_col].rolling(20).std()
bb_l = sma(df[price_col], 20) - 2 * df[price_col].rolling(20).std()
df["bb_width"] = (bb_u - bb_l) / (sma(df[price_col], 20) + 1e-9)

# lags
for lag in [1,2,3,5,10]:
    df[f"lag_close_{lag}"] = df[price_col].shift(lag)
    df[f"lag_ret_{lag}"] = df["return_1d"].shift(lag)

# drop nan rows from indicators
df = df.dropna().copy()

# ------------------ Labels ------------------
df["close_next"] = df[price_col].shift(-1)
df["y"] = (df["close_next"] > df[price_col]).astype(int)
df = df.dropna().copy()

# features list - keep moderately sized list
FEATURES = [
    "return_1d","return_5d","volatility_10d",
    "sma_10","sma_20","ema_12","ema_26","rsi_14",
    "macd","macd_signal","macd_hist","bb_width",
    "lag_ret_1","lag_ret_2","lag_ret_3","lag_ret_5"
]
# Ensure features exist
FEATURES = [f for f in FEATURES if f in df.columns]
X = df[FEATURES].values
y = df["y"].values

print("Features used:", FEATURES)
print("Dataset size:", X.shape)

# ------------------ Train/Test time-split ------------------
n = len(df)
split = int(n * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
dates_train = df.index[:split]
dates_test = df.index[split:]

# ------------------ Scaling & PCA ------------------
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# reduce dimension for quantum model (small #qubits)
pca_components = 4  # change to 2 for faster simulation
pca = PCA(n_components=pca_components, random_state=42)
X_train_p = pca.fit_transform(X_train_s)
X_test_p = pca.transform(X_test_s)

# ------------------ Classical baselines ------------------
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay

def eval_and_save(model, name, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    y_pred = model.predict(Xte)
    acc = accuracy_score(yte, y_pred)
    prec = precision_score(yte, y_pred, zero_division=0)
    rec = recall_score(yte, y_pred, zero_division=0)
    f1 = f1_score(yte, y_pred, zero_division=0)
    try:
        y_score = model.predict_proba(Xte)[:,1]
    except Exception:
        try:
            y_score = model.decision_function(Xte)
        except Exception:
            y_score = y_pred
    try:
        auc = roc_auc_score(yte, y_score)
    except Exception:
        auc = np.nan

    # save confusion matrix and ROC
    disp = ConfusionMatrixDisplay.from_predictions(yte, y_pred)
    disp.ax_.set_title(f"Confusion Matrix - {name}")
    cm_path = os.path.join(OUTDIR, f"confusion_matrix_{name}.png")
    plt.savefig(cm_path, bbox_inches="tight"); plt.close()

    try:
        RocCurveDisplay.from_predictions(yte, y_score)
        plt.title(f"ROC Curve - {name}")
        roc_path = os.path.join(OUTDIR, f"roc_curve_{name}.png")
        plt.savefig(roc_path, bbox_inches="tight"); plt.close()
    except Exception:
        pass

    return {"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}

metrics = []
print("Training classical baselines...")
metrics.append(eval_and_save(LogisticRegression(max_iter=1000), "logreg", X_train_p, y_train, X_test_p, y_test))
metrics.append(eval_and_save(SVC(kernel="rbf", probability=True), "svm_rbf", X_train_p, y_train, X_test_p, y_test))
metrics.append(eval_and_save(RandomForestClassifier(n_estimators=200, random_state=42), "rf", X_train_p, y_train, X_test_p, y_test))

# ------------------ Quantum Model using Qiskit (EstimatorQNN) ------------------
print("Setting up Quantum Neural Network (EstimatorQNN) ...")
has_qiskit = True
try:
    # Qiskit imports (modern)
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.primitives import Estimator
    # use Aer simulator if available (qiskit-aer)
    try:
        from qiskit_aer import AerSimulator
        backend_sim = AerSimulator()
    except Exception:
        # Fall back to no backend object; Estimator may use default provider
        backend_sim = None

    from qiskit.circuit.library import TwoLocal, ZZFeatureMap
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from qiskit_machine_learning.connectors import TorchConnector
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as e:
    print("Qiskit or PyTorch imports failed. Quantum model will be skipped.")
    print("Error:", e)
    has_qiskit = False

if has_qiskit:
    # number of qubits = pca_components
    num_qubits = pca_components

    # build circuit: feature encoding (rotations) + trainable ansatz (TwoLocal)
    x = ParameterVector("x", num_qubits)
    theta = ParameterVector("θ", 2 * num_qubits * 2)  # we'll not rely on exact count; use ansatz.num_parameters later

    # Simple encoding circuit: Ry(x_i)
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(x[i], i)

    # Use a TwoLocal ansatz and append
    ansatz = TwoLocal(num_qubits=num_qubits, rotation_blocks=["ry","rz"], entanglement_blocks="cz", entanglement="linear", reps=2)
    # append ansatz (it has parameters we will treat as weight params)
    qc.compose(ansatz, inplace=True)

    # set observable (Z on first qubit)
    observable = SparsePauliOp.from_list([("Z" + "I"*(num_qubits-1), 1.0)])

    # create Estimator primitive (optionally with AerSimulator)
    if backend_sim is not None:
        estimator = Estimator(options={"backend": backend_sim})
    else:
        estimator = Estimator()

    # build EstimatorQNN (it expects the circuit, observables, and parameter groups)
    # find parameter lists: inputs are x (we used ParameterVector "x"), weights are ansatz.parameters
    # However qc.parameters ordering: combine by name; to be safe, explicitly recompute parameter lists:
    input_params = list(x)
    weight_params = [p for p in qc.parameters if p.name.startswith("θ") is False and p.name not in [ip.name for ip in input_params]]
    # fallback: if weight_params empty, get ansatz params
    if len(weight_params) == 0:
        weight_params = list(ansatz.parameters)

    qnn = EstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
        input_gradients=False
    )

    # connect to PyTorch
    class QNNBinaryClassifier(nn.Module):
        def __init__(self, qnn):
            super().__init__()
            self.qnn = TorchConnector(qnn)  # returns expectation in [-1,1]
        def forward(self, x):
            y = self.qnn(x)  # shape (batch, 1) maybe
            # map (-1..1) to probability (0..1)
            prob = (y + 1.0) / 2.0
            return prob

    # Prepare data as torch tensors
    Xtr_t = torch.tensor(X_train_p, dtype=torch.float32)
    Xte_t = torch.tensor(X_test_p, dtype=torch.float32)
    ytr_t = torch.tensor(y_train.reshape(-1,1), dtype=torch.float32)
    yte_t = torch.tensor(y_test.reshape(-1,1), dtype=torch.float32)

    model = QNNBinaryClassifier(qnn)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    # Train (small number of epochs for demo)
    epochs = 12
    print("Training quantum model (this may be slow depending on hardware)...")
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        probs = model(Xtr_t)
        loss = criterion(probs, ytr_t)
        loss.backward()
        optimizer.step()
        if epoch % 2 == 0:
            print(f"Epoch {epoch}/{epochs}  Loss: {loss.item():.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        probs_te = model(Xte_t).numpy().ravel()
    y_pred_q = (probs_te >= 0.5).astype(int)

    # metrics and save plots
    acc = accuracy_score(y_test, y_pred_q)
    prec = precision_score(y_test, y_pred_q, zero_division=0)
    rec = recall_score(y_test, y_pred_q, zero_division=0)
    f1 = f1_score(y_test, y_pred_q, zero_division=0)
    try:
        auc = roc_auc_score(y_test, probs_te)
    except Exception:
        auc = np.nan

    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_q)
    disp.ax_.set_title("Confusion Matrix - qnn_vqc")
    cm_path = os.path.join(OUTDIR, "confusion_matrix_qnn_vqc.png")
    plt.savefig(cm_path, bbox_inches="tight"); plt.close()

    try:
        RocCurveDisplay.from_predictions(y_test, probs_te)
        plt.title("ROC Curve - qnn_vqc")
        roc_path = os.path.join(OUTDIR, "roc_curve_qnn_vqc.png")
        plt.savefig(roc_path, bbox_inches="tight"); plt.close()
    except Exception:
        pass

    metrics.append({"model": "qnn_vqc", "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc})

# ------------------ Save metrics ------------------
metrics_df = pd.DataFrame(metrics).sort_values(by="accuracy", ascending=False)
metrics_path = os.path.join(OUTDIR, "metrics.csv")
metrics_df.to_csv(metrics_path, index=False)
print("\n=== Results ===")
print(metrics_df.to_string(index=False))
print(f"\nSaved metrics to: {metrics_path}")
print(f"Plots (if produced) saved in: {OUTDIR}")

# also save a small sample of predictions and dates for inspection
try:
    sample_df = pd.DataFrame({
        "date": np.array(dates_test).astype(str),
        "close": df[price_col].values[split:],
        "y_true": y_test
    }).reset_index(drop=True)
    # if quantum preds exist, attach
    if any(m["model"] == "qnn_vqc" for m in metrics):
        sample_df["y_pred_qnn"] = y_pred_q
    sample_path = os.path.join(OUTDIR, "test_sample.csv")
    sample_df.to_csv(sample_path, index=False)
    print(f"Saved test sample to: {sample_path}")
except Exception:
    pass
