import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

# === Custom Sigmoid ===
def Sigmoid(Z):
    Z = np.clip(Z, -500, 500)
    return 1 / (1 + np.exp(-Z))

# === Custom Logistic Regression Class ===
class My_Logistic_Regression():
    def __init__(self, max_iter=1000, learning_rate=0.01):
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def fit(self, X, y):
        m, n = X.shape
        y = np.array(y).reshape(-1, 1)
        X = np.hstack((np.ones((m, 1)), X))  # Add bias term
        self.W = np.random.randn(n + 1, 1)

        for _ in range(self.max_iter):
            y_hat = Sigmoid(X @ self.W)
            gradient = (1 / m) * X.T @ (y_hat - y)
            self.W -= self.learning_rate * gradient
        return self

    def predict(self, X):
        m = X.shape[0]
        X = np.hstack((np.ones((m, 1)), X))  # Add bias
        y_hat = Sigmoid(X @ self.W)
        return (y_hat >= 0.5).astype(int).flatten()

# === Main training logic wrapped in a function ===
def train_logistic_regression():
    # === Load and preprocess dataset ===
    columns = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
        "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
        "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
        "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
        "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
        "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
    ]

    df = pd.read_csv("Train_data.csv", header=None, names=columns, low_memory=False)
    df = df[df["duration"] != "duration"]  # Skip header-in-row error if any

    X = df[columns[:-1]].copy()
    y = df["label"]

    # === Encode categorical features ===
    le_protocol = LabelEncoder()
    le_service = LabelEncoder()
    le_flag = LabelEncoder()

    X["protocol_type"] = le_protocol.fit_transform(X["protocol_type"])
    X["service"] = le_service.fit_transform(X["service"])
    X["flag"] = le_flag.fit_transform(X["flag"])

    # Save encoders
    os.makedirs("python_models/models/LR_models", exist_ok=True)
    joblib.dump(le_protocol, "python_models/models/LR_models/le_protocol.pkl")
    joblib.dump(le_service, "python_models/models/LR_models/le_service.pkl")
    joblib.dump(le_flag, "python_models/models/LR_models/le_flag.pkl")

    # Encode target (0 = normal, 1 = anomaly)
    y = y.apply(lambda x: 0 if x == "normal" else 1)

    # === Shuffle, Split, Scale ===
    X, y = shuffle(X, y, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "python_models/models/LR_models/scaler.pkl")

    # === Polynomial Features (degree=1 = linear) ===
    poly = PolynomialFeatures(degree=1)
    X_train_trf = poly.fit_transform(X_train_scaled)
    X_test_trf = poly.transform(X_test_scaled)

    # === Train Model ===
    model = My_Logistic_Regression(max_iter=1000)
    model.fit(X_train_trf, y_train)

    # === Evaluate ===
    y_pred = model.predict(X_test_trf)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

    # Save the model (just the __dict__ for joblib compatibility)
    joblib.dump(model, "python_models/models/LR_models/classifier.pkl")
    print("✅ Logistic Regression Model trained and saved successfully.")

# === Entry Point ===
if __name__ == "__main__":
    train_logistic_regression()
