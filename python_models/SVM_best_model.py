import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

def safe_label_transform(le, col_values):
    known_classes = set(le.classes_)
    unknowns = set(col_values.unique()) - known_classes
    if unknowns:
        print(f"[Warning] Unknown categories in column: {unknowns}")
        col_values = col_values.apply(lambda x: x if x in known_classes else "unknown")
        if "unknown" not in le.classes_:
            le.classes_ = np.append(le.classes_, "unknown")
    return le.transform(col_values)

def rbf_kernel(X1, X2, gamma):
    K = np.zeros((X1.shape[0], X2.shape[0]))
    for i in range(X1.shape[0]):
        K[i, :] = np.exp(-gamma * np.linalg.norm(X1[i] - X2, axis=1)**2)
    return K

class CustomSVM:
    def __init__(self, C=1.0, gamma=0.1, lr=0.001, epochs=100):
        self.C = C
        self.gamma = gamma
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        n = X.shape[0]
        self.alpha = np.zeros(n)
        self.X = X
        self.y = y

        K = rbf_kernel(X, X, self.gamma)

        for epoch in range(self.epochs):
            for i in range(n):
                gradient = 1 - self.y[i] * np.sum(self.alpha * self.y * K[:, i])
                self.alpha[i] += self.lr * gradient
                self.alpha[i] = np.clip(self.alpha[i], 0, self.C)

    def project(self, X_new):
        K = rbf_kernel(X_new, self.X, self.gamma)
        return np.dot(K, self.alpha * self.y)

    def predict(self, X_new):
        return np.sign(self.project(X_new))

def train_svm():
    # === Load and preprocess data ===
    train_data = pd.read_csv('Train_data.csv')
    train_data.drop(['num_outbound_cmds'], axis=1, inplace=True)

    # === Encode categorical features ===
    cat_cols = ['protocol_type', 'service', 'flag']
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(train_data[col].astype(str))
        train_data[col] = safe_label_transform(le, train_data[col].astype(str))
        encoders[col] = le
    joblib.dump(encoders, 'python_models/models/SVM_models/label_encoders.pkl')

    # === Encode target ===
    target_encoder = LabelEncoder()
    target_encoder.fit(train_data['class'].astype(str))
    train_data['class'] = safe_label_transform(target_encoder, train_data['class'].astype(str))
    joblib.dump(target_encoder, 'python_models/models/SVM_models/target_encoder.pkl')
    print("Target classes:", dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_))))

    # === Select binary classes ===
    binary_classes = [0, 1]  # Change as needed
    train_data_binary = train_data[train_data['class'].isin(binary_classes)]

    X = train_data_binary.drop('class', axis=1)
    y = train_data_binary['class']
    y_binary = np.where(y == binary_classes[0], -1, 1)

    # Save column order for test alignment
    os.makedirs("python_models/models/SVM_models", exist_ok=True)
    joblib.dump(X.columns.tolist(), 'python_models/models/SVM_models/feature_order.pkl')

    # === Scale Features ===
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'python_models/models/SVM_models/scaler.pkl')

    # === Train/Test Split ===
    x_train, x_test, y_train, y_test = train_test_split(X_scaled, y_binary, test_size=0.3, random_state=42)

    # === Train and Evaluate ===
    svm = CustomSVM(C=1.0, gamma=0.1, lr=0.01, epochs=10)
    svm.fit(x_train, y_train)
    preds = svm.predict(x_test)

    acc = accuracy_score(y_test, preds)
    print("Custom RBF SVM Accuracy:", acc)

    joblib.dump(svm, 'python_models/models/SVM_models/SVM_best_model.pkl')

if __name__ == "__main__":
    train_svm()
