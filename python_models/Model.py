import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

class MyGaussianNaiveBayes:
    def __init__(self, epsilon=1e-9): 
        self.class_probabilities = {}
        self.class_means = {}
        self.class_variances = {}
        self.epsilon = epsilon

    def fit(self, X, y):
        classes = np.unique(y)
        self.class_probabilities = {class_label: np.sum(y == class_label) / len(y) for class_label in classes}
        
        self.class_means = {}
        self.class_variances = {}
        
        for class_label in classes:
            X_class = X[y == class_label]
            self.class_means[class_label] = np.mean(X_class, axis=0)
            self.class_variances[class_label] = np.var(X_class, axis=0) + self.epsilon  

    def gaussian_pdf(self, x, mean, var):
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * ((x - mean) ** 2) / var)

    def predict(self, X):
        predictions = []
        for x in X:
            class_probs = {}
            for class_label in self.class_probabilities:
                prob = np.log(self.class_probabilities[class_label])
                for i, feature_value in enumerate(x):
                    mean = self.class_means[class_label][i]
                    var = self.class_variances[class_label][i]
                    pdf_val = self.gaussian_pdf(feature_value, mean, var)
                    prob += np.log(pdf_val + self.epsilon)  # Add epsilon here too
                class_probs[class_label] = prob
            predicted_class = max(class_probs, key=class_probs.get)
            predictions.append(predicted_class)
        return np.array(predictions)


# Step 1: Load training and test data
train = pd.read_csv('Train_data.csv')
test = pd.read_csv('Test_data.csv')

# Step 2: Label Encoding for categorical columns
encoding_dict = {
    "protocol_type": LabelEncoder(),
    "service": LabelEncoder(),
    "flag": LabelEncoder()
}

# Function to apply label encoding
def label_encode(df, encoding_dict):
    for col, encoder in encoding_dict.items():
        df[col] = encoder.fit_transform(df[col])
    return encoding_dict

# Apply label encoding on both train and test data
encoding_dict = label_encode(train, encoding_dict)
label_encode(test, encoding_dict)

# Step 3: Drop unnecessary columns
train.drop(['num_outbound_cmds'], axis=1, inplace=True, errors='ignore')
test.drop(['num_outbound_cmds'], axis=1, inplace=True, errors='ignore')

# Step 4: Encode target 'class' into 0 (normal) and 1 (anomaly)
train['class'] = train['class'].map({'normal': 0, 'anomaly': 1})

# Step 5: Prepare feature matrix and target
X = train.drop(['class'], axis=1)
y = train['class']

# Step 6: Feature Selection using RandomForest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Get top N important features (we choose top 20)
top_n = 20
importances = rf_model.feature_importances_
top_features_idx = np.argsort(importances)[::-1][:top_n]
selected_features = X.columns[top_features_idx]
X_selected = X[selected_features]

# Step 7: Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Step 8: Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 9: Train Naive Bayes
nb_model = MyGaussianNaiveBayes()
nb=GaussianNB()
nb_model.fit(X_train, y_train)
nb.fit(X_train, y_train)


# Step 10: Evaluate the model on training and validation data
train_acc = accuracy_score(y_train, nb.predict(X_train))
val_acc = accuracy_score(y_val, nb.predict(X_val))
print(f"Training Accuracy: {train_acc * 100:.2f}%")
print(f"Validation Accuracy: {val_acc * 100:.2f}%")

# Step 11: Retrain the Naive Bayes model on the full training data
nb_model.fit(X_scaled, y)

# Step 12: Make predictions on the real test data
X_test = test[selected_features]
X_test_scaled = scaler.transform(X_test)
test_predictions = nb_model.predict(X_test_scaled)

# Step 13: Save predictions to CSV
pred_df = pd.DataFrame(test_predictions, columns=['predicted_class'])

# Step 14: Save models, encoders, and transformers
joblib.dump(nb, 'python_models/models/Navie_bayes_Model/naive_bayes_model.pkl')
joblib.dump(rf_model, 'python_models/models/Navie_bayes_Model/random_forest_model.pkl')
joblib.dump(scaler, 'python_models/models/Navie_bayes_Model/scaler.pkl')
joblib.dump(selected_features.tolist(), 'python_models/models/Navie_bayes_Model/selected_features.pkl')

# Save the combined LabelEncoders in a single dictionary and pickle it
joblib.dump(encoding_dict, 'python_models/models/Navie_bayes_Model/all_label_encoders.pkl')

print("All models and files saved successfully!")
