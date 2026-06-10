import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from python_models.SVM_best_model import CustomSVM
from python_models.LogisticRegression import My_Logistic_Regression
from python_models.Model import MyGaussianNaiveBayes
from python_models.randomforestprml import MyRandomForest
from python_models.model_knn import CustomKNN
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import argparse

encoders = joblib.load('python_models/models/SVM_models/label_encoders.pkl')
target_encoder = joblib.load('python_models/models/SVM_models/target_encoder.pkl')
scaler = joblib.load('python_models/models/SVM_models/scaler.pkl')
model = joblib.load('python_models/models/SVM_models/SVM_best_model.pkl')
feature_order = joblib.load('python_models/models/SVM_models/feature_order.pkl')

def safe_label_transform(le, col_values):
    known_classes = set(le.classes_)
    unknowns = set(col_values.unique()) - known_classes
    if unknowns:
        print(f"[Warning] Unknown categories in column: {unknowns}")
        col_values = col_values.apply(lambda x: x if x in known_classes else "unknown")
        if "unknown" not in le.classes_:
            le.classes_ = np.append(le.classes_, "unknown")
    return le.transform(col_values)

def check_and_predict_SVM(output_csv='predictions.csv'):
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--input', help='Path to input CSV file')
        args = parser.parse_args()

        input_file = args.input if args.input else 'real_time_nids_features.csv'
        print(f"[*] Using input file: {input_file}")

        # === Step 2: Load test data ===
        x_test = pd.read_csv(input_file)
        x_test.drop(['num_outbound_cmds'], axis=1, errors='ignore', inplace=True)

        # === Step 3: Encode categorical columns ===
        cat_cols = ['protocol_type', 'service', 'flag']
        for col in cat_cols:
            le = encoders[col]
            known_classes = set(le.classes_)
            x_test[col] = x_test[col].astype(str).apply(lambda val: val if val in known_classes else 'unknown')
            if 'unknown' not in le.classes_:
                le.classes_ = np.append(le.classes_, 'unknown')
            x_test[col] = le.transform(x_test[col])

        # === Step 4: Align column order ===
        x_test = x_test[feature_order]

        # === Step 5: Scale test features ===
        x_test_scaled = scaler.transform(x_test)

        # === Step 6: Predict using custom SVM ===
        predictions = model.predict(x_test_scaled)
        pred_labels = ["normal" if p == 1 else "anomaly" for p in predictions]

        # === Step 7: Save predictions ===
        df_out = pd.DataFrame(pred_labels, columns=["svm"])
        df_out.to_csv(output_csv, index=False)

        normal_count = pred_labels.count("normal")
        anomaly_count = pred_labels.count("anomaly")

        summary = []
        summary.append("    === SVM Model Predicts ===")

        print("Predictions complete from SVM")
        print("Normal : ", normal_count)
        print("Anomaly: ", anomaly_count)
        summary.append(f"Normal : {normal_count}")
        summary.append(f"Anomaly: {anomaly_count}")

        for line in summary:
            print(line)
        # Save to file for frontend
        with open("prediction_summary.txt", mode='w', newline='') as f:
            for line in summary:
                f.write(line + "\n")

    except Exception as e:
        return [f"Error: {str(e)}"]

def check_and_predict_LR(output_csv='predictions.csv'):
    clf = joblib.load("python_models/models/LR_models/classifier.pkl")
    le_protocol = joblib.load("python_models/models/LR_models/le_protocol.pkl")
    le_service = joblib.load("python_models/models/LR_models/le_service.pkl")
    le_flag = joblib.load("python_models/models/LR_models/le_flag.pkl")
    scaler = joblib.load("python_models/models/LR_models/scaler.pkl")

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--input', help='Path to input CSV file')
        args = parser.parse_args()

        input_file = args.input if args.input else 'real_time_nids_features.csv'
        print(f"[*] Using input file: {input_file}")

        df = pd.read_csv(input_file)
        print("✅ Data read")

        df["protocol_type"] = safe_label_transform(le_protocol, df["protocol_type"])
        df["service"] = safe_label_transform(le_service, df["service"])
        df["flag"] = safe_label_transform(le_flag, df["flag"])

        feature_cols = [
            "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
            "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
            "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
            "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
            "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
            "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
            "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
            "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
            "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
        ]

        X = df[feature_cols]
        X_scaled = scaler.transform(X)
        X_trf = PolynomialFeatures(degree=1).fit_transform(X_scaled)
        preds = clf.predict(X_trf)

        pred_labels = ["normal" if p == 0 else "anomaly" for p in preds]

        # Append to existing predictions.csv
        if os.path.exists(output_csv):
            combined = pd.read_csv(output_csv)
            combined["lr"] = pred_labels
            combined.to_csv(output_csv, index=False)
        else:
            pd.DataFrame(pred_labels, columns=["lr"]).to_csv(output_csv, index=False)

        normal_count = pred_labels.count("normal")
        anomaly_count = pred_labels.count("anomaly")

        summary = []
        summary.append("    === Logistic Regression Model Predicts ===")

        print("✅ Predictions complete from Logistic Regression")
        print("Normal : ", normal_count)
        print("Anomaly: ", anomaly_count)
        summary.append(f"Normal : {normal_count}")
        summary.append(f"Anomaly: {anomaly_count}")

        for line in summary:
            print(line)
        # Save to file for frontend
        with open("prediction_summary.txt", mode='a', newline='') as f:
            for line in summary:
                f.write(line + "\n")
        return df, normal_count, anomaly_count

    except Exception as e:
        print("❌ Error during prediction:", e)
        raise

def preprocess_and_predict_NV(output_csv='predictions.csv'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Path to input CSV file')
    args = parser.parse_args()

    input_file = args.input if args.input else 'real_time_nids_features.csv'
    print(f"[*] Using input file: {input_file}")

    df = pd.read_csv(input_file)
    
    nb_model = joblib.load('python_models/models/Navie_bayes_Model/naive_bayes_model.pkl')
    scaler = joblib.load('python_models/models/Navie_bayes_Model/scaler.pkl')
    selected_features = joblib.load('python_models/models/Navie_bayes_Model/selected_features.pkl')
    encoding_dict = joblib.load('python_models/models/Navie_bayes_Model/all_label_encoders.pkl')

    # Apply safe label transform to handle unseen categories
    for col, encoder in encoding_dict.items():
        if col in df.columns:
            df[col] = safe_label_transform(encoder, df[col].astype(str))
        else:
            print(f"[Warning] Column {col} missing from input data")

    # Select and scale features
    df = df[selected_features]
    df_scaled = scaler.transform(df)

    # Predict
    predictions = nb_model.predict(df_scaled)
    pred_labels = ["normal" if p == 0 else "anomaly" for p in predictions]

    # Append to existing predictions.csv
    if os.path.exists(output_csv):
        combined = pd.read_csv(output_csv)
        combined["nv"] = pred_labels
        combined.to_csv(output_csv, index=False)
    else:
        pd.DataFrame(pred_labels, columns=["nv"]).to_csv(output_csv, index=False)

    # Summary
    normal = pred_labels.count("normal")
    anomaly = pred_labels.count("anomaly")
    print("✅ Predictions complete from Naive Bayes")
    print("Normal :", normal)
    print("Anomaly:", anomaly)
    summary = []
    summary.append("    === Naive Bayes Model Predicts ===")
    summary.append(f"Normal : {normal}")
    summary.append(f"Anomaly: {anomaly}")

    for line in summary:
            print(line)
        # Save to file for frontend
    with open("prediction_summary.txt", mode='a', newline='') as f:
            for line in summary:
                f.write(line + "\n")

    return normal, anomaly

def predict_knn(output_csv='predictions.csv'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Path to input CSV file')
    args = parser.parse_args()

    input_file = args.input if args.input else 'real_time_nids_features.csv'
    print(f"[*] Using input file: {input_file}")

    real_data = pd.read_csv(input_file)

    cat_cols = ['protocol_type', 'service', 'flag']

    def load_or_fit_encoder(file_name, column_name):
        if os.path.exists(file_name):
            return joblib.load(file_name)
        else:
            print(f"[Warning] '{file_name}' not found. Creating and fitting a new LabelEncoder on column '{column_name}'")
            le = LabelEncoder()
            real_data[column_name] = le.fit_transform(real_data[column_name])
            joblib.dump(le, file_name)
            return le

    def safe_label_transform(le, series):
        valid_labels = set(le.classes_)
        return series.apply(lambda x: le.transform([x])[0] if x in valid_labels else -1)

    # Encode and mark invalid rows
    for col in cat_cols:
        le = load_or_fit_encoder(f"python_models/models/Knn_model/{col}_le.pkl", col)
        real_data[col] = safe_label_transform(le, real_data[col])

    # Keep mask of valid rows (to also filter predictions.csv if needed)
    valid_mask = (real_data[cat_cols] != -1).all(axis=1)
    real_data = real_data[valid_mask]

    scaler = joblib.load("python_models/models/Knn_model/scaler.pkl")
    selected_indices = joblib.load("python_models/models/Knn_model/selected_feature_indices.pkl")

    real_data_scaled = scaler.transform(real_data.values)
    real_data_selected = real_data_scaled[:, selected_indices]

    knn_model = joblib.load("python_models/models/Knn_model/knn_model.pkl")
    predictions = knn_model.predict(real_data_selected)

    # ✅ Convert numeric predictions to labels
    label_map = {0: 'normal', 1: 'anomaly'}
    predicted_labels = [label_map[pred] for pred in predictions]

    # Append predictions to output_csv if lengths match
    if os.path.exists(output_csv):
        combined = pd.read_csv(output_csv)

        if len(combined) == len(predicted_labels):
            combined["knn"] = predicted_labels
            combined.to_csv(output_csv, index=False)
        else:
            print(f"❌ Length mismatch after filtering: combined={len(combined)}, predictions={len(predicted_labels)}")
    else:
        pd.DataFrame(predicted_labels, columns=["knn"]).to_csv(output_csv, index=False)

    # Summary
    unique_labels, counts = np.unique(predicted_labels, return_counts=True)
    print("DEBUG >> unique_labels:", type(unique_labels), unique_labels)
    print("DEBUG >> counts:", type(counts), counts)
    print("\nPrediction Summary from KNN:")
    sum = 0
    for label, count in zip(unique_labels, counts):
        sum += count
        l = count
        print(f"{label}: {count}")

    summary = []
    summary.append("    === kNN Model Predicts ===")
    summary.append(f"Normal : {l}")
    summary.append(f"Anomaly: {sum - l}")

    for line in summary:
            print(line)
        # Save to file for frontend
    with open("prediction_summary.txt", mode='a', newline='') as f:
            for line in summary:
                f.write(line + "\n")

def check_and_predict_RF(output_csv='predictions.csv'):
    model = joblib.load('python_models/models/RF_models/random_forest.pkl')  

    # Load unlabeled data
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Path to input CSV file')
    args = parser.parse_args()

    input_file = args.input if args.input else 'real_time_nids_features.csv'
    print(f"[*] Using input file: {input_file}")

    unlabeled_data = pd.read_csv(input_file)

    for col in unlabeled_data.columns:
        if unlabeled_data[col].dtype == 'object':
            le = LabelEncoder()
            unlabeled_data[col] = le.fit_transform(unlabeled_data[col].astype(str))

    preds = model.predict(unlabeled_data)
    pred_labels = ["normal" if p == 1 else "anomaly" for p in preds]

    # Save predictions to CSV
    if os.path.exists(output_csv):
        combined = pd.read_csv(output_csv)
        combined["rf"] = pred_labels
        combined.to_csv(output_csv, index=False)
    else:
        pd.DataFrame(pred_labels, columns=["rf"]).to_csv(output_csv, index=False)

    prediction_counts = Counter(pred_labels)
    print("Predictions from Random Forest")
    for label, count in prediction_counts.items():
        print(f"{label}: {count}")

    summary = []
    summary.append("    === Random Forest Model Predicts ===")
    summary.append(f"Normal : {prediction_counts["normal"]}")
    summary.append(f"Anomaly: {prediction_counts["anomaly"]}")

    for line in summary:
            print(line)
        # Save to file for frontend
    with open("prediction_summary.txt", mode='a', newline='') as f:
            for line in summary:
                f.write(line + "\n")

    if "anomaly" in prediction_counts and prediction_counts["anomaly"] > 150:
        print("Final Say: 🚨 Anomaly Detected")
    else:
        print("Final Say: ✅ Normal Behavior")


def predict_with_bgmm(output_csv='predictions.csv'):
    # Load trained model and preprocessing tools
    bgmm = joblib.load("python_models/models/BGMM_model/bgmm_model.pkl")
    scaler = joblib.load("python_models/models/BGMM_model/scaler.pkl")
    threshold = joblib.load("python_models/models/BGMM_model/threshold.pkl")
    le_protocol = joblib.load("python_models/models/BGMM_model/le_protocol_type.pkl")
    le_service = joblib.load("python_models/models/BGMM_model/le_service.pkl")
    le_flag = joblib.load("python_models/models/BGMM_model/le_flag.pkl")
    feature_order = joblib.load("python_models/models/BGMM_model/feature_order.pkl")

    def safe_label_transform(label_encoder, values):
        known_classes = set(label_encoder.classes_)
        return np.array([
            label_encoder.transform([val])[0] if val in known_classes else -1
            for val in values
        ])

    try:
        # Load input features
        parser = argparse.ArgumentParser()
        parser.add_argument('--input', help='Path to input CSV file')
        args = parser.parse_args()

        input_file = args.input if args.input else 'real_time_nids_features.csv'
        print(f"[*] Using input file: {input_file}")

        df = pd.read_csv(input_file)
        # print(df.info())
        # Encode categorical features safely
        df["protocol_type"] = safe_label_transform(le_protocol, df["protocol_type"].astype(str))
        df["service"] = safe_label_transform(le_service, df["service"].astype(str))
        df["flag"] = safe_label_transform(le_flag, df["flag"].astype(str))

        # Align feature order
        X = df[feature_order]
        X_scaled = scaler.transform(X)
        # print(len(X_scaled))
        # Get anomaly scores and predict
        scores = -bgmm.score_samples(X_scaled)
        predictions = (scores > threshold).astype(int)
        print(len(predictions))
        labels = ["normal" if pred == 0 else "anomaly" for pred in predictions]

        # Save predictions
        if os.path.exists(output_csv):
            existing = pd.read_csv(output_csv)
            new = pd.DataFrame(labels, columns=["bgmm"])
            combined = pd.concat([existing, new], ignore_index=True)
            combined.to_csv(output_csv, index=False)

        else:
            df_with_preds = df.copy()
            df_with_preds["bgmm"] = labels
            df_with_preds.to_csv(output_csv, index=False)

        # Print summary
        print("BGMM Prediction Summary:")
        print(f"Normal  : {labels.count('normal')}")
        print(f"Anomaly : {labels.count('anomaly')}")

        summary = []
        summary.append("    === Gaussian Mixture Model Predicts ===")
        summary.append(f"Normal : {labels.count("normal")}")
        summary.append(f"Anomaly: {labels.count("anomaly")}")

        for line in summary:
                print(line)
            # Save to file for frontend
        with open("prediction_summary.txt", mode='a', newline='') as f:
                for line in summary:
                    f.write(line + "\n")

    except Exception as err:
        print("Prediction failed due to error:", err)
        raise

if __name__ == "__main__":
    check_and_predict_SVM()
    check_and_predict_LR()
    preprocess_and_predict_NV()
    check_and_predict_RF()
    predict_knn()
    predict_with_bgmm()