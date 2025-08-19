# src/train_model.py

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
import os

print("--- Bắt đầu quá trình huấn luyện mô hình (22 đặc trưng) ---")

df = pd.read_csv('./data/pd_speech_features.csv')
print("Tải dữ liệu thành công.")

print("Bắt đầu tiền xử lý dữ liệu...")

feature_columns_map = {
    'DFA': 'DFA', 'RPDE': 'RPDE',
    'Jitter_local': 'locPctJitter', 'Jitter_rap': 'rapJitter', 'Jitter_ppq5': 'ppq5Jitter',
    'Shimmer_local': 'locShimmer', 'Shimmer_apq3': 'apq3Shimmer', 'Shimmer_apq5': 'apq5Shimmer',
    'HNR': 'meanHarmToNoiseHarmonicity',
    'MFCC_0': 'mean_MFCC_0th_coef', 'MFCC_1': 'mean_MFCC_1st_coef', 'MFCC_2': 'mean_MFCC_2nd_coef',
    'MFCC_3': 'mean_MFCC_3rd_coef', 'MFCC_4': 'mean_MFCC_4th_coef', 'MFCC_5': 'mean_MFCC_5th_coef',
    'MFCC_6': 'mean_MFCC_6th_coef', 'MFCC_7': 'mean_MFCC_7th_coef', 'MFCC_8': 'mean_MFCC_8th_coef',
    'MFCC_9': 'mean_MFCC_9th_coef', 'MFCC_10': 'mean_MFCC_10th_coef', 'MFCC_11': 'mean_MFCC_11th_coef',
    'MFCC_12': 'mean_MFCC_12th_coef'
}

important_cols_in_csv = list(feature_columns_map.values())
X = df[important_cols_in_csv]
y = df['class']
print(f"Đã chọn {len(important_cols_in_csv)} đặc trưng để huấn luyện.")

output_dir = './model_assets'
os.makedirs(output_dir, exist_ok=True)

feature_columns_path = os.path.join(output_dir, 'feature_columns.json')
with open(feature_columns_path, 'w') as f:
    json.dump(important_cols_in_csv, f)
print(f"Đã lưu danh sách các cột đặc trưng vào '{feature_columns_path}'")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
scaler_path = os.path.join(output_dir, 'scaler.joblib')
joblib.dump(scaler, scaler_path)
print(f"Đã tạo và lưu scaler vào '{scaler_path}'")

print("Bắt đầu huấn luyện mô hình XGBoost...")
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train_scaled, y_train)
model_path = os.path.join(output_dir, 'parkinson_model.joblib')
joblib.dump(model, model_path)
print(f"Đã huấn luyện và lưu mô hình thành công vào '{model_path}'")

print("\n--- Đánh giá hiệu suất mô hình trên tập kiểm tra ---")
y_pred = model.predict(X_test_scaled)
print(f"Độ chính xác (Accuracy): {accuracy_score(y_test, y_pred):.4f}\n")
print("Báo cáo Phân loại (Classification Report):")
print(classification_report(y_test, y_pred, target_names=['Healthy', 'Parkinson\'s']))
print("Ma trận Nhầm lẫn (Confusion Matrix):")
print(confusion_matrix(y_test, y_pred))
print("\n--- Quá trình huấn luyện hoàn tất ---")