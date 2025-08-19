# src/predict.py

import pandas as pd
import joblib
import os
import json

def diagnose_voice_sample(input_csv_path):
    print("Đang tải mô hình, scaler và danh sách đặc trưng...")
    
    model_dir = './model_assets'
    try:
        model = joblib.load(os.path.join(model_dir, 'parkinson_model.joblib'))
        scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
        with open(os.path.join(model_dir, 'feature_columns.json'), 'r') as f:
            feature_columns_expected_by_model = json.load(f)
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy tệp cần thiết trong thư mục '{model_dir}'. Vui lòng chạy 'train_model.py' trước.")
        return

    try:
        new_data = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy tệp đầu vào '{input_csv_path}'.")
        return

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
    
    new_data_renamed = new_data.rename(columns=feature_columns_map)

    try:
        X_new = new_data_renamed[feature_columns_expected_by_model]
    except KeyError as e:
        print(f"Lỗi: Tệp CSV đầu vào thiếu các cột đặc trưng. Cột bị thiếu: {e}")
        return

    X_new_scaled = scaler.transform(X_new)

    prediction = model.predict(X_new_scaled)
    probabilities = model.predict_proba(X_new_scaled)
    
    predicted_class = prediction[0]
    probability_of_predicted_class = probabilities[0][predicted_class] * 100
    diagnosis = "Bệnh Parkinson" if predicted_class == 1 else "Khỏe mạnh"

    print("\n--- KẾT QUẢ CHẨN ĐOÁN ---")
    print(f"Chẩn đoán: {diagnosis}")
    print(f"Độ chắc chắn của mô hình: {probability_of_predicted_class:.2f}%")
    print("\nLưu ý: Kết quả này chỉ mang tính tham khảo và không thay thế cho chẩn đoán y tế chuyên nghiệp.")

if __name__ == "__main__":
    features_file = './data/extract_features/extracted_voice_features.csv'
    if not os.path.exists(features_file):
        print(f"Lỗi: Không tìm thấy tệp '{features_file}'. Vui lòng chạy 'extract_features.py' trước.")
    else:
        diagnose_voice_sample(features_file)