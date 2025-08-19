# src/extract_features.py

import parselmouth
from parselmouth.praat import call
import numpy as np
import pandas as pd
import librosa
import os
import fathon
from fathon import fathonUtils as fu
from pyrpde import rpde

def preprocess_audio(file_path, target_sr=22050):
    try:
        signal, sr = librosa.load(file_path, sr=target_sr, mono=True)
        signal = librosa.util.normalize(signal)
        return signal, sr
    except Exception as e:
        print(f"Lỗi khi tải tệp {file_path}: {e}")
        return None, None

def calculate_jitter_shimmer_hnr(sound, f0min, f0max, point_process):
    local_features = {}
    local_features['Jitter_local'] = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    local_features['Jitter_rap'] = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    local_features['Jitter_ppq5'] = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    local_features['Shimmer_local'] = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    local_features['Shimmer_apq3'] = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    local_features['Shimmer_apq5'] = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
    local_features['HNR'] = call(harmonicity, "Get mean", 0, 0)
    return local_features

def extract_all_features(file_path):
    features = {}
    signal, sr = preprocess_audio(file_path)
    if signal is None: return None
    sound = parselmouth.Sound(signal, sampling_frequency=sr)
    f0min, f0max = 75, 500
    
    try:
        aggregated_signal = fu.toAggregated(signal)
        pydfa = fathon.DFA(aggregated_signal)
        winSizes = np.logspace(1, 3, 50).astype(np.int64)
        n, F = pydfa.computeFlucVec(winSizes)
        alpha, _ = pydfa.fitFlucVec()
        features['DFA'] = alpha
    except Exception as e:
        print(f"Lỗi khi tính DFA: {e}"); features['DFA'] = np.nan

    try:
        signal_rpde = signal[:5000] if len(signal) > 5000 else signal
        rpde_result = rpde(signal_rpde, dim=3, tau=1, epsilon=0.02, tmax=1)
        features['RPDE'] = rpde_result[0] if isinstance(rpde_result, tuple) else rpde_result
    except Exception as e:
        print(f"Lỗi khi tính RPDE: {e}"); features['RPDE'] = np.nan

    try:
        point_process = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
        features.update(calculate_jitter_shimmer_hnr(sound, f0min, f0max, point_process))
    except Exception as e:
        print(f"Lỗi khi tính Jitter/Shimmer/HNR: {e}")
        keys = ['Jitter_local', 'Jitter_rap', 'Jitter_ppq5', 'Shimmer_local', 'Shimmer_apq3', 'Shimmer_apq5', 'HNR']
        for k in keys: features[k] = np.nan

    try:
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        for i in range(13): 
            features[f'MFCC_{i}'] = mfccs_mean[i]
    except Exception as e:
        print(f"Lỗi khi tính MFCCs: {e}")
        for i in range(13): features[f'MFCC_{i}'] = np.nan
            
    return features

if __name__ == "__main__":
    audio_file = input("Vui lòng nhập đường dẫn đến tệp âm thanh (.wav): ").strip().replace('"', '')
    if not os.path.exists(audio_file):
        print(f"Lỗi: Không tìm thấy tệp tại '{audio_file}'")
    else:
        print(f"Đang xử lý tệp: {audio_file}")
        all_extracted_features = extract_all_features(audio_file)
        if all_extracted_features:
            important_cols = [
                'DFA', 'RPDE', 'Jitter_local', 'Jitter_rap', 'Jitter_ppq5', 
                'Shimmer_local', 'Shimmer_apq3', 'Shimmer_apq5', 'HNR', 
                'MFCC_0', 'MFCC_1', 'MFCC_2', 'MFCC_3', 'MFCC_4', 'MFCC_5', 'MFCC_6', 'MFCC_7',
                'MFCC_8', 'MFCC_9', 'MFCC_10', 'MFCC_11', 'MFCC_12'
            ]
            df_features = pd.DataFrame([all_extracted_features])
            df_features['filename'] = os.path.basename(audio_file)
            final_cols = ['filename'] + important_cols
            df_features = df_features[final_cols]
            output_path = './data/extract_features/extracted_voice_features.csv'
            df_features.to_csv(output_path, index=False)
            print("\n--- KẾT QUẢ TRÍCH XUẤT ĐẶC TRƯNG ---")
            print(df_features.to_string())
            print(f"\nĐã lưu các đặc trưng vào tệp '{output_path}'")