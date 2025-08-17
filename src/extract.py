import pandas as pd
import numpy as np
import parselmouth
import librosa
from scipy.stats import entropy
import os
from pathlib import Path

# Định nghĩa hàm trích xuất đặc trưng
def extract_features_from_wav(file_path, sample_id):
    try:
        # Khởi tạo dictionary để lưu đặc trưng
        features = {'id': sample_id}

        # 1. Jitter và Shimmer (parselmouth)
        sound = parselmouth.Sound(file_path)
        point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 50, 600)  # Điều chỉnh fmin, fmax
        
        # Kiểm tra số điểm trong PointProcess
        num_points = parselmouth.praat.call(point_process, "Get number of points")
        if num_points < 2:
            print(f"Warning: Insufficient pitch points for {file_path}. Skipping shimmer features.")
            features['locPctJitter'] = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            features['locAbsJitter'] = parselmouth.praat.call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
            features['rapJitter'] = parselmouth.praat.call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
            features['ppq5Jitter'] = parselmouth.praat.call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
            features['ddpJitter'] = parselmouth.praat.call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
            features['locShimmer'] = np.nan
            features['locDbShimmer'] = np.nan
            features['apq5Shimmer'] = np.nan
            features['apq11Shimmer'] = np.nan
        else:
            features['locPctJitter'] = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            features['locAbsJitter'] = parselmouth.praat.call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
            features['rapJitter'] = parselmouth.praat.call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
            features['ppq5Jitter'] = parselmouth.praat.call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
            features['ddpJitter'] = parselmouth.praat.call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
            try:
                features['locShimmer'] = parselmouth.praat.call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                features['apq3Shimmer'] = parselmouth.praat.call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                features['apq5Shimmer'] = parselmouth.praat.call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                features['apq11Shimmer'] = parselmouth.praat.call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                features['ddaShimmer'] = parselmouth.praat.call([sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            except:
                print(f"Warning: Shimmer calculation failed for {file_path}. Assigning NaN.")
                features['locShimmer'] = np.nan
                features['ddaShimmer'] = np.nan
                features['locDbShimmer'] = np.nan
                features['apq5Shimmer'] = np.nan
                features['apq11Shimmer'] = np.nan

        # 2. PPE (librosa)
        y, sr = librosa.load(file_path, sr=None)
        y = y / np.max(np.abs(y))
        f0, _, _ = librosa.pyin(y=y, fmin=50, fmax=600, sr=sr)  # Điều chỉnh fmin, fmax
        f0 = f0[~np.isnan(f0)]
        if len(f0) < 10:
            print(f"Warning: Insufficient pitch values for PPE in {file_path}. Assigning NaN.")
            features['PPE'] = np.nan
        else:
            hist, _ = np.histogram(f0, bins=50, density=True)
            ppe_norm = entropy(hist, base=2) / np.log2(len(hist))
            features['PPE'] = ppe_norm

        # 3. DFA
        if len(f0) < 10:
            features['DFA'] = np.nan
        else:
            f0_norm = (f0 - np.mean(f0)) / np.std(f0)
            def dfa(signal):
                n_vals = np.floor(np.logspace(np.log10(4), np.log10(len(signal)//4), num=20)).astype(int)
                n_vals = np.unique(n_vals)
                flucts = []
                for n in n_vals:
                    if n < 4:
                        continue
                    segments = len(signal) // n
                    rms_vals = []
                    for seg in range(segments):
                        segment_data = signal[seg*n : (seg+1)*n]
                        t = np.arange(n)
                        coeffs = np.polyfit(t, segment_data, 1)
                        trend = np.polyval(coeffs, t)
                        detrended = segment_data - trend
                        rms_vals.append(np.sqrt(np.mean(detrended**2)))
                    flucts.append(np.mean(rms_vals))
                flucts = np.array(flucts)
                n_vals = np.array(n_vals[:len(flucts)])
                coeffs = np.polyfit(np.log(n_vals), np.log(flucts), 1)
                return coeffs[0]
            features['DFA'] = dfa(f0_norm)

        # 4. RPDE
        def rpde(signal, embedding_dim=3, tau=1):
            N = len(signal) - (embedding_dim - 1) * tau
            if N <= 0:
                print(f"Warning: Insufficient signal length for RPDE in {file_path}. Assigning NaN.")
                return np.nan
            embedded = np.array([signal[i:i + N] for i in range(0, embedding_dim * tau, tau)]).T
            distances = np.sqrt(np.sum((embedded[:, None] - embedded[None, :])**2, axis=2))
            recurrence_times = []
            for i in range(len(distances)):
                idx = np.where(distances[i] < 0.1)[0]
                if len(idx) > 1:
                    recurrence_times.append(idx[1] - idx[0])
            if not recurrence_times:
                print(f"Warning: No recurrence times for RPDE in {file_path}. Assigning NaN.")
                return np.nan
            hist, _ = np.histogram(recurrence_times, bins=50, density=True)
            return entropy(hist)
        features['RPDE'] = rpde(y)

        # 5. Mean period và Std dev period
        times = [parselmouth.praat.call(point_process, "Get time from index", i + 1)
                 for i in range(parselmouth.praat.call(point_process, "Get number of points"))]
        periods = np.diff(times)
        periods_ms = np.array(periods)
        features['meanPeriodPulses'] = np.mean(periods_ms) if len(periods_ms) > 0 else np.nan
        features['stdDevPeriodPulses'] = np.std(periods_ms) if len(periods_ms) > 0 else np.nan

        # 6. Harmonicity features
        def harmonicity_features(path, time_step=0.01, min_pitch=50, silence_threshold=0.03, periods_per_window=1.0):
            snd = parselmouth.Sound(path)
            harm = snd.to_harmonicity_ac(time_step=time_step, minimum_pitch=min_pitch,
                                         silence_threshold=silence_threshold, periods_per_window=periods_per_window)
            hnr_db = harm.values.ravel()
            valid = hnr_db > -200
            if not np.any(valid):
                harm = snd.to_harmonicity_ac(time_step=time_step, minimum_pitch=40,
                                             silence_threshold=0.0, periods_per_window=periods_per_window)
                hnr_db = harm.values.ravel()
                valid = hnr_db > -200
                if not np.any(valid):
                    print(f"Warning: No voiced frames for harmonicity in {path}. Assigning NaN.")
                    return np.nan, np.nan, np.nan
            hnr_db_v = hnr_db[valid]
            mean_hnr = float(np.mean(hnr_db_v))
            nhr = 10.0 ** (-hnr_db_v / 10.0)
            mean_nhr = float(np.mean(nhr))
            rpow = 10.0 ** (hnr_db_v / 10.0)
            r = rpow / (1.0 + rpow)
            mean_auto_corr = float(np.mean(r))
            return mean_auto_corr, mean_hnr, mean_nhr
        mean_auto_corr, mean_hnr, mean_nhr = harmonicity_features(file_path)
        features['meanAutoCorrHarmonicity'] = mean_auto_corr
        features['meanHarmToNoiseHarmonicity'] = mean_hnr
        features['meanNoiseToHarmHarmonicity'] = mean_nhr

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Đọc file Demographics_age_sex.csv
demographics = pd.read_csv("./data/Demographics.csv")
audio_folder = "./data/raw_audio"  # Điều chỉnh đường dẫn thư mục

# Trích xuất đặc trưng cho tất cả file WAV
all_features = []
for _, row in demographics.iterrows():
    sample_id = row['Sample ID']
    wav_path = os.path.join(audio_folder, f"{sample_id}.wav")
    
    if not os.path.exists(wav_path):
        print(f"File not found: {wav_path}")
        continue
    
    features = extract_features_from_wav(wav_path, sample_id)
    if features is not None:
        # Thêm thông tin từ demographics
        features['class'] = row['Label']
        features['gender'] = 1 if row['Sex'] == 'M' else 0
        all_features.append(features)

# Tạo DataFrame và lưu vào CSV
features_df = pd.DataFrame(all_features)

pd_speech_df = pd.read_csv("./data/pd_speech_features_scaled.csv")
column_orders = pd_speech_df.columns.tolist()
features_df = features_df[column_orders]
# Lưu DataFrame vào file CSV
output_csv = "extracted_features.csv"
features_df.to_csv(output_csv, index=False)
print(f"Features saved to {output_csv}")

# Kiểm tra cột
print("\nColumns in output CSV:")
print(features_df.columns.tolist())