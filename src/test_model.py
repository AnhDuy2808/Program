import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler

# === Đường dẫn file ===
CSV_PATH = "./extracted_features.csv"       # file test bạn muốn dùng
SCALER_PATH = "./data/scaler/scaler.pkl"    # scaler từ data_processing
MODEL_PATH = "./notebook/random_forest_model.pkl"             # model từ models_training

# 1. Đọc dữ liệu test
df = pd.read_csv(CSV_PATH)


# Chuyển HC -> 0, PD -> 1
df["class"] = df["class"].apply(lambda x: 0 if str(x).strip().upper() == "HC" else 1)

df['PPE'] = df['PPE'].fillna(df['PPE'].mean())
df = df.drop(columns=['DFA'], errors='ignore')

id_col = df['id'] if 'id' in df.columns else None
class_col = df['class'] if 'class' in df.columns else None

extract_features = df.drop(columns=['id', 'class'], errors='ignore')


# 2. Load scaler
scaler = joblib.load(SCALER_PATH)
extract_features_scaled = scaler.transform(extract_features)

df_scaled = pd.DataFrame(extract_features_scaled, columns=extract_features.columns)
if id_col is not None:
    df_scaled.insert(0, 'id', id_col)

if class_col is not None:
    df_scaled['class'] = class_col

df_scaled.to_csv('./data/extract_features_scaled.csv', index=False)


X = df_scaled.drop(columns=['id', 'class'], errors="ignore")

y = df_scaled["class"]

# 3. Load model
model = joblib.load(MODEL_PATH)

# 4. Dự đoán
y_pred = model.predict(X)

# 5. Đánh giá
acc = accuracy_score(y, y_pred)
cm = confusion_matrix(y, y_pred)
report = classification_report(y, y_pred)

print(f"Accuracy: {acc:.4f}")
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(report)
