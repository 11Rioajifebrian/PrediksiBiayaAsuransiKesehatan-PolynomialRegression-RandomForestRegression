import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

# Lokasi dataset (pastikan file ada di folder yang sama dengan app.py)
DATASET_PATH = "Regression.csv"

# Menambahkan CSS untuk mempercantik font dan menambahkan ikon
st.markdown("""
    <style>
        /* Gaya font untuk seluruh aplikasi */
        body {
            font-family: 'Arial', sans-serif;
        }

        /* Gaya font untuk judul halaman */
        .css-1d391kg {
            font-family: 'Verdana', sans-serif;
            font-size: 30px;
            font-weight: bold;
            color: #4CAF50;
        }

        /* Gaya font untuk elemen sidebar */
        .sidebar .sidebar-content {
            background: #f4f7fa;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            font-family: 'Courier New', monospace;
        }

        /* Gaya font untuk teks di dalam sidebar */
        .sidebar .sidebar-content h3 {
            font-size: 18px;
            font-weight: bold;
            color: #333333;
        }

        /* Gaya font untuk subheader di Streamlit */
        .css-1v3fvcr {
            font-family: 'Georgia', serif;
            font-weight: normal;
            font-size: 24px;
            color: #3F51B5;
        }
    </style>
""", unsafe_allow_html=True)

# Judul Halaman
st.title("Prediksi Biaya Asuransi Kesehatan menggunakan Kombinasi Algoritma Polynomial Regression dan Random Forest Regression")

# Memuat dataset
st.write("Memuat dataset... :floppy_disk:")
try:
    data = pd.read_csv(DATASET_PATH)
    st.write("Dataset berhasil dimuat! :white_check_mark:")
    st.write(data.head())
except FileNotFoundError:
    st.error(f"File dataset '{DATASET_PATH}' tidak ditemukan. Pastikan file berada di folder yang sama dengan script ini. :x:")
    st.stop()

# Preprocessing
st.write("Melakukan preprocessing pada dataset... :gear:")
encoded_data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)
X = encoded_data.drop('charges', axis=1)
y = encoded_data['charges']

# Menyimpan nama kolom fitur
feature_names = X.columns

# Membagi dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Polynomial Regression
st.write("Melatih model Polynomial Regression... :chart_with_upwards_trend:")
poly = PolynomialFeatures(degree=3)  # Menggunakan derajat polinomial 3
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

random_forest_model = RandomForestRegressor(random_state=42, n_estimators=100)
random_forest_model.fit(X_train, y_train)

# Prediksi pada data uji
poly_pred = poly_model.predict(X_test_poly)
rf_pred = random_forest_model.predict(X_test)
combined_pred = (poly_pred + rf_pred) / 2

# Evaluasi model
mse_poly = mean_squared_error(y_test, poly_pred)
mse_rf = mean_squared_error(y_test, rf_pred)
mse_combined = mean_squared_error(y_test, combined_pred)

mae_poly = mean_absolute_error(y_test, poly_pred)
mae_rf = mean_absolute_error(y_test, rf_pred)
mae_combined = mean_absolute_error(y_test, combined_pred)

rmse_poly = np.sqrt(mse_poly)
rmse_rf = np.sqrt(mse_rf)
rmse_combined = np.sqrt(mse_combined)

r2_poly = r2_score(y_test, poly_pred)
r2_rf = r2_score(y_test, rf_pred)
r2_combined = r2_score(y_test, combined_pred)

# Menampilkan hasil evaluasi
st.subheader("Hasil Evaluasi Model :bar_chart:")
st.write(f"Polynomial Regression - MSE: {mse_poly:.2f}, R²: {r2_poly:.2f}")
st.write(f"Random Forest - MSE: {mse_rf:.2f}, R²: {r2_rf:.2f}")
st.write(f"Model Gabungan - MSE: {mse_combined:.2f}, R²: {r2_combined:.2f}")

# Menampilkan hasil akurasi tambahan
st.subheader("Hasil Akurasi Model :star:")
st.write(f"Polynomial Regression - MAE: {mae_poly:.2f}, RMSE: {rmse_poly:.2f}")
st.write(f"Random Forest - MAE: {mae_rf:.2f}, RMSE: {rmse_rf:.2f}")
st.write(f"Model Gabungan - MAE: {mae_combined:.2f}, RMSE: {rmse_combined:.2f}")

# Tambahan evaluasi dengan print
st.subheader("Tambahan Evaluasi dengan Print :clipboard:")
y_predPoly = poly_model.predict(X_test_poly)
y_predRandom = random_forest_model.predict(X_test)
st.text("Polynomial Regression")
st.text(f"Mean Absolute Error: {metrics.mean_absolute_error(y_test, y_predPoly):.2f}")
st.text(f"Mean Squared Error: {metrics.mean_squared_error(y_test, y_predPoly):.2f}")
st.text(f"Root Mean Squared Error: {np.sqrt(metrics.mean_squared_error(y_test, y_predPoly)):.2f}")

st.text("Random Forest Regression")
st.text(f"Mean Absolute Error: {metrics.mean_absolute_error(y_test, y_predRandom):.2f}")
st.text(f"Mean Squared Error: {metrics.mean_squared_error(y_test, y_predRandom):.2f}")
st.text(f"Root Mean Squared Error: {np.sqrt(metrics.mean_squared_error(y_test, y_predRandom)):.2f}")

# Precision, Recall, F1-Score
st.subheader("Precision, Recall, F1-Score")

def regression_classification_report(y_true, y_pred):
    residual = y_true - y_pred
    threshold = residual.std()  # Menggunakan standar deviasi residu sebagai batas
    residual_class = (abs(residual) < threshold).astype(int)
    y_class_true = np.ones_like(residual_class)
    report = metrics.classification_report(y_class_true, residual_class, target_names=['Outlier', 'Inlier'], output_dict=True)
    return report

# Polynomial Regression Classification Report
st.write("**Polynomial Regression**")
poly_report = regression_classification_report(y_test, y_predPoly)
st.write(f"Precision: {poly_report['Inlier']['precision']:.2f}")
st.write(f"Recall: {poly_report['Inlier']['recall']:.2f}")
st.write(f"F1-Score: {poly_report['Inlier']['f1-score']:.2f}")

# Random Forest Regression Classification Report
st.write("**Random Forest Regression**")
random_report = regression_classification_report(y_test, y_predRandom)
st.write(f"Precision: {random_report['Inlier']['precision']:.2f}")
st.write(f"Recall: {random_report['Inlier']['recall']:.2f}")
st.write(f"F1-Score: {random_report['Inlier']['f1-score']:.2f}")

# Visualisasi MSE
st.subheader("Perbandingan MSE :bar_chart:")
fig, ax = plt.subplots()
methods = ["Polynomial Regression", "Random Forest", "Gabungan"]
mse_values = [mse_poly, mse_rf, mse_combined]
ax.bar(methods, mse_values, color=['blue', 'green', 'orange'])
ax.set_ylabel("Mean Squared Error (MSE)")
ax.set_title("Perbandingan Performa Model")
st.pyplot(fig)

# Visualisasi RMSE
st.subheader("Perbandingan RMSE :bar_chart:")
fig, ax = plt.subplots()
rmse_values = [rmse_poly, rmse_rf, rmse_combined]
ax.bar(methods, rmse_values, color=['blue', 'green', 'orange'])
ax.set_ylabel("Root Mean Squared Error (RMSE)")
ax.set_title("Perbandingan RMSE Model")
st.pyplot(fig)

# Analisis Rinci Data
st.subheader("Analisis Rinci Data :mag:")
# Menampilkan distribusi data dengan KDE plot
st.write("Distribusi Data (KDE Plot) :chart:")
fig, ax = plt.subplots()
sns.kdeplot(y, fill=True, ax=ax, color="skyblue")
ax.set_title("Distribusi Biaya Asuransi")
ax.set_xlabel("Biaya Asuransi")
st.pyplot(fig)

# Menampilkan heatmap untuk korelasi fitur
st.write("Heatmap Korelasi :fire:")
fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = encoded_data.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Korelasi Antar Fitur")
st.pyplot(fig)

# Visualisasi catplot untuk fitur kategorikal
st.write("Visualisasi Catplot :bar_chart:")
fig = sns.catplot(data=data, x="smoker", y="charges", hue="sex", kind="box", aspect=2)
fig.set_axis_labels("Perokok", "Biaya Asuransi")
fig.fig.suptitle("Biaya Asuransi Berdasarkan Kebiasaan Merokok dan Jenis Kelamin")
st.pyplot(fig)

# Plot garis regresi sempurna
st.write("Garis Regresi Sempurna (Ideal) :straight_ruler:")
fig, ax = plt.subplots()
ax.scatter(y_test, poly_pred, color='blue', alpha=0.5, label='Polynomial Regression')
ax.scatter(y_test, rf_pred, color='green', alpha=0.5, label='Random Forest')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, label='Perfect Prediction')
ax.set_xlabel("Biaya Asli")
ax.set_ylabel("Biaya Prediksi")
ax.set_title("Perbandingan Prediksi vs Realita")
ax.legend()
st.pyplot(fig)

# Input pengguna untuk prediksi
# Sidebar card style
st.sidebar.markdown("""
    <style>
        .sidebar .sidebar-content {
            background: #f4f7fa;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        .sidebar .sidebar-content .stTextInput, 
        .sidebar .sidebar-content .stSlider, 
        .sidebar .sidebar-content .stSelectbox {
            margin-bottom: 10px;
        }
        .sidebar .sidebar-content h3 {
            margin-top: 0;
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.header("Input Parameter Pengguna :person:")

# Fitur input pengguna
def user_input_features():
    age = st.sidebar.slider("Umur", 18, 100, 30)
    sex = st.sidebar.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
    children = st.sidebar.slider("Jumlah Anak", 0, 10, 1)
    smoker = st.sidebar.selectbox("Perokok", ["Ya", "Tidak"])
    region = st.sidebar.selectbox("Wilayah", ["Northeast", "Northwest", "Southeast", "Southwest"])

    input_data = {
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex_male": 1 if sex == "Laki-laki" else 0,
        "smoker_yes": 1 if smoker == "Ya" else 0,
        "region_northwest": 1 if region == "Northwest" else 0,
        "region_southeast": 1 if region == "Southeast" else 0,
        "region_southwest": 1 if region == "Southwest" else 0,
    }

    for col in feature_names:
        if col not in input_data:
            input_data[col] = 0

    return pd.DataFrame(input_data, index=[0])

input_df = user_input_features()

# Input nilai tukar USD ke IDR
st.sidebar.header("Konversi Mata Uang :moneybag:")
exchange_rate = st.sidebar.number_input("Nilai Tukar USD ke IDR", value=15000.0, step=100.0)

# Pindahkan tombol prediksi ke sidebar
predict_button = st.sidebar.button("Prediksi Biaya :rocket:")

if predict_button:
    input_poly = poly.transform(input_df)
    poly_pred = poly_model.predict(input_poly)[0]
    rf_pred = random_forest_model.predict(input_df)[0]
    combined_pred = (poly_pred + rf_pred) / 2

    # Konversi ke rupiah
    poly_pred_idr = poly_pred * exchange_rate
    rf_pred_idr = rf_pred * exchange_rate
    combined_pred_idr = combined_pred * exchange_rate

    st.subheader("Hasil Prediksi Biaya :dollar:")
    st.write(f"Prediksi Polynomial Regression: ${poly_pred:.2f} (Rp{poly_pred_idr:,.2f})")
    st.write(f"Prediksi Random Forest: ${rf_pred:.2f} (Rp{rf_pred_idr:,.2f})")
    st.write(f"Prediksi Gabungan: ${combined_pred:.2f} (Rp{combined_pred_idr:,.2f})")

    # Visualisasi prediksi
    fig, ax = plt.subplots()
    methods = ["Polynomial Regression", "Random Forest", "Gabungan"]
    predictions = [poly_pred_idr, rf_pred_idr, combined_pred_idr]
    ax.bar(methods, predictions, color=['blue', 'green', 'orange'])
    ax.set_ylabel("Prediksi Biaya (IDR)")
    ax.set_title("Perbandingan Prediksi (Dalam Rupiah)")
    st.pyplot(fig)
