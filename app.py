import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# 1. Load & Clean Dataset
# ============================
@st.cache_data
def load_data():
    df = pd.read_csv("Car details v3.csv")  # Update with your CSV
    df = df.dropna(subset=['engine', 'mileage', 'max_power', 'seats'])
    df['mileage'] = df['mileage'].str.extract(r'(\d+\.?\d*)').astype(float)
    df['engine'] = df['engine'].str.extract(r'(\d+\.?\d*)').astype(float)
    df['max_power'] = df['max_power'].str.extract(r'(\d+\.?\d*)').astype(float)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df['age'] = 2025 - df['year']
    df['brand'] = df['name'].str.split(' ').str[0]
    df.drop('name', axis=1, inplace=True)
    return df

df = load_data()

# ============================
# 2. Encode Categorical Columns
# ============================
categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner', 'brand']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ============================
# 3. Features & Target
# ============================
features = ['age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats'] + categorical_cols
X = df[features]
y = df['selling_price']

# Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Auto-fill defaults
default_mileage = round(df['mileage'].mean(), 2)
default_engine = round(df['engine'].mean(), 2)
default_max_power = round(df['max_power'].mean(), 2)
default_seats = int(df['seats'].mode()[0])
default_seller = "Individual"
default_owner = "First Owner"

# ============================
# 4. Streamlit Layout
# ============================
st.title("🚗 Car Price Prediction App")
menu = ["Predict", "Graphs", "View Data", "About"]
choice = st.radio("Navigation", menu, horizontal=True)

# ============================
# 5. Predict Section
# ============================
if choice == "Predict":
    st.header("🔮 Predict Car Selling Price")
    col1, col2 = st.columns(2)

    with col1:
        brand = st.text_input("Brand (e.g., Maruti, Hyundai)")
        year = st.number_input("Year of Manufacture", min_value=1980, max_value=2025, value=2019)
        km_driven = st.number_input("KM Driven", min_value=0, value=50000)

    with col2:
        fuel = st.selectbox("Fuel Type", df['fuel'].astype(str).unique())
        transmission = st.selectbox("Transmission", df['transmission'].astype(str).unique())
        condition = st.selectbox("Car Condition", ["Excellent", "Good", "Average", "Poor"])

    if st.button("Predict"):
        age = 2025 - year
        brand_encoded = safe_encode(encoders['brand'], brand)
        fuel_encoded = safe_encode(encoders['fuel'], fuel)
        seller_encoded = safe_encode(encoders['seller_type'], default_seller)
        transmission_encoded = safe_encode(encoders['transmission'], transmission)
        owner_encoded = safe_encode(encoders['owner'], default_owner)


        # Condition factor
        condition_factor = {"Excellent": 1.05, "Good": 1.0, "Average": 0.9, "Poor": 0.8}
        condition_multiplier = condition_factor.get(condition, 1.0)

        # Input vector
        input_vec = np.array([[age, km_driven, default_mileage, default_engine,
                               default_max_power, default_seats,
                               fuel_encoded, seller_encoded, transmission_encoded,
                               owner_encoded, brand_encoded]])
        
        base_price = model.predict(input_vec)[0]
        final_price = base_price * condition_multiplier

        st.subheader(f"Predicted Selling Price: ₹{round(final_price, 2)}")

        st.subheader("Cleaned Dataset")
        st.dataframe(df, use_container_width=True)

# ============================
# 6. Graphs Section
# ============================
elif choice == "Graphs":
    st.header("📊 Graphs & Scatter Plots")
    x_axis = st.selectbox("Select X-axis", df.columns)
    y_axis = st.selectbox("Select Y-axis", df.columns)
    
    plt.figure(figsize=(8,5))
    plt.scatter(df[x_axis], df[y_axis], alpha=0.6, color='green')
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(f"{x_axis} vs {y_axis}")
    st.pyplot(plt)

# ============================
# 7. View Data Section
# ============================
elif choice == "View Data":
    st.header("📑 Full Cleaned Dataset")
    st.dataframe(df, use_container_width=True)

# ============================
# 8. About Section
# ============================
elif choice == "About":
    st.header("ℹ️ About this App")
    st.write("""
    This **Car Price Prediction App** predicts the selling price of used cars based on important features.
    
    ### Features:
    - Predict car price using **Random Forest Regressor**
    - Auto-fill default values for mileage, engine, power, seats
    - Adjust price based on car condition (Excellent / Good / Average / Poor)
    - Explore dataset and scatter plots
    
    **Developed with ❤️ using Streamlit, Scikit-Learn, Pandas, Matplotlib & Seaborn**
    """)


