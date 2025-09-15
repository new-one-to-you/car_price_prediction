import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# Safe encode function
# ============================
def safe_encode(encoder, value):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        return 0  # Default encoding for unseen labels

# ============================
# 1. Load & Clean Dataset
# ============================
@st.cache_data
def load_data():
    df = pd.read_csv("Car details v3.csv")
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
# Encode Categorical Columns
# ============================
categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner', 'brand']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ============================
# Features & Target
# ============================
features = ['age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats'] + categorical_cols
X = df[features]
y = df['selling_price']

# Train Linear Regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Defaults
default_mileage = round(df['mileage'].mean(), 2)
default_engine = round(df['engine'].mean(), 2)
default_max_power = round(df['max_power'].mean(), 2)
default_seats = int(df['seats'].mode()[0])
default_seller = "Individual"
default_owner = "First Owner"

# ============================
# Page Config
# ============================
st.set_page_config(
    page_title="🚗 Car Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("🚗 Car Price Prediction")
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/743/743007.png", width=120)
menu = ["Predict", "Graphs", "View Data", "About"]
choice = st.sidebar.radio("Navigation", menu)

# ============================
# Predict Section
# ============================
if choice == "Predict":
    st.markdown("## 🔮 Predict Car Selling Price")
    
    # Input cards
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            brand = st.text_input("Brand (e.g., Maruti, Hyundai)")
            year = st.number_input("Year of Manufacture", min_value=1980, max_value=2025, value=2019)
            km_driven = st.number_input("KM Driven", min_value=0, value=50000)
        with col2:
            fuel = st.text_input("Fuel Type (Petrol, Diesel, CNG)")
            transmission = st.text_input("Transmission Type (Manual/Automatic)")
            condition = st.selectbox("Car Condition", ["Excellent", "Good", "Average", "Poor"])
            seats = st.number_input("Seats (optional)", min_value=2, max_value=10, value=default_seats)

    if st.button("Predict Price"):
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
                               default_max_power, seats,
                               fuel_encoded, seller_encoded, transmission_encoded,
                               owner_encoded, brand_encoded]])
        base_price = model.predict(input_vec)[0]
        final_price = base_price * condition_multiplier

        # Display result
        st.markdown("### 💰 Predicted Selling Price")
        st.metric(label="Price (INR)", value=f"₹ {round(final_price,2)}")

        # Display Car Details with defaults
        st.markdown("### 🚗 Car Details (User Input & Defaults)")
        details = {
            "Brand": brand,
            "Year": year,
            "KM Driven": km_driven,
            "Fuel Type": fuel,
            "Transmission": transmission,
            "Seats": seats,
            "Mileage (Default)": default_mileage,
            "Engine CC (Default)": default_engine,
            "Max Power (Default)": default_max_power,
            "Seller Type (Default)": default_seller,
            "Owner (Default)": default_owner,
            "Condition": condition
        }
        col1, col2 = st.columns(2)
        for i, (key, value) in enumerate(details.items()):
            if i % 2 == 0:
                col1.markdown(f"**{key}:** {value}")
            else:
                col2.markdown(f"**{key}:** {value}")

# ============================
# Graphs Section
# ============================
elif choice == "Graphs":
    st.markdown("## 📊 Explore Graphs")
    x_axis = st.selectbox("Select X-axis", df.columns, index=df.columns.get_loc('km_driven'))
    y_axis = st.selectbox("Select Y-axis", df.columns, index=df.columns.get_loc('selling_price'))

    fig, ax = plt.subplots(figsize=(8,5))
    sns.scatterplot(data=df, x=x_axis, y=y_axis, hue='fuel', palette='Set2', s=80)
    plt.title(f"{x_axis} vs {y_axis}")
    st.pyplot(fig)

# ============================
# View Data Section
# ============================
elif choice == "View Data":
    st.markdown("## 📑 Full Cleaned Dataset")
    st.dataframe(df, use_container_width=True)

# ============================
# About Section
# ============================
elif choice == "About":
    st.markdown("## ℹ️ About this App")
    st.markdown("""
This **Car Price Prediction App** predicts the selling price of used cars using **Linear Regression**.

**Features:**
- Predict car price with optional condition-based adjustment
- Auto-fill defaults for mileage, engine, power, and seats
- Explore dataset with interactive scatter plots
- User-friendly layout with metric boxes and side navigation

**Developed with ❤️ using:**  
- Streamlit  
- Pandas & NumPy  
- Scikit-Learn  
- Matplotlib & Seaborn
    """)
