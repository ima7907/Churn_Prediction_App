import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt

# ------------------------------
# Load model + files
# ------------------------------
path = os.path.dirname(__file__)

model = pickle.load(open(os.path.join(path, "model.pkl"), "rb"))
columns = pickle.load(open(os.path.join(path, "columns.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(path, "scaler.pkl"), "rb"))
accuracy = pickle.load(open(os.path.join(path, "accuracy.pkl"), "rb"))
cm = pickle.load(open(os.path.join(path, "cm.pkl"), "rb"))

# ------------------------------
# UI
# ------------------------------
st.set_page_config(page_title="Churn Prediction", layout="wide")

st.title("📊 Customer Churn Prediction App")
st.markdown("### Predict whether a customer will churn or not")

# ------------------------------
# Sidebar Inputs
# ------------------------------
st.sidebar.header("Enter Customer Details")

tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly = st.sidebar.number_input("Monthly Charges", 0.0, 10000.0, 500.0)
total = st.sidebar.number_input("Total Charges", 0.0, 100000.0, 5000.0)

contract = st.sidebar.selectbox("Contract Type", [
    "Month-to-month", "One year", "Two year"
])

internet = st.sidebar.selectbox("Internet Service", [
    "DSL", "Fiber optic", "No"
])

payment = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer", "Credit card"
])

security = st.sidebar.selectbox("Online Security", ["Yes", "No"])
support = st.sidebar.selectbox("Tech Support", ["Yes", "No"])

# ------------------------------
# Input Validation
# ------------------------------
if monthly < 50:
    st.warning("⚠️ Monthly charges too low. Enter value above 50.")
    st.stop()

if total < 100:
    st.warning("⚠️ Total charges too low. Enter value above 100.")
    st.stop()

# ------------------------------
# Prepare Input
# ------------------------------
input_data = {
    "tenure": tenure,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
    "Contract": contract,
    "InternetService": internet,
    "PaymentMethod": payment,
    "OnlineSecurity": security,
    "TechSupport": support
}

df = pd.DataFrame([input_data])
df = pd.get_dummies(df)
df = df.reindex(columns=columns, fill_value=0)

# Scale
df_scaled = scaler.transform(df)

# ------------------------------
# Prediction
# ------------------------------
if st.sidebar.button("🚀 Predict Churn"):

    prob = model.predict_proba(df_scaled)[0][1]

    if prob >= 0.40:
        st.error(f"⚠️ Customer is likely to CHURN ({prob:.2f})")
    else:
        st.success(f"✅ Customer will NOT churn ({1-prob:.2f})")

    # ------------------------------
    # Probability Graph
    # ------------------------------
    st.subheader("📊 Churn Probability")

    fig, ax = plt.subplots()
    ax.bar(["Not Churn", "Churn"], [1-prob, prob])
    ax.set_title("Prediction Probability")

    st.pyplot(fig)

# ------------------------------
# Model Metrics
# ------------------------------
st.markdown("---")
st.subheader("📊 Model Performance")

st.write(f"✅ Accuracy: {accuracy:.2f}")

st.subheader("Confusion Matrix")

fig2, ax2 = plt.subplots()
ax2.imshow(cm)

for i in range(len(cm)):
    for j in range(len(cm)):
        ax2.text(j, i, cm[i][j], ha="center", va="center")

ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")

st.pyplot(fig2)