import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title("Credit Card Fraud Detection System")

# load dataset
data = pd.read_csv("creditcard.csv")

st.subheader("Dataset Preview")
st.write(data.head())

# features and target
X = data.drop("Class", axis=1)
y = data["Class"]

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# prediction accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

st.subheader("Model Accuracy")
st.write(accuracy)

st.subheader("Fraud Prediction Example")

amount = st.number_input("Transaction Amount", 0.0)

if st.button("Check Transaction"):

    sample = X_test.iloc[0:1].copy()
    sample["Amount"] = amount

    prediction = model.predict(sample)

    if prediction[0] == 1:
        st.error("⚠ Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")