import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

st.title("Tiny Iris Classifier")
st.write("Train a quick Logistic Regression model on the Iris dataset.")

X, y = load_iris(return_X_y=True, as_frame=True)
df = pd.concat([X, pd.Series(y, name="target")], axis=1)
st.dataframe(df.head())

test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
if st.button("Train"):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42)
    clf = LogisticRegression(max_iter=200).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    st.success(f"Accuracy: {accuracy_score(yte, pred):.3f}")

