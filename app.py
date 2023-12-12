# Importing packages
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import streamlit as st

# Function to replace values
def clean_sm(x):
    return np.where(x == 1, 1, 0)

# Create a Streamlit app
def main():
    st.title("LinkedIn User Classifier App")

    # Read data
    s = pd.read_csv('social_media_usage.csv')

    # Create a new dataframe "ss"
    ss = pd.DataFrame({
        "sm_li": clean_sm(s['web1h']),
        "income": np.where(s['income'] > 9, np.nan, s['income']),
        "education": np.where(s['educ2'] > 8, np.nan, s['educ2']),
        "parent": np.where(s["par"] == 1, 1, 0),
        "married": np.where(s["marital"] == 1, 1, 0),
        "gender": np.where(s["gender"] == 2, 1, 0),
        "age": np.where(s['age'] > 98, np.nan, s['age'])
    })

    # Drop missing data
    ss = ss.dropna()



    # Target (y) and feature(s) selection (X)
    y = ss["sm_li"]
    X = ss[["income", "education", "parent", "married", "gender", "age"]]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=987)

    # Initialize logistic regression model
    lr = LogisticRegression(class_weight='balanced')

    # Fit model to training data
    lr.fit(X_train, y_train)

    # # Evaluate the model on test data
    # y_pred = lr.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)

    # # Display model accuracy
    # st.write(f"The model accuracy is: {accuracy:.2%}")

    # # Create a confusion matrix
    # conf_matrix = confusion_matrix(y_test, y_pred)
    # conf_matrix_df = pd.DataFrame(conf_matrix, columns=["Predicted Negative", "Predicted Positive"],
    #                               index=["Actual Negative", "Actual Positive"])

    # # Display the confusion matrix
    # st.subheader("Confusion Matrix:")
    # st.write(conf_matrix_df)

    # # Calculate and display precision, recall, and F1 score
    # precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
    # recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
    # f1_score = 2 * (precision * recall) / (precision + recall)

    # st.subheader("Precision, Recall, and F1 Score:")
    # st.write(f"Precision: {precision:.2%}")
    # st.write(f"Recall: {recall:.2%}")
    # st.write(f"F1 Score: {f1_score:.2%}")

    # User input for prediction
    st.subheader("Make Predictions:")
    income = st.slider("Income", 1, 9, 5)
    education = st.slider("Education", 1, 8, 4)
    parent = st.checkbox("Parent")
    married = st.checkbox("Married")
    gender = st.checkbox("Female")
    age = st.slider("Age", 18, 98, 30)

    # Make prediction for user input
    user_input = np.array([income, education, int(parent), int(married), int(gender), age]).reshape(1, -1)
    prediction_proba = lr.predict_proba(user_input)[:, 1]

    st.subheader("Prediction Result:")
    st.subheader("Prediction Result:")
    st.write(f"Probability of being a LinkedIn user: {prediction_proba[0]:.2%}")


if __name__ == "__main__":
    main()
