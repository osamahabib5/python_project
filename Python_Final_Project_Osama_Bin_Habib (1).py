#!/usr/bin/env python
# coding: utf-8

# # Final Project
# ## Osama Bin Habib
# ### Date:5th Dec 2023

# ***

# In[123]:


#Importing packages
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import streamlit as st

# #### Q1 Read in the data, call the dataframe "s"  and check the dimensions of the dataframe

# In[124]:


s = pd.read_csv('social_media_usage.csv')

# using dataframe.shape
shape = s.shape


print("Shape = {}".format(shape))


# ***

# #### Q2 Define a function called clean_sm that takes one input, x, and uses `np.where` to check whether x is equal to 1. If it is, make the value of x = 1, otherwise make it 0. Return x. Create a toy dataframe with three rows and two columns and test your function to make sure it works as expected

# In[125]:


#Function to replace s
def clean_sm(x):
    return np.where(x == 1, 1, 0)

# Create a toy dataframe
toy = {'Column1': [1, 2, 3],
        'Column2': [0, 1, 1]}

df = pd.DataFrame(toy)

# Apply the clean_sm function to each element in the dataframe
df_cleaned = df.applymap(clean_sm)

# Display the original and cleaned dataframes
print("Original DataFrame:")
print(df)

print("\nCleaned DataFrame:")
print(df_cleaned)


# ***

# #### Q3 Create a new dataframe called "ss". The new dataframe should contain a target column called sm_li which should be a binary variable ( that takes the value of 1 if it is 1 and 0 otherwise (use clean_sm to create this) which indicates whether or not the individual uses LinkedIn, and the following features: income (ordered numeric from 1 to 9, above 9 considered missing), education (ordered numeric from 1 to 8, above 8 considered missing), parent (binary), married (binary), female (binary), and age (numeric, above 98 considered missing). Drop any missing values. Perform exploratory analysis to examine how the features are related to the target.

# In[126]:


# import pandas as pd
# import numpy as np

ss = pd.DataFrame({
    "sm_li":clean_sm(s['web1h']),
    "income": np.where(s['income'] > 9, np.nan, s['income']),
    "education":np.where(s['educ2'] > 8, np.nan, s['educ2']),
    "parent":np.where(s["par"] == 1,1,0),
    "married":np.where(s["marital"] == 1,1,0),
    "gender":np.where(s["gender"] == 2,1,0),
    "age":np.where(s['age'] > 98, np.nan, s['age'])})


# Drop missing data (or impute it)
ss = ss.dropna()


# + Target: sm_li (Potential Linkedin Customer)
#     + The user uses Linkedin (=1)
#     + The user doesn't use Linkedin  (=0)
# + Features:
#     + income (numeric)
#     + education (numeric)
#     + parent (binary)
#     + married (binary)
#     + gender (binary)
#     + age (numeric)

# In[127]:


#Exploratory data analysis
alt.Chart(ss.groupby(["income", "education"], as_index=False)["sm_li"].mean()).\
mark_circle().\
encode(x="income",
      y="sm_li",
      color="education:N")


# ***

# #### Q4 Create a target vector (y) and feature set (X)

# In[128]:


# Target (y) and feature(s) selection (X)
y = ss["sm_li"]
X = ss[["income", "education", "parent", "married","gender","age"]]


# ***

# #### Q5 Split the data into training and test sets. Hold out 20% of the data for testing. Explain what each new object contains and how it is used in machine learning

# In[129]:


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=987) # set for reproducibility


# - X_train contains 80% of the data and contains the features used to predict the target when training the model. 
# - X_test contains 20% of the data and contains the features used to test the model on unseen data to evaluate performance. 
# - y_train contains 80% of the the data and contains the target that we will predict using the features when training the model. 
# - y_test contains 20% of the data and contains the target we will predict when testing the model on unseen data to evaluate performance.

# ***

# #### Q6 Instantiate a logistic regression model and set class_weight to balanced. Fit the model with the training data.

# In[130]:


# Initialize algorithm 
lr = LogisticRegression(class_weight='balanced')


# In[131]:


# Fit algorithm to training data
lr.fit(X_train, y_train)


# ***

# #### Q7 Evaluate the model using the testing data. What is the model accuracy for the model? Use the model to make predictions and then generate a confusion matrix from the model. Interpret the confusion matrix and explain what each number means.

# In[133]:


# Make predictions using the model and the testing data
y_pred = lr.predict(X_test)

print(f'Model Accuracy: ', accuracy_score(y_test, y_pred))


# In[118]:


# Make a confusion matrix
pd.DataFrame(confusion_matrix(y_test, y_pred))


# - Cell 1 (111) represents True Negative values ( Non-users of Linkedin were marked as non-users)
# - Cell 2 (57) represents False Positive values (Non-Users of LinkedIn platform were marked as Users)
# - Cell 3 (21) represents False Negative values (Users of Linkedin were labelled as non-users)
# - Cell 4 (63) represents True Positive values (The users of LinkedIn platform were labelled correctly)

# ***

# #### Q8 Create the confusion matrix as a dataframe and add informative column names and index names that indicate what each quadrant represents

# In[134]:


# Compare those predictions to the actual test data using a confusion matrix (positive class=1)

#confusion_matrix(y_test, y_pred)

df_conf_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred),
            columns=["Predicted negative", "Predicted positive"],
            index=["Actual negative","Actual positive"]).style.background_gradient(cmap="PiYG")

df_conf_matrix


# ***

# #### Q9 Aside from accuracy, there are three other metrics used to evaluate model performance: precision, recall, and F1 score. Use the results in the confusion matrix to calculate each of these metrics by hand. Discuss each metric and give an actual example of when it might be the preferred metric of evaluation. After calculating the metrics by hand, create a classification_report using sklearn and check to ensure your metrics match those of the classification_report.

# + **Accuracy**
# + Other evaluation metrics:
#     + **Recall**: Recall is calculated as $\frac{TP}{(TP+FN)}$ and is important when the goal is to minimze the chance of missing positive cases. E.g. fraud
#     + **Precision**: calculated as $\frac{TP}{(TP+FP)}$ and is important when the goal is to minimize incorrectly predicting positive cases. E.g. cancer screening
#     + **F1 score**: F1 score is the weighted average of recall and precision calculated as $2\times\frac{(precision x recall)}{(precision+recall)}$

# In[135]:


## recall: TP/(TP+FN)
recall = 63/(63+21)

## precision: TP/(TP+FP)
precision = 63/(63+57)

f1_score = 2*((precision*recall) / (precision+recall))

print(f'Recall: ', recall, "Precision: ", precision, "F1-score: ", f1_score)


# In[136]:


# Get other metrics with classification_report
print(classification_report(y_test, y_pred))


# ***

# #### Q10 Use the model to make predictions. For instance, what is the probability that a high income (e.g. income=8), with a high level of education (e.g. 7), non-parent who is married female and 42 years old uses LinkedIn? How does the probability change if another person is 82 years old, but otherwise the same?

# In[146]:


# New data for features: age, college, high_income, ideology
user1 = [8, 7, 2, 1,2, 42]

# Predict class, given input features
predicted_class = lr.predict([user1])

# # Generate probability of positive class (=1)
probs = lr.predict_proba([user1])


# In[143]:


# Print predicted class and probability
print(f"Predicted class: {predicted_class[0]}") # 0=not pro-environment, 1=pro-envronment
print(f"Probability that this person uses LinkedIn: {probs[0][1]}")


# In[148]:


# New data for features: age, college, high_income, ideology
user2 = [8, 7, 2, 1,2, 84]

# Predict class, given input features
predicted_class2 = lr.predict([user2])

# # Generate probability of positive class (=1)
probs2 = lr.predict_proba([user2])


# In[149]:


# Print predicted class and probability
print(f"Predicted class: {predicted_class2[0]}") # 0=not pro-environment, 1=pro-envronment
print(f"Probability that this person uses LinkedIn: {probs2[0][1]}")

