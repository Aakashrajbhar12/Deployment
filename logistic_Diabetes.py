#Importing The Required Library
import pandas as pd
import streamlit as st 
from sklearn.linear_model import LogisticRegression

#Title of the Model
st.title('Model Deployment: Logistic Regression ~ Diabetes')

#Sidebar Title
st.sidebar.header('User Input Parameters')

#User_Defined Function For Input Parameter
def user_input_features():
    PREG = st.sidebar.number_input('Insert the Preg')
    PLAS = st.sidebar.number_input('Insert the Plasma')
    PRES = st.sidebar.number_input('Insert the Pres')
    SKIN = st.sidebar.number_input('Insert the Skin')
    TEST = st.sidebar.number_input('Insert the Test')
    MASS = st.sidebar.number_input('Insert the Mass')
    PEDI = st.sidebar.number_input('Insert the Pedi')
    AGE = st.sidebar.number_input("Insert the Age")
    data = {'preg':PREG,
            'plas':PLAS,
            'pres':PRES,
            'skin':SKIN,
            'test':TEST,
            'mass':MASS,
            'pedi':PEDI,
            'age':AGE}
    features = pd.DataFrame(data,index = [0])
    return features 
#Saving the input Parameter into Dataframes
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

#Loading The Dataset
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
diabetes = pd.read_csv("pima-indians-diabetes_data.csv",names = names)

# Dividing our data into input and output variables 
X = diabetes.iloc[:,0:8]
Y = diabetes.iloc[:,8]

#Logistic regression and fit the model
clf = LogisticRegression(max_iter = 400)
clf.fit(X,Y)

#Predicting the Users Input which are Stored in df
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Predicted Result')
st.write('No Diabetes' if prediction_proba[0][1] < 0.5 else 'Yes Diabetes')

st.subheader('Prediction Probability')
st.write(prediction_proba)