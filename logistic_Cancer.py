#Importing The Required Library
import pandas as pd
import streamlit as st 
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

#Title of the Model
st.title('Model: Logistic Regression ~ Cancer')

#Sidebar Title
st.sidebar.header('User Input Parameters')

#User_Defined Function For Input Parameter
def user_input_features():
    RADIUS_MEAN = st.sidebar.number_input('Insert the Radius_mean')
    TEXTURE_MEAN = st.sidebar.number_input('Insert the Texture_mean')
    PERIMETER_MEAN = st.sidebar.number_input('Insert the Perimeter_mean')
    AREA_MEAN = st.sidebar.number_input('Insert the Area_mean')
    SMOOTHNESS_MEAN = st.sidebar.number_input('Insert the Smoothness_mean')
    COMPACTNESS_MEAN = st.sidebar.number_input('Insert the Compactness_mean')
    CONCAVITY_MEAN = st.sidebar.number_input('Insert the Concavity_mean')
    CONCAVE_POINTS_MEAN = st.sidebar.number_input("Insert the Concave points_mean")
    SYMMETRY_MEAN = st.sidebar.number_input("Insert the Symmetry_mean")
    FRACTAL_DIMENSION_MEAN = st.sidebar.number_input("Insert the Fractal_dimension_mean")
    RADIUS_SE = st.sidebar.number_input("Insert the Radius_se")
    TEXTURE_SE = st.sidebar.number_input("Insert the Texture_se")
    PERIMETER_SE = st.sidebar.number_input("Insert the Perimeter_se")
    AREA_SE = st.sidebar.number_input("Insert the Area_se")
    SMOOTHNESS_SE = st.sidebar.number_input("Insert the Smoothness_se")
    COMPACTNESS_SE = st.sidebar.number_input("Insert the Compactness_se")
    CONCAVITY_SE = st.sidebar.number_input("Insert the Concavity_se")
    CONCAVE_POINTS_SE = st.sidebar.number_input("Insert the Concave points_se")
    SYMMETRY_SE = st.sidebar.number_input("Insert the Symmetry_se")
    FRACTAL_DIMENSION_SE = st.sidebar.number_input("Insert the Fractal_dimension_se")
    RADIUS_WORST = st.sidebar.number_input("Insert the Radius_worst")
    TEXTURE_WORST = st.sidebar.number_input("Insert the Texture_worst")
    PERIMETER_WORST = st.sidebar.number_input("Insert the Perimeter_worst")
    AREA_WORST = st.sidebar.number_input("Insert the Area_worst")
    SMOOTHNESS_WORST = st.sidebar.number_input("Insert the Smoothness_worst")
    COMPACTNESS_WORST = st.sidebar.number_input("Insert the Compactness_worst")
    CONCAVITY_WORST = st.sidebar.number_input("Insert the Concavity_worst")
    CONCAVE_POINTS_WORST = st.sidebar.number_input("Insert the Concave points_worst")
    SYMMETRY_WORST = st.sidebar.number_input("Insert the Symmetry_worst")
    FRACTAL_DIMENSION_WORST = st.sidebar.number_input("Insert the Fractal_dimension_worst")

    data = { 'radius_mean':RADIUS_MEAN,
             'texture_mean':TEXTURE_MEAN,
             'perimeter_mean':PERIMETER_MEAN,
             'area_mean':AREA_MEAN,
             'smoothness_mean':SMOOTHNESS_MEAN,
             'compactness_mean':COMPACTNESS_MEAN,
             'concavity_mean':CONCAVITY_MEAN,
             'concave points_mean':CONCAVE_POINTS_MEAN,
             'symmetry_mean':SYMMETRY_MEAN,
             'fractal_dimension_mean':FRACTAL_DIMENSION_MEAN,
             'radius_se':RADIUS_SE,
             'texture_se':TEXTURE_SE,
             'perimeter_se':PERIMETER_SE,
             'area_se':AREA_SE,
             'smoothness_se':SMOOTHNESS_SE,
             'compactness_se':COMPACTNESS_SE,
             'concavity_se':CONCAVITY_SE,
             'concave points_se':CONCAVE_POINTS_SE,
             'symmetry_se':SYMMETRY_SE,
             'fractal_dimension_se':FRACTAL_DIMENSION_SE,
             'radius_worst':RADIUS_WORST,
             'texture_worst':TEXTURE_WORST,
             'perimeter_worst':PERIMETER_WORST,
             'area_worst':AREA_WORST,
             'smoothness_worst':SMOOTHNESS_WORST,
             'compactness_worst':COMPACTNESS_WORST,
             'concavity_worst':CONCAVITY_WORST,
             'concave points_worst':CONCAVE_POINTS_WORST,
             'symmetry_worst':SYMMETRY_WORST,
             'fractal_dimension_worst':FRACTAL_DIMENSION_WORST}
    features = pd.DataFrame(data,index = [0])
    return features 
#Saving the input Parameter into Dataframes
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

#Loading The Dataset
cancer = pd.read_csv("cancer.csv")

#Complete Cancer dataset - applying lable encoder to species column
label_encoder = preprocessing.LabelEncoder()
cancer['diagnosis'] = label_encoder.fit_transform(cancer['diagnosis']) 


# Dividing our data into input and output variables 
X = cancer.iloc[:,1:]
Y = cancer.iloc[:,0]

#Logistic regression and fit the model
clf = LogisticRegression()
clf.fit(X,Y)

#Predicting the Users Input which are Stored in df
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Predicted Result')
st.write('B - No Diabetes' if prediction_proba[0][1] < 0.5 else ' M - Yes Diabetes')

st.subheader('Prediction Probability')
st.write(prediction_proba)



