import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Iris Tahmin Uygulaması

Girdiğiniz değerlere göre çiceğin türünü tahmin eden bir uygulama!
""")

st.sidebar.header('Parametreleri Girin')

def user_input_features():
    sepal_length = st.sidebar.slider('Çanak yaprak genişliği', 4.3, 7.9, 5.4)
    st.sidebar.write("---")
    sepal_width = st.sidebar.slider('Çanak yaprak uzunluğu', 2.0, 4.4, 3.4)
    st.sidebar.write("---")
    petal_length = st.sidebar.slider('Taç yaprak genişliği', 1.0, 6.9, 1.3)
    st.sidebar.write("---")
    petal_width = st.sidebar.slider('Taç yaprak uzunluğu', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('Girdiğiniz Değerler')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Sınıflar')
st.write(iris.target_names)

st.subheader('Tahmin : '+iris.target_names[prediction][0])

st.subheader('Tahmin Olasılığı')
st.write(prediction_proba)