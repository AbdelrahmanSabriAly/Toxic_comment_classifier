import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.layers import TextVectorization

st.set_page_config(page_title="Toxic Comments Classifier",layout='wide')

hide_st_style = """
<style>
MainMenu {visibility: hidden;}
footer{visibility: hidden;}
</style>
"""
st.markdown(hide_st_style,unsafe_allow_html=True)


model = tf.keras.models.load_model('model.h5')
labels = ['Toxic','Severe Toxic','Obsence','Threat','Insult','Identity hate']
from_desk = pickle.load(open("v_layer.pkl","rb"))

vectorizer = TextVectorization.from_config(from_desk['config'])
vectorizer.set_weights(from_desk['weights'])


st.title("Toxic Comment Classifier")

about,app,contact = st.tabs(['About','App','Contact'])

#  ------------------------ About --------------------------- #
about.subheader("Toxic comments are highly dangerous and have significant negative effects.")
about.subheader("They harm individuals, create toxic online environment, and hinder meaningful discussions")
about.write("In this project, a recurrent neural netork is trained on Jigsaw dataset. You can find the dataset in the following link")
about.markdown('[Jigsaw dataset](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge)')
about.write("You can find the link of the project in the following GitHub repository")
about.markdown('[GitHub Repo](https://github.com/AbdelrahmanSabriAly/ToxicCommentClassifier.git)')
#  ------------------------  App  --------------------------- #
app.subheader("Enter a sentence to examine")
text = app.text_input("i.e. : I hate you")
if text:
    input_text = vectorizer(text)
    predictions = model.predict(np.expand_dims(input_text,0))
    results = {}
    for i in range(6):
        results[labels[i]] = predictions[0][i]

    updated_data = {}

    for key, value in results.items():
        if float(value) > 0.5:
            updated_data[key] = True
        else:
            updated_data[key] = False
    app.dataframe(updated_data)

#  ----------------------- Contact --------------------------- #
contact.subheader("Abdelrahman Sabri Aly")
contact.write("Email: aaly6995@gmail.com")
contact.write("Phone: +201010681318")
contact.markdown("[WhatsApp:]( https://wa.me/+201010681318)")
contact.markdown("[Linkedin](https://www.linkedin.com/in/abdelrahman-sabri)")
