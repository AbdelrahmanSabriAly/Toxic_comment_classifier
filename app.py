import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.layers import TextVectorization

# Set Streamlit page configuration
st.set_page_config(page_title="Toxic Comments Classifier", layout='wide')

# Hide Streamlit style elements
hide_st_style = """
<style>
MainMenu {visibility: hidden;}
footer{visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Define the labels for classification
labels = ['Toxic', 'Severe Toxic', 'Obscene', 'Threat', 'Insult', 'Identity hate']

# Load the vectorizer configuration from pickle file
from_desk = pickle.load(open("v_layer.pkl", "rb"))

# Create a text vectorizer using the loaded configuration
vectorizer = TextVectorization.from_config(from_desk['config'])
vectorizer.set_weights(from_desk['weights'])

# Set up the Streamlit app layout with tabs
st.title("Toxic Comment Classifier")
about, app, contact = st.tabs(['About', 'App', 'Contact'])

# ------------------------ About --------------------------- #
about.subheader("Toxic comments are highly dangerous and have significant negative effects.")
about.subheader("They harm individuals, create toxic online environment, and hinder meaningful discussions")
about.write("In this project, a recurrent neural network is trained on the Jigsaw dataset.")
about.markdown('[Jigsaw dataset](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge)')
about.write("You can find the link to the project in the following GitHub repository")
about.markdown('[GitHub Repo](https://github.com/AbdelrahmanSabriAly/ToxicCommentClassifier.git)')

# ------------------------  App  --------------------------- #
app.subheader("Enter a sentence to examine")
text = app.text_input("i.e. : I hate you")
if text:
    # Vectorize the input text using the text vectorizer
    input_text = vectorizer(text)
    
    # Make predictions using the trained model
    predictions = model.predict(np.expand_dims(input_text, 0))
    
    results = {}
    
    # Map the prediction probabilities to the corresponding labels
    for i in range(6):
        results[labels[i]] = predictions[0][i]
    
    updated_data = {}
    
    # Threshold the prediction probabilities to determine the presence of toxicity
    for key, value in results.items():
        if float(value) > 0.5:
            updated_data[key] = True
        else:
            updated_data[key] = False
    
    # Display the results in a dataframe
    app.dataframe(updated_data)

# ----------------------- Contact --------------------------- #
contact.subheader("Abdelrahman Sabri Aly")
contact.write("Email: aaly6995@gmail.com")
contact.write("Phone: +201010681318")
contact.markdown("[WhatsApp:](https://wa.me/+201010681318)")
contact.markdown("[LinkedIn](https://www.linkedin.com/in/abdelrahman-sabri)")
