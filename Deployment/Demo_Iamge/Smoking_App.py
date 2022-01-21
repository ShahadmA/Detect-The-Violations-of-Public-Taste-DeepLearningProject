import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import cv2

import keras
from keras.models import load_model
from scipy.spatial import distance
import scipy


from skimage.transform import resize


st.set_page_config(page_title='Smoking Detector', layout='wide', initial_sidebar_state='expanded')

model = load_model("smoking_modelMN200.h5")





st.title("Smoking Detector 🚬")


def predict(image_data, Model):

        size = (224, 224)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(224, 224),    interpolation=cv2.INTER_CUBIC))/255.

        img_reshape = img_resize[np.newaxis,...]

        prediction = model.predict(img_reshape)

        return prediction



uploaded_file = st.file_uploader(" Upload here ", type=['jpg','png', 'jpeg'])


if uploaded_file is not None:

    u_img = Image.open(uploaded_file)
    st.image(u_img, 'Uploaded Image', use_column_width=True)
 
    prediction = predict(u_img, model)

    if np.argmax(prediction) == 0:
       
        m1 = '<p style="color:Green; font-size: 30px;"> Not smoking! </p>'
        st.markdown(m1, unsafe_allow_html=True)
    
    else:
        m3 = '<p style="color:Red; font-size: 30px;"> Smoking </p>'
        st.markdown(m3, unsafe_allow_html=True)


   
    prob = '<p style=" color:Black; font-size: 20px;">Probability </p>'  
    st.markdown(prob, unsafe_allow_html=True)


    if np.argmax(prediction) == 0:
        st.write(("not smoking:  "),np.round(prediction[0][0])*100)

     
    else:
        st.write((" smoking :   "),np.round(prediction[0][1]*100))

       

