import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import json
with open("sugg.json", 'r') as f:
    sugg = json.loads(f.read())
model_to_predict = tf.keras.models.load_model('plant_disease_3.h5')
def predict_covid(test_image):
    img = cv2.imread(test_image)
    img = img / 255.0
    img = cv2.resize(img, (224, 224))
    img = img.reshape(1,224,224,3)
    prediction = model_to_predict.predict(img)
    pred_class = np.argmax(prediction, axis = -1)
    return pred_class

def load_image(image_file):
    img = Image.open(image_file)
    return img


st.write("Plant disease using CNN")



pic = st.file_uploader("Upload a picture!")
submit = st.button('submit')



if submit:
    pic_details = {"filename":pic.name, 'filetype':pic.type, 'filesize':pic.size}
    st.write(pic_details)

    st.image(load_image(pic), width=250)

    with open('test.jpg', 'wb') as f:
        f.write(pic.getbuffer())
    pred = predict_covid('test.jpg')
    if pred[0] in [3, 4, 6, 10, 14, 17, 19, 22, 27, 23, 24, 37]:
        sugg_ = sugg['Healthy']
        st.write(sugg_)
    elif pred[0] in [5, 25]:
        sugg_ = sugg['Powdery disease']
        st.write(sugg_)
    elif pred[0] in [2, 8]:
        sugg_ = sugg['Rust disease']
        st.write(sugg_)
    elif pred[0] in [35]:
        sugg_ = sugg['Yellow Curl Leaf']
        st.write(sugg_)
    elif pred[0] in [9, 13, 20, 21, 30]:
        sugg_ = sugg['Late Blight']
        st.write(sugg_)
    else:
        sugg_ = sugg['Black rot']
        st.write(sugg_)
  
