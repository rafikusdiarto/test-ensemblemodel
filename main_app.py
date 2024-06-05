# Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import tensorflow as tf

# Loading the Model
model = load_model('ensemble_model.pkl')
                    
# Name of Classes
CLASS_NAMES = ('Corn_Blight', 'Corn_Common_Rust', 'Corn_Gray_Spot', 'Healthy')

# Setting Title of App
st.title("Corn Leaf Disease Detection Using Ensemble CNN")
st.markdown("Upload an image of the corn leaf")

# Uploading the dog image
plant_image = st.file_uploader("Choose an image...", type = "jpg")
submit = st.button('predict Disease')

# On predict button click
if submit:
    if plant_image is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray (bytearray(plant_image.read()), dtype = np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        
        # Displaying the image
        st.image(opencv_image, channels="BGR")
        st.write(opencv_image.shape)
        
        # Resizing the image
        opencv_image = cv2.resize(opencv_image, (256, 256))
        
        # Convert image to 4 Dimension
        opencv_image.shape = (1, 256, 256, 3)
        
        #Make Prediction
        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]
        st.title(str("This is "+result.split('-')[0]+ "corn leaf with " +  result.split('-')[1]))