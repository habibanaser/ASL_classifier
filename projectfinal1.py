import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import cv2
from PIL import Image, ImageOps
import numpy as np
from skimage.transform import resize

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('D:\mnetclassifier.h5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Image Classification
         """
         )

file = st.file_uploader("Upload the image to be classified", type=["jpg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)

def upload_predict(upload_image, model):

        size = (64,64)
        image = np.asarray(upload_image)/255
        img_resize = cv2.resize(image, dsize=(64,64),interpolation=cv2.INTER_CUBIC)

        img_reshape = img_resize[np.newaxis,...]
        names=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
        prediction = model.predict(img_reshape)
        prediction = np.argmax(prediction, axis=-1)
        pred_class=names[int(prediction)]

        return pred_class
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    image_class = upload_predict(image, model)
    st.write("The image is classified as",image_class)
    print("The image is classified as ",image_class)

