import sys
import numpy as np 
from PIL import Image

from src.logger import logging
from src.exception import CustomException
import keras
try:
    logging.info("Prediction")

    model = keras.models.load_model("model.keras")

    img_path = 'C:\\weapon_detection\\test2.jpg'
    img = keras.preprocessing.image.load_img(img_path,target_size=(100,100))

    input_arr = keras.preprocessing.image.img_to_array(img)
    input_arr = np.expand_dims(input_arr,axis=0)

    pred = model.predict(input_arr)
    pr = np.argmax(pred,axis=-1)
    print(pr)
except Exception as e:
    raise CustomException(e,sys)