import sys
import numpy as np 
from PIL import Image

from src.logger import logging
from src.exception import CustomException
import keras
try:
    logging.info("Prediction")

    model = keras.models.load_model("model.keras")
    #img = keras.utils.load_img('C:\\weapon_detection\\test5.jpg')
    img_path = 'C:\\weapon_detection\\knife_374.jpg'
    img = keras.preprocessing.image.load_img(img_path,target_size=(256,256))
    #img = keras.layers.Resizing(256,256)
    #img = keras.layers.Rescaling(1./255)
    #print(img.shape)
    input_arr = keras.preprocessing.image.img_to_array(img)
    input_arr = np.array([input_arr])
    input_arr = input_arr.astype('float32') / 255.
    
    #print(model.evaluate())
    pred = model.predict(input_arr)
    print(pred)
    pr = np.argmax(pred,axis=-1)
    print(pr)
except Exception as e:
    raise CustomException(e,sys)