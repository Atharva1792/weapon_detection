import sys
import numpy as np 
import matplotlib.pyplot as plt

from src.components.data_ingestion import data_ingestion
from src.logger import logging
from src.exception import CustomException
import keras
try:
    logging.info("Prediction")

    model = keras.models.load_model("model.keras")
    class_names = ['Gun','Knife']
    for i in range(20):
        img_path = f'C:\\weapon_detection\\test_data\\test{i}.jpg'
        img = keras.preprocessing.image.load_img(img_path,target_size=(180,180))
        plt.imshow(img)
        
        input_arr = keras.preprocessing.image.img_to_array(img)
        input_arr = np.expand_dims(input_arr,axis=0)

        pred = model.predict(input_arr)
        prob = max(pred[0])
        pr = np.argmax(pred,axis=-1)
        print(f"Predicted Class : {class_names[pr[0]]} \tProbability : {prob}")
        
        plt.show()
except Exception as e:
    raise CustomException(e,sys)