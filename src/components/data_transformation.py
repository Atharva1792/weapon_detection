import sys
import os
import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from src.exception import CustomException
from src.logger import logging

class DataTransformation:
    
    def data_transform(self,train_data,valid_data):
        try:
            logging.info("Preprocessing started")
            self.train_data = keras.layers.Resizing(180,180)
            self.valid_data = keras.layers.Resizing(180,180)
            self.train_data = keras.layers.Rescaling(scale=1./255,offset=0.0)
            self.valid_data = keras.layers.Rescaling(scale=1./255,offset=0.0)
            
            logging.info("Preprocessing done")
            print(train_data)
            print(valid_data)
            return (self.train_data , self.valid_data)

        except Exception as e:
            raise CustomException(e,sys)