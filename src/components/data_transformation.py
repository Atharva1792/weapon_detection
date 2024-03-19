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


            AUTOTUNE = tf.data.AUTOTUNE

            train_data = train_data.cache().prefetch(buffer_size=AUTOTUNE)
            valid_data = valid_data.cache().prefetch(buffer_size=AUTOTUNE)
            logging.info("Preprocessing done")
            #print(train_data)
            #print(valid_data)
            return (train_data , valid_data)

        except Exception as e:
            raise CustomException(e,sys)