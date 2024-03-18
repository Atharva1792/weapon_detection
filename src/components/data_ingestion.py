import tensorflow
import keras
import sys
import matplotlib.pyplot as plt
import numpy as np

from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class data_ingestion:

    def get_data(self):

        try:
            logging.info("Getting dataset from the image directory")
            self.train_data = keras.utils.image_dataset_from_directory(
                directory='train_data/',
                labels='inferred',
                label_mode='categorical',
                class_names= ['guns','knife'],
                batch_size=32,
                image_size=(100,100)
            )

            self.valid_data = keras.utils.image_dataset_from_directory(
                directory='valid_data/',
                labels='inferred',
                label_mode='categorical',
                class_names= ['guns','knife'],
                batch_size=32,
                image_size=(100,100)
            )
            class_names =  self.train_data.class_names
            print(class_names)
            for image_batch, labels_batch in self.train_data:
                print(image_batch.shape)
                print(labels_batch.shape)

                break
            #self.train_data = self.train_data / 255.0
            #self.valid_data = self.valid_data / 255.0
            #print(self.train_data[0])
            return (self.train_data , self.valid_data)    
        
        except Exception as e:
            raise CustomException(e,sys)
        
        
        
if __name__ == "__main__":
    obj = data_ingestion()
    train_data,valid_data = obj.get_data()

    '''data_transform = DataTransformation()
    train,valid = data_transform.data_transform(train_data=train_data,valid_data=valid_data)
    class_name = train_data.class_names'''
    
    model_trainer = ModelTrainer()
    model_trainer.model_trainer(train_data=train_data,valid_data=valid_data)