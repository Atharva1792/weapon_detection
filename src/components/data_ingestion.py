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
                validation_split=0.2,
                subset='training',
                seed=123,
                batch_size=32,
                image_size=(180,180)
            )

            self.valid_data = keras.utils.image_dataset_from_directory(
                directory='train_data/',
                labels='inferred',
                label_mode='categorical',
                validation_split=0.2,
                subset='validation',
                seed=123,
                class_names= ['guns','knife'],
                batch_size=32,
                image_size=(180,180)
            )
            
            for image_batch, labels_batch in self.train_data:
                print("Shape of image and label:")
                print(image_batch.shape)
                print(labels_batch.shape)

                break

            return (self.train_data , self.valid_data)    
        
        except Exception as e:
            raise CustomException(e,sys)
        
        
        
if __name__ == "__main__":
    obj = data_ingestion()
    train_data,valid_data = obj.get_data()

    #data_transform = DataTransformation()
    #train,valid = data_transform.data_transform(train_data=train_data,valid_data=valid_data)
    
    model_trainer = ModelTrainer()
    model_trainer.model_trainer(train_data=train_data,valid_data=valid_data)
    