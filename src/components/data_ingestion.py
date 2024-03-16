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
                batch_size=32
            )

            self.valid_data = keras.utils.image_dataset_from_directory(
                directory='test_data/',
                labels='inferred',
                label_mode='categorical',
                class_names= ['guns','knife'],
                batch_size=32
            )
            class_names =  self.train_data.class_names
            print(class_names)
            plt.figure(figsize=(10, 10))
            for images, labels in self.train_data.take(1):
                for i in range(1):
                    ax = plt.subplot(3, 3, i + 1)
                    plt.imshow(images[i].numpy().astype("uint8"))
                    #plt.title(class_names[labels[i]])
                    plt.axis("off")
            #data_dir = '/train_data'
            return (self.train_data , self.valid_data)    
        
        except Exception as e:
            raise CustomException(e,sys)
        
        
        
if __name__ == "__main__":
    obj = data_ingestion()
    train_data,valid_data = obj.get_data()
    
    #data_transform = DataTransformation()
    #train,valid = data_transform.data_transform(train_data=train_data,valid_data=valid_data)
    
    #model_trainer = ModelTrainer()
    #print(model_trainer.model_trainer(train_data=train,valid_data=valid))
    