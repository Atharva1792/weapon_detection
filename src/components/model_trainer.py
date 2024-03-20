import numpy as np
import sys
import keras
from keras.models import Sequential
from keras.layers import MaxPooling2D,Conv2D,Dense,Flatten

from src.exception import CustomException
from src.logger import logging

class ModelTrainer:
    def model_trainer(self,train_data,valid_data):
        try:
            logging.info("Training model started")
            model = Sequential()

            model.add(keras.layers.Rescaling((1./255.0),input_shape=(180,180,3)))
            model.add(Conv2D(32, (3,3), activation = 'relu'))
            model.add(MaxPooling2D((2,2,)))
            
            model.add(Conv2D(32, (3,3), activation = 'relu'))
            model.add(MaxPooling2D((2,2)))

            model.add(Flatten())
            model.add(Dense(64,activation = 'relu'))
            model.add(Dense(2,activation= 'sigmoid'))
            model.summary()
            model.compile(
                loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
                )
            model.fit(train_data,validation_data=valid_data,epochs=15,batch_size=64)
            
            _,acc = model.evaluate(valid_data)
            print("Accuracy of Model: ",acc)
            
            logging.info("Model Training Complete")
        
            model.save("model.keras")

        
        except Exception as e:
            raise CustomException(e,sys)