import os
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
            model.add(Conv2D(32, (3,3), activation = 'relu', input_shape = (256 , 256 , 3)))
            model.add(MaxPooling2D((2,2,)))
            
            model.add(Conv2D(32, (3,3), activation = 'relu'))
            model.add(MaxPooling2D((2,2)))

            model.add(Flatten())
            model.add(Dense(64,activation = 'relu'))
            model.add(Dense(2, activation = 'sigmoid'))

            model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
            model.fit(train_data,epochs=5,batch_size=64)
            
            #print(model.evaluate(valid_data))
            
            logging.info("Model Training Complete")
        except Exception as e:
            raise CustomException(e,sys)