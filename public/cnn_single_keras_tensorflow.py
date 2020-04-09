#coding:utf-8


import numpy as np
np.random.seed(1337)  # for reproducibility
from public.keras_load_data import *
import tensorflow as tf
from keras.models import Sequential,Model
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras import backend as K
from keras.layers import Dropout
import shutil

import os
import math

def pictureSize_afterPool(picture_size,stride):
    return math.ceil(float(picture_size)/float(stride))

def getTrainSet(k):
    l_train_x=[]
    l_train_y=[]
    l=[]
    for i in range(k):
        xi=np.random.randint(0,len(train_x_all))
        l.append(xi)
        l_train_x.append(train_x_all[xi])
        l_train_y.append(train_y_all[xi])
    x_train=np.array(l_train_x)
    y_train=np.array(l_train_y)
    # x_train = l_train_x_array.reshape(-1, 1, 28, 28) / 255.
    # y_train = np_utils.to_categorical(l_train_y_array, num_classes=10)
    return x_train, y_train, l


def getValidationSet(l):
    l_index = list(set(range(len(train_x_all))) ^ set(l))
    # print(len(l_index))
    l_validation_x = []
    l_validation_y = []

    for i in range(len(l_index)):
        l_validation_x.append(train_x_all[l_index[i]])
        l_validation_y.append(train_y_all[l_index[i]])
    return np.array(l_validation_x),np.array(l_validation_y)


class cnn():
    def __init__(self, x,model_save_path, model_name):
        self.x=x

        self.conv1_kernel_size = x[0]
        self.conv1_kernel_numbers = x[1]
        self.pool1_size = x[2]
        self.pool1_stride = x[3]
        self.conv2_kernel_size = x[4]
        self.conv2_kernel_numbers = x[5]
        self.pool2_size = x[6]
        self.pool2_stride = x[7]
        self.fullconnection_numbers = x[8]
        self.learning_rate = x[9]
        self.dropout=x[10]

        self.pictureSize_afterPool2 = pictureSize_afterPool(pictureSize_afterPool(28,self.pool1_stride), self.pool2_stride)

        self.model_save_path = model_save_path
        self.model_name = model_name

        self.train_x, self.train_y, l = getTrainSet(55000)
        self.validation_x, self.validation_y = getValidationSet(l)
        # print(len(set(l)))
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        with open(os.path.join(self.model_save_path, "variables.txt"), "w")as f:
            f.write(str(self.x))

    def cnn_run(self, lun):
        K.clear_session()
        graph = tf.Graph()
        with graph.as_default():
            model = Sequential()
            # Conv layer 1 output shape (32, 28, 28)
            model.add(Convolution2D(
                batch_input_shape=(None, 1, 28, 28),
                filters=self.conv1_kernel_numbers,
                kernel_size=(self.conv1_kernel_size, self.conv1_kernel_size),
                strides=(1, 1),
                padding='same',     # Padding method
                data_format='channels_first'
            ))
            model.add(Activation('relu'))

            # Pooling layer 1 (max pooling) output shape (32, 14, 14)
            model.add(MaxPooling2D(
                pool_size=(self.pool1_size, self.pool1_size),
                strides=(self.pool1_stride, self.pool1_stride),
                padding='same',    # Padding method
                data_format='channels_first'
            ))

            # Conv layer 2 output shape (64, 14, 14)
            model.add(Convolution2D(filters=self.conv2_kernel_numbers, kernel_size=(self.conv2_kernel_size,self.conv2_kernel_size), strides=(1,1), padding='same', data_format='channels_first'))
            model.add(Activation('relu'))

            # Pooling layer 2 (max pooling) output shape (64, 7, 7)
            model.add(MaxPooling2D(pool_size=(self.pool2_size, self.pool2_size), strides=(self.pool2_stride,self.pool2_stride),padding= 'same', data_format='channels_first'))

            # Fully connected layer 1 input shape (64 * 7 * 7) = (3136), output shape (1024)
            model.add(Flatten())
            model.add(Dense(self.fullconnection_numbers))
            model.add(Activation('relu'))
            model.add(Dropout(self.dropout))

            # Fully connected layer 2 to shape (10) for 10 classes
            model.add(Dense(10,name="fc2"))
            model.add(Activation('softmax'))

            # Another way to define your optimizer
            adam = Adam(lr=self.learning_rate)

            # We add metrics to get more results you want to see
            model.compile(optimizer=adam,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            print('Training ------------')
            # Another way to train the model
            # 用单个模型自己的训练集去训练
            model.fit(self.train_x, self.train_y, epochs=lun, batch_size=100,validation_data=(self.validation_x, self.validation_y))

            # 用原始训练集获得该模型训练集上的准确率
            last_layer=Model(inputs=model.input,outputs=model.get_layer("fc2").output)
            y_prediction=last_layer.predict(validation_x_all)

            # loss,train_acc=model.evaluate(train_x_all,train_y_all)
            loss,acc = model.evaluate(validation_x_all, validation_y_all)
            # loss,test_acc = model.evaluate(test_x_all, test_y_all)
            with open(os.path.join(self.model_save_path,"train_acc.txt"),"w")as f:
                # f.write(str(train_acc)+","+str(acc)+","+str(test_acc))
                f.write(str(acc))

            # model.save(os.path.join(self.model_save_path, self.model_name))  # creates a HDF5 file 'my_model.h5'
            if (acc < 0.3 and lun == 1) or (lun == 10):
                model.save(os.path.join(self.model_save_path, self.model_name))  # creates a HDF5 file 'my_model.h5'
        tf.reset_default_graph()
        return acc, y_prediction


# print('\nTesting ------------')
# # Evaluate the model with the metrics we defined earlier
# loss, accuracy = model.evaluate(X_test, y_test)
#
# print('\ntest loss: ', loss)
# print('\ntest accuracy: ', accuracy)

if __name__ == "__main__":
    for i in range(10):
        model_save_path = os.path.join("E:\pc_model\small_single\\2018_5_14",str(i))
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        else:
            shutil.rmtree(model_save_path)
        model_name = 'model_' + str(i) + '.h5'
        x = [3, 16, 2, 2, 3, 64, 2, 2, 1000, 0.001, 0.5]
        c = cnn(x, model_save_path, model_name)
        print(c.cnn_run(10)[0])

