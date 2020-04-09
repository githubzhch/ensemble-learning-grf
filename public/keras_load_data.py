#coding:utf-8

# training X shape (60000, 28x28), Y shape (60000, ). test X shape (10000, 28x28), Y shape (10000, )
from keras.datasets import mnist
from keras.utils import np_utils

(X_train, Y_train),(X_test, y_test) = mnist.load_data()

test_x_all = X_test.reshape(-1, 1, 28, 28) / 255.
test_y_all = np_utils.to_categorical(y_test, num_classes=10)

train_validation_x_all = X_train.reshape(-1, 1, 28, 28) / 255.
train_validation_y_all = np_utils.to_categorical(Y_train, num_classes=10)


train_x_all=train_validation_x_all[:55000]
train_y_all=train_validation_y_all[:55000]

validation_x_all=train_validation_x_all[55000:]
validation_y_all=train_validation_y_all[55000:]



if __name__ == "__main__":
    print(train_x_all.shape)
    print(validation_x_all.shape)
    print(test_x_all.shape)
