# -*- coding: utf-8 -*-
"""
 @Author  : yangzh
 @Email   : 1725457378@qq.com
 @File    : train_gen.py
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Sequential
# from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint,EarlyStopping
from OurModel.ConvLSTM import ConvLSTM2D
from data_detail import getdata
from keras.optimizers import Adam


def model(input_size, Len):
    input_shape = (Len, input_size, input_size, 3)
    model = Sequential( )
    model.add(ConvLSTM2D(32, kernel_size = (3, 3), activation = 'sigmoid', padding = 'same', input_shape = input_shape,
                         return_sequences = True))
    model.add(ConvLSTM2D(32, kernel_size = (3, 3), activation = 'sigmoid', padding = 'same'), )
    model.add(GlobalAveragePooling2D( ))
    model.add(Dense(5, activation = 'softmax'))
    model.summary( )
    optimizer = Adam(lr = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return model


def main():
    train_path = "./dataset/csv/train_csv/"
    test_path = "./dataset/csv/test_csv/"
    Len = 2
    batch_size = 4
    input_size = 70
    epochs = 250
    classes = 5

    train_sequence, class_weight_t = getdata(train_path, batch_size = batch_size, input_size = input_size, Len = Len,
                                             classes = classes)
    validation_sequence, class_weight_v = getdata(test_path, batch_size = batch_size, input_size = input_size,
                                                  Len = Len, classes = classes)

    o_model = model(input_size, Len)

    m_p = 'O_ConvLSTM2D_1_len(3)_bs(4)_is(50)_exp_lr/'

    log_path = './log_file/' + m_p
    modelpath = './model_snapshots/' + m_p
    os.makedirs(modelpath)

    modelpath = './model_snapshots/' + m_p + 'weights_best.h5'
    tensorBoard = TensorBoard(log_dir = log_path)
    checkpoint = ModelCheckpoint(modelpath, monitor = 'val_acc', verbose = 1, save_best_only = True, mode = 'max',
                                 period = 2)
    earlystopping = EarlyStopping(monitor='val_loss', patience=30, verbose=0)


    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 10, mode = 'auto')
    o_model.evaluate_generator(train_sequence, steps_per_epoch = train_sequence.__len__( ), epochs = epochs,
                          validation_data = validation_sequence, callbacks = [tensorBoard, reduce_lr, checkpoint,earlystopping],
                          verbose = 1)


if __name__ == '__main__':
    main( )
