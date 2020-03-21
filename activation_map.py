# -*- coding: utf-8 -*-
"""
 @Author  : yangzh
 @Email   : 1725457378@qq.com
 @File    : activation_map.py
"""
import matplotlib.pyplot as plt
import seaborn as sns
from keras import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Sequential
from keras.optimizers import Adam

# from keras.layers.convolutional_recurrent import ConvLSTM2D
from OurModel.ConvLSTM import ConvLSTM2D
from data_detail import getdata


def ConvLSTM2D_model(input_size, Len):
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


def get_map(input_size, Len):
    train_sequence = getdata(path = '/code/dataset/test_mp/Cyclone/',
                             batch_size = 1, input_size = input_size, Len = Len, classes = 5)

    model = ConvLSTM2D_model(input_size, Len)
    model.load_weights(filepath = '/code/model/weights-improvement-140-0.9015.hdf5')
    gmp_layer_model = Model(inputs = model.input, outputs = model.get_layer('conv_lst_m2d_2').output)
    preds = gmp_layer_model.predict_generator(train_sequence)[0][0]

    img = train_sequence.__getitem__(0)
    ax = sns.heatmap(preds, cmap = plt.cm.Blues)

    # plt.imshow(img[0][0][0,:])
    # plt.imshow(img[0][0][1,:])
    plt.axis('off')
    plt.savefig("/code/dataset/test_mp/Cyclone/img/Floods_Conv.png")
    plt.show( )


if __name__ == '__main__':
    input_size = 70
    Len = 2

    get_map(input_size, Len)
