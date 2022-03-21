import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import os




os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

char_list =    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

def CRNN(img_args, valid = False):
        input = keras.Input(shape=(32,128,1))
        conv_1 = keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(input)
        pool_1 = keras.layers.MaxPool2D(pool_size=(2,2), strides=2)(conv_1)
        conv_2 = keras.layers.Conv2D(filters=128, kernel_size=(3,3),activation='relu',padding='same')(pool_1)
        pool_2 = keras.layers.MaxPool2D(pool_size=(2,2), strides=2)(conv_2)
        conv_3 = keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same')(pool_2)
        conv_4 = keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same')(conv_3)
        pool_4 = keras.layers.MaxPool2D(pool_size=(2,1))(conv_4)
        conv_5 = keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same')(pool_4)
        batch_norm_5 = keras.layers.BatchNormalization()(conv_5)
        conv_6 = keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same')(batch_norm_5)
        batch_norm_6 = keras.layers.BatchNormalization()(conv_6)
        pool_6 = keras.layers.MaxPool2D(pool_size=(2,1))(batch_norm_6)
        conv_7 = keras.layers.Conv2D(filters=512, kernel_size=(2,2), activation='relu')(pool_6)

        map_to_sequence = keras.layers.Lambda(lambda x: keras.backend.squeeze(x, 1))(conv_7)
        blstm_1 = keras.layers.Bidirectional(keras.layers.LSTM(units=128, return_sequences=True, dropout = 0.2))(map_to_sequence)
        blstm_2 = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout = 0.2))(blstm_1)
        output = keras.layers.Dense(len(char_list) + 1, activation='softmax')(blstm_2)

        test_model = keras.Model(inputs=input, outputs=output, name="CRNN_Test_Model")


        if(valid == True):
                print(test_model.summary())
                return test_model


        training_img, train_input_length, train_label_length, valid_img, valid_input_length, valid_label_length, training_txt, valid_txt, maxLen = img_args
        # training_img = np.array(training_img, dtype=np.float)
        # train_input_length = np.array(train_input_length, dtype=np.int)
        # train_label_length = np.array(train_label_length, dtype=np.int)
        
        # valid_img = np.array(valid_img, dtype=np.float)
        # valid_input_length = np.array(valid_input_length, dtype=np.int)
        # valid_label_length = np.array(valid_label_length, dtype=np.int)
        training_img = np.array(training_img)#, dtype=np.float)
        train_input_length = np.array(train_input_length)# dtype=np.int)
        train_label_length = np.array(train_label_length)# dtype=np.int)

        valid_img = np.array(valid_img)# dtype=np.float)
        valid_input_length = np.array(valid_input_length)# dtype=np.int)
        valid_label_length = np.array(valid_label_length)# dtype=np.int)

        labels = keras.Input(name='the_labels', shape=[maxLen], dtype='float32')
        input_length = keras.Input(name='input_length', shape=[1], dtype='int64')
        label_length = keras.Input(name='label_length', shape=[1], dtype='int64')




        def ctcLambda(args):
                y_pred, labels, input_length, label_length = args
                return keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

        loss_out = keras.layers.Lambda(ctcLambda, output_shape=(1,), name='ctc')([output, labels, input_length, label_length])
        model = keras.Model(inputs=[input, labels, input_length, label_length], outputs=loss_out)
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer = 'adam')

        filepath="overnight.hdf5"
        checkpoint = keras.callbacks.ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        callbacks_list = [checkpoint]






        train_size = train_input_length.size
        test_size = valid_input_length.size
        print(train_size)


        train_padded_txt = keras.preprocessing.sequence.pad_sequences(training_txt, maxlen=maxLen, padding='post', value = len(char_list))
        valid_padded_txt = keras.preprocessing.sequence.pad_sequences(valid_txt, maxlen=maxLen, padding='post', value = len(char_list))
        model.fit(x=[training_img, train_padded_txt, train_input_length, train_label_length], y=np.zeros(train_size),  epochs = 100, validation_data = ([valid_img, valid_padded_txt, valid_input_length, valid_label_length], [np.zeros(test_size)]), verbose = 1, callbacks = callbacks_list)


        print(model.summary())



# //MaxLen args should always be sent last from test.py

