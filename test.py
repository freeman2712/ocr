import scipy.io
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import cv2
import fnmatch
import string
import os
from model.model import CRNN

training_img = []
training_txt = []
train_input_length = []
train_label_length = []
orig_txt = []

s = 0
maxLen = 0

test_img = []
test_txt = []
test_input_length = []
test_label_length = []
test_orig_txt = []
d_test = dict()

d = dict()

mat = scipy.io.loadmat('dataset/IIIT5K/trainCharBound.mat')
char_list = string.ascii_letters+string.digits


for i in mat['trainCharBound'][0]:
        d[i[0][0]] = i[1][0]

print(d['train/644_2.png'])
print(len(d['train/644_2.png']))

def encode_to_labels(text):
        encoded_list = []
        for i, char in enumerate(text):
                try:
                        encoded_list.append(char_list.index(char))
                except:
                        print(char)
        return encoded_list




for root, dirs, files in os.walk("./dataset/IIIT5K/train"):
        # print(root, files)
        for f_name in fnmatch.filter(files, '*.png'):
                img = cv2.imread(os.path.join(root, f_name))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                aspect_ratio = img.shape[1]/img.shape[0]
                img = cv2.resize(img, (int(aspect_ratio*32), 32), interpolation=cv2.INTER_AREA)
                if(img.shape[1] > 128):
                        continue
                if(img.shape[1] < 128):
                        add_zeros = np.ones((32, 128 - img.shape[1]))*255
                        img = np.concatenate((img, add_zeros), axis = 1)
                img = np.expand_dims(img, axis=2)
                img = img/255.0
                dict_index = "train/" + f_name
                # print(dict_index)
                s = s + 1
                orig_txt.append(d[dict_index])
                training_img.append(img)
                txt = (d[dict_index])
                txt = str(txt)
                if(len(txt) > maxLen):
                        maxLen = len(txt)
                train_label_length.append(len(txt))
                train_input_length.append(31)
                training_txt.append(encode_to_labels(txt))
# img = cv2.resize(img, )



mat_test = scipy.io.loadmat('dataset/IIIT5K/testCharBound.mat')
for i in mat_test['testCharBound'][0]:
        d_test[i[0][0]] = i[1][0]


for root, dirs, files in os.walk("./dataset/IIIT5K/test"):
        # print(root, files)
        for f_name in fnmatch.filter(files, '*.png'):
                img = cv2.imread(os.path.join(root, f_name))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                aspect_ratio = img.shape[1]/img.shape[0]
                img = cv2.resize(img, (int(aspect_ratio*32), 32), interpolation=cv2.INTER_AREA)
                if(img.shape[1] > 128):
                        continue
                if(img.shape[1] < 128):
                        add_zeros = np.ones((32, 128 - img.shape[1]))*255
                        img = np.concatenate((img, add_zeros), axis = 1)
                img = np.expand_dims(img, axis=2)
                img = img/255.0
                dict_index = "test/" + f_name
                # print(dict_index)

                test_orig_txt.append(d_test[dict_index])
                test_img.append(img)
                txt = (d_test[dict_index])
                txt = str(txt)
                if(len(txt) > maxLen):
                        maxLen = len(txt)
                test_label_length.append(len(txt))
                test_input_length.append(31)
                test_txt.append(encode_to_labels(txt))

# print(test_img[10])

print(s)
print(training_img[10].shape)
print(training_txt[10])
print(orig_txt[10])
print(train_label_length[10])
print(train_input_length[10])
# CRNN([training_img, train_input_length, train_label_length, test_img, test_input_length, test_label_length, training_txt, test_txt, maxLen])

valid_model = CRNN([training_img, train_input_length, train_label_length, test_img, test_input_length, test_label_length, training_txt, test_txt, maxLen], True)
valid_model.load_weights('crnn_model.h5')

training_img = np.array(training_img)#, dtype=np.float)
train_input_length = np.array(train_input_length)# dtype=np.int)
train_label_length = np.array(train_label_length)# dtype=np.int)

test_img = np.array(test_img)# dtype=np.float)
# test_input_length = np.array(test_input_length)# dtype=np.int)
# test_label_length = np.array(test_label_length)# dtype=np.int)

prediction = valid_model.predict(test_img[:10])
out = keras.backend.get_value(keras.backend.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                         greedy=True)[0][0])


print(prediction[0][2])
i = 0
correct = 0
incorrect = 0
total = 0
for x in out:
        print(test_orig_txt[i], end = ' ')
        temp = ""
        for p in x:
                if int(p)!=-1:
                        temp = temp + char_list[int(p)]
        if(temp == test_orig_txt[i]):
                correct += 1
        else:
                incorrect += 1
        total += 1
        print(temp)
        i = i + 1


print("Total {}".format(total))
print("Correct {}".format(correct))
print("Incorrect {}".format(incorrect))




