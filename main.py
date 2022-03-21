import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from model.model import CRNN

import string
from line_segmentation import segmentLines
import tensorflow as tf
from tensorflow import keras

# char_list = string.ascii_letters+string.digits

char_list =    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
FILENAME = 'test_4.png'
image = cv2.imread(FILENAME)

lines = segmentLines(image)
image_data = list()
for img in lines:
        # print(img.shape)
        gray = img

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        gray_ = cv2.medianBlur(gray, 5)

        thresh = cv2.adaptiveThreshold(gray, 255, 1,1,11,2)
        thresh_colour = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        kernel_word = np.ones((3,3), np.uint8)
        thresh = cv2.dilate(thresh,kernel_word,iterations = 3)
        kernel_line = np.ones((5,100), np.uint8)
        thresh = cv2.erode(thresh,None,iterations = 2)

        contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contour_list = list()
        # print(hierarchy[0][1][1])
        for i in range(len(contours)):
                if(hierarchy[0][i][3] == -1):
                        contour_list.append(contours[i])

        contours = tuple(contour_list)
        # cv2.imshow('l', img)
        # cv2.waitKey(0)


        for cnt in contours:
                x,y,w,h = cv2.boundingRect(cnt)
        #     image_data_temp = gray[y:y+h, x:x+w]
        #     aspect_ratio = image_data_temp.shape[1]/image_data_temp.shape[0]
        #     # print(image_data_temp.shape)
        #     image_data_temp = cv2.resize(image_data_temp, (int(aspect_ratio*32), 32), interpolation=cv2.INTER_AREA)
        #     if(image_data_temp.shape[1] > 128):
        #             image_data_temp = cv2.resize(image_data_temp, (128, 32), interpolation=cv2.INTER_AREA)
        #     elif image_data_temp.shape[1] < 128:
        #             add_zeros = np.ones((32, 128 - image_data_temp.shape[1]))*255
        #             image_data_temp = np.concatenate((image_data_temp, add_zeros), axis = 1)
        #     image_data_temp = np.expand_dims(image_data_temp, axis=2)
        #     image_data.append(image_data_temp)

                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.rectangle(thresh_colour,(x,y),(x+w,y+h),(0,255,0),2)

        points = []
        for cnt in contours:
                x,y,w,h = cv2.boundingRect(cnt)
                points.append((x,y,w,h))

        points = sorted(points, key = lambda data:math.sqrt(data[0]**2 + data[1]**2))

        for (x,y,w,h) in points:
                # print(x,y)
                
                image_data_temp = gray[y:y+h, x:x+w]
                aspect_ratio = image_data_temp.shape[1]/image_data_temp.shape[0]
                print(image_data_temp.shape)
                image_data_temp = cv2.resize(image_data_temp, (int(aspect_ratio*32), 32), interpolation=cv2.INTER_AREA)
                if(image_data_temp.shape[1] > 128):
                        image_data_temp = cv2.resize(image_data_temp, (128, 32), interpolation=cv2.INTER_AREA)
                elif image_data_temp.shape[1] < 128:
                        # add_zeros = np.ones((32, 128 - image_data_temp.shape[1]))*255
                        # image_data_temp = np.concatenate((image_data_temp, add_zeros), axis = 1)
                        image_data_temp = cv2.copyMakeBorder(image_data_temp, 0,0,0, 128 - image_data_temp.shape[1], cv2.BORDER_CONSTANT, value=[255,255,255])
                image_data_temp = np.expand_dims(image_data_temp, axis=2)
                image_data.append(image_data_temp)


# for i in image_data:
#         cv2.imshow('k', i)
#         cv2.waitKey(0)
# ii = image_data[2]
# ar = image_data[2].shape[1]/image_data[2].shape[0]
# print(ar)
# print(ii.shape)
# ii = cv2.resize(ii, (int(ar*32), 32), interpolation=cv2.INTER_NEAREST)
# ii = cv2.copyMakeBorder(ii, 0,0,0, 128 - ii.shape[1], cv2.BORDER_CONSTANT, value=[255,255,255])

# ii = np.expand_dims(ii, axis=2)
# print(ii.shape)
# cv2.imshow('c', ii)

# cv2.waitKey(0)
        # Finally show the image
        # for i in image_data:
        #             print(i.shape)
        # print("Shape: {}".format(image_data[-2].shape))

###########FJlkdjfks
        cv2.imshow('img',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        plt.rcParams['figure.figsize'] = (16,16)
        plt.subplot(121),plt.imshow(img),plt.title('Word')


# for i in image_data:
#         cv2.imshow('img', i)
#         cv2.waitKey(0)

valid_model = CRNN([], True)
valid_model.load_weights('trained_models/crnn_model.h5')

image_data = np.array(image_data)
prediction = valid_model.predict(image_data)
out = keras.backend.get_value(keras.backend.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                         greedy=True)[0][0])

# # i = 0
# # correct = 0
# # incorrect = 0
# # total = 0
for x in out:
        temp = ""
        for p in x:
                if int(p)!=-1:
                        temp = temp + char_list[int(p)]
        print(temp)
# # print("Total {}".format(total))
# # print("Correct {}".format(correct))
# # print("Incorrect {}".format(incorrect))