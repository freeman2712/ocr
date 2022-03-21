import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

# Load the image
def segmentLines(img):
        # img = cv2.imread('test_4.png')

        # convert to grayscale
        img_ = img
        gray_ = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # print(gray_.shape)
        # smooth the image to avoid noises
        gray = cv2.medianBlur(gray_,5)

        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
        thresh_color = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)

        # apply some dilation and erosion to join the gaps
        kernel = np.ones((3,15), np.uint8)
        thresh = cv2.dilate(thresh,kernel,iterations = 3)
        thresh = cv2.erode(thresh,None,iterations = 2)

        # Find the contours
        contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        points = list()
        lines = list()
        for cnt in contours:
                x,y,w,h = cv2.boundingRect(cnt)
                # lines.append(gray_[y:y+h, x:x+w])
                points.append((x,y,w,h))
                # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                # cv2.rectangle(thresh_color,(x,y),(x+w,y+h),(0,255,0),2)
        

        points = sorted(points, key = lambda data:math.sqrt(data[0]**2 + data[1]**2))
        for (x,y,w,h) in points:
                lines.append(img_[y-3:y+h+3, x:x+w])


        
        # cv2.imshow('img', lines[-2])
        # cv2.waitKey(0)
        # return lines
         
        # cv2.imshow('img',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # plt.rcParams['figure.figsize'] = (16,16)
        # plt.subplot(121),plt.imshow(img),plt.title('Line')

        # for i in lines:
        #         print(i.shape)
        #         cv2.imshow('img', i)
        #         cv2.waitKey(0)
        # print(lines[0].shape)
        return lines

if __name__ == "__main__":
        img = cv2.imread('test_4.png')
        segmentLines(img)