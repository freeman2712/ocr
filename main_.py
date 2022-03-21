import sys
import sys
import numpy as np 
import cv2


if(len(sys.argv) == 1):
    print('Enter name of the file in the cmd line arg')
    sys.exit()


imgName = sys.argv[1]


img = cv2.imread(imgName)
cv2.imshow('Test', img)

print(img.shape)
cv2.waitKey(0)