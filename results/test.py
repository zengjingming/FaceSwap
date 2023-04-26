import cv2
import numpy as np

s=cv2.imread("./marlon.jpg")
t=cv2.imread("./testA.jpg")
print(s.shape)
print(t.shape)
difference=cv2.subtract(s,t)

result=not np.any(difference)
if result is True:
    print("yes")
