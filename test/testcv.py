
class test:
    def __init__(self, name, age):
      self.name = name
      self.age = age

    def func1(self):
        self.func2()
        print("work 1")

    def func2(self):
        print("work 2")

import cv2

a=cv2.imread('z.PNG')
cv2.imshow('temp',a)
cv2.waitKey(0)
