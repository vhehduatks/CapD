import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

detector=dlib.get_frontal_face_detector()
shaper=dlib.shape_predictor('model_68')
facerec=dlib.face_recognition_model_v1()

img_path=list()
desc_dec=list()

def face_finder(img):
    face_marks=list()
    face_rect=list()

    detected=detector(img,1)

    if not detected:
        return face_marks,face_rect

    for face in detected:
        rect=(face.left(),face.right(),face.top(),face.bottom())
        shape=shaper(img,face)

        face_marks.append(shape)
        face_rect.append(rect)
    
    return face_marks,face_rect


def face_encode(img,face_marks):
    face_descriptors=list()
    
    for mark in face_marks:
        descriptor=facerec.compute_face_descriptor(img,mark)
        face_descriptors.append(np.array(descriptor))

    return np.array(face_descriptors)

def face_recognition(face_descriptors,img,rects):
        
    for i,desc in enumerate(face_descriptors):
        rect=patches.Rectangle(rects[i][0],rects[i][1][1]-rects[i][0][1],rects[i][1][0]-rects[i][0][0],linewidth=2,edgecolor='w',facecolor='none')
