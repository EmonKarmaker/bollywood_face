from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import pickle
import cv2
from mtcnn import MTCNN
from PIL import Image
feature_list =np.array(pickle.load(open('embedding.pkl','rb')))
filenames=pickle.load(open('filenames.pkl','rb'))

model=VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
detector=MTCNN()
sample_img=cv2.imread('sample/download (2).jpeg')
results=detector.detect_faces(sample_img)
x,y,width,height=results[0]['box']
face=sample_img[y:y+height,x:x+width]
cv2.imshow('output',face)
cv2.waitKey(0)
