from keras_vggface.utils import preprocess_input
from keras_vggface import VGGFaceimport 
import numpy as np
from sklearn.metrics.pairwise import cosine_simolarity
import pickle
import cv2
from mtcnn import MTCNN
from PIL import Image
feature_list -np.array(pickle.load(open('embedding.pkl','rb')))
filenames=pickle.load(open('filenames.pkl','rb'))

model=VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
detector=MTCNN()
