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

# Load model
model = VGGFace(model='resnet50', include_top=False, input_shape=(224,224,3), pooling='avg')

# Initialize MTCNN
detector = MTCNN()

# Load sample image
sample_img = cv2.imread('sample/20230114_041411.jpg')

# Detect face
results = detector.detect_faces(sample_img)

if results:
    x, y, width, height = results[0]['box']

    # Crop the face
    face = sample_img[y:y+height, x:x+width]

    # Convert to PIL and resize
    image = Image.fromarray(face)
    image = image.resize((224, 224))

    # Convert to numpy array
    face_array = np.asarray(image)

    # Convert to float32 and preprocess
    face_array = face_array.astype('float32')
    preprocessed_img = preprocess_input(np.expand_dims(face_array, axis=0))

    # Extract features
    result = model.predict(preprocessed_img).flatten()

   # print("✅ Feature extracted successfully!")
    #print(result.shape)

#else:
 #   print("❌ No face detected!")
    similarity=[]
    for i in range(len(feature_list)):

        similarity.append(cosine_similarity(result.reshape(1,-1),feature_list[i].reshape(1,-1)))
    index_pos=sorted(list(enumerate(similarity)),reverse=True,key=lambda x:x[1])[0][0]
    temp_img=cv2.imread(filenames[index_pos])

    cv2.imshow('output',temp_img)
    cv2.waitKey(0)