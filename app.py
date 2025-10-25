# app.py
import streamlit as st
import os
import pickle
import cv2
import numpy as np
from PIL import Image
from mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Initialize model and detector
# ---------------------------
st.set_option('deprecation.showfileUploaderEncoding', False)
detector = MTCNN()
model = VGGFace(model='resnet50', include_top=False, input_shape=(224,224,3), pooling='avg')

# ---------------------------
# Load embeddings and filenames
# ---------------------------
feature_list = pickle.load(open('embedding.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# ---------------------------
# Utility function: save uploaded image
# ---------------------------
def save_uploaded_image(uploaded_file):
    os.makedirs('uploads', exist_ok=True)
    file_path = os.path.join('uploads', uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# ---------------------------
# Utility function: extract features from an image
# ---------------------------
def extract_features(img_path):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)

    if results:
        x, y, width, height = results[0]['box']
        face = img[y:y+height, x:x+width]

        # Convert to PIL Image and resize
        face_image = Image.fromarray(face)
        face_image = face_image.resize((224,224))

        # Convert to numpy array and preprocess
        face_array = np.asarray(face_image).astype('float32')
        face_array = np.expand_dims(face_array, axis=0)
        preprocessed = preprocess_input(face_array)
        features = model.predict(preprocessed).flatten()
        return features
    else:
        return None

# ---------------------------
# Utility function: recommend the most similar celebrity
# ---------------------------
def recommend(features):
    similarity = []
    for f in feature_list:
        sim = cosine_similarity(features.reshape(1,-1), f.reshape(1,-1))[0][0]
        similarity.append(sim)
    index = np.argmax(similarity)
    return index

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Which Bollywood Celebrity Are You?")

uploaded_file = st.file_uploader("Upload your image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Save uploaded image
    img_path = save_uploaded_image(uploaded_file)

    # Extract features
    features = extract_features(img_path)

    if features is not None:
        # Recommend most similar celebrity
        index = recommend(features)
        predicted_name = os.path.basename(filenames[index]).split('_')[0]

        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.header("Your uploaded image")
            st.image(Image.open(img_path), use_column_width=True)
        with col2:
            st.header(f"Looks like: {predicted_name}")
            st.image(Image.open(filenames[index]), use_column_width=True)
    else:
        st.error("No face detected. Please upload a clear frontal face image.")
