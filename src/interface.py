import streamlit as st
import torch
import cv2
import numpy as np
from cnn_model import MobileNetV2FER

# Carregue o modelo treinado
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MobileNetV2FER(num_classes=7).to(device)
model.load_state_dict(torch.load("facial_expression_mobilenetv2.pth", map_location=device))
model.eval()

# Labels do FER2013
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

st.title("Reconhecimento de Expressões Faciais em Tempo Real")

run = st.button('Iniciar Webcam')

FRAME_WINDOW = st.image([])

def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48, 48))
    face = face.astype(np.float32) / 255.0
    face = (face - 0.5) / 0.5  # Normalização
    face = np.expand_dims(face, axis=0)  # (1, 48, 48)
    face = np.expand_dims(face, axis=0)  # (1, 1, 48, 48)
    face = torch.from_numpy(face).float().to(device)
    return face

if run:
    cap = cv2.VideoCapture(0)
    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Não foi possível acessar a câmera.")
            break
        face_tensor = preprocess(frame)
        with torch.no_grad():
            output = model(face_tensor)
            pred = torch.argmax(output, 1).item()
            label = labels[pred]
        # Mostra o label na imagem
        cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
else:
    st.write('Clique em "Iniciar Webcam" para começar.')