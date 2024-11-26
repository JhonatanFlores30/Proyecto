import streamlit as st
from PIL import Image
import plotly.express as px
import pandas as pd
import cv2
import time
import tensorflow as tf
import numpy as np
from collections import Counter

# Título de la aplicación
st.title("Página de la Webcam con Detección de Emociones")

# Cargar tu modelo preentrenado
model = tf.keras.models.load_model("modelEmocion.h5")

# Lista de clases de emociones (con las clases que mencionaste)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Placeholder para reproducir la cámara
frame_placeholder = st.empty()

# Inicia la cámara
cap = cv2.VideoCapture(0)

# Lista para almacenar las emociones detectadas
emotion_counts = {emotion: 0 for emotion in emotion_labels}

# Preprocesamiento de la imagen (según cómo tu modelo haya sido entrenado)
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Si tu modelo fue entrenado con imágenes en escala de grises
    image = cv2.resize(image, (48, 48))  # Ajusta el tamaño si es necesario (48x48 es común en muchos modelos)
    image = image / 255.0  # Normalizar la imagen
    image = np.expand_dims(image, axis=-1)  # Añadir una dimensión extra para la profundidad de color
    image = np.expand_dims(image, axis=0)  # Añadir una dimensión extra para el batch size
    return image

# Loop para tomar frames de la cámara y analizarlos
if cap.isOpened():
    for _ in range(100):  # Limite de 100 frames para terminar el loop
        ret, frame = cap.read()
        if not ret:
            st.write("Error: No frame available.")
            break
        
        # Preprocesamiento de la imagen
        processed_image = preprocess_image(frame)

        # Realizar la predicción de todas las emociones
        emotion_probs = model.predict(processed_image)
        
        # Contar las ocurrencias de todas las emociones detectadas
        for i, emotion in enumerate(emotion_labels):
            # Asignar la probabilidad para cada emoción
            emotion_prob = emotion_probs[0][i]
            emotion_counts[emotion] += emotion_prob  # Sumar la probabilidad para cada emoción

        # Mostrar el frame y la emoción detectada
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertir el frame a RGB
        frame_pil = Image.fromarray(frame_rgb)  # Convertir a imagen de PIL para Streamlit
        caption = f"Probabilidades de emociones detectadas."
        
        # Mostrar la imagen
        frame_placeholder.image(frame_pil, caption=caption)

        # Pausar brevemente para permitir que otros elementos se rendericen
        time.sleep(0.1)
else:
    st.write("Error: Unable to open webcam.")

# Liberar la cámara
cap.release()

# Convertir los resultados de la detección de emociones en un DataFrame
emotion_data = pd.DataFrame(list(emotion_counts.items()), columns=['Emotion', 'Probability Sum'])

# Crear gráfico de barras para las emociones detectadas
fig = px.bar(
    emotion_data,
    x='Emotion',
    y='Probability Sum',
    title='Frecuencia y Probabilidad de Emociones Detectadas',
    labels={'Emotion': 'Emoción', 'Probability Sum': 'Suma de Probabilidades'},
    color='Probability Sum',
    color_continuous_scale='Spectral'
)

# Mostrar el gráfico en Streamlit
st.title("Gráfico de Emociones Detectadas")
st.plotly_chart(fig)
