import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Cargar el modelo previamente entrenado
model = tf.keras.models.load_model("modelo_emociones.h5")

# Clases (ajusta si usaste nombres distintos)
class_names = ['Enojado', 'feliz', 'neutral', 'triste']

st.title("Clasificador de emociones faciales 😊😠😐😢")

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_column_width=True)

    # Preprocesamiento: redimensionar e invertir canales para el modelo
    img = image.resize((224, 224))  # asegúrate que este tamaño sea el correcto para tu modelo
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]  # Quitar canal alfa si existe
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predicción
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.markdown(f"### Emoción detectada: **{predicted_class}**")
