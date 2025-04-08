import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Cargar modelo
model = tf.keras.models.load_model("modelo_emociones.h5")

# Etiquetas de las clases (modifica según tu dataset)
class_names = ['feliz', 'neutral', 'triste']

st.title("Clasificación de emociones faciales")
st.write("Sube una imagen y el modelo predecirá la emoción.")

# Cargar imagen
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Imagen cargada.', use_column_width=True)

    # Preprocesamiento
    img = image.resize((224, 224))  # Tamaño usado en el entrenamiento
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predicción
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    st.write(f"**Emoción detectada:** {predicted_class}")
