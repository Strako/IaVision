import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflowjs as tfjs
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Función para preprocesar las imágenes
def preprocess_image(image, target_size=(64, 64)):
    if isinstance(image, str):  # Si es una ruta, carga la imagen
        image = cv2.imread(image)
        if image is None:
            raise FileNotFoundError(f"No se pudo leer la imagen en la ruta: {image}")
    # Procesa la imagen como una matriz
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    return image / 255.0  # Normalizar entre 0 y 1

# Generación de datos de ejemplo (se necesitan imágenes etiquetadas)
def load_dataset(image_paths, labels, target_size=(64, 64)):
    images = [preprocess_image(img, target_size) for img in image_paths]
    return np.array(images), np.array(labels)

# Ejemplo de datos (modifica con tus propios datos)
image_paths = [
    './img/tb01.png',
    './img/tb02.png',
    './img/tb03.png',
    './img/tb04.png',
    './img/tb05.png',
    './img/tb06.png',
    './img/tm01.png',
    './img/tm02.png',
    './img/tm03.png',
    './img/tm04.png',
    './img/tm05.png',
    './img/tm06.png',
    # Agrega más imágenes aquí
]

labels = [0,0,0,0,0,0,1,1,1,1,1,1]  # Etiquetas: 0-Tumor benigno, 1-Tumor maligno
X, y = load_dataset(image_paths, labels)
y = to_categorical(y, num_classes=3)  # Codificación one-hot

# Dividir datos en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo
model = Sequential([
    Flatten(input_shape=(64, 64, 3)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')  # Tres clases: triángulo, cuadrado, círculo
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=16)

# Guardar el modelo entrenado
model.save('shape_detector_model.h5')

# Convertir modelo a tensorflowjs
tfjs.converters.save_keras_model(model, 'model')

# Usar el modelo en tu script original
def predict_shape(image, model, target_size=(64, 64)):
    img = preprocess_image(image, target_size)  # Ahora acepta matrices
    img = np.expand_dims(img, axis=0)  # Añadir dimensión para el batch
    predictions = model.predict(img)
    class_idx = np.argmax(predictions)
    confidence = predictions[0][class_idx]  # Obtener el valor de confianza para la clase predicha
    return class_idx, confidence

# Cargar el modelo guardado
model = tf.keras.models.load_model('shape_detector_model.h5')

# Procesar la imagen para detección de contornos y usar el modelo
image = cv2.imread('./img/tt.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Usar Canny con umbral ajustado
canny = cv2.Canny(gray, 50, 150)  # Umbral ajustado
canny = cv2.dilate(canny, None, iterations=3)  # Dilatación para ampliar los contornos
canny = cv2.erode(canny, None, iterations=2)  # Erosión para limpiar ruidos

# Encontrar todos los contornos
cnts, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Si hay contornos encontrados, procesarlos
if cnts:
    # Crear una máscara para todo el contorno
    mask = np.zeros_like(image)
    
    # Ordenar los contornos por área en orden descendente
    sorted_cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:3]  # Los tres contornos más grandes
    
    for cnt in sorted_cnts:
        cv2.drawContours(mask, [cnt], -1, (255, 255, 255), thickness=cv2.FILLED)
    
    # Aplicar la máscara para extraer las manchas del tumor
    tumor_region = cv2.bitwise_and(image, mask)

    # Procesar los tres contornos más grandes
    for cnt in sorted_cnts:
        # Obtener el rectángulo que rodea cada contorno
        x, y, w, h = cv2.boundingRect(cnt)
        roi = tumor_region[y:y+h, x:x+w]  # Región de interés (máscara aplicada)

        if roi.size > 0:
            class_idx, confidence = predict_shape(roi, model)
            label = ["Tumor benigno", "Tumor maligno"][class_idx]
            confidence_text = f"{confidence * 100:.2f}%"  # Formatear la confianza como porcentaje
            label_text = f"{label} ({confidence_text})"
            cv2.putText(image, label_text, (x, y-5), 1, 1, (0, 255, 0), 1)

        cv2.drawContours(image, [cnt], 0, (0, 255, 0), 2)  # Dibuja el contorno actual

# Mostrar la imagen resultante
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
