import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# 1. Cargar el dataset
data_path = "processed_dataset_with_gene_present.csv"
df = pd.read_csv(data_path)

# Separar variables (X) y objetivo (y)
X = df.drop(["gene_present"], axis=1)  # Eliminar la variable objetivo
y = df["gene_present"]                 # Variable objetivo

# Identificar columnas numéricas y aplicar escalamiento
numerical_columns = ["average_amount_per_animal"]
scaler = StandardScaler()
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

# Convertir X a numpy array (requerido por TensorFlow)
X = X.values
y = y.values

# 2. Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Construir el modelo de Deep Learning
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),  # Capa de entrada: cantidad de features
    tf.keras.layers.Dense(128, activation="relu"),    # Capa oculta 1 con 128 neuronas
    tf.keras.layers.Dropout(0.3),                     # Regularización con Dropout
    tf.keras.layers.Dense(64, activation="relu"),     # Capa oculta 2 con 64 neuronas
    tf.keras.layers.Dropout(0.3),                     # Otro Dropout
    tf.keras.layers.Dense(1, activation="sigmoid")    # Capa de salida binaria (1 neurona)
])

# 4. Compilar el modelo
model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

# 5. Entrenar el modelo
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=50,       # Número de épocas
                    batch_size=32)   # Tamaño de batch

# 6. Evaluar el modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# 7. Predicción y reporte de resultados
y_pred = (model.predict(X_test) > 0.5).astype("int32")  # Convertir probabilidades a clases (0 o 1)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
