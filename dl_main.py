import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
import joblib

dataset = pd.read_csv("./data/dl_data/standardized_encoded_dataset.csv")

dataset["gene_name_blaCTX-M_present"] = dataset["gene_name_blaCTX-M"].notna().astype(int)
dataset["gene_name_ermB_present"] = dataset["gene_name_ermB"].notna().astype(int)
dataset["gene_name_tetA_present"] = dataset["gene_name_tetA"].notna().astype(int)

X = dataset.drop(columns=[
    "gene_name_blaCTX-M", "gene_name_ermB", "gene_name_tetA",
    "gene_name_blaCTX-M_present", "gene_name_ermB_present", "gene_name_tetA_present"
])
y = dataset[["gene_name_blaCTX-M_present", "gene_name_ermB_present", "gene_name_tetA_present"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(3, activation="sigmoid")
])

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=50, batch_size=32)

loss, accuracy = model.evaluate(X_test, y_test)

model.save("./model/multi_output_gene_prediction_model.keras")
joblib.dump(scaler, "./model/scaler.pkl")
print("El modelo ha sido guardado como './model/multi_output_gene_prediction_model.keras'.")