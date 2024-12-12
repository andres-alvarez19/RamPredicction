import pandas as pd
import tensorflow as tf
import joblib

model = tf.keras.models.load_model("./model/multi_output_gene_prediction_model.keras")
scaler = joblib.load("./model/scaler.pkl")

new_data = pd.DataFrame([{
    "number_of_animals": 200,
    "average_weight": 450.5,
    "sample_ph": 7.2,
    "sample_temperature": 25.0,
    "sample_humidity": 60,
    "organic_matter_concentration": 40.5,
    "microbial_load": 500000,
    "microbial_diversity": 3.2,
    "treatment_days": 15,
    "average_amount_per_animal": 20.0,
    "ambient_temperature": 18.0,
    "relative_humidity": 70,
    "farm_size": 120.0
}])

expected_columns = scaler.feature_names_in_

for col in expected_columns:
    if col not in new_data.columns:
        new_data[col] = 0

new_data = new_data[expected_columns]

new_data_scaled = scaler.transform(new_data)

predictions = model.predict(new_data_scaled)

print("Probabilidad de presencia de genes:")
print(f"blaCTX-M: {predictions[0][0]:.2%}")
print(f"ermB: {predictions[0][1]:.2%}")
print(f"tetA: {predictions[0][2]:.2%}")
