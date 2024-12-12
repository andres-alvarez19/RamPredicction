import pandas as pd
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("../data/dl_data/dataset_prueba_generado.csv")

numerical_columns = [
    "number_of_animals", "average_weight", "average_age", "sample_ph",
    "sample_temperature", "sample_humidity", "organic_matter_concentration",
    "microbial_load", "microbial_diversity", "treatment_days",
    "average_amount_per_animal", "ambient_temperature",
    "relative_humidity", "farm_size", "gene_concentration", "resistance_percentage"
]

dataset_scaled = dataset.copy()

scaler = StandardScaler()
dataset_scaled[numerical_columns] = scaler.fit_transform(dataset[numerical_columns])

output_path = "./data/dl_data/standardized_dataset.csv"
dataset_scaled.to_csv(output_path, index=False)

from sklearn.preprocessing import OneHotEncoder

dataset = pd.read_csv("./data/dl_data/standardized_dataset.csv")

categorical_columns = [
    "animal_name", "animal_diet", "antibiotic_use", "animal_type",
    "antibiotic_name", "antibiotic_frequency", "geographic_location", "gene_name"
]

encoder = OneHotEncoder(sparse_output=False, drop="first")
encoded_columns = encoder.fit_transform(dataset[categorical_columns])


encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(categorical_columns))

dataset_encoded = pd.concat([dataset.drop(columns=categorical_columns), encoded_df], axis=1)

output_path = "../data/dl_data/standardized_encoded_dataset.csv"
dataset_encoded.to_csv(output_path, index=False)

print(f"El dataset con One-Hot Encoding ha sido guardado en: {output_path}")
print(dataset_encoded.head())

import os

os.remove("./data/dl_data/standardized_dataset.csv")