import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

dataset_path = "dataset_prueba.csv"
df = pd.read_csv(dataset_path)

categorical_columns = ["antibiotic_name", "gene_name", "animal_name", "antibiotic_frequency", "bacteria_name"]
numerical_columns = ["average_amount_per_animal"]

categorical_transformer = OneHotEncoder(sparse_output=False)
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_columns),
        ("cat", categorical_transformer, categorical_columns),
    ]
)

# Aplicar el preprocesamiento
X_processed = preprocessor.fit_transform(df)

# Convertir el resultado a un DataFrame
# Obtener nombres de columnas generadas
cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_columns)
new_columns = list(numerical_columns) + list(cat_feature_names)
X_processed_df = pd.DataFrame(X_processed, columns=new_columns)

# Guardar el dataset procesado
processed_dataset_path = "processed_dataset.csv"
X_processed_df.to_csv(processed_dataset_path, index=False)

print(f"Dataset procesado guardado en: {processed_dataset_path}")

# Cargar el dataset procesado
data_path = "processed_dataset.csv"
df = pd.read_csv(data_path)

# Identificar las columnas de 'gene_name' usando sus nombres
gene_columns = ["gene_name_aadA1", "gene_name_blaCTX-M", "gene_name_ermB", "gene_name_tetA"]

# Crear la columna 'gene_present' sumando las columnas de genes
df["gene_present"] = df[gene_columns].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

# Verificar los resultados
print(df[["gene_present"] + gene_columns].head())

# Guardar el dataset con la nueva columna
output_path = "processed_dataset_with_gene_present.csv"
df.to_csv(output_path, index=False)
print(f"Dataset actualizado guardado en: {output_path}")
