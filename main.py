import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# Cargar el dataset
data_path = "processed_dataset_with_gene_present.csv"
df = pd.read_csv(data_path)

# Separar variables (X) y objetivo (y)
X = df.drop(["gene_present"], axis=1)  # No necesitamos 'gene_name', solo 'gene_present'
y = df["gene_present"]

# Identificar columnas categóricas y numéricas
categorical_columns = ["antibiotic_name_eritromicina", "antibiotic_name_estreptomicina",
                       "antibiotic_name_penicilina", "antibiotic_name_tetraciclina",
                       "animal_name_cerdos", "animal_name_ovejas", "animal_name_pollos",
                       "animal_name_vacas", "antibiotic_frequency_diaria",
                       "antibiotic_frequency_mensual", "antibiotic_frequency_semanal",
                       "bacteria_name_E. coli", "bacteria_name_Enterococcus",
                       "bacteria_name_Salmonella", "bacteria_name_Staphylococcus"]

numerical_columns = ["average_amount_per_animal"]

# Preprocesamiento: StandardScaler para numéricas y OneHotEncoder ya no es necesario
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_columns),
    ]
)

# Construir el pipeline del modelo
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
pipeline.fit(X_train, y_train)

# Evaluar el modelo
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
