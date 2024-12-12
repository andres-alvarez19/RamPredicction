import pandas as pd
import random

data = {
    "animal_name": random.choices(["vacas", "cerdos", "pollos", "ovejas"], k=100),
    "number_of_animals": [random.randint(10, 1000) for _ in range(100)],
    "animal_diet": random.choices(["forraje", "concentrado", "mixto"], k=100),
    "antibiotic_use": random.choices([0, 1], k=100),
    "animal_type": random.choices(["bovino", "porcino", "avicola", "ovino"], k=100),
    "average_weight": [round(random.uniform(50.0, 800.0), 2) for _ in range(100)],
    "average_age": [random.randint(6, 120) for _ in range(100)],
    "sample_ph": [round(random.uniform(5.5, 8.5), 2) for _ in range(100)],
    "sample_temperature": [round(random.uniform(15.0, 40.0), 2) for _ in range(100)],
    "sample_humidity": [random.randint(20, 90) for _ in range(100)],
    "organic_matter_concentration": [round(random.uniform(10.0, 70.0), 2) for _ in range(100)],
    "microbial_load": [random.randint(1000, 1000000) for _ in range(100)],
    "microbial_diversity": [round(random.uniform(1.0, 5.0), 2) for _ in range(100)],
    "antibiotic_name": random.choices(["penicilina", "tetraciclina", "estreptomicina", "eritromicina"], k=100),
    "treatment_days": [random.randint(1, 30) for _ in range(100)],
    "average_amount_per_animal": [round(random.uniform(5.0, 50.0), 2) for _ in range(100)],
    "antibiotic_frequency": random.choices(["diaria", "semanal", "mensual"], k=100),
    "geographic_location": random.choices(["Temuco, Chile", "Santiago, Chile", "Valdivia, Chile", "Osorno, Chile"],
                                          k=100),
    "ambient_temperature": [round(random.uniform(10.0, 35.0), 2) for _ in range(100)],
    "relative_humidity": [random.randint(30, 80) for _ in range(100)],
    "farm_size": [round(random.uniform(10.0, 500.0), 2) for _ in range(100)],
    "gene_name": random.choices(["blaCTX-M", "tetA", "aadA1", "ermB", None], k=100),
    "gene_concentration": [round(random.uniform(0.1, 10.0), 2) if random.random() > 0.2 else None for _ in range(100)],
    "resistance_percentage": [round(random.uniform(5.0, 100.0), 2) if random.random() > 0.2 else None for _ in
                              range(100)]
}

dataset = pd.DataFrame(data)

dataset.to_csv("./data/dl_data/dataset_prueba_generado.csv", index=False)

print("Dataset generado:")
print(dataset.head())
