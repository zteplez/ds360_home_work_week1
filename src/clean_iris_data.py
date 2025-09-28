import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def clean_iris_data(input_path='data/raw/iris.csv',
                    output_path='data/processed/iris_processed.csv'):
    
    df = pd.read_csv(input_path)

    df_clean = df.copy()

    le_species = LabelEncoder()
    df_clean['species'] = le_species.fit_transform(df_clean['species'])

    df_clean['sepal_area'] = df_clean['sepal_length'] * df_clean['sepal_width']
    df_clean['petal_area'] = df_clean['petal_length'] * df_clean['petal_width']
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_clean.to_csv(output_path, index=False)
        
    print("Veri temizlendi ve kaydedildi:", output_path)
    print(f"Orijinal boyut: {df.shape}")
    print(f"Temizlenmiş boyut: {df_clean.shape}")

    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species', 'petal_area', 'sepal_area']
    print(f"Model özellikleri: {features}")

    return df_clean, features

if __name__ == "__main__":
    clean_iris_data()
