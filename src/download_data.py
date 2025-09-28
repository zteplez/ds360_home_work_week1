import pandas as pd
import seaborn as sns
import os

def download_iris_data():
    df = sns.load_dataset("iris")

    df.to_csv('data/raw/iris.csv', index=False)

    print("Iris data seti indirildi ?")
    print(f"Veri boyutu: {df.shape}")
    print(f"Kolonlar: {list(df.columns)}")

if __name__ == "__main__":
    download_iris_data()