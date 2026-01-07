# src/features.py
import pandas as pd

def preparar_features(dataset):

    categoricas = [
        'gender', 'education_level', 'marital_status',
        'income_category', 'card_category'
    ]
    categoricas = [c for c in categoricas if c in dataset.columns]

    df = pd.get_dummies(
        dataset,
        columns=categoricas,
        drop_first=False,
        dtype=int
    )

    X = df.drop(columns=['clientnum','attrition_flag'], errors='ignore')
    y = df['attrition_flag'].astype(int)

    return X, y
