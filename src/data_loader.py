# src/data_loader.py
import pandas as pd
import numpy as np

def cargar_dataset():
    """
    Carga y limpia el dataset BankChurners de forma robusta.
    Retorna un DataFrame limpio con attrition_flag binario.
    """

    url = "https://raw.githubusercontent.com/WelmanRosa/tia_python/main/BankChurners.csv"

    dataset = pd.read_csv(url, sep=';', encoding='utf-8', engine='python')

    dataset.columns = (
        dataset.columns
        .str.strip()
        .str.lower()
        .str.replace(r'\s+', '_', regex=True)
    )

    for col in dataset.select_dtypes(include=['object']).columns:
        dataset[col] = dataset[col].str.strip()

    cols_nb = [
        'naive_bayes_classifier_attrition_flag_card_category_contacts_count_12_mon_dependent_count_education_level_months_inactive_12_mon_1',
        'naive_bayes_classifier_attrition_flag_card_category_contacts_count_12_mon_dependent_count_education_level_months_inactive_12_mon_2'
    ]
    dataset = dataset.drop(columns=[c for c in cols_nb if c in dataset.columns], errors='ignore')

    if 'attrition_flag' not in dataset.columns:
        alias_attr = [c for c in dataset.columns if ('attrition' in c and 'flag' in c)]
        if alias_attr:
            dataset['attrition_flag'] = dataset[alias_attr[0]]
        else:
            raise KeyError("No se encuentra la variable objetivo attrition_flag")

    def map_attrition(x):
        if pd.isna(x): 
            return np.nan
        s = str(x).strip()
        if s == 'Existing Customer': 
            return 0
        if s == 'Attrited Customer': 
            return 1
        try:
            val = int(float(s))
            if val in (0, 1):
                return val
        except:
            pass
        return np.nan

    dataset['attrition_flag'] = dataset['attrition_flag'].apply(map_attrition)
    dataset['attrition_flag'] = pd.to_numeric(dataset['attrition_flag'], errors='coerce')

    num_candidates = [
        'customer_age','dependent_count','months_on_book','total_relationship_count',
        'months_inactive_12_mon','contacts_count_12_mon','credit_limit',
        'total_revolving_bal','avg_open_to_buy','total_amt_chng_q4_q1',
        'total_trans_amt','total_trans_ct','total_ct_chng_q4_q1','avg_utilization_ratio'
    ]
    for nc in num_candidates:
        if nc in dataset.columns:
            dataset[nc] = pd.to_numeric(dataset[nc], errors='coerce')

    dataset = dataset[~dataset['attrition_flag'].isna()].reset_index(drop=True)
    dataset['attrition_flag'] = dataset['attrition_flag'].astype(int)

    return dataset
