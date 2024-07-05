import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data():
    radiomics = pd.read_excel('data/OpenBTAI_RADIOMICS.xlsx')
    morphological = pd.read_excel('data/OpenBTAI_MORPHOLOGICAL_MEASUREMENTS.xlsx')
    return radiomics, morphological

def clean_data(radiomics, morphological):
    radiomics.drop(["Timepoint", "Label", "Lesion", "Segment"], axis=1, inplace=True)
    morphological = morphological[["PATIENT", "Primary Tumor"]]
    dataset_merged = pd.merge(morphological, radiomics, on='PATIENT')
    return dataset_merged

def encode_data(dataset_merged):
    df_encoded = dataset_merged.copy()
    df_encoded.rename(columns={"Primary Tumor": "Target"}, inplace=True)
    df_encoded.drop("PATIENT", axis=1, inplace=True)
    target_encoder = LabelEncoder()
    df_encoded["Target"] = target_encoder.fit_transform(df_encoded["Target"])
    return df_encoded, target_encoder

if __name__ == "__main__":
    radiomics, morphological = load_data()
    dataset_merged = clean_data(radiomics, morphological)
    df_encoded, target_encoder = encode_data(dataset_merged)
    print(df_encoded.head())
