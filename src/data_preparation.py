import pandas as pd

def load_data():
    dataset_radiomics = pd.read_excel('data/OpenBTAI_RADIOMICS.xlsx')
    dataset_morphological = pd.read_excel('data/OpenBTAI_MORPHOLOGICAL_MEASUREMENTS.xlsx')
    return dataset_radiomics, dataset_morphological

def prepare_data(datasets):
    dataset_radiomics, dataset_morphological = datasets
    dataset_radiomics.drop(["Timepoint", "Label", "Lesion", "Segment"], axis=1, inplace=True)
    dataset_morphological = dataset_morphological[["PATIENT", "Primary Tumor"]]
    dataset_merged = pd.merge(dataset_morphological, dataset_radiomics, on='PATIENT')
    return dataset_merged
