# BrainMetDetect

## Project Overview

BrainMetDetect is a machine learning project aimed at detecting brain metastases using radiomic and morphological measurements. This repository contains the scripts necessary to prepare the data, engineer features, train machine learning models, and explain the models' decisions. The dataset used in this project is publicly available and can be cited using the provided DOI.

## Directory Structure

```
BrainMetDetect/
│
├── data/
│   ├── OpenBTAI_RADIOMICS.xlsx
│   ├── OpenBTAI_MORPHOLOGICAL_MEASUREMENTS.xlsx
│
├── src/
│   ├── data_preparation.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_explanation.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── main.py
```

## Data Description

The dataset used in this project is sourced from the publication:

- Renz, D. M., et al. (2023). OpenBTAI: A public dataset for radiomic and morphological measurements of brain metastases. *Scientific Data*, 10(1), 123. [https://doi.org/10.1038/s41597-023-02123-0](https://doi.org/10.1038/s41597-023-02123-0)

### Files

- `OpenBTAI_RADIOMICS.xlsx`: Contains radiomic features of brain metastases.
- `OpenBTAI_MORPHOLOGICAL_MEASUREMENTS.xlsx`: Contains morphological measurements and patient information.

## Project Components

### 1. Data Preparation (`src/data_preparation.py`)

This module handles loading and merging the datasets, cleaning the data, and encoding categorical variables.

### 2. Feature Engineering (`src/feature_engineering.py`)

This module includes steps for feature selection using the Gini index, normalizing the data, and splitting it into training and test sets.

### 3. Model Training (`src/model_training.py`)

This module contains code for training multiple models including Random Forest and XGBoost classifiers. It also includes hyperparameter optimization using the FOX algorithm.

### 4. Model Explanation (`src/model_explanation.py`)

This module uses SHAP (SHapley Additive exPlanations) to provide explanations for the model predictions, helping to understand the feature importances and the decision-making process of the models.

## Installation

To set up the project, clone the repository and install the required packages:

```bash
git clone https://github.com/hamidreza-s-salehi/BrainMetDetect.git
cd BrainMetDetect
pip install -r requirements.txt
```

## Usage

### Running the Project

1. **Prepare the Data:**

   ```bash
   python src/data_preparation.py
   ```

2. **Engineer Features:**

   ```bash
   python src/feature_engineering.py
   ```

3. **Train Models:**

   ```bash
   python src/model_training.py
   ```

4. **Explain Models:**

   ```bash
   python src/model_explanation.py
   ```

### Main Script

You can also run the entire pipeline using the `main.py` script:

```bash
python main.py
```

## Citation

If you use this project in your research, please cite it as follows:

```
Sadeghsalehi. (2024). BrainMetDetect: A machine learning approach for detecting brain metastases using radiomic and morphological measurements.
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
