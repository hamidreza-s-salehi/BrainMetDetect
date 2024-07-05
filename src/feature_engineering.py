import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from .data_preparation import encode_categorical_features 

TOP_N_FEATURES = 50

def select_features(df_encoded):
    X = df_encoded.drop(columns='Target', axis=1)
    y = df_encoded['Target']

    X, _ = encode_categorical_features(X)

    gini_model = DecisionTreeClassifier()
    gini_model.fit(X, y)
    feature_importances = pd.Series(gini_model.feature_importances_, index=X.columns)
    top_features_tree = feature_importances.nlargest(TOP_N_FEATURES)
    top_selected_feature = pd.concat([pd.Series(top_features_tree.index), pd.Series(["Target"])])
    df_final = df_encoded.loc[:, top_selected_feature]
    return df_final

def normalize_data(df):
    numeric_columns = df.select_dtypes(include=[float, int]).columns
    scaler = MinMaxScaler()
    std_df = pd.DataFrame(scaler.fit_transform(df[numeric_columns]), columns=numeric_columns)
    df.loc[:, numeric_columns] = std_df
    return df

if __name__ == "__main__":
    from .data_preparation import load_data, clean_data, encode_data, encode_categorical_features

    radiomics, morphological = load_data()
    dataset_merged = clean_data(radiomics, morphological)
    df_encoded, target_encoder = encode_data(dataset_merged)
    df_encoded, label_encoders = encode_categorical_features(df_encoded)
    df_final = select_features(df_encoded)
    df_final_normalized = normalize_data(df_final)
    print(df_final_normalized.head())
