import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

TOP_N_FEATURES = 50

def select_features(df_encoded):
    X = df_encoded.drop(columns='Target', axis=1)
    y = df_encoded['Target']
    gini_model = DecisionTreeClassifier()
    gini_model.fit(X, y)
    feature_importances = pd.Series(gini_model.feature_importances_, index=X.columns)
    top_features_tree = feature_importances.nlargest(TOP_N_FEATURES)
    top_selected_feature = pd.Series(list(set(top_features_tree.index)))
    top_selected_feature = top_selected_feature.append(pd.Series(["Target"]))
    df_final = df_encoded[top_selected_feature]
    return df_final

def normalize_data(df):
    scaler = MinMaxScaler()
    std_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return std_df

if __name__ == "__main__":
    from data_preparation import load_data, clean_data, encode_data

    radiomics, morphological = load_data()
    dataset_merged = clean_data(radiomics, morphological)
    df_encoded, target_encoder = encode_data(dataset_merged)
    df_final = select_features(df_encoded)
    df_final_normalized = normalize_data(df_final)
    print(df_final_normalized.head())
