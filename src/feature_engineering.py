import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def select_features(data):
    df_encoded = data.copy()
    df_encoded.rename(columns={"Primary Tumor": "Target"}, inplace=True)
    df_encoded.drop("PATIENT", axis=1, inplace=True)

    target_encoder = LabelEncoder()
    df_encoded["Target"] = target_encoder.fit_transform(df_encoded["Target"])

    cat_columns = df_encoded.select_dtypes(exclude='number').columns
    for cat_column in cat_columns:
        categorical_encoder = LabelEncoder()
        not_null_index = df_encoded[cat_column].notnull()
        df_encoded.loc[not_null_index, cat_column] = categorical_encoder.fit_transform(df_encoded[cat_column][not_null_index])
    
    gini_model = DecisionTreeClassifier()
    X = df_encoded.drop(columns='Target', axis=1)
    y = df_encoded['Target']
    gini_model.fit(X, y)
    feature_importances = pd.Series(gini_model.feature_importances_, index=X.columns)
    top_features = feature_importances.nlargest(50).index.tolist()
    top_features.append('Target')
    df_final = df_encoded[top_features]
    return df_final.drop(columns='Target'), df_final['Target']

def split_data(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
