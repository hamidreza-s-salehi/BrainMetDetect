from src.data_preparation import load_data, clean_data, encode_data
from src.feature_engineering import select_features, normalize_data
from src.model_training import train_random_forest, optimize_random_forest, train_xgboost
from src.model_explanation import explain_model

if __name__ == "__main__":
    radiomics, morphological = load_data()
    dataset_merged = clean_data(radiomics, morphological)
    df_encoded, target_encoder = encode_data(dataset_merged)

    df_final = select_features(df_encoded)
    df_final_normalized = normalize_data(df_final)

    X = df_final_normalized.drop(columns='Target', axis=1)
    y = df_final_normalized['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = train_random_forest(X_train, y_train, X_test, y_test)
    rf_opt_model = optimize_random_forest(X_train, y_train, X_test, y_test)
    gb_model = train_xgboost(X_train, y_train, X_test, y_test)

    labels = target_encoder.classes_
    explain_model(rf_model, X_train, X_test, labels)
    explain_model(rf_opt_model, X_train, X_test, labels)
    explain_model(gb_model, X_train, X_test, labels)
