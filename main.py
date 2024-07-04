from src import data_preparation, feature_engineering, model_training, model_explanation

if __name__ == "__main__":
    # Data Preparation
    data = data_preparation.load_data()
    prepared_data = data_preparation.prepare_data(data)

    # Feature Engineering
    features, target = feature_engineering.select_features(prepared_data)
    X_train, X_test, y_train, y_test = feature_engineering.split_data(features, target)

    # Model Training
    rf_model, best_rf_params = model_training.train_rf_model(X_train, y_train)
    xgb_model, best_xgb_params = model_training.train_xgb_model(X_train, y_train)

    # Model Explanation
    model_explanation.explain_model(rf_model, X_test, y_test, 'Random Forest')
    model_explanation.explain_model(xgb_model, X_test, y_test, 'XGBoost')
