import shap
import matplotlib.pyplot as plt

def explain_model(model, X_train, X_test, labels):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, class_names=labels)

if __name__ == "__main__":
    from model_training import train_random_forest, optimize_random_forest, train_xgboost

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
