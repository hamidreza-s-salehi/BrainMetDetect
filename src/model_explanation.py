import shap
import matplotlib.pyplot as plt

def explain_model(model, X_test, y_test, model_name):
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test)
    plt.title(f'{model_name} Feature Importance')
    plt.show()
    print(f'Classification Report for {model_name}')
    print(classification_report(y_test, model.predict(X_test)))
