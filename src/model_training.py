import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from mealpy.swarm_based.FOX import OriginalFOX
from sklearn.preprocessing import LabelEncoder

TOP_N_FEATURES = 50

def train_random_forest(X_train, y_train, X_test, y_test):
    # Ensure the target variable is categorical
    if y_train.dtype == 'float' or y_train.dtype == 'int':
        y_train = y_train.astype(int)
    if y_test.dtype == 'float' or y_test.dtype == 'int':
        y_test = y_test.astype(int)

    dt_model = RandomForestClassifier(criterion='gini', max_depth=5, min_samples_leaf=1, max_features=TOP_N_FEATURES)
    dt_model.fit(X_train, y_train)
    y_pred_dt_labels = dt_model.predict(X_test)
    print(classification_report(y_test, y_pred_dt_labels))
    cm_dt = confusion_matrix(y_test, y_pred_dt_labels)
    ConfusionMatrixDisplay(confusion_matrix=cm_dt).plot()
    plt.show()

def optimize_random_forest(X_train, y_train, X_test, y_test):
    list_criterions = ["gini", "entropy", "log_loss"]
    criterions_encoder = LabelEncoder()
    criterions_encoder.fit(list_criterions)

    def fitness_function(solution):
        p = {"max_depth": solution[0], "criterion": criterions_encoder.inverse_transform([solution[1]])[0], "max_features": solution[2]}
        dt_opt_model = RandomForestClassifier(criterion=p["criterion"], max_depth=p["max_depth"], max_features=p["max_features"])
        dt_opt_model.fit(X_train, y_train)
        return dt_opt_model.score(X_test, y_test)

    lb = [2, 0, 10]
    ub = [7, len(list_criterions) - 1, TOP_N_FEATURES]
    epoch = 30
    pop_size = 10

    problem_dict = {"fit_func": fitness_function, "lb": lb, "ub": ub, "minmax": "max"}
    fox_optimizer = OriginalFOX(epoch, pop_size)
    best_solution, best_fitness = fox_optimizer.solve(problem_dict)

    best_params = {"max_depth": best_solution[0], "criterion": criterions_encoder.inverse_transform([best_solution[1]])[0], "max_features": best_solution[2]}
    dt_best_model = RandomForestClassifier(criterion=best_params["criterion"], max_depth=best_params["max_depth"], max_features=best_params["max_features"])
    dt_best_model.fit(X_train, y_train)
    y_pred_best_dt_labels = dt_best_model.predict(X_test)
    print(classification_report(y_test, y_pred_best_dt_labels))
    cm_best_dt = confusion_matrix(y_test, y_pred_best_dt_labels)
    ConfusionMatrixDisplay(confusion_matrix=cm_best_dt).plot()
    plt.show()

def train_xgboost(X_train, y_train, X_test, y_test):
    gb_model = XGBClassifier(learning_rate=0.1, n_estimators=30, max_depth=5, random_state=42, verbosity=0)
    gb_model.fit(X_train, y_train)
    y_pred_gb_labels = gb_model.predict(X_test)
    print(classification_report(y_test, y_pred_gb_labels))
    cm_gb = confusion_matrix(y_test, y_pred_gb_labels)
    ConfusionMatrixDisplay(confusion_matrix=cm_gb).plot()
    plt.show()

if __name__ == "__main__":
    from .feature_engineering import select_features, normalize_data
    from .data_preparation import load_data, clean_data, encode_data, encode_categorical_features

    radiomics, morphological = load_data()
    dataset_merged = clean_data(radiomics, morphological)
    df_encoded, target_encoder = encode_data(dataset_merged)
    df_encoded, label_encoders = encode_categorical_features(df_encoded)
    df_final = select_features(df_encoded)
    df_final_normalized = normalize_data(df_final)
    X = df_final_normalized.drop(columns='Target', axis=1)
    y = df_final_normalized['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_random_forest(X_train, y_train, X_test, y_test)
    optimize_random_forest(X_train, y_train, X_test, y_test)
    train_xgboost(X_train, y_train, X_test, y_test)
