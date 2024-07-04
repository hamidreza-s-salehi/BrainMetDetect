from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from mealpy.swarm_based.FOX import OriginalFOX
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

def train_rf_model(X_train, y_train):
    list_criterions = ["gini", "entropy", "log_loss"]
    criterions_encoder = LabelEncoder()
    criterions_encoder.fit(list_criterions)

    def fitness_function(solution):
        p = {
            "max_depth": solution[0],
            "criterion": criterions_encoder.inverse_transform([solution[1]])[0],
            "max_features": solution[2]
        }
        model = RandomForestClassifier(
            criterion=p["criterion"],
            max_depth=p["max_depth"],
            max_features=p["max_features"]
        )
        model.fit(X_train, y_train)
        return model.score(X_train, y_train)

    problem_dict = {
        "fit_func": fitness_function,
        "lb": [2, 0, 10],
        "ub": [7, 2, 50],
        "minmax": "max",
        "amend_position": lambda x: [int(xi) for xi in x]
    }

    optimizer = OriginalFOX(epoch=30, pop_size=10)
    best_solution, _ = optimizer.solve(problem_dict)

    best_params = {
        "max_depth": best_solution[0],
        "criterion": criterions_encoder.inverse_transform([best_solution[1]])[0],
        "max_features": best_solution[2]
    }

    model = RandomForestClassifier(
        criterion=best_params["criterion"],
        max_depth=best_params["max_depth"],
        max_features=best_params["max_features"]
    )
    model.fit(X_train, y_train)
    return model, best_params

def train_xgb_model(X_train, y_train):
    gb_list_criterions = ["friedman_mse", "squared_error"]
    gb_criterions_encoder = LabelEncoder()
    gb_criterions_encoder.fit(gb_list_criterions)

    def gb_fitness_function(solution):
        p = {
            "max_depth": int(solution[0]),
            "learning_rate": round(solution[1], 3),
            "criterion": gb_criterions_encoder.inverse_transform([int(solution[2])])[0],
            "n_estimator": int(solution[3])
        }
        model = XGBClassifier(
            learning_rate=p["learning_rate"],
            n_estimators=p["n_estimator"],
            criterion=p["criterion"],
            max_depth=p["max_depth"],
            random_state=42,
            verbosity=0
        )
        model.fit(X_train, y_train)
        return model.score(X_train, y_train)

    gb_problem_dict = {
        "fit_func": gb_fitness_function,
        "lb": [4, 0.05, 0, 10],
        "ub": [5, 0.15, 1, 50],
        "minmax": "max"
    }

    gb_optimizer = OriginalFOX(epoch=20, pop_size=10)
    gb_best_solution, _ = gb_optimizer.solve(gb_problem_dict)

    best_params = {
        "max_depth": int(gb_best_solution[0]),
        "learning_rate": gb_best_solution[1],
        "criterion": gb_criterions_encoder.inverse_transform([int(gb_best_solution[2])])[0],
        "n_estimator": int(gb_best_solution[3])
    }

    model = XGBClassifier(
        learning_rate=best_params["learning_rate"],
        n_estimators=best_params["n_estimator"],
        criterion=best_params["criterion"],
        max_depth=best_params["max_depth"],
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)
    return model, best_params
