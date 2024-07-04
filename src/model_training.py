from sklearn.ensemble import RandomForestClassifier
from mealpy.swarm_based.FOX import OriginalFOX
from sklearn.preprocessing import LabelEncoder
from mealpy.utils.space import IntegerVar

def train_rf_model(X_train, y_train):
    list_criterions = ["gini", "entropy"]  # Valid criteria for RandomForestClassifier
    criterions_encoder = LabelEncoder()
    criterions_encoder.fit(list_criterions)

    def fitness_function(solution):
        p = {
            "max_depth": int(solution[0]),
            "criterion": criterions_encoder.inverse_transform([int(solution[1])])[0],
            "max_features": int(solution[2])
        }
        model = RandomForestClassifier(
            criterion=p["criterion"],
            max_depth=p["max_depth"],
            max_features=p["max_features"]
        )
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_train, y_train)
        return accuracy

    def obj_func(solution):
        return fitness_function(solution)

    problem_dict = {
        "fit_func": fitness_function,
        "obj_func": obj_func,
        "lb": [2, 0, 10],
        "ub": [7, 1, 50],  # Updated upper bound for criterion index (0 or 1)
        "minmax": "max",
        "amend_position": lambda x: [int(xi) for xi in x],
        "bounds": [
            IntegerVar(2, 7),    # max_depth bounds
            IntegerVar(0, 1),    # criterion index (0 for 'gini', 1 for 'entropy')
            IntegerVar(10, 50)   # max_features bounds
        ]
    }

    optimizer = OriginalFOX(epoch=30, pop_size=10)
    best_solution, _ = optimizer.solve(problem_dict)

    best_params = {
        "max_depth": int(best_solution[0]),
        "criterion": criterions_encoder.inverse_transform([int(best_solution[1])])[0],
        "max_features": int(best_solution[2])
    }

    model = RandomForestClassifier(
        criterion=best_params["criterion"],
        max_depth=best_params["max_depth"],
        max_features=best_params["max_features"]
    )
    model.fit(X_train, y_train)
    return model, best_params
