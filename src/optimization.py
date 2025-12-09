import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class JOA:
    def __init__(self, obj_func, bounds, pop_size=20, iterations=20):
        self.obj_func = obj_func
        self.bounds = np.array(bounds)
        self.pop_size = pop_size
        self.iterations = iterations
        self.num_params = len(bounds)

    def optimize(self):
        # Initialize jellyfish population randomly
        pop = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.pop_size, self.num_params))
        fitness = np.array([self.obj_func(ind) for ind in pop])

        for it in range(self.iterations):
            # Sort population by fitness (minimization problem)
            sorted_idx = np.argsort(fitness)
            pop, fitness = pop[sorted_idx], fitness[sorted_idx]

            # Best jellyfish position
            best_jelly = pop[0].copy()

            # Update positions
            for i in range(1, self.pop_size):
                rand_jelly = np.random.randint(self.pop_size)  # Select a random jellyfish
                direction = np.random.uniform(-1, 1, self.num_params)  # Random movement

                new_jelly = pop[i] + direction * (best_jelly - pop[rand_jelly])
                new_jelly = np.clip(new_jelly, self.bounds[:, 0], self.bounds[:, 1])  # Keep within bounds

                # Evaluate new position
                new_fitness = self.obj_func(new_jelly)

                # Accept better solution
                if new_fitness < fitness[i]:
                    pop[i] = new_jelly
                    fitness[i] = new_fitness

        # Return best parameters and fitness
        return pop[0], fitness[0]

def get_model_config(model_name):
    """
    Returns the configuration (bounds, parameter names, model class) for a given model.
    """
    if model_name == "Logistic Regression":
        return {
            "class": LogisticRegression,
            "params": ["C"],
            "bounds": [(0.01, 10.0)],
            "types": [float],
            "fixed_params": {"max_iter": 1000, "solver": "liblinear"}
        }
    elif model_name == "Naive Bayes":
        return {
            "class": GaussianNB,
            "params": ["var_smoothing"],
            "bounds": [(1e-10, 1e-8)],
            "types": [float],
            "fixed_params": {}
        }
    elif model_name == "Support Vector Machine":
        return {
            "class": SVC,
            "params": ["C", "gamma"],
            "bounds": [(0.1, 100.0), (0.001, 1.0)],
            "types": [float, float],
            "fixed_params": {"kernel": "rbf", "probability": True}
        }
    elif model_name == "Random Forest":
        return {
            "class": RandomForestClassifier,
            "params": ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"],
            "bounds": [(10, 200), (3, 50), (2, 20), (1, 10)],
            "types": [int, int, int, int],
            "fixed_params": {"random_state": 42}
        }
    elif model_name == "Neural Network":
        return {
            "class": MLPClassifier,
            "params": ["hidden_layer_sizes", "alpha"],
            "bounds": [(5, 100), (0.0001, 0.1)],
            "types": [int, float],
            "fixed_params": {"activation": "relu", "solver": "adam", "max_iter": 1000, "random_state": 42}
        }
    else:
        return None

def optimize_model(model_name, X_train, y_train):
    """
    Optimizes hyperparameters for a specific model using JOA.
    """
    config = get_model_config(model_name)
    if not config:
        return None

    # Split training data further for validation during optimization
    X_opt_train, X_opt_val, y_opt_train, y_opt_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    def objective_function(params):
        current_params = config["fixed_params"].copy()
        
        for i, param_name in enumerate(config["params"]):
            val = params[i]
            if config["types"][i] == int:
                val = int(round(val))
            
            # Special handling for tuple params like hidden_layer_sizes
            if param_name == "hidden_layer_sizes":
                current_params[param_name] = (val,)
            else:
                current_params[param_name] = val

        model = config["class"](**current_params)
        model.fit(X_opt_train, y_opt_train)
        y_pred = model.predict(X_opt_val)
        return -accuracy_score(y_opt_val, y_pred)

    # Run JOA
    joa = JOA(objective_function, bounds=config["bounds"], pop_size=10, iterations=10)
    best_params_array, _ = joa.optimize()

    # Construct best params dictionary
    best_params = config["fixed_params"].copy()
    for i, param_name in enumerate(config["params"]):
        val = best_params_array[i]
        if config["types"][i] == int:
            val = int(round(val))
        
        if param_name == "hidden_layer_sizes":
            best_params[param_name] = (val,)
        else:
            best_params[param_name] = val
            
    return config["class"](**best_params)
