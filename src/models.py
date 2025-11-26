from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def get_models():
    """
    Returns a dictionary of models to be trained.
    
    Returns:
        dict: Dictionary where keys are model names and values are model instances.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(kernel='linear', probability=True),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(11,), activation='relu', solver='adam', max_iter=1000, random_state=42)
    }
    return models
