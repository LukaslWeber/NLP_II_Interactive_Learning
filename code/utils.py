import os, random, pickle
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
import numpy as np


def initialize_random_number_generators(seed):
    np.random.seed(seed)
    random.seed(seed)


def load_MNIST():
    mnist = fetch_openml('mnist_784', version=1, parser="auto")

    X, y = mnist.data.to_numpy(), mnist.target.to_numpy()

    # Normalize as data ranges between 0 and 255
    X = X / 255
    # Convert type of y from string to integer
    y = y.astype(int)

    X_train, y_train = X[:60000], y[:60000]
    X_test, y_test = X[60000:], y[60000:]

    return X_train, y_train, X_test, y_test, X, y


def log_metrics(step: int, model, X_train, y_train, X_test, y_test, metrics: dict):
    train_loss = log_loss(y_train, model.predict_proba(X_train))
    test_loss = log_loss(y_test, model.predict_proba(X_test))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    metrics['train_loss'].append(train_loss)
    metrics['test_loss'].append(test_loss)
    metrics['test_acc'].append(test_accuracy)
    print(f"Iteration {step}: "
          f"- Train Loss: {train_loss:.4f} "
          f"- Test Loss: {test_loss:.4f} "
          f"- Test Accuracy: {test_accuracy}")
    return


def create_model(model_params, random_seed):
    return LogisticRegression(solver='sag',
                              penalty=model_params["regularization"],
                              C=model_params["regularization_strength"],
                              multi_class='multinomial',
                              max_iter=model_params["max_iterations_per_epoch"],
                              warm_start=True,
                              random_state=random_seed)


def save_model_and_metrics(experiment:str, dataset_name:str, name: str, model, metrics: dict):
    base_path = os.path.join("../results", dataset_name, f"exp{experiment}")

    # save model
    save_file(os.path.join(base_path, "models",  f"{name}_model.pkl"), model)

    # save metrics
    save_file(os.path.join(base_path, "metrics",  f"{name}_metrics.pkl"), metrics)


def load_model_and_metrics(experiment:str, dataset_name:str,  name: str):
    base_path = os.path.join("../results", dataset_name, f"exp{experiment}")

    loaded_model = load_file(os.path.join(base_path, "models", f"{name}_model.pkl"))

    loaded_metrics = load_file(os.path.join(base_path, "metrics", f"{name}_metrics.pkl"))

    return loaded_model, loaded_metrics

def save_file(path:str, dict):
    with open(path, 'wb') as file:
        pickle.dump(dict, file)

def load_file(path:str):
    with open(path, 'rb') as file:
        loaded_dictionary = pickle.load(file)

    return loaded_dictionary