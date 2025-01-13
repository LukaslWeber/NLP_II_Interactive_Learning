import os, random, pickle
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
import numpy as np
import tensorflow as tf
import torch
from sklearn.model_selection import train_test_split
from skorch import NeuralNetClassifier
from torch import nn
from cnn_model import CNN


def initialize_random_number_generators(seed):
    np.random.seed(seed)
    random.seed(seed)
    # Cuda should not be used but I found it hard to produce reproducible results with
    # Pytorch models so I'll set this as a precaution
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For CUDA
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Ensure reproducibility


def load_MNIST(random_seed, validation_split=0.2):
    mnist = fetch_openml('mnist_784', version=1, parser="auto")

    X, y = mnist.data.to_numpy(), mnist.target.to_numpy()

    # Normalize as data ranges between 0 and 255
    X = X / 255.0
    # Convert type of y (aka the label) from string to integer
    y = y.astype(int)

    X_train_full, y_train_full = X[:60000], y[:60000] # This is the default way of extracting train and test data
    X_test, y_test = X[60000:], y[60000:]

    # Create validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=validation_split, random_state=random_seed
    )

    return X_train, y_train, X_test, y_test, X_val, y_val, X, y


def load_CIFAR():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalize pixel values as they are rgb values between 0 and 255
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Transpose the datasets to have the channel (3) first and then height (1) and width (2)
    X_train = np.transpose(X_train, (0, 3, 1, 2))
    X_test = np.transpose(X_test, (0, 3, 1, 2))

    # Convert labels to 1D array
    y_train, y_test = y_train.flatten(), y_test.flatten()

    X_whole = np.concatenate((X_train, X_test), axis=0)
    y_whole = np.concatenate((y_train, y_test), axis=0)

    # Check if concatenation of train and test doesn't change data order
    assert (X_whole[0:50000] == X_train).all(), "X_train does not match the first 50000 samples of X_whole."
    assert (X_whole[50000:] == X_test).all(), "X_test does not match the last samples of X_whole."
    assert (y_whole[0:50000] == y_train).all(), "y_train does not match the first 50000 labels of y_whole."
    assert (y_whole[50000:] == y_test).all(), "y_test does not match the last labels of y_whole."

    return X_train, y_train, X_test, y_test, X_whole, y_whole


def log_metrics(step: int, model, X_train, y_train, X_test, y_test, metrics: dict, is_cnn=False, device="cpu"):
    if is_cnn:
        criterion = torch.nn.CrossEntropyLoss()

        model.estimator.module_.eval()  # Put the model into evaluation mode

        with torch.no_grad():  # Disable gradient computation for efficiency
            train_logits = model.estimator.forward(torch.tensor(X_train, dtype=torch.float32, device=device)).detach()
            test_logits = model.estimator.forward(torch.tensor(X_test, dtype=torch.float32, device=device)).detach()

            y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=device)
            y_test_tensor = torch.tensor(y_test, dtype=torch.long, device=device)

            train_loss = criterion(train_logits, y_train_tensor).item()
            test_loss = criterion(test_logits, y_test_tensor).item()

            test_accuracy = accuracy_score(y_test, model.predict(X_test))

        model.estimator.module_.train()  # Restore training mode
    else:
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


def create_log_reg_model(model_params, random_seed, device="cpu"):
    return LogisticRegression(solver=model_params['solver'],
                              penalty=model_params["regularization"],
                              C=model_params["regularization_strength"],
                              multi_class='multinomial',
                              max_iter=model_params["max_iterations_per_epoch"],
                              tol=model_params["early_stopping_tol"],
                              warm_start=True,
                              random_state=random_seed)


def create_cnn_model(model_params, random_seed, device="cpu"):
    initialize_random_number_generators(random_seed)
    cnn = CNN(model_params)
    return NeuralNetClassifier(cnn,
               criterion=nn.CrossEntropyLoss, # don't i need to use another loss?
               optimizer=torch.optim.Adam,  # Pass the optimizer class, not an instance
               optimizer__lr=model_params["learning_rate"],  # Set learning rate
               optimizer__weight_decay=model_params["weight_decay"],
               train_split=None,  # this disables an internal validation split
               verbose=0,
               device=device,
               warm_start=True)


def save_model_and_metrics(experiment: str, dataset_name: str, name: str, model, metrics: dict):
    base_path = os.path.join("../results", dataset_name, f"exp{experiment}")

    # save model
    save_file(os.path.join(base_path, "models", f"{name}_model.pkl"), model)

    # save metrics
    save_file(os.path.join(base_path, "metrics", f"{name}_metrics.pkl"), metrics)


def load_model_and_metrics(experiment: str, dataset_name: str, name: str):
    base_path = os.path.join("../results", dataset_name, f"exp{experiment}")

    loaded_model = load_file(os.path.join(base_path, "models", f"{name}_model.pkl"))

    loaded_metrics = load_file(os.path.join(base_path, "metrics", f"{name}_metrics.pkl"))

    return loaded_model, loaded_metrics


def save_file(path: str, dict):
    with open(path, 'wb') as file:
        pickle.dump(dict, file)


def load_file(path: str):
    with open(path, 'rb') as file:
        loaded_dictionary = pickle.load(file)

    return loaded_dictionary
