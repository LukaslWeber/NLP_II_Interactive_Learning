import os, random, pickle
import time

from modAL import ActiveLearner
from modAL.uncertainty import classifier_margin
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

    X_train_full, y_train_full = X[:60000], y[:60000]  # This is the default way of extracting train and test data
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
    cnn = CNN(**model_params)
    return NeuralNetClassifier(cnn,
                               criterion=nn.CrossEntropyLoss,  # don't i need to use another loss?
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


# Training scripts:
""" 
    Method for pool-based active learning query strategies. 
"""


def train_active_learner(model_params, query_strat, n_query_instances: int, epochs: int, random_seed: int,
                         datasets, create_model, device="cpu"):
    initialize_random_number_generators(seed=random_seed)

    dataset_name = datasets['dataset_name']
    X_initial, y_initial = datasets['X_initial'], datasets['y_initial']
    X_train, y_train = datasets['X_train'], datasets['y_train']
    X_test, y_test = datasets['X_test'], datasets['y_test']
    pool_idx = datasets['pool_idx']
    X_pool, y_pool = X_train[pool_idx], y_train[pool_idx]

    is_cnn = (dataset_name == "CIFAR")

    model = create_model(model_params, random_seed=random_seed, device=device)

    learner = ActiveLearner(estimator=model,
                            query_strategy=query_strat,
                            X_training=X_initial,
                            y_training=y_initial)

    # Passing X_training and y_training to the ActiveLearner automatically calls the fit method for log_reg with these data points

    metrics = {'queries': [], 'train_loss': [], 'train_loss_current': [], 'test_loss': [], 'test_acc': []}

    start = time.time()
    for epoch in range(epochs):  # epochs=n_queries to have both models trained on the same number of overall epochs
        query_idx, query_inst = learner.query(X_pool, n_instances=n_query_instances)

        # Simulate labeling
        learner.teach(X_pool[query_idx], y_pool[query_idx], only_new=False)

        # Remove queried point(s)
        X_pool, y_pool = np.delete(X_pool, query_idx, axis=0), np.delete(y_pool, query_idx, axis=0)

        metrics['queries'].append(query_idx)
        # log_metrics logs train loss for the whole train dataset which doesn't reflect the actual value in the current step but gives the ability to compare both models on the training set.
        # To log training state on the actual (current) training set, do this additionally
        if is_cnn:
            criterion = torch.nn.CrossEntropyLoss()
            learner.estimator.module_.eval()  # Put the model into evaluation mode

            with torch.no_grad():  # Disable gradient computation for efficiency
                X_training_tensor = torch.tensor(learner.X_training, dtype=torch.float32, device=device)
                curr_train_logits = learner.estimator.forward(X_training_tensor)
                curr_train_y_tensor = torch.tensor(learner.y_training, dtype=torch.long, device=device)
                train_loss_current = criterion(curr_train_logits,
                                               curr_train_y_tensor).item()

            learner.estimator.module_.train()  # Restore training mode

        else:
            train_loss_current = log_loss(learner.y_training, learner.predict_proba(learner.X_training))
        metrics['train_loss_current'].append(train_loss_current)

        log_metrics(epoch + 1, learner, X_train, y_train, X_test, y_test, metrics, is_cnn=is_cnn, device=device)
        print(
            f"       Current train loss: {train_loss_current:.4f}    number of train samples: {len(learner.X_training)}")
    print(f"Training time: {time.time() - start:.2f} seconds")

    return learner.estimator, metrics


""" 
    Method for stream-based active learning query strategies. 

    arguments:
    - model_params: dict
    - query_strat: dict
    - query_score_threshold: float
    - epochs: int
    - random_seed: int
    - X_stream: np.ndarray, typically full dataset
    - y_stream: np.ndarray, typically full dataset
    - X_initial: np.ndarray, initial training points
    - y_initial: np.ndarray, initial training points
"""


def train_active_learner_stream(model_params, query_score_fn, n_query_instances: int, query_score_threshold: float,
                                epochs: int, random_seed: int, datasets, create_model, device="cpu"):
    initialize_random_number_generators(seed=random_seed)

    X_stream, y_stream = datasets['X_train'].copy(), datasets['y_train'].copy()
    X_initial, y_initial = datasets['X_initial'], datasets['y_initial']
    X_train, y_train = datasets['X_train'], datasets['y_train']
    X_test, y_test = datasets['X_test'], datasets['y_test']

    model = create_model(model_params, random_seed=random_seed, device=device)

    learner = ActiveLearner(estimator=model,
                            X_training=X_initial,
                            y_training=y_initial)
    # Passing X_training and y_training to the ActiveLearner automatically calls the fit method for log_reg with these data points

    metrics = {'queries': [], 'train_loss': [], 'train_loss_current': [], 'test_loss': [], 'test_acc': []}

    start = time.time()

    # In pool based training, I get n_query_instances in each epoch. To have a comparable amount of data points for retraining the classifier, I use max_instances which equates to the amount of instances, a model in pool based approaches has seen.
    max_instances = n_query_instances * epochs
    used_instances = 0

    # Prevent infinite looping in case that no sample fulfills the query condition
    retry_count, max_retries = 0, 10000

    # Use a random permutation of the whole train data set to mimic a data stream.
    stream_indices, stream_pointer = np.random.permutation(len(X_stream)), 0

    while used_instances < max_instances:
        if stream_pointer >= len(
                stream_indices):  # Restart stream simulation if the loop went through the whole list of data points
            stream_indices = np.random.permutation(len(X_stream))
            stream_pointer = 0

        stream_idx = stream_indices[stream_pointer]
        x_instance, y_instance = X_stream[stream_idx].reshape(1, -1), y_stream[stream_idx].reshape(-1, )
        stream_pointer += 1
        retry_count += 1

        print((X_train == X_stream).all())

        query_score = query_score_fn(learner, x_instance)

        # Depending on the function, we want to select samples with either a high score (uncertainty) or a low score (margin)

        if query_score_fn == classifier_margin:
            query_condition = query_score < query_score_threshold
        else:  # classifier_uncertainty, classifier_entropy
            query_condition = query_score > query_score_threshold

        if query_condition:
            learner.teach(x_instance, y_instance)

            metrics['queries'].append(stream_idx)
            # log_metrics logs train loss for the whole train dataset which doesn't reflect the actual value in the current step but gives the ability to compare both models on the training set.
            # To log training state on the actual (current) training set, do this additionally
            train_loss_current = log_loss(learner.y_training, learner.predict_proba(learner.X_training))
            metrics['train_loss_current'].append(train_loss_current)

            log_metrics((used_instances + 1), learner, X_train, y_train, X_test, y_test, metrics)
            print(f"       Current train loss: {train_loss_current:.4f}")

            used_instances += 1
            retry_count = 0

        if retry_count > max_retries:
            print(f"No suitable example could be found after {max_retries} retries, so training is stopped early.")
            break

    print(f"Training time: {time.time() - start:.2f} seconds")

    return learner.estimator, metrics
