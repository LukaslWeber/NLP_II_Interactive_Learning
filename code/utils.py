import os, random, pickle
import time

from modAL import ActiveLearner, Committee
from modAL.uncertainty import classifier_margin, classifier_uncertainty
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, log_loss, pairwise_distances
from sklearn.linear_model import LogisticRegression
import numpy as np
import tensorflow as tf
import torch
from sklearn.model_selection import train_test_split
from skorch import NeuralNetClassifier
from skorch.callbacks import GradientNormClipping
from torch import nn
from cnn_model import CNN

is_cnn = False
device = "cpu"


def initialize_random_number_generators(seed):
    """
    Initializes random number generators for NumPy, Python's random module, and PyTorch
    to ensure reproducibility.

    Args:
        seed (int): The seed value to initialize the random number generators.

    Notes:
        - Sets the seed for NumPy's random number generator.
        - Sets the seed for Python's built-in random module.
        - Sets the seed for PyTorch's random number generator (both CPU and CUDA).
        - deterministic behavior in PyTorch by setting `torch.backends.cudnn.deterministic = True`.
        - Disables CuDNN benchmarking (`torch.backends.cudnn.benchmark = False`)
    """
    np.random.seed(seed)
    random.seed(seed)
    # Cuda should not be used but I found it hard to produce reproducible results with
    # Pytorch models so I'll set this as a precaution
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For CUDA
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Ensure reproducibility


def load_MNIST(random_seed, validation_split=0.2):
    """
    Loads the MNIST dataset from OpenML, normalizes the pixel values, and optionally splits
    the training data into training and validation sets.

    Args:
        random_seed (int): The seed for random number generation to ensure reproducibility
                           when splitting the dataset.
        validation_split (float, optional): The proportion of the training data to be used
                                            as validation data. Defaults to 0.2.

    Returns:
        tuple: A tuple containing:
            - X_train (numpy.ndarray): Training set features.
            - y_train (numpy.ndarray): Training set labels.
            - X_test (numpy.ndarray): Test set features.
            - y_test (numpy.ndarray): Test set labels.
            - X_val (numpy.ndarray): Validation set features (empty if validation_split=0.0).
            - y_val (numpy.ndarray): Validation set labels (empty if validation_split=0.0).
            - X (numpy.ndarray): Full dataset features.
            - y (numpy.ndarray): Full dataset labels.
    """
    mnist = fetch_openml('mnist_784', version=1, parser="auto")

    X, y = mnist.data.to_numpy(), mnist.target.to_numpy()

    # Normalize as data ranges between 0 and 255
    X = X / 255.0
    # Convert type of y (aka the label) from string to integer
    y = y.astype(int)

    X_train_full, y_train_full = X[:60000], y[:60000]  # This is the default way of extracting train and test data
    X_test, y_test = X[60000:], y[60000:]

    if validation_split == 0.0:
        X_train = X_train_full
        y_train = y_train_full
        X_val, y_val = [], []
    else:
        # Create validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=validation_split, random_state=random_seed
        )

    return X_train, y_train, X_test, y_test, X_val, y_val, X, y


def log_metrics(step: int, learner, X_train, y_train, X_test, y_test, metrics: dict, is_cnn=False, device="cpu"):
    """
        Logs training and test metrics for a given learner and updates the provided metrics dictionary.

        Args:
            step (int): The current iteration or training step.
            learner: The model used for training and evaluation, supporting `predict_proba` and `predict`.
            X_train (numpy.ndarray): Training data features.
            y_train (numpy.ndarray): Training data labels.
            X_test (numpy.ndarray): Test data features.
            y_test (numpy.ndarray): Test data labels.
            metrics (dict): Dictionary storing lists of logged metrics ('train_loss', 'test_loss', 'test_acc').
            is_cnn (bool, optional): Whether the learner is a CNN model requiring `argmax` for predictions. Defaults to False.
            device (str, optional): The computing device ('cpu' or 'cuda'). Defaults to 'cpu'.

        Returns:
            None
    """
    train_loss = compute_loss(y_hat=learner.predict_proba(X_train), y_data=y_train)
    test_logits = learner.predict_proba(X_test)
    test_loss = compute_loss(y_hat=test_logits, y_data=y_test)
    if is_cnn:
        test_preds = test_logits.argmax(axis=1)
        test_accuracy = accuracy_score(y_test, test_preds)
    else:
        test_accuracy = accuracy_score(y_test, learner.predict(X_test))

    metrics['train_loss'].append(train_loss)
    metrics['test_loss'].append(test_loss)
    metrics['test_acc'].append(test_accuracy)
    print(f"After iteration {step}: \n"
          f"  - Train Loss: {train_loss:.4f} \n"
          f"  - Test Loss: {test_loss:.4f} \n"
          f"  - Test Accuracy: {test_accuracy}")
    return


########## Utility methods regarding models ##########
def create_log_reg_model(model_params, random_seed, device="cpu"):
    """
    Creates a Logistic Regression model

    :param model_params:
    :param random_seed:
    :param device:
    Returns:
        Logistic Regression model
    """
    return LogisticRegression(solver=model_params['solver'],
                              penalty=model_params["regularization"],
                              C=model_params["regularization_strength"],
                              multi_class='multinomial',
                              max_iter=model_params["max_iterations_per_epoch"],
                              tol=model_params["early_stopping_tol"],
                              warm_start=True,
                              random_state=random_seed)


def save_model_and_metrics(experiment: str, dataset_name: str, name: str, model, metrics: dict, base_path=None):
    """
    Saves model and the training metrics dictionary

    :param experiment:
    :param dataset_name:
    :param name:
    :param model:
    :param metrics:
    :param base_path:
    :return:
    """
    if base_path is None:
        base_path = os.path.join("../results", dataset_name, f"exp{experiment}")

    # save model
    save_file(os.path.join(base_path, "models", f"{name}_model.pkl"), model)

    # save metrics
    save_file(os.path.join(base_path, "metrics", f"{name}_metrics.pkl"), metrics)


def load_model_and_metrics(experiment: str, dataset_name: str, name: str, base_path=None):
    """
    Loads model and training metrics dictionary

    :param experiment:
    :param dataset_name:
    :param name:
    :param base_path:
    :return:
        model,
        metrics
    """
    if base_path is None:
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


########## Training scripts ##########
def random_sampling(classifier, X_pool, n_instances):
    n_samples = len(X_pool)
    query_idx = np.random.choice(range(n_samples), size=n_instances, replace=False)
    return query_idx, X_pool[query_idx]


def ranked_uc_and_dv_score(learner, X, alpha_uc_dv=0.5):
    uncertainty = classifier_uncertainty(learner, X)
    diversity = np.min(pairwise_distances(X, learner.X_training), axis=1)
    combined_scores = alpha_uc_dv * uncertainty + (1 - alpha_uc_dv) * diversity
    return combined_scores


def ranked_uc_and_dv_query(learner, X, n_instances=1):
    uc_dv_scores = ranked_uc_and_dv_score(learner, X)
    # Sort them in descending order
    ranked_indices = np.argsort(uc_dv_scores)[::-1]
    selected_indices = ranked_indices[:n_instances]
    selected_instances = X[selected_indices]

    return selected_indices, selected_instances


def extract_datasets_from_dict(datasets):
    dataset_name = datasets['dataset_name']
    X_initial, y_initial = datasets['X_initial'], datasets['y_initial']
    X_train, y_train = datasets['X_train'], datasets['y_train']
    X_val, y_val = datasets['X_val'], datasets['y_val']
    X_test, y_test = datasets['X_test'], datasets['y_test']
    pool_idx = datasets['pool_idx']
    X_pool, y_pool = X_train[pool_idx], y_train[pool_idx]
    return (dataset_name,
            (X_initial, y_initial),
            (X_train, y_train),
            (X_test, y_test),
            (X_val, y_val),
            pool_idx, (X_pool, y_pool))


def compute_loss(y_hat, y_data):
    if is_cnn:
        criterion = torch.nn.CrossEntropyLoss().to(device)
        loss = criterion(torch.tensor(y_hat, dtype=torch.float32, device=device),
                         torch.tensor(y_data, dtype=torch.long, device=device)).item()
    else:
        loss = log_loss(y_data, y_hat)
    return loss


def change_max_epochs(learner, new_max):
    if is_cnn:
        print(learner.estimator.get_params()['max_epochs'])
        learner.estimator.set_params(max_epochs=new_max)
        print(learner.estimator.get_params()['max_epochs'])
    else:
        learner.estimator.set_params(max_iter=new_max)


def train_active_learner(model_params, query_strat, n_query_instances: int, n_query_epochs: int, random_seed: int,
                         datasets, create_model, n_iter, patience, device="cpu"):
    """
    Method to train a pool-based active learner

    :param model_params:
    :param query_strat:
    :param n_query_instances:
    :param n_query_epochs:
    :param random_seed:
    :param datasets:
    :param create_model:
    :param n_iter:
    :param patience:
    :param device:
    :return: model, training metrics
    """
    device = device
    initialize_random_number_generators(seed=random_seed)
    # Dict to log metrics
    metrics = {'queries': [], 'train_loss': [], 'test_loss': [], 'test_acc': []}
    #Variables for early stopping
    #best_loss = np.inf
    #no_improvement_count = 0
    # Get datasets from dictionary
    (dataset_name, (X_initial, y_initial), (X_train, y_train), (X_test, y_test),
     (X_val, y_val), pool_idx, (X_pool, y_pool)) = extract_datasets_from_dict(datasets)
    # Convenience variable to indicate if the CNN is used.
    global is_cnn
    is_cnn = (dataset_name == "CIFAR")
    # Create model with the respective method
    model = create_model(model_params, random_seed=random_seed, device=device)
    # Passing X_training and y_training to the ActiveLearner automatically calls the fit method for log_reg with these data points
    learner = ActiveLearner(estimator=model,
                            query_strategy=query_strat,
                            X_training=X_initial,
                            y_training=y_initial)
    # After initial training, only fit n_iter times after querying new saples
    change_max_epochs(learner, n_iter)

    start = time.time()
    for epoch in range(
            n_query_epochs):  # epochs=n_queries to have both models trained on the same number of overall epochs
        query_idx, query_inst = learner.query(X_pool, n_instances=n_query_instances)  # Query samples
        print(y_pool[query_idx])
        learner.teach(X_pool[query_idx], y_pool[query_idx], only_new=False)  # Simulate labeling
        X_pool, y_pool = (np.delete(X_pool, query_idx, axis=0),
                          np.delete(y_pool, query_idx, axis=0))  # Remove queried point(s) from the unlabeled pool
        # log_metrics logs train loss for the whole train dataset which doesn't reflect the current loss value
        # in the current step but gives the ability to compare both models on the training set.
        # To log training state on the actual (current) training set, do this additionally
        #val_loss = compute_loss(y_hat=learner.predict_proba(X_val), y_data=y_val)

        metrics['queries'].append(query_idx)
        log_metrics(epoch, learner, X_train, y_train, X_test, y_test, metrics)
        print(f"  - number of train samples: {len(learner.X_training)}")

        # This is for early stopping.
        #if val_loss < best_loss:
        #    best_loss = val_loss
        #    no_improvement_count = 0
        #else:
        #    no_improvement_count += 1

        #if no_improvement_count >= patience:
        #    print("Early stopping triggered.")
        #    break
    print(f"Training time: {time.time() - start:.2f} seconds")

    return learner.estimator, metrics


def change_max_committee_epochs(_committee, n_iter, is_cnn):
    for l in _committee.learner_list:
        if is_cnn:
            l.estimator.set_params(max_epochs=n_iter)
        else:
            l.estimator.set_params(max_iter=n_iter)


def create_committee(n_learners, n_initial, X_initial, y_initial, create_model, model_params, query_strat, random_seed,
                     device="cpu"):
    """
    Creates a committee of models

    :param n_learners:
    :param n_initial:
    :param X_initial:
    :param y_initial:
    :param create_model:
    :param model_params:
    :param query_strat:
    :param random_seed:
    :param device:
    :return:
    """
    learners_list = list()
    for member_idx in range(n_learners):
        np.random.seed(
            random_seed + member_idx)  # random_seed+member_idx is a robust way to always get the same subsets and starting points across query methods
        _sample_idx = np.random.choice(n_initial, size=int(0.6 * n_initial), replace=False)
        _X_train = X_initial[_sample_idx]
        _y_train = y_initial[_sample_idx]
        model = create_model(model_params, random_seed=random_seed + member_idx, device=device)
        # Passing X_training and y_training to the ActiveLearner automatically calls the fit method for log_reg with these data points
        learner = ActiveLearner(estimator=model,
                                query_strategy=query_strat,
                                X_training=X_initial,
                                y_training=y_initial)
        learners_list.append(learner)
    return Committee(learner_list=learners_list, query_strategy=query_strat)


def train_committee_learner(model_params, query_strat, n_query_instances: int, n_query_epochs: int, random_seed: int,
                            datasets, create_model, n_iter, n_learners, patience, device="cpu"):
    """
    Trains a committee of models and logs metrics. Like train_active_learner but with a committee!

    :param model_params:
    :param query_strat:
    :param n_query_instances:
    :param n_query_epochs:
    :param random_seed:
    :param datasets:
    :param create_model:
    :param n_iter:
    :param n_learners:
    :param patience:
    :param device:
    :return: committee, metrics
    """
    device = device
    initialize_random_number_generators(seed=random_seed)
    # Dict to log metrics
    metrics = {'queries': [], 'train_loss': [], 'train_loss_current': [], 'test_loss': [], 'test_acc': []}
    # Variables for early stopping
    #best_loss = np.inf
    #no_improvement_count = 0
    # Get datasets from dictionary
    (dataset_name, (X_initial, y_initial), (X_train, y_train), (X_test, y_test),
     (X_val, y_val), pool_idx, (X_pool, y_pool)) = extract_datasets_from_dict(datasets)
    # Convenience variable to indicate if the CNN is used.
    global is_cnn
    is_cnn = (dataset_name == "CIFAR")
    # Create model with the respective method
    _committee = create_committee(n_learners=n_learners, n_initial=len(X_initial), X_initial=X_initial,
                                  y_initial=y_initial, create_model=create_model, model_params=model_params,
                                  query_strat=query_strat, random_seed=random_seed, device=device)
    # After initial training, only fit n_iter times after querying new saples
    change_max_committee_epochs(_committee, n_iter, is_cnn)

    start = time.time()
    for epoch in range(
            n_query_epochs):  # epochs=n_queries to have both models trained on the same number of overall epochs
        query_idx, query_inst = _committee.query(X_pool, n_instances=n_query_instances)  # Query samples
        _committee.teach(X_pool[query_idx], y_pool[query_idx], only_new=False, bootstrap=False)
        X_pool, y_pool = (np.delete(X_pool, query_idx, axis=0),
                          np.delete(y_pool, query_idx, axis=0))  # Remove queried point(s) from the unlabeled pool
        # log_metrics logs train loss for the whole train dataset which doesn't reflect the current loss value
        # in the current step but gives the ability to compare both models on the training set.
        # To log training state on the actual (current) training set, do this additionally
        #val_loss = compute_loss(y_hat=_committee.predict_proba(X_val), y_data=y_val)

        metrics['queries'].append(query_idx)
        log_metrics(epoch, _committee, X_train, y_train, X_test, y_test, metrics)
        print(f"  - number of train samples: {len(_committee.learner_list[0].X_training)}")

        # This is for early stopping.
        #if val_loss < best_loss:
        #    best_loss = val_loss
        #    no_improvement_count = 0
        #else:
        #    no_improvement_count += 1

        #if no_improvement_count >= patience:
        #    print("Early stopping triggered.")
        #    break
    print(f"Training time: {time.time() - start:.2f} seconds")

    return _committee, metrics


########## UNUSED METHODS THAT CAN BE USED FOR FURTHER EXPERIMENTS ##########
### For instance, when a CNN should be tested ###
### or stream-based active learning should be emulated ###
def load_CIFAR(random_seed, validation_split=0.2):
    (X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalize pixel values as they are rgb values between 0 and 255
    X_train_full = X_train_full.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Transpose the datasets to have the channel (3) first and then height (1) and width (2)
    X_train_full = np.transpose(X_train_full, (0, 3, 1, 2))
    X_test = np.transpose(X_test, (0, 3, 1, 2))

    # Convert labels to 1D array
    y_train_full, y_test = y_train_full.flatten(), y_test.flatten()

    X_whole = np.concatenate((X_train_full, X_test), axis=0)
    y_whole = np.concatenate((y_train_full, y_test), axis=0)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=validation_split, random_state=random_seed
    )

    # Check if concatenation of train and test doesn't change data order
    assert (X_whole[0:50000] == X_train_full).all(), "X_train does not match the first 50000 samples of X_whole."
    assert (X_whole[50000:] == X_test).all(), "X_test does not match the last samples of X_whole."
    assert (y_whole[0:50000] == y_train_full).all(), "y_train does not match the first 50000 labels of y_whole."
    assert (y_whole[50000:] == y_test).all(), "y_test does not match the last labels of y_whole."

    return X_train, y_train, X_test, y_test, X_val, y_val, X_whole, y_whole


def create_cnn_model(model_params, random_seed, device="cpu"):
    par = model_params.copy()
    lr = par.pop("lr")
    weight_decay = par.pop("weight_decay")
    max_epochs = par.pop("max_epochs")
    batch_size = par.pop("batch_size")
    initialize_random_number_generators(random_seed)
    cnn = CNN(**par)
    gradient_clipping = GradientNormClipping(gradient_clip_value=1.0)
    return NeuralNetClassifier(cnn,
                               criterion=nn.CrossEntropyLoss,  # don't i need to use another loss?
                               optimizer=torch.optim.Adam,  # Pass the optimizer class, not an instance
                               optimizer__lr=lr,  # Set learning rate
                               optimizer__weight_decay=weight_decay,
                               max_epochs=max_epochs,
                               batch_size=batch_size,
                               train_split=None,  # this disables an internal validation split
                               verbose=3,
                               device=device,
                               warm_start=True,
                               callbacks=[gradient_clipping])

def train_active_learner_stream(model_params, query_score_fn, n_query_instances: int, query_score_threshold: float,
                                epochs: int, random_seed: int, datasets, create_model, device="cpu"):
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
