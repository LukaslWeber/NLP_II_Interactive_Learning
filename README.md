# NLP II: Interactive Learning
 
This is the corresponding Repository to my project on Interactive Learning. It belongs to the course MLNLP2 taught by Stefano Teso.

### Task 
> Choose a subset of query strategies from the modal active learning library and compare how well they perform for a shallow classifier (e.g., logistic regression) trained on MNIST and optionally for a deep neural network (e.g., a CNN with few hidden layers) trained on a reasonable subset of CIFAR 10.

---
## Additional notes:
- Technically both train scripts could be compressed into one but due to the sake of readability, they are separate!
- Upon installing the environment, a file has to be changed because modAL does not work with the numpy version that it recommends: The modAL expected_error_reduction 
  - Change "p_subsample: np.float = 1.0" to "p_subsample: float = 1.0"
  - Change "if loss is 'binary': nloss = _proba_uncertainty(refitted_proba) elif loss is 'log': nloss = _proba_entropy(refitted_proba)" to "if loss == 'binary': nloss = _proba_uncertainty(refitted_proba) elif loss == 'log': nloss = _proba_entropy(refitted_proba)"
  
---

# Citations
- modAL: T. Danka and P. Horvath, "modAL: A modular active learning framework for Python," available on arXiv: https://arxiv.org/abs/1805.00979. [Online]. Available: https://github.com/modAL-python/modAL

--- 

# Roadmap and ToDos:
- [x] Install necessary packages
- [x] Make a list of query strategies that modAL supports
- [x] Read up on modAL
- [x] Look at [Logistic Regression on MNIST](https://github.com/michelucci/Logistic-Regression-Explained/blob/master/MNIST%20with%20Logistic%20Regression%20from%20scratch.ipynb)
- [ ] Create shallow classifier on MNIST
  - [x] Loss-function: Cross Entropy/ multinomial logistic loss for multi-class classification
  - [x] L2 Regularization
  - [x] Train on all data without query strategies (To check and benchmark)
  - [ ] Be able to Train with query strategies
    - [x] Random Querying
    - [x] Uncertainty Sampling
      - [x] Classification Uncertainty
      - [x] Classification margin
      - [x] Classification Entropy
    - [x] Ranked batch-mode samlping
    - [x] expected error reduction: binary and log loss (Roy and McCallum)
    - [x] Information density
    - [X] Steam-Based Sampling
  - [x] Train on MNIST with part of the data (starting points for the query strategies)
- [x] Set random number generators to always provide the same models and pseudo-randomness
- [ ] instead of random X_pool try stratified  sampling (e.g. X_initial, _, y_initial, _ = train_test_split(X_train, y_train, test_size=0.9, stratify=y_train, random_state=random_seed))
- [ ] Shuffle Dataset maybe after each epoch?
- [ ] Maybe: [use Committee based methods](https://modal-python.readthedocs.io/en/latest/content/models/Committee.html#query-strategies)
- [ ] Cross-Validation verwenden um sicherzustellen dass es nicht an unterschiedlichen Random-splits am Dataset liegt?
- [ ] Evaluation
  - [x] Plot Accuracy
  - [x] Per-Class-Accuracy
  - [ ] Plot Per-Class-Accuracy
  - [x] Plot train error
  - [x] Plot test error
  - [x] ROC Curve
  - [x] Precision
  - [x] Recall
  - [x] F1-Score: Useful if MNIST classes are imbalanced.
  - [x] Use confusion matrices for a class-wise error analysis.
  - [ ] Graph which points (how to show this in a graph) were added, which points had the most impact
  - [x] use TüPlots
  - [ ] [Display misclassified images](https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a)
- [x] Check if all possible query strategies were used
- [ ] Implement CNN and do the whole thing with a CNN instead of the shallow classifier
  - [x] load CIFAR
  - [ ] Update n_initial parameter to set initial dataset size to something moderate
  - [x] Define CNN
  - [ ] Image processing: Zero-padding, stride, ...
  - [x] Update training routine
  - [ ] Shuffle datasets each iteration
  - [ ] initialization of weights
  - [ ] check if neuralnet has a max_iter
  - [ ] [train in batches](https://skorch.readthedocs.io/en/stable/user/FAQ.html#how-do-i-shuffle-my-train-batches
  - [ ] update the n_itiial dataset sizes


---

# Questions to Stefano: