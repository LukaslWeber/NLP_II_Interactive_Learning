# NLP II: Interactive Learning
 
This is the corresponding Repository to my project on Interactive Learning. It belongs to the course MLNLP2 taught by Stefano Teso.

### Task 
> Choose a subset of query strategies from the modal active learning library and compare how well they perform for a shallow classifier (e.g., logistic regression) trained on MNIST and optionally for a deep neural network (e.g., a CNN with few hidden layers) trained on a reasonable subset of CIFAR 10.

---
# Installation
- I used a venv and can only test it on Windows. In case it doesn't work, feel free to write me a Mail (lukas.weber@studenti.unitn.it)
- Install with the "requirements.txt" file in the repo with "pip install -r requirements.txt"
---
## Additional notes:
- "Shallow classifier" describes the file in which the Logistic Regression models are trained. They are used in the report. "Deep classifier" describes the CNN files which are not used in the paper but left in the repo for further experiments. 
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
- [x] Create shallow classifier on MNIST
  - [x] Loss-function: Cross Entropy/ multinomial logistic loss for multi-class classification
  - [x] L2 Regularization
  - [x] Train on all data without query strategies (To check and benchmark)
  - [x] Be able to Train with query strategies
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
- [x] [use Committee based methods](https://modal-python.readthedocs.io/en/latest/content/models/Committee.html#query-strategies)
- [x] Cross-Validation to search for parameters
- [x] Evaluation
  - [x] Plot Accuracy
  - [x] Per-Class-Accuracy
  - [x] Plot train error
  - [x] Plot test error
  - [x] ROC Curve
  - [x] Precision
  - [x] Recall
  - [x] F1-Score
  - [x] Use confusion matrices for a class-wise error analysis.
  - [x] use TüPlots
- [x] Check if all possible query strategies were used
- [x] Implement CNN and do the whole thing with a CNN instead of the shallow classifier
  - [x] load CIFAR
  - [x] Update n_initial parameter to set initial dataset size to something moderate
  - [x] Define CNN
  - [x] Image processing: Zero-padding, stride, ...
  - [x] Update training routine
  - [x] Shuffle datasets each iteration
  - [x] initialization of weights