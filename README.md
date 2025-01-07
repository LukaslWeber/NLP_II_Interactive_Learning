# NLP II: Interactive Learning
 
This is the corresponding Repository to my project on Interactive Learning. It belongs to the course MLNLP2 taught by Stefano Teso.

### Task 
> Choose a subset of query strategies from the modal active learning library and compare how well they perform for a shallow classifier (e.g., logistic regression) trained on MNIST and optionally for a deep neural network (e.g., a CNN with few hidden layers) trained on a reasonable subset of CIFAR 10.


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
    - [ ] Steam-Based Sampling
  - [x] Train on MNIST with part of the data (starting points for the query strategies)
  - [ ] Use pretrained net and comprare how well finetuning with and without modAL works
- [x] Set random number generators to always provude the same models
- [ ] instead of random X_pool try stratified  sampling (e.g. X_initial, _, y_initial, _ = train_test_split(X_train, y_train, test_size=0.9, stratify=y_train, random_state=random_seed))
- [ ] Shuffle Dataset maybe after each epoch?
- [ ] Maybe: [use Committee based methods](https://modal-python.readthedocs.io/en/latest/content/models/Committee.html#query-strategies)
- [ ] Cross-Validation verwenden um sicherzustellen dass es nicht an unterschiedlichen Random-splits am Dataset liegt?
- [ ] Evaluation
  - [ ] Accuracy
  - [ ] Percentage error
  - [ ] ROC Curve
  - [ ] Precision, Recall, F1-Score: Useful if MNIST classes are imbalanced.
  - [ ] Learning Curve: Plot accuracy against the number of training samples queried.
  - [ ] Maybe even compare to state of the art [from the Benchmark](https://paperswithcode.com/sota/image-classification-on-mnist?metric=Trainable%20Parameters)
  - [ ] Maybe Explainability
  - [ ] Use confusion matrices for a class-wise error analysis.
  - [ ] Graph which points (how to show this in a graph) were added, which points had the most impact
  - [ ] Maybe: use TüPlots
- [ ] Check if all possible query strategies were used
- [ ] Implement CNN and do the whole thing with a CNN instead of the shallow classifier
- [ ] Rework citations
  - [ ] MNIST
  - [ ] CIFAR 
  - [ ] Adapt Structure/ Looks
- [ ] [Display misclassified images](https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a)


- [ ] Disagreement Sampling (for classifiers) (für Committees gedacht)
- [ ] Acquisition Functions sind vermutlich raus, weil sie einen BayesianOptimizer erfordern und nicht mit einem ActiveLearner funktionieren
   - [ ] Probability of Improvement
   - [ ] Expected Improvement
   - [ ] Upper Confidence Bound

---

# Questions to Stefano Teso:
- Should I use a OneVsRestClassifier, i.e. should each class be treated as a binary classification problem?