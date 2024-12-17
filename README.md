# NLP II: Interactive Learning
 
This is the corresponding Repository to my project on Interactive Learning. It belongs to the course MLNLP2 taught by Stefano Teso.

### Task 
> Choose a subset of query strategies from the modal active learning library and compare how well they perform for a shallow classifier (e.g., logistic regression) trained on MNIST and optionally for a deep neural network (e.g., a CNN with few hidden layers) trained on a reasonable subset of CIFAR 10.


---

# Citations
- modAL: T. Danka and P. Horvath, "modAL: A modular active learning framework for Python," available on arXiv: https://arxiv.org/abs/1805.00979. [Online]. Available: https://github.com/modAL-python/modAL

--- 

# Roadmap and ToDos:
- [ ] Install necessary packages
- [ ] Make a list of query strategies that modAL supports
- [ ] Read up on modAL
- [ ] Look at [Logistic Regression on MNIST](https://github.com/michelucci/Logistic-Regression-Explained/blob/master/MNIST%20with%20Logistic%20Regression%20from%20scratch.ipynb)
- [ ] Create shallow classifier on MNIST
  - [ ] Loss-function: Cross Entropy? mean_absolute_percentage_error? MSE?
  - [ ] Train on all data without query strategies (To check and benchmark)
  - [ ] Train with query strategies
    - [ ] Random Querying
    - [ ] Bayesian Optimization
  - [ ] Train on MNIST with part of the data (starting points for the query strategies)
- [ ] Evaluation
  - [ ] Accuracy
  - [ ] Percentage error
  - [ ] Maybe even compare to state of the art [from the Benchmark](https://paperswithcode.com/sota/image-classification-on-mnist?metric=Trainable%20Parameters)
  - [ ] Maybe Explainability
- [ ] Implement CNN and do the whole thing with a CNN instead of the shallow classifier
- [ ] Rework citations
  - [ ] MNIST
  - [ ] CIFAR 
  - [ ] Adapt Structure/ Looks