# NHL Match Outcome Predictor

A **work-in-progress** machine learning project that predicts the outcome of NHL games using historical match data. Built with a deep learning architecture, the system is designed to handle a wide variety of game metrics and team statistics.

**NOTE:** There are two different branches of the repository, the main branch is a working predictor based on the averages of previous statistics and data. 

## Technical Overview

This project leverages a custom-tuned deep feedforward neural network architecture with dense layers, ReLU activations, and dropout regularization to mitigate overfitting. Data undergoes extensive preprocessing including label encoding of categorical features and MinMax normalization. Hyperparameter tuning is manually configured for learning rate, batch size, and hidden layer dimensionality. The training pipeline utilizes TensorFlow's Keras API and integrates scikit-learn utilities for efficient data transformation. Future iterations will explore cross-validation, model persistence with HDF5, and potentially Bayesian optimization for hyperparameter search.

## Features

- TensorFlow-based neural network for classification
- Data preprocessing with Pandas and scikit-learn
- Label encoding and feature scaling
- Modular codebase with support for further tuning and experimentation
- Train/test split for model evaluation

## Setup

```bash
pip install -r requirements.txt
python main.py
```

## Status
**Actively in development!** Currently resolving input encoding issues and refining preprocessing steps. Neural net architecture and training loop are functional but unvalidated.
Please feel free to contribute as needed! Submit an issue or pull request.

## License

MIT License
