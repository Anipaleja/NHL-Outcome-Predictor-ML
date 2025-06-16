# NHL Match Outcome Predictor
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)  [![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)  [![Stars](https://img.shields.io/github/stars/anipaleja/NHL-Outcome-Predictor-ML?style=social)](https://github.com/anipaleja/NHL-Outcome-Predictor-ML/stargazers)  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)  [![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)


A **work-in-progress** machine learning project that predicts the outcome of NHL games using historical match data. Built with a deep learning architecture, the system is designed to handle a wide variety of game metrics and team statistics.

**PLEASE NOTE:** There are two different branches of the repository, the main branch is a working predictor based on the averages of previous statistics and data. The Upcoming-Game-Prediction branch is for a seperate part of the project that uses the NHL API for real time statistics, although, I am trying to get the API to send the real time statistics to a `.csv` file to get the data more locally. 

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
Output should look like this: 

![NHL Rec-3](https://github.com/user-attachments/assets/0d334184-a3c3-493d-86f7-c5c5fa71544c)


<img width="1000" alt="Screenshot 2025-05-29 at 2 19 30 PM" src="https://github.com/user-attachments/assets/ac58c14e-64bf-4513-86e3-43b4eba66fb2" />


## Status
**Actively in development!** Currently resolving input encoding issues and refining preprocessing steps. Neural net architecture and training loop are functional but unvalidated.
Please feel free to contribute as needed! Submit an issue or pull request.

## License

GNU AFFERO GENERAL PUBLIC LICENSE
