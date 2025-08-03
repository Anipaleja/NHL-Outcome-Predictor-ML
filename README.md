# NHL Outcome Predictor
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)  [![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)  [![Stars](https://img.shields.io/github/stars/anipaleja/NHL-Outcome-Predictor-ML?style=social)](https://github.com/anipaleja/NHL-Outcome-Predictor-ML/stargazers)  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)  [![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

A machine learning system that predicts NHL game outcomes using advanced AI architectures. Built with transformer neural networks, quantum-inspired ensembles, and comprehensive feature engineering for professional-grade accuracy.

## Features

- **Transformer Neural Networks**: Multi-head attention mechanisms for complex pattern recognition
- **Quantum-Inspired Ensembles**: Advanced ensemble methods with LightGBM, XGBoost, and CatBoost
- **261 Advanced Features**: Comprehensive feature engineering from team statistics to venue analysis
- **Interactive Web Interface**: Beautiful, responsive UI for team vs team predictions
- **Real-time Analytics**: Live NHL API integration with intelligent caching
- **Venue Intelligence**: Home/away advantages and playoff mode considerations

## Technical Architecture

This system employs a sophisticated multi-layer approach:

- **Core ML Framework**: PyTorch transformers with 512-dimensional embeddings and 16 attention heads
- **Ensemble Pipeline**: LightGBM, XGBoost, CatBoost with intelligent weight optimization
- **Feature Engineering**: 43 → 261 features including rolling averages, momentum indicators, and venue analytics
- **Data Pipeline**: Robust NHL API integration with CSV fallback (62,890+ historical games)
- **Web Framework**: Flask-based interactive interface with real-time predictions

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Quick Demo
```bash
python quick_demo.py
```

### 3. Launch Web Interface
```bash
python web_interface.py
```
Then open: http://localhost:5000

### 4. Advanced Usage
```bash
python advanced_nhl_predictor.py
```
Output should look like this: 

![NHL Rec-3](https://github.com/user-attachments/assets/0d334184-a3c3-493d-86f7-c5c5fa71544c)

After installing the required prerequisites, the model initiates its self-training process, monitoring the Epoch and loss metrics as displayed below:  

<p align="center"> <img width="900" alt="Training Screenshot" src="https://github.com/user-attachments/assets/ac58c14e-64bf-4513-86e3-43b4eba66fb2" style="border-radius: 12px;" /> </p>

## Status

**Actively in development!** Currently resolving input encoding issues and refining preprocessing steps. Neural net architecture and training loop are functional but unvalidated.
Please feel free to contribute as needed! Submit an issue or pull request.

## License

GNU AFFERO GENERAL PUBLIC LICENSE

See the [license](LICENSE.md) for more details
