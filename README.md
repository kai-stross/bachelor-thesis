# Bachelor Thesis
==============================

This repository contains the code of my bachelor thesis `Using Graph Neural Networks for Web Tracker Detection`. It includes transforming t.ex web crawl datasets into Pytoch Geometric Data Objects and then training Graph Neural Network models on the given node classification problem. 

# Installation

Install the conda environment listed in `environment.yml` with the following command:

`conda env create -f environment.yml`

Afterwards you can simply execute the source code from the `src` folder using the Jupyter Notebooks from the  `notebooks` folder.

# Project Organization

    ├── LICENSE                                             <- The project license
    ├── README.md                                           <- This README for developers using this project
    │   
    ├── data
    │   ├── interim                                         <- Intermediate data that has been transformed
    │   ├── processed                                       <- The final datasets for modeling
    │   └── raw                                             <- The original, immutable data dump from `https://github.com/t-ex-tools/t.ex`
    │
    ├── graphgym                                            <- The GNN exploration (GraphGym) platfrom from
    │                                                       `https://github.com/pyg-team/pytorch_geometric/tree/master/graphgym`
    │
    ├── models                                              <- Model class definitions and trained models
    │   ├── definitions                                     <- The class definitions of the models
    │   └── trained                                         <- The trained models
    │
    ├── notebooks                                           <- Jupyter notebooks
    │   ├── 1.0-data-preprocessing.ipynb                    <- Data preprocessing
    │   ├── 2.x-analysis-...ipynb                           <- Data analysis
    │   ├── 3.1/2.x-architecture/hyperparameters...ipynb    <- GNN model exploration, training and hyperparameter tuning
    │   └── 4.x-final-...ipynb                              <- Final GNN models evaluation and explanation
    │
    ├── references                                          <- Data dictionaries and other configurations
    │   ├── dictionary                                      <- Centrality metrics dictionary
    │   └── tex_config                                      <- t.ex feature configuration
    │   
    ├── reports                                             <- Generated analysis
    │   ├── figures                                         <- Data analysis figures
    │   ├── gcn                                             <- GCN model training logs
    │   └── variant                                         <- GNN variant training logs
    │
    ├── results                                             <- Generated results
    │   ├── gcn                                             <- GCN model results      
    │   ├── variant                                         <- GNN variant (GraphSAGE) model results
    │   └── machine learning                                <-  Machine learning results from
    │                                                       `https://github.com/t-ex-tools/t.ex-graph-2.0-classifier`
    │ 
    ├── src                                                 <- Source code for use in this project
    │   ├── __init__.py                                     <- File to make src a Python module
    │   │
    │   ├── data                                            <- Scripts to generate data
    │   │   └── preprocess.py
    │   │
    │   ├── features                                        <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models                                          <- Scripts to train models and then use trained models to make
    │   │   │                                               predictions
    │   │   ├── evaluate_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization                                   <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── environment.yml                                     <- File to install conda environment (`conda env create -f environment.yml`)
                           










--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
