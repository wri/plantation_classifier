# Plantation Classifier
Research method to spatially differentiate planted and natural trees using a transfer learning approach for image classification.


## Directory Structure
```
├── LICENSE
├── README.md                       <- The top-level README for developers using this project.
├── contributing.md                 <- Contribution guidelines
├── requirements.txt                <- The requirements file for reproducing the analysis environment, e.g.
│                                      generated with `pip freeze > requirements.txt`
│
├── setup.py                        <- makes project pip installable (pip install -e .) so src can be imported
├── environment.yaml                <- xx 
├── params.yaml                     <- xx 
├── config.yaml                     <- xx 
├── dvc.yaml                        <- xx 
├── src                             <- Source code for use in this project.
│   ├── __init__.py                 <- Makes src a Python module       
│   ├── stage_load_data.py          <- xx
│   ├── stage_prep_features.py      <- xx
│   ├── stage_train_model.py        <- xx
│   ├── stage_evaluate.py           <- xx
│   ├── transfer_learning.py        <- deployment script
│   │
│   ├── load_data                   <- Scripts to download or generate data
│   │   ├── __init__.py             <- Makes src a Python module 
│   │   └── s3_download.py          <- xx
│   │
│   ├── features                    <- Scripts to import and prepare raw data into features for modeling
│   │   ├── __init__.py             <- Makes src a Python module 
│   │   ├── create_xy.py            <- xx 
│   │   ├── texture_analysis.py     <- xx 
│   │   ├── slow_glcm.py            <- xx
│   │   └── fast_glcm.py            <- xx 
│   │    
│   ├── model                       <- Scripts to train models, perform feature selection and tune hyperparameters
│   │   ├── __init__.py             <- Makes src a Python module 
│   │   ├── ptype_run_preds.py      <- will be removed
│   │   ├── score_classifier.py     <- will be removed
│   │   ├── feature_selection.py    <- xx
│   │   ├── train.py                <- xx           
│   │   └── tune.py                 <- xx
│   │    
│   ├── evaluation                         <- Graphics and figures from model evaluation
│   │   ├── confusion_matrix_data.csv       <- xx
│   │   ├── confusion_matrix.png            <- xx
│   │   ├── metrics.json                    <- xx
│   │   └── validation_visuals.py           <- creates confusion matrix
│   │
│   └── utils                       <- Scripts for utility functions
│       ├── __init__.py             <- Makes src a Python module 
│       ├── cloud_removal.py        <- xx 
│       ├── interpolation.py        <- xx  
│       ├── proximal_steps.py       <- xx  
│       ├── indices.py              <- xx  
│       ├── logs.py                 <- xx  
│       ├── preprocessing.py        <- xx  
│       ├── validate_io.py          <- xx
│       ├── quick_viz.py            <- handy visualizations for geospatial data
│       └── mosaic.py               <- xx 
│
├── notebooks                           <- Jupyter notebooks.           
│   ├── exploratory_data_analysis.ipynb <- xx
│   ├── extract_features.ipynb          <- xx
│   ├── modeling_approaches.ipynb       <- xx
│   ├── mvp-pilots.ipynb                <- xx
│   ├── post_processing.ipynb           <- xx
│   ├── prototype.ipynb                 <- xx
│   ├── scaling_workflow.ipynb          <- xx
│   ├── texture_analysis.ipynb          <- xx
│   ├── training_data_eda.ipynb          <- xx
│   └── tuning-feature-selection.ipynb  <- xx
│
│
├── .gitignore                     <- xx
├── .dockerignore                  <- xx
└── .dvcignore                     <- xx
```

## Contributing
We welcome contributions! [See the docs for guidelines](https://github.com/wri/plantation_classifier/blob/dvc/contributing.md).
