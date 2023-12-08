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
├── params.yaml                     <- xx contributing.md Dockerfile
├── src                             <- Source code for use in this project.
│   ├── __init__.py                 <- Makes src a Python module       
│   ├── stage_load_data.py          <- xx
│   ├── stage_featurize.py          <- xx
│   ├── stage_train.py              <- xx
│   ├── stage_evaluate.py           <- xx
│   ├── validate_io.py              <- xx
│   │
│   ├── load_data                   <- Scripts to download or generate data
│   │   ├── __init__.py             <- Makes src a Python module 
│   │   ├── identify_tiles.py       <- xx 
│   │   └── s3_download.py          <- xx
│   │
│   ├── features                    <- Scripts to turn raw data into features for modeling
│   │   ├── __init__.py             <- Makes src a Python module 
│   │   ├── large_feats.py          <- xx (update name) 
│   │   ├── prepare_data.py         <- xx (update name) 
│   │   ├── texture_analysis.py     <- xx 
│   │   ├── slow_glcm.py            <- xx
│   │   └── fast_glcm.py            <- xx 
│   │    
│   ├── model                       <- Scripts to train models and then use trained models to make predictions
│   │   ├── __init__.py             <- Makes src a Python module 
│   │   ├── run_preds.py            <- xx (update name) 
│   │   ├── train.py                <- xx (will replace run_preds)
│   │   ├── feature_selection.py    <- xx 
│   │   ├── tune.py                 <- xx 
│   │   └── score_classifier.py     <- xx
│   │
│   └── utils                       <- Scripts for utility functions
│       ├── __init__.py             <- Makes src a Python module 
│       ├── cloud_removal.py        <- xx 
│       ├── interpolation.py        <- xx  
│       ├── logs.py                 <- xx  
│       ├── utils.py                <- xx  
│       └── mosaic.py               <- xx 
│
├── notebooks                           <- Jupyter notebooks.           
│   ├── exploratory_data_analysis.ipynb <- xx
│   ├── modeling_approaches.ipynb       <- xx
│   ├── mvp-pilots.ipynb                <- xx
│   ├── post_processing.ipynb           <- xx
│   ├── prototype.ipynb                 <- xx
│   ├── scaling_workflow.ipynb          <- xx
│   ├── texture_analysis.ipynb          <- xx
│   └── tuning-feature-selection.ipynb  <- xx
│
├── reports                        <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures                    <- Generated graphics and figures to be used in reporting
│
├── .gitignore                     <- xx
├── .dockerignore                  <- xx
└── .dvcignore                     <- xx
```

## Contributing
We welcome contributions! [See the docs for guidelines](https://github.com/wri/plantation_classifier/blob/dvc/contributing.md).
