# Plantation Classifier
Research method to spatially differentiate planted and natural trees using a transfer learning approach for image classification.


## Directory Structure
```
├── LICENSE
├── README.md                       <- The top-level README for developers using this project.
├── contributing.md                 <- Contribution guidelines
├── requirements.txt                <- The requirements file for reproducing the analysis environment
├── Dockerfile 
├── setup.py                        <- makes project pip installable (pip install -e .) so src can be imported
├── environment.yaml                 
├── params.yaml                      
├── config.yaml                      
├── dvc.yaml 
├── dvc.lock                        
├── src                             <- Source code for use in this project.
│   ├── __init__.py                 <- Makes src a Python module       
│   ├── stage_load_data.py          <- xx
│   ├── stage_prep_features.py      <- xx
│   ├── stage_select_and_tune.py    <- xx
│   ├── stage_train_model.py        <- xx
│   ├── stage_evaluate_model.py     <- xx
│   ├── transfer_learning.py        <- deployment script
│   │
│   ├── load_data                   <- Scripts to download or generate data
│   │   ├── __init__.py             <- Makes src a Python module 
│   │   └── s3_download.py          <- xx
│   │
│   ├── features                    <- Scripts to import and prepare raw data into features for modeling
│   │   ├── __init__.py             <- Makes src a Python module 
│   │   ├── PlantationsData.py      <- xx
│   │   ├── create_xy.py            <- xx 
│   │   ├── feature_selection.py    <- xx 
│   │   ├── texture_analysis.py     <- xx 
│   │   ├── slow_glcm.py            <- xx
│   │   └── fast_glcm.py            <- xx 
│   │    
│   ├── model                       <- Scripts to train models, perform feature selection and tune hyperparameters
│   │   ├── __init__.py             <- Makes src a Python module 
│   │   ├── train.py                <- xx           
│   │   └── tune.py                 <- xx
│   │    
│   ├── evaluation                          <- Graphics and figures from model evaluation
│   │   ├── confusion_matrix_data.csv       <- xx
│   │   ├── confusion_matrix.png            <- xx
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
│       ├── validate_io.py          <- validation/assertions of pipeline inputs and outputs
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
