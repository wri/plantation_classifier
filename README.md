# Transfer Learning for Tree Cover Classification in Ghana

## Overview

This repository contains the code for a novel transfer learning-based approach to classify tree systems in Ghana using Sentinel-1 and Sentinel-2 imagery. By leveraging pre-trained convolutional neural networks and feature engineering, this method distinguishes between natural forests, monoculture plantations, and agroforestry systems—a critical step for monitoring deforestation and facilitating effective landscape management.

* Why it matters: Traditional classification methods struggle to distinguish natural and planted tree systems in cloud-prone, heterogeneous landscapes like Ghana.
* What’s new: We integrate [deep learning-based tree features](https://github.com/wri/sentinel-tree-cover) and texture analysis to improve classification accuracy.
* Who is this for? Data scientists, geospatial analysts, and researchers working with remote sensing data for land use classification.

The application of the method is illustrated for 26 priority administrative districts throughout Ghana. The final product is a 10m resolution land use map of Ghana for the year 2020 that distinguishes between natural, monoculture and agroforestry systems.  

![Pixel-based Land Use Classification Results](data/figures/image.png)

## ML Pipeline Overview

### Step 1: Data Preprocessing
- Sentinel-1/2 analysis-ready data preparation

### Step 2: Feature Engineering
- Extracts tree features from pre-trained CNN
- Computes texture metrics 

### Step 3: Model Training
- Trains CatBoost classifier
- Performs feature selection
- Performs hyperparameter tuning

### Step 4: Inference & Mapping
- Applies trained model to generate 10m-resolution land use maps
- Post processing steps

## Contributing
See our [contribution guidelines](https://github.com/wri/plantation_classifier/blob/master/contributing.md).

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.

## Repository Organization
```
├── LICENSE
├── README.md                      
├── contributing.md                  
├── requirements.txt               
├── Dockerfile                      
├── environment.yaml                 
├── params.yaml                      
├── config.yaml                      
├── dvc.yaml 
├── dvc.lock                        
├── src                                 <- Source code for use in this project.
│   ├── __init__.py                        
│   ├── stage_load_data.py          
│   ├── stage_prep_features.py      
│   ├── stage_select_and_tune.py    
│   ├── stage_train_model.py        
│   ├── stage_evaluate_model.py     
│   ├── transfer_learning.py        
│   │
│   ├── transfer                        <- Scripts/steps to perform feature extraction
│   │
│   ├── load_data                       <- Scripts to download or generate data
│   │   ├── __init__.py            
│   │   └── s3_download.py           
│   │
│   ├── features                        <- Scripts to import and prepare modeling inputs
│   │   ├── __init__.py             
│   │   ├── PlantationsData.py      
│   │   ├── create_xy.py            
│   │   ├── feature_selection.py    
│   │   ├── texture_analysis.py    
│   │   ├── slow_glcm.py            
│   │   └── fast_glcm.py            
│   │    
│   ├── model                           <- Scripts to train models, select features, tune
│   │   ├── __init__.py             
│   │   ├── train.py                   
│   │   └── tune.py               
│   │    
│   ├── evaluation                      <- Graphics and figures from model evaluation
│   │   ├── confusion_matrix_data.csv       
│   │   ├── confusion_matrix.png            
│   │   └── validation_visuals.py           
│   │
│   └── utils                           <- Scripts for utility functions
│       ├── __init__.py             
│       ├── cloud_removal.py         
│       ├── interpolation.py          
│       ├── proximal_steps.py        
│       ├── indices.py                
│       ├── logs.py                   
│       ├── preprocessing.py         
│       ├── validate_io.py          
│       ├── quick_viz.py             
│       └── mosaic.py               
│
├── notebooks                           <- Jupyter notebooks                         
│   ├── analyses         
│   ├── features     
│   ├── modeling      
│   └── training_data
│
├── .gitignore                     
├── .dockerignore                  
└── .dvcignore                   
```