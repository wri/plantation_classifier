## Overview
This research and code repository present a method for detection and separation of tree systems in Sentinel-2 satellite imagery. Using a transfer learning approach, learned tree features are extracted from the convolutional neural network used to produce Brandt et al.’s (2023) [Tropical Tree Cover](https://github.com/wri/sentinel-tree-cover) dataset and applied in a post-classification exercise. The application of the method is illustrated for 26 priority administrative districts throughout Ghana, given its highly heterogenous agricultural and natural landscape. The final product is a 10m resolution land use map of Ghana for the year 2020 that distinguishes between natural, monoculture and agroforestry tree systems.  

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Models](#models)
- [Results](#results)
- [Contributing](#contributing)
- [Repository Organization](#repository-organization)
- [Citations](#citations)
- [License](#license)
- [Contact](#contact)

## Contributing
1. Fork the repo
2. Create your feature branch `git checkout -b feature/fooBar`
3. Commit your changes `git commit -am 'Commit message'`
4. Push to the branch `git push origin feature/fooBar`
5. Create a new pull request

## Repository Organization
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
│   ├── stage_load_data.py          
│   ├── stage_prep_features.py      
│   ├── stage_select_and_tune.py    
│   ├── stage_train_model.py        
│   ├── stage_evaluate_model.py     
│   ├── transfer_learning.py        <- deployment script
│   │
│   ├── transfer                    <- Scripts to extract features from NN
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
│   │   ├── confusion_matrix_data.csv       
│   │   ├── confusion_matrix.png            
│   │   └── validation_visuals.py           
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
├── notebooks                           <- Jupyter notebooks           
│   ├── exploratory_data_analysis.ipynb 
│   ├── extract_features.ipynb          
│   ├── modeling_approaches.ipynb       
│   ├── mvp-pilots.ipynb                
│   ├── post_processing.ipynb           
│   ├── prototype.ipynb                 
│   ├── scaling_workflow.ipynb          
│   ├── texture_analysis.ipynb        
│   ├── training_data_eda.ipynb        
│   └── tuning-feature-selection.ipynb 
│
│
├── .gitignore                     
├── .dockerignore                  
└── .dvcignore                   
```

## Citations
Brandt, J., Ertel, J., Spore, J., & Stolle, F. (2023). Wall-to-wall mapping of tree extent in the tropics with Sentinel-1 and Sentinel-2. Remote Sensing of Environment, 292, 113574. doi:10.1016/j.rse.2023.113574

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.