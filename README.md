## Overview
This research and code repository present a method for detection and separation of tree systems in Sentinel-2 satellite imagery. Using a transfer learning approach, learned tree features are extracted from Brandt et al.’s (2023) [Tropical Tree Cover](https://github.com/wri/sentinel-tree-cover) convolutional neural network and applied in a post-classification exercise. The application of the method is illustrated for 26 priority administrative districts throughout Ghana, given its highly heterogenous agricultural and natural landscape. The final product is a 10m resolution land use map of Ghana for the year 2020 that distinguishes between natural, monoculture and agroforestry tree systems.  

## Table of Contents

- [Overview](#overview)
- [Data](#data)
- [Models](#models)
- [Results](#results)
- [Contributing](#contributing)
- [Citations](#citations)
- [License](#license)
- [Repository Organization](#repository-organization)

## Data
coming soon.

## Models
coming soon.

## Contributing
See our [contribution guidelines](https://github.com/wri/plantation_classifier/blob/master/contributing.md).

## Citations
Brandt, J., Ertel, J., Spore, J., & Stolle, F. (2023). Wall-to-wall mapping of tree extent in the tropics with Sentinel-1 and Sentinel-2. Remote Sensing of Environment, 292, 113574. doi:10.1016/j.rse.2023.113574

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
│   ├── exploratory_data_analysis.ipynb 
│   ├── extract_features.ipynb          
│   ├── modeling_approaches.ipynb       
│   ├── mvp-pilots.ipynb                
│   ├── post_processing.ipynb           
│   ├── prototype.ipynb   
│   ├── resegmentation_analysis.ipynb                
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