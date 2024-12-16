## Overview
Monitoring ecosystem services, commodity-driven deforestation and progress towards international restoration commitments requires separate treatment of natural and agricultural trees in earth observation datasets. Satellite-based remote sensing data, combined with machine learning techniques, can offer a cost effective, automated, and transparent option for monitoring these systems at scale. In Ghana, this effort is complicated by persistent cloud cover, haze, and the intrinsic heterogeneity of agroforestry systems.
The objective of this work is to spatially differentiate tree cover into 4 types of systems, or land use classes, using a transfer learning approach. The application of the method is illustrated for 26 priority administrative districts throughout Ghana, given its highly heterogenous agricultural and natural landscape. The final product is a 10m resolution land use map of Ghana for the year 2020 that distinguishes between natural, monoculture and agroforestry systems.  
The results highlight the value in incorporating texture information and extracted tree features from Brandt et al.’s (2023) [Tropical Tree Cover](https://github.com/wri/sentinel-tree-cover) to improve pixel-based classification accuracy. We identify [xx] ha of planted area across 26 districts in Ghana in the year 2020. Land use maps that disaggregate planted and natural tree cover can facilitate effective decision-making for integrated landscape management plans and restoration interventions.

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Data & Model](#data-and-model)
- [Contributing](#contributing)
- [License](#license)
- [Repository Organization](#repository-organization)
- [Citations](#citations)

## Data and Model
The data and model will be released following publication of our technical note in 2025.

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

## Citations
Brandt, J., Ertel, J., Spore, J., & Stolle, F. (2023). Wall-to-wall mapping of tree extent in the tropics with Sentinel-1 and Sentinel-2. Remote Sensing of Environment, 292, 113574. doi:10.1016/j.rse.2023.113574