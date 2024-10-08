
import geopandas as gpd
import numpy as np
import rasterio as rs
import pandas as pd
from shapely.geometry import Point
import os
import yaml


def calculate_class_distribution(raster_file):
    '''
    Get the counts and proportions of each land use class (0, 1, 2, 3)
    bincount() counts occurrences of non-negative integers. can optionally include 
    zeros or no-data values, depending on the minlength argument (4 or 256).

    Class distribution: {0: 269113871, 1: 1272342, 2: 138084189, 3: 99051836}
    Class proportions: {0: 53.03, 1: 0.25, 2: 27.21, 3: 19.52}
    '''    
    land_use_map = rs.open(raster_file).read(1)
    classes = [0, 1, 2, 3]
    class_dist = {}
    class_prop = {}
    counts = np.bincount(land_use_map.flatten(), minlength=3) 
    valid_counts = counts[classes]
    total_pixels = valid_counts.sum()
    for value in classes:
        count = counts[value]
        percentage = (count / total_pixels) * 100
        class_dist[value] = count
        class_prop[value] = round(percentage,2)

    print(f"Class distribution: {class_dist}")
    print(f"Class proportions: {class_prop}")
    print(f"Total count (pixels): {total_pixels}")
        
    return class_dist, class_prop

def buffer(buffer_dist, train_batches):
    '''
    This function needs to ensure that none of the identified validation samples
    overlap with the input training plots or fall within a 1km buffer zone of the
    training samples. 
    This would have to take the plot data as input..
    '''
    # after v23 integrated have it check locally for file
    # if not os.file_exists('../data/validation/training_data_buffers.shp'):

    # if buffer file doesn't exist, create new one
    # create single df of all training samples
    df_list = []
    for i in train_batches:
        df = pd.read_csv(f'../data/collect_earth_surveys/ceo-plantations-train-v{i}/ceo-plantations-train-v{i}-plot.csv')    
        df_list.append(df)
    df_master = pd.concat(df_list, ignore_index=True)

    # print stats per class
    plantation_counts = df_master['PLANTATION'].value_counts()
    plantation_percs = df_master['PLANTATION'].value_counts(normalize=True) * 100
    for pclass, count in plantation_counts.items():
        percentage = round(plantation_percs[pclass], 2) 
        print(f"Class {pclass}: {count} training points, {percentage}% of total")

    # Create buffer zone around each plot (calc plot radius + buffer)
    plot_radius = 65
    gdf_train = gpd.GeoDataFrame(df_master, 
                                 geometry=gpd.points_from_xy(df_master['LON'], 
                                                             df_master['LAT']), 
                                                             crs='EPSG:3857') 
    
    gdf_train['buffer'] = gdf_train.geometry.buffer(plot_radius + buffer_dist)
    
    # Merge all the buffered areas into a single geometry
    buffer_zone = gdf_train['buffer'].unary_union
    buffer_zone = buffer_zone.to_crs(epsg='4326') # crs switch needed?
    buffer_zone.to_file(f'../data/validation/buffer.shp')
    
    return buffer_zone


def sample_raster_by_class(raster_file, 
                           total_samples, 
                           class_proportions,
                           buffer_zone,
                           outfile):
    '''
    Sample a raster for a specific land cover class and return geo-referenced points.
    GeoDataFrame containing sampled points with geographic coordinates and raster values
    '''
    with rs.open(raster_file) as src:
        raster = src.read(1)  # Read the first band
        transform = src.transform
        crs = src.crs

    gdf = gpd.GeoDataFrame(columns=['geometry', 'value'], crs=crs)

    # overlay the buffer with the raster and then sample?
    # or only append points that are outside the buffer zone?

    for cls, proportion in class_proportions.items():
        class_mask = (raster == cls)
        num_samples = int((proportion / 100) * total_samples)
        class_indices = np.argwhere(class_mask)
        
        print(f"Sampling {num_samples} points for class {cls} out of {class_indices.shape[0]} available pixels.")
        
        # Randomly sample pixel indices from the available class pixels
        sampled_indices = np.random.choice(class_indices.shape[0], 
                                        size=num_samples,
                                        replace=False)
        sampled_pixel_coords = class_indices[sampled_indices]
        
        # Convert sampled pixel indices to geographic coordinates
        geo_points = []
        for row, col in sampled_pixel_coords:
            lon, lat = rs.transform.xy(transform, row, col)
            geo_points.append(Point(lon, lat))
        
        temp_gdf = gpd.GeoDataFrame(geometry=geo_points, crs=crs)
        
        # Sample raster values at the sampled points
        temp_gdf['value'] = [raster[row, col] for row, col in sampled_pixel_coords]
        gdf = pd.concat([gdf, temp_gdf], ignore_index=True)

    gdf.to_file(outfile)
    
    return gdf

def run_validation_workflow(raster_file, 
                            total_samples, 
                            outfile, 
                            buffer_distance,
                            params_path):

    '''
    Calculates class distribution and samples the
    provided input raster based on the number of total_samples.
    The input raster contains the following values: [0,1,2,3,255]
    Outputs a geodataframe of sample points.
    Steps:
        1. Calculates total area of map and class distribution
        2. Performs stratified random sampling
    '''
    with open(params_path) as file:
        params = yaml.safe_load(file)
        
    train_surveys = params['data_load']['ceo_survey']
    class_dist, class_prop = calculate_class_distribution(raster_file)
    buffer_zone = buffer(buffer_distance, train_surveys)
    sampled_points = sample_raster_by_class(raster_file, 
                                            total_samples, 
                                            class_prop,
                                            buffer_zone,
                                            outfile)
    return sampled_points



## Legacy ##


# def stratified_random_sample(land_use_map, 
#                              total_samples, 
#                              class_distribution, 
#                              class_proportions):
#     '''
#     Perform stratified random sample using np.random
#     returns pixel values 
#     '''
#     # option to flatten to 1d array which is faster
#     # need to convert 1d indices back to 2d using np.unravel_index()
#     sample_points = []  

#     # Samples per class based on proportion
#     for cls, proportion in class_proportions.items():
#         cls_sample_count = int((proportion/100) * total_samples)
#         cls_mask = land_use_map[land_use_map == cls]
#         print(f"Class {cls}: sampling {cls_sample_count} of {cls_mask.size} pixels available.")
        
#         sampled_indices = np.random.choice(cls_mask, 
#                                            size=cls_sample_count, 
#                                            replace=False
#                                           )
#         sample_points.extend(sampled_indices.tolist())
    

#     return sample_points


# def stratified_random_sample_new(land_use_map, 
#                              total_samples, 
#                              class_proportions):
#     '''
#     Perform stratified random sample using np.random.choice
#     returns the indices of the random samples (rather than the pixel values)
#     '''
#     sample_points = []  # List to store sampled indices

#     # Flatten the land use map for easier sampling (optional)
#     flattened_map = land_use_map.flatten()
#     n_rows, n_cols = land_use_map.shape
    
#     # Samples per class based on proportion
#     for cls, proportion in class_proportions.items():
#         # Calculate the number of samples for the current class
#         cls_sample_count = int((proportion / 100) * total_samples)
        
#         # Find the indices where the land use map matches the current class
#         cls_indices = np.argwhere(flattened_map == cls).flatten()
#         total_cls_pixels = cls_indices.size
        
#         print(f"Class {cls}: sampling {cls_sample_count} of {total_cls_pixels} pixels available.")
        
#         # Ensure that we don't sample more than the available number of pixels
#         if cls_sample_count > total_cls_pixels:
#             cls_sample_count = total_cls_pixels
        
#         # Randomly sample the indices
#         sampled_indices = np.random.choice(cls_indices, size=cls_sample_count, replace=False)
        
#         # Convert the flat indices back to 2D indices (row, col)
#         sampled_points_2d = np.unravel_index(sampled_indices, (n_rows, n_cols))
        
#         # Append the 2D indices to the sample_points list
#         sample_points.extend(zip(sampled_points_2d[0], sampled_points_2d[1]))
    
#     return sample_points