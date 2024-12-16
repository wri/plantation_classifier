
import geopandas as gpd
import numpy as np
import rasterio as rs
import pandas as pd
from shapely.geometry import Point
from rasterio.features import geometry_mask
import os
import yaml


def calculate_class_distribution(raster_file):
    '''
    Get the counts and proportions of each land use class (0, 1, 2, 3)
    bincount() counts occurrences of non-negative integers. 
    Option to include zeros or no-data values, depending on bincount's 
    minlength argument (change to 4 or 256).
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
        
    return total_pixels, class_prop

def calculate_smpl_size(population, size):
    '''
    adapted from: https://github.com/flaboss/python_stratified_sampling/tree/master
    A function to compute the sample size. If not informed, a sampling 
    size will be calculated using Cochran adjusted sampling formula:
        cochran_n = (Z**2 * p * q) /e**2
        where:
            - Z is the z-value. In this case we use 1.96 representing 95%
            - p is the estimated proportion of the population which has an
                attribute. In this case we use 0.5
            - q is 1-p
            - e is the margin of error. In this case we use 3%

        This formula is adjusted as follows:
        adjusted_cochran = cochran_n / 1+((cochran_n -1)/N)
        where:
            - cochran_n = result of the previous formula
            - N is the population size
    '''
    print("Computing sample size...")
    if size is None:
        cochran_n = np.round(((1.96)**2 * 0.5 * 0.5)/ 0.03**2)
        n = np.round(cochran_n/(1+((cochran_n -1) /population)))
    elif size >= 0 and size < 1:
        n = np.round(population * size)
    elif size < 0:
        raise ValueError('Parameter "size" must be an integer or a proportion between 0 and 0.99.')
    elif size >= 1:
        n = size
    return n

def buffer_training_pts(buffer_dist, 
                        train_batches,
                        valv1):
    '''
    This function creates a buffer zone around each training plot to prevent
    overlap between the training and validation samples. First, all plot level CEO surveys
    are combined. The native plot size is calculated with a radius of 65m and added to the 
    buffer_dist. To apply the buffer in meters, the gdf is converted to a meter-based
    CRS (EPSG:3857) to ensure correct distance measurements.
    Saves the buffer zone shapefile to local dir.

    if valv1 is not none, imports the last validation schema to create 
    a buffer around pts that have already been used for validation
    '''
    df_list = []
    print(f"Creating buffer zone with {train_batches} batches")
    for i in train_batches:
        df = pd.read_csv(f'../../data/collect_earth_surveys/plantations-train-{i}/ceo-plantations-train-{i}-plot.csv')    
        df_list.append(df)
        if valv1 != None:
            val = pd.read_csv(valv1) 
            df_list.append(val)
    df_master = pd.concat(df_list, ignore_index=True)

    # Create buffer zone around each plot (calc plot radius + buffer)
    gdf_train = gpd.GeoDataFrame(df_master, 
                                geometry=gpd.points_from_xy(df_master['center_lon'], 
                                                             df_master['center_lat']), 
                                                             crs='EPSG:4326') 
    gdf_train = gdf_train.to_crs(epsg=3857)
    plot_radius = 65
    gdf_train['buffer'] = gdf_train.geometry.buffer(plot_radius + buffer_dist)
    buffer_zone = gdf_train['buffer'].unary_union
    buffer_zone = gpd.GeoSeries([buffer_zone], crs='EPSG:3857').to_crs(epsg=4326)
    buffer_zone.to_file(f'../../data/validation/buffer.shp')
    return buffer_zone


def sample_raster_by_class(raster_file, 
                           total_samples, 
                           class_proportions,
                           buffer_zone,
                           outfile):
    '''
    Overlays the provided raster with the buffer zone. Samples the mask for each land cover class 
    and calculates total_samples count of geo-referenced points given class_proporations.
    Returns a GeoDataFrame containing sampled points with geographic coordinates and raster values
    '''
    with rs.open(raster_file) as src:
        raster = src.read(1)
        crs = src.crs
        transform=src.transform

    buffer_mask = geometry_mask(buffer_zone.geometry,
                                transform=transform,
                                invert=False,  # areas INSIDE the geometries will be False
                                out_shape=src.shape)

    gdf = gpd.GeoDataFrame(columns=['geometry', 'value'], crs=crs)
    print(f"Total_samples: {total_samples}")
    for cls, proportion in class_proportions.items():
        class_mask = (raster == cls) & (buffer_mask)
        num_samples = int((proportion / 100) * total_samples)
        class_indices = np.argwhere(class_mask)

        print(f"Sampling {num_samples} points for class {cls} out of {class_indices.shape[0]} available pixels.")
        
        # Randomly sample pixel indices from the available 
        # class pixels until num_samples is reached
        geo_points = []
        while len(geo_points) < num_samples:
            sampled_indices = np.random.choice(class_indices.shape[0], 
                                            size=num_samples,
                                            replace=False)

            sampled_pixel_coords = class_indices[sampled_indices]

            # Convert sampled pixel indices to geographic coordinates
            for row, col in sampled_pixel_coords:
                lon, lat = rs.transform.xy(transform, row, col)
                point = Point(lon, lat)
                geo_points.append(point)
                    
        temp_gdf = gpd.GeoDataFrame(geometry=geo_points, crs=crs)

        # Now sample raster values at the sampled points
        temp_gdf['value'] = [raster[row, col] for row, col in sampled_pixel_coords]
        gdf = pd.concat([gdf, temp_gdf], ignore_index=True)

    gdf.to_file(outfile)
    
    return gdf

def run_validation_workflow(raster_file, 
                            outfile, 
                            buffer_distance,
                            params_path,
                            valv1=None,
                            total_samples=None,
                            ):

    '''
    Calculates distribution of validation samples per class and samples the
    provided input raster based on the number of total_samples.
    The input raster contains the following values: [0,1,2,3,255]
    Outputs a geodataframe of sample points.
    Steps:
        1. Calculates total area of map and class distribution
        2. Creates a buffer zone around training plots (and previous
        validation survey if filename provided)
        3. Performs stratified random sampling for each class

    '''
    with open(params_path) as file:
        params = yaml.safe_load(file)
    
    train_surveys = params['data_load']['ceo_survey']
    total_pixels, class_prop = calculate_class_distribution(raster_file)
    if total_samples is None:
        total_samples = calculate_smpl_size(total_pixels, size=None)
    buffer_zone = buffer_training_pts(buffer_distance, train_surveys, valv1)
    class_prop = {0: 20, 1: 25, 2: 25, 3: 30}
    sampled_points = sample_raster_by_class(raster_file, 
                                            total_samples, 
                                            class_prop,
                                            buffer_zone,
                                            outfile)
    return sampled_points