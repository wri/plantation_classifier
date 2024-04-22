
import numpy as np
from scipy import ndimage


def remove_small_patches(arr, thresh):
    
    '''
    Finds patches of of size thresh in a given array.
    Label these features using ndimage.label() and counts
    pixels for each label. If the count doesn't meet provided
    threshold, make the label 0. Return an updated array

    (option to add a 3x3 structure which considers features connected even 
    if they touch diagonally - probably dnt want this)
    
    '''

    # creates arr where each unique feature (non zero value) has a unique label
    # num features are the number of connected patches
    labeled_array, num_features = ndimage.label(arr)

    # get pixel count for each label
    label_size = [(labeled_array == label).sum() for label in range(num_features + 1)]

    for label,size in enumerate(label_size):
        if size < thresh:
            arr[labeled_array == label] = 0
    
    return arr

def cleanup_noisy_zeros(preds, ttc_treecover):
    '''
    One option for correcting false positives in the predicted "no tree" class.
    Replaces zero predictions with a more representative value calculated 
    using a median filter.
    the median filter kernel represents the window of pixels that will be considered,
    for example:
        5x5 Kernel: includes the target pixel and the two layers of 
        neighboring pixels. It smooths data over a wider area and is 
        more effective in reducing noise but might also remove more detail.
        (better for concentrated, large scale)
        3x3 Kernel: smaller kernel focuses on the target pixel and its 
        immediate neighbors. It provides less smoothing, retains more detail, 
        and is less aggressive in noise reduction. (better for dispersed 
        smallholder systems)
    
    '''
    # create a boolean mask in areas where preds is 0 but tree cover >10
    is_fp_zero = np.logical_and(preds == 0, ttc_treecover > .10)

    # apply median filter on the array
    preds_flt = ndimage.median_filter(np.copy(preds), 5)

    # Replace noisy zeros in the original preds with the filtered values
    preds[is_fp_zero] = preds_flt[is_fp_zero]

    return preds


def clean_tile(arr: np.array, feature_select: list, ttc: np.array):

    '''
    Applies the no data and no tree flag if TTC tree cover predictions are used
    in feature selection. The NN produces a float32 continuous prediction.

    Performs a connected component analysis to remove positive predictions 
    where the connected pixel count is < thresh. 

    preds = preds * (ttc_treecover > .10)
    '''
    # create no data and no tree flag (boolean mask)
    # where TML probability is 255 or 0, pass along to preds
    # note that the feats shape is (x, x, 65)
    no_data_flag = ttc[...,0] == 255.
    no_tree_flag = ttc[...,0] <= 0.1

    # FLAG: this step requires feature selection
    if 13 in feature_select:
        arr[no_data_flag] = 255.
        arr[no_tree_flag] = 0.

    # postprocess_mono = remove_small_patches(arr == 1, thresh = 20)
    # postprocess_af = remove_small_patches(arr == 2, thresh = 15)
    
    # # multiplying by boolean will turn every False into 0 
    # # and keep every True as the original label
    # arr[arr == 1] *= postprocess_mono[arr == 1]
    # arr[arr == 2] *= postprocess_af[arr == 2]

    # # clean up pixels in the non-tree class
    output = cleanup_noisy_zeros(arr, ttc[...,0])
  
    # del postprocess_af, postprocess_mono

    return output