import numpy as np
import hickle as hkl
import pytest


##### GROUP 4
# separate test file for testing functions are performing as expected

# ideas:
## given input 10, it should be scaled to output 0
## apply it to a test numpy array
## if the array is sorted before and after, its logical
## reshaping - should always be multiple of xx and dimensions should always be smaller
## For the adjust_shape assertions, ensure s1.shape[1:-1] == s2_10.shape[1:-1] == s2_20.shape[1:-1] == dem.shape