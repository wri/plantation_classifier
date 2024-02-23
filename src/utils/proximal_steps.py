import datetime
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import scipy.sparse as sparse
from collections import Counter


# Documentation


def calculate_proximal_steps(date: int, satisfactory: list):
    """Returns proximal steps that are cloud and shadow free
         Parameters:
          date (int): current time step
          satisfactory (list): time steps with no clouds or shadows
         Returns:
          arg_before (str): index of the prior clean image
          arg_after (int): index of the next clean image
    """
    arg_before, arg_after = None, None
    if date > 0:
        idx_before = satisfactory - date
        arg_before = idx_before[np.where(idx_before < 0, idx_before,
                                         -np.inf).argmax()]
    if date < np.max(satisfactory):
        idx_after = satisfactory - date
        arg_after = idx_after[np.where(idx_after > 0, idx_after,
                                       np.inf).argmin()]
    if not arg_after and not arg_before:
        arg_after = date
        arg_before = date
    if not arg_after:
        arg_after = arg_before
    if not arg_before:
        arg_before = arg_after
    return arg_before, arg_after


def calculate_proximal_steps_two(date: int, satisfactory: list):
    """Returns proximal steps that are cloud and shadow free
         Parameters:
          date (int): current time step
          satisfactory (list): time steps with no clouds or shadows
         Returns:
          arg_before (str): index of the prior clean image
          arg_after (int): index of the next clean image
    """
    arg_before, arg_after = [], []
    if date > 0:
        idx_before = satisfactory - date

        arg_before = np.array(
            np.where(idx_before < 0, idx_before, -np.inf).flatten())
        to_print = np.copy(arg_before)
        n_before = 2  #if date < np.max(satisfactory) else 3

        if np.sum(arg_before > -np.inf) == 0:
            arg_before = np.empty((0))
        elif np.sum(arg_before > -np.inf) > n_before:
            arg_before = np.argpartition(arg_before, -n_before)[-n_before:]
        elif np.sum(arg_before > -np.inf) == n_before:
            arg_before = np.argwhere(arg_before > -np.inf).flatten()
        else:
            arg_before = np.array(arg_before.argmax())
        if arg_before != np.empty((0)):
            arg_before = list(idx_before[arg_before])
    if date < np.max(satisfactory):
        idx_after = satisfactory - date
        arg_after = np.array(
            np.where(idx_after > 0, idx_after, np.inf).flatten())
        n_after = 2  # if date > 0 else 3

        if np.sum(arg_after < np.inf) == 0:
            arg_after = np.empty((0))
        if np.sum(arg_after < np.inf) > n_after:
            arg_after = np.argpartition(arg_after, n_after)[:n_after]
        elif np.sum(arg_after < np.inf) == n_after:
            arg_after = np.argwhere(arg_after < np.inf).flatten()
        else:
            arg_after = np.array(arg_after.argmin())
        if arg_after != np.empty((0)):
            arg_after = list(idx_after[arg_after])

    if arg_after == np.empty((0)) and arg_before == np.empty((0)):
        arg_after = date
        arg_before = date
    elif arg_after == np.empty((0)):
        arg_after = arg_before
    elif arg_before == np.empty((0)):
        arg_before = arg_after

    return np.array(arg_before).astype(int), np.array(arg_after).astype(int)