#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:31:35 2020

@author: jkemmer
"""
from juliet_mods.data_handling import Prior
import numpy as np


def predict_observed_transits(t0,
                              P,
                              t_min,
                              t_max,
                              uncert=0,
                              tgap_min=0,
                              tgap_max=0):
    """Function to predict transits. Can consider a data gap (useful for TESS).

    Parameters
    ----------
    t0 : float
        Transit time in JD
    P : float
        Period in days
    t_min : float
        Start time in JD
    t_max : float
        End time in JD
    uncert : float, optional
        Uncertainty in t0, by default 0
    tgap_min : float, optional
        Start of data gap in JD, by default 0
    tgap_max : int, optional
        End of data gap in JD, by default 0

    Returns
    -------
    array,array
        Arrays of the transit times and transit numbers starting with
        0 from t0.
    """
    min_ind = np.floor((t_min - t0) / P)
    max_ind = np.ceil((t_max - t0) / P)
    times = t0 + P * np.arange(min_ind + 1, max_ind + 1, 1)
    times = times[(t_min - uncert <= times) & (times <= t_max - uncert)]
    transits = []
    for n, time in zip(np.arange(min_ind + 1, max_ind + 1, 1), times):
        if time >= tgap_min and time <= tgap_max:
            times = np.delete(times, n)
        else:
            transits.append(int(n))
    return transits, times


def add_ttv_prior(plist, transits, times, ttv_std, pl_idx, instr):
    """convenience function to add ttv priors to a PriorList.
       Considers individual transit times for each transit.


    Parameters
    ----------
    plist : object
        A PriorList object from the juliet_mods
    transits : array_like
        A list containing the transit numbers
    times : array_like
        A list containing the individual transit times
    ttv_std : float
        Std-dev for the normal prior that is used
    pl_idx : int
        Identifiert of the planet for which the transits hold
    instr : str
        Name of the instrument for which the TTVs are considered

    Returns
    -------
    _type_
        _description_
    """
    for n, tc in zip(transits, times):
        prior = Prior(f'T_p{pl_idx}_{instr}_{n}', 'normal', [tc, ttv_std])
        plist.add(prior)
    return plist


def add_ind_ttv_prior(plist, transits, times, ttv_std, pl_idx, instr):
    """Identical to "add_ttv_prior" but consideres a shift in transit time (classical measure of TTVs) instead of individual transit times.
    """
    for n, tc in zip(transits, times):
        prior = Prior(f'dt_p{pl_idx}_{instr}_{n}', 'normal', [0, ttv_std])
        plist.add(prior)
    return plist
