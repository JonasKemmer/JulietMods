# -*- coding: utf-8 -*-
from math import ceil, floor
from copy import deepcopy
import tkinter as tk
from tkinter import ttk

import numpy as np
import pandas as pd

__author__ = "Jonas Kemmer @ ZAH, Landessternwarte Heidelberg"


def flatten_dict(dictio):
    """Function to return a list of the dictionary entries."""
    flat = []
    for key in dictio.keys():
        flat.append(np.atleast_1d(dictio[key]))
    flat = np.hstack(flat)
    return np.array(flat)


def float_upround(num, places=0):
    """Function to upround a number to the given decimals."""
    return ceil(num * (10**places)) / float(10**places)


def float_round(num, places=0):
    """Function to round a number to the given decimals."""
    return round(num * (10**places)) / float(10**places)


def get_significant_decimals(x):
    """Returns the significant decimals for an input uncertainty."""
    return -(int(np.floor(np.log10(abs(x)))) - 1)


def get_posterior_string(posterior):
    """A function to create a latex string for the posterior distribution
    of an parameter.

    Parameters
    ----------
    posterior : array
        The array that contains the posterior distribution from a fit.

    Returns
    -------
    str
        The string is like:
        "$median^+upper_uncertainty_-lower_uncertainty$"
    """
    if posterior is None:
        return r"\dots"
    elif isinstance(posterior, (list, tuple, np.ndarray)):
        try:
            median, lower, upper = np.nanquantile(posterior.value,
                                                  [0.5, 0.16, 0.84])
        except AttributeError:
            median, lower, upper = np.nanquantile(posterior, [0.5, 0.16, 0.84])
        lower = median - lower
        upper = upper - median
        significant_decimals = max(get_significant_decimals(lower),
                                   get_significant_decimals(upper))
        median = float_round(median, significant_decimals)
        lower = float_upround(lower, significant_decimals)
        upper = float_upround(upper, significant_decimals)
        # return f"${median}^{{+{upper}}}_{{-{lower}}}$"
        return rf"$\num{{{median}}}^{{+\num{{{upper}}}}}_{{-\num{{{lower}}}}}$"
    else:
        return f"{posterior} (fixed)"


def sample_model_times(results, samplingfreq, category, instrument=None):
    """Function to create model times based on the data times.

    Parameters
    ----------
    results : object from juliet.fit
    samplingfreq : str
        Samplingfrequency of the model, by default '0.25D' (pandas frequency)
    category: str
        Can be either "RV" or "LC".
    instrument: str
        If only the times of a specific instrument should be considered, an
        instrument string should be given, by default None

    Returns
    -------
    array
        model times in JD
    """
    if instrument is None:
        if category == 'RV':
            times = results.data.t_rv
        elif category == 'LC':
            times = results.data.t_lc
        else:
            raise KeyError('Category must be either "RV" or "LC".')
    else:
        if category == 'RV':
            times = results.data.times_rv[instrument]
        elif category == 'LC':
            times = results.data.times_lc[instrument]
        else:
            raise KeyError('Category must be either "RV" or "LC".')

    min_t = np.min(times)
    max_t = np.max(times)
    model_times = pd.date_range(pd.to_datetime(min_t, origin='julian',
                                               unit='D'),
                                pd.to_datetime(max_t, origin='julian',
                                               unit='D'),
                                freq=samplingfreq)
    model_times = model_times.to_julian_date().values
    return model_times


def get_period_and_t0(results, pnum):
    """Function to determine the period and t0 from a fit."""
    try:
        P = np.median(results.posteriors['posterior_samples'][f'P_p{pnum}'])
    except KeyError:
        try:
            P = results.data.priors[f'P_p{pnum}']['hyperparameters'][0]
        except IndexError:
            P = results.data.priors[f'P_p{pnum}']['hyperparameters']
    try:
        t0 = np.median(results.posteriors['posterior_samples'][f't0_p{pnum}'])
    except KeyError:
        try:
            t0 = results.data.priors[f't0_p{pnum}']['hyperparameters'][0]
        except IndexError:
            t0 = results.data.priors[f't0_p{pnum}']['hyperparameters']
    return P, t0


def bin_data(times, values, binlength):
    """Function to bin evenly sampled data.
       Binned uncertainty is  std(values)/sqrt(len(values)).

    Parameters
    ----------
    times : array
        The timestamps.
    values : array
        The data.
    binlength : str
        A pandas frequency, example: '0.5D' for 0.5 days.

    Returns
    -------
    pandas DataFrame
        The DataFrame contains the binned "jd", "values" and "errors"
        as columns.
    """
    df = pd.DataFrame(np.array([times, values]).T, columns=['times', 'values'])
    df['times'] = pd.to_datetime(df['times'], origin='julian', unit='D')
    df.set_index('times', inplace=False)

    bins = df.groupby(pd.Grouper(key='times', freq=binlength))
    binned = bins.mean()
    binned['errors'] = bins['values'].std() / np.sqrt(bins.size())
    binned = binned[binned['errors'].notna()]
    binned.insert(0, 'jd', binned.index.to_julian_date())
    return binned[['jd', 'values', 'errors']]


def bin_phased_data(phases, values, bin_length):
    """Function to bin phased data.
       Binned uncertainty is  std(values)/sqrt(len(values)).

    Parameters
    ----------
    phases : array
        The timestamps.
    values : array
        The data.
    binlength : float
        The bin length in phase.

    Returns
    -------
    pandas DataFrame
        The DataFrame contains the binned "phases", "values" and "errors"
        as columns.
    """
    df = pd.DataFrame(np.array([phases, values]).T,
                      columns=['phases', 'values'])
    min_t = np.min(phases)
    max_t = np.max(phases)
    bin_edges = [min_t]
    time = min_t
    while time <= max_t:
        time += (bin_length)
        bin_edges.append(time)
    bins = df.groupby(pd.cut(x=df['phases'], bins=bin_edges))
    binned = bins.mean()
    # standard error of mean
    binned['errors'] = bins['values'].std() / np.sqrt(bins.size())
    binned = binned[binned['errors'].notna()]
    return binned[['phases', 'values', 'errors']]


def get_rv_residuals(results):
    """Function to calculate the RV residuals from a juliet fit and save them
       in a file.

    Parameters
    ----------
    results : the outcome from juliet.fit


    Returns
    -------
    Saves a file in the juliet out_folder that contains the residual RVs
    after subtracting the best fit model. The columns are the times, residual
    rvs and uncertainties. Jitter is not considered in the uncertainties,
    so that the residuals can be used to create a residual GLS.
    """
    residuals = []
    for instrument in np.unique(results.data.instruments_rv):
        resultsdummy = deepcopy(results)
        try:
            lm_args = results.data.lm_rv_arguments[instrument]
        except (KeyError, TypeError):
            lm_args = None
        model_rv, components = resultsdummy.rv.evaluate(
            instrument,
            t=results.data.times_rv[instrument],
            GPregressors=results.data.times_rv[instrument],
            LMregressors=lm_args,
            return_components=True)
        residuals.append(
            pd.DataFrame(
                np.transpose([
                    results.data.times_rv[instrument],
                    results.data.data_rv[instrument] - model_rv,
                    results.data.errors_rv[instrument],
                    np.tile(instrument,
                            len(results.data.times_rv[instrument])), model_rv
                ])))
    residuals = pd.concat(residuals)
    residuals.to_csv(f'{results.data.out_folder}/residuals_rvs.dat',
                     header=False,
                     index=False,
                     float_format='%.6f',
                     sep=' ')


def get_lc_residuals(results):
    """Function to calculate the light curve residuals from a juliet fit and save them
       in a file.

    Parameters
    ----------
    results : the outcome from juliet.fit


    Returns
    -------
    Saves a file in the juliet out_folder that contains the residual fluxes
    after subtracting the best fit model. The columns are the times, residual
    flux and uncertainties. Jitter is not considered in the uncertainties.
    """
    residuals = []
    for instrument in np.unique(results.data.instruments_lc):
        resultsdummy = deepcopy(results)
        try:
            lm_args = results.data.lm_lc_arguments[instrument]
        except (KeyError, TypeError):
            lm_args = None
        model_lc = resultsdummy.lc.evaluate(
            instrument,
            t=results.data.times_lc[instrument],
            LMregressors=lm_args,
            GPregressors=results.data.times_lc[instrument])

        residuals.append([
            results.data.times_lc[instrument],
            results.data.data_lc[instrument] - model_lc + 1,
            results.data.errors_lc[instrument],
            np.repeat(instrument, len(results.data.times_lc[instrument]))
        ])
    np.savetxt(f'{results.data.out_folder}/residuals_lcs.dat',
               np.concatenate(residuals, axis=1).T,
               fmt='%s')


def append_lnZ(results):
    """Function to add the lnZ of a model to the posterior.dat that is created
       by juliet."""
    lnz = results.posteriors['lnZ']
    lnz_err = results.posteriors['lnZerr']
    with open(f'{results.data.out_folder}/posteriors.dat',
              "r+") as posteriorfile:
        if "dlogZ" in posteriorfile.read():
            return
        posteriorfile.write(
            f'\n# dlogZ                {lnz:.3f} +/- {lnz_err:.3g}')


def notification_pop_up(msg='Fit finished!', sound=True):
    popup = tk.Tk()
    popup.wm_title("!")
    if sound:
        popup.bell()
    label = ttk.Label(
        popup,
        text=msg,
    )
    label.pack(side="top", fill="x", pady=10)
    B1 = ttk.Button(popup, text="Okay", command=popup.destroy)
    B1.pack()
    popup.mainloop()
