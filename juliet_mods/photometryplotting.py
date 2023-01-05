# -*- coding: utf-8 -*-
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import juliet_mods.utils as utils
import juliet_mods.transitplotting as tp

__author__ = "Jonas Kemmer @ ZAH, Landessternwarte Heidelberg"

sns.set(context='paper',
        font_scale=0.9,
        style='ticks',
        rc={
            "lines.linewidth": 1,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True
        })


def find_data_chunks(times, gaplength='0.33Y'):
    deltas = pd.to_datetime(pd.Series(times), origin='julian',
                            unit='D').diff()[1:]
    gaps = deltas > pd.Timedelta(gaplength)
    return gaps.cumsum().values


def _plot_instrument_(results, instrument, color, ax, res, jd_offset, nsamples,
                      show_binned, binlength, interpolate, samplingfreq,
                      max_gap, show_lm):
    # Datapoints and model
    try:
        if instrument in results.data.lm_lc_arguments:
            print('Linear Model detected, setting interpolate to "False"')
            interpolate = False
    except TypeError:
        pass

    if interpolate is True:
        model_times = utils.sample_model_times(results, samplingfreq, 'LC',
                                               instrument)
    else:
        model_times = results.data.times_lc[instrument]

    model_lc, model_components, _ = tp.get_lc_model(results, nsamples,
                                                    instrument, model_times)
    try:
        lc_offset = np.median(
            results.posteriors['posterior_samples'][f'mflux_{instrument}'])
    except IndexError:
        lc_offset = results.data.priors[f'mflux_{instrument}'][
            'hyperparameters']
    if show_binned:
        marker = '.'
        s = 8
        alpha = 0.8
        edgecolor = None
        color = color
    else:
        marker = 'o'
        s = 15
        alpha = 1
        edgecolor = 'black'
        color = color

    if show_lm or interpolate == True:
        lm_correction = 0
    else:
        lm_correction = model_components['lm']

    ax.scatter(results.data.times_lc[instrument] - jd_offset,
               results.data.data_lc[instrument] + lc_offset - lm_correction,
               marker=marker,
               s=s,
               alpha=alpha,
               edgecolor=edgecolor,
               color=color)
    model_lc += lc_offset

    if interpolate:
        ax.plot(model_times - jd_offset,
                model_lc - lm_correction,
                color='black',
                zorder=5)
    else:
        gaps = find_data_chunks(model_times)
        for idx in range(max(gaps) + 1):
            mask = gaps == idx
            if idx == 0:
                mask = np.insert(mask, 0, True)
            else:
                mask = np.insert(mask, 0, False)
            ax.plot(model_times[mask] - jd_offset,
                    model_lc[mask] - lm_correction[mask],
                    color='black',
                    zorder=5)

    # if any("GP" in param for param in results.model_parameters):
    #     try:
    #         model_gp = model_lc - model_components['lm']
    #         ax.plot(model_times - jd_offset, model_gp, color='#DBA039', lw=1.5)
    #     except Exception as e:
    #         print(e)
    if show_lm:
        try:
            model_lm = 1 + model_components['lm'] + lc_offset
            ax.plot(model_times - jd_offset, model_lm, color='#AD4332', lw=1.5)
        except Exception as e:
            print(e)
        # except KeyError:
        #     pass

    # Residual axis
    instr_lc, _, _ = tp.get_lc_model(results, nsamples, instrument,
                                     results.data.times_lc[instrument])

    res.scatter(results.data.times_lc[instrument] - jd_offset,
                results.data.data_lc[instrument] - instr_lc,
                marker=marker,
                s=s,
                alpha=alpha,
                edgecolor=edgecolor,
                color=color)

    # Confidence intervals
    model_quantiles = tp.get_lc_confidence_intervals(results, nsamples,
                                                     instrument, model_times)

    model_quantiles += lc_offset
    for (upper, lower), color in zip(model_quantiles[::-1],
                                     ['grey', 'darkgrey', 'dimgrey']):
        if interpolate:
            ax.fill_between(model_times - jd_offset,
                            lower - lm_correction,
                            upper - lm_correction,
                            color=color,
                            zorder=0)
        else:
            for idx in range(max(gaps) + 1):
                mask = gaps == idx
                if idx == 0:
                    mask = np.insert(mask, 0, True)
                else:
                    mask = np.insert(mask, 0, False)
                ax.fill_between(model_times[mask] - jd_offset,
                                lower[mask] - lm_correction[mask],
                                upper[mask] - lm_correction[mask],
                                color=color,
                                zorder=0)

    if show_binned:
        binned_data = utils.bin_data(results.data.times_lc[instrument],
                                     results.data.data_lc[instrument],
                                     binlength=binlength)
        binned_model = utils.bin_data(results.data.times_lc[instrument],
                                      instr_lc,
                                      binlength=binlength)
        ax.errorbar(binned_data['jd'] - jd_offset,
                    binned_data['values'] + lc_offset,
                    yerr=binned_data['errors'],
                    fmt='o',
                    markersize=3.5,
                    markeredgecolor='black',
                    markerfacecolor="None",
                    ecolor='black')
        res.errorbar(binned_data['jd'] - jd_offset,
                     binned_data['values'] - binned_model['values'],
                     yerr=binned_data['errors'],
                     fmt='o',
                     markersize=3.5,
                     markeredgecolor='black',
                     markerfacecolor="None",
                     ecolor='black')


def plot_photometry_indv_panels(results,
                                jd_offset=2457000,
                                interpolate=False,
                                samplingfreq='0.5D',
                                max_gap='0.3Y',
                                nsamples=1000,
                                show_binned=False,
                                binlength='10D',
                                show_lm=False,
                                show=False,
                                saveformat='pdf'):
    """Similar to plot_lc_indv_panels but with the plot optimised for long-term
        photometric monitoring.

    Parameters
    ----------
    results : object
        Output from a juliet fit that contains all neccesary data.
    jd_offset : int, optional
        Offset for the x-axis, by default 2457000
    interpolate : bool, optional
        Whether to plot an interpolated light curve or use the original
        sampling, by default False
    samplingfreq : str, optional
        Samplingfrequency of the model, by default '0.5D' = 0.5 days
    max_gap : str, optional
        Points between data gaps larger than the given value will not be
        connected (if interpolate is False).
    nsamples : int, optional
        Number of models which are drawn from the posterior to derive
        the confidence intervals, by default 1000
    nsamples : int, optional
        Number of models which are drawn from the posterior to derive
        the confidence intervals, by default 1000
    show_binned : bool, optional
        Whether to overlay binned datapoints.
        Errorbars are the standart deviation of the data bins, by default False
    binlength : str, optional
        Timespan for the data bins, by default '10D'
    show_lm : bool, optional
        If False, the linear model will be subtracted from the model before
        plotting, else the linear model will be depicted in the plot, by default
        False
    show : bool, optional
        Whether the plot is shown after the function call, by default False
    saveformat : str, optional
        Saveformat passed to plt.savefig, by default 'pdf'
    """
    for instrument in sorted(list(results.data.data_lc.keys())):
        fig, (ax, res) = plt.subplots(2,
                                      1,
                                      figsize=(7, 2),
                                      sharex=True,
                                      gridspec_kw={'height_ratios': [3, 1]})
        res.set_xlabel(f'BJD -{jd_offset}')
        res.set_ylabel('(O-C)')
        res.axhline(0, ls='--', color='grey')
        ax.set_ylabel('Relative flux')

        _plot_instrument_(results, instrument, '#0E9CA1', ax, res, jd_offset,
                          nsamples, show_binned, binlength, interpolate,
                          samplingfreq, max_gap, show_lm)

        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        fig.savefig(
            f'{results.data.out_folder}/{instrument}_photometry_plot.{saveformat}',
            dpi=400,
            bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)


def plot_photometry_with_shared_parameters(results,
                                           instruments,
                                           jd_offset=2457000,
                                           samplingfreq='0.5D',
                                           nsamples=1000,
                                           show_binned=False,
                                           binlength='10D',
                                           show=False,
                                           saveformat='pdf'):
    """Equivalent to a single plot from plot_photometry_indv_panels
       but with "instruments" keyword that allows to specify a number of
       instruments that will be plotted toghether in the panel, assuming they
       share the same model (individual offsets are considered!). Useful for
       example if you have multiple cameras from one instrument.
       Note: the shown model corresponds to the first instrument from the list.

    Parameters
    ----------
    results : object
        Output from a juliet fit that contains all neccesary data.
    instruments: list
        A list that contains the instruments considered for the panel
    jd_offset : int, optional
        Offset for the x-axis, by default 2457000
    interpolate : bool, optional
        Whether to plot an interpolated light curve or use the original
        sampling, by default False
    samplingfreq : str, optional
        Samplingfrequency of the model, by default '0.5D' = 0.5 days
    nsamples : int, optional
        Number of models which are drawn from the posterior to derive
        the confidence intervals, by default 1000
    nsamples : int, optional
        Number of models which are drawn from the posterior to derive
        the confidence intervals, by default 1000
    show_binned : bool, optional
        Whether to overlay binned datapoints.
        Errorbars are the standart deviation of the data bins, by default False
    binlength : str, optional
        Timespan for the data bins, by default '10D'
    show : bool, optional
        Whether the plot is shown after the function call, by default False
    saveformat : str, optional
        Saveformat passed to plt.savefig, by default 'pdf'
    """
    fig, (ax, res) = plt.subplots(2,
                                  1,
                                  figsize=(7, 2),
                                  sharex=True,
                                  gridspec_kw={'height_ratios': [3, 1]})
    res.set_xlabel(f'BJD -{jd_offset}')
    res.set_ylabel('(O-C)')
    res.axhline(0, ls='--', color='grey')
    ax.set_ylabel('Relative flux')

    min_t = 1e10
    max_t = 1e-10
    for instrument in instruments:
        if np.min(results.data.times_lc[instrument]) < min_t:
            min_t = np.min(results.data.times_lc[instrument])
        if np.max(results.data.times_lc[instrument]) > max_t:
            max_t = np.max(results.data.times_lc[instrument])
    model_times = pd.date_range(pd.to_datetime(min_t, origin='julian',
                                               unit='D'),
                                pd.to_datetime(max_t, origin='julian',
                                               unit='D'),
                                freq=samplingfreq)
    model_times = model_times.to_julian_date().values
    resultsdummy = deepcopy(results)
    model_lc = resultsdummy.lc.evaluate(instruments[0],
                                        t=model_times,
                                        GPregressors=model_times,
                                        nsamples=nsamples)

    for idx, instrument in enumerate(instruments):
        try:
            lc_offset = np.median(
                results.posteriors['posterior_samples'][f'mflux_{instrument}'])
        except IndexError:
            lc_offset = results.data.priors[f'mflux_{instruments}'][
                'hyperparameters']
        if idx == 0:
            ax.plot(model_times - jd_offset,
                    model_lc + lc_offset,
                    color='black',
                    zorder=5)
            model_quantiles = tp.get_lc_confidence_intervals(
                results, nsamples, instrument, model_times)
            model_quantiles += lc_offset
            for (upper, lower), color in zip(model_quantiles[::-1],
                                             ['grey', 'darkgrey', 'dimgrey']):
                ax.fill_between(model_times - jd_offset,
                                lower,
                                upper,
                                color=color,
                                zorder=0)
        if show_binned:
            marker = '.'
            s = 8
            alpha = 0.8
            edgecolor = None

        else:
            marker = 'o'
            s = 15
            alpha = 1
            edgecolor = 'black'
        ax.scatter(results.data.times_lc[instrument] - jd_offset,
                   results.data.data_lc[instrument] + lc_offset,
                   marker=marker,
                   s=s,
                   alpha=alpha,
                   edgecolor=edgecolor,
                   color='#0E9CA1')
        instr_lc, _, _ = tp.get_lc_model(results, nsamples, instrument,
                                         results.data.times_lc[instrument])

        # Residual axis
        res.scatter(
            results.data.times_lc[instrument] - jd_offset,
            results.data.data_lc[instrument] - instr_lc,  # + lc_offset,
            marker=marker,
            s=s,
            alpha=alpha,
            edgecolor=edgecolor,
            color='#0E9CA1')

        if show_binned:
            binned_data = utils.bin_data(results.data.times_lc[instrument],
                                         results.data.data_lc[instrument],
                                         binlength=binlength)
            binned_model = utils.bin_data(results.data.times_lc[instrument],
                                          instr_lc,
                                          binlength=binlength)
            ax.errorbar(binned_data['jd'] - jd_offset,
                        binned_data['values'] + lc_offset,
                        yerr=binned_data['errors'],
                        fmt='o',
                        markersize=3.5,
                        markeredgecolor='black',
                        markerfacecolor="None",
                        ecolor='black')
            res.errorbar(binned_data['jd'] - jd_offset,
                         binned_data['values'] - binned_model['values'],
                         yerr=binned_data['errors'],
                         fmt='o',
                         markersize=3.5,
                         markeredgecolor='black',
                         markerfacecolor="None",
                         ecolor='black')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    fig.savefig(
        f'{results.data.out_folder}/{"_".join(instruments)}_photometry_plot.{saveformat}',
        dpi=400,
        bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_photometry(results,
                    jd_offset=2457000,
                    interpolate=False,
                    samplingfreq='0.5D',
                    nsamples=1000,
                    indv_models=False,
                    show_binned=False,
                    binlength='1D',
                    show=False,
                    saveformat='pdf'):
    """Plots a broad light curve of all data and the best model.
    CAUTION if a non-global GP model is used and "indv_models"=True: The plotted model corresponds to the instrument whose initial letter appears first in the alphabet.

    Parameters
    ----------
    results : object
        Output from a juliet fit that contains all neccesary data.
    jd_offset : int, optional
        Offset for the x-axis, by default 2457000
    interpolate : bool, optional
        Whether to plot an interpolated light curve or use the original
        sampling, by default False
    samplingfreq : str, optional
        Samplingfrequency of the model, by default '0.5D' = 0.5 days
    nsamples : int, optional
        Number of models which are drawn from the posterior to derive
        the confidence intervals, by default 1000
    indv_models : bool, optional
        Whether to consider individual models for each instrument or a global
        model. Offsets are considered in either case, by default False
    show_binned : bool, optional
        Whether to overlay binned datapoints.
        Errorbars are the standart deviation of the data bins, by default False
    binlength : str, optional
        Timespan for the data bins, by default '1D'
    show : bool, optional
        Whether the plot is shown after the function call, by default False
    saveformat : str, optional
        Saveformat passed to plt.savefig, by default 'pdf'
    """
    instruments = sorted(list(results.data.data_lc.keys()))
    colors = ['#DBA039', '#0E9CA1', '#AD4332', '#394F82', '#2C6A99']
    if len(instruments) > len(colors):
        colors = sns.color_palette("husl", len(instruments))

    fig, (ax, res) = plt.subplots(2,
                                  1,
                                  figsize=(7, 2),
                                  sharex=True,
                                  gridspec_kw={'height_ratios': [3, 1]})
    res.set_xlabel(f'BJD -{jd_offset}')
    res.set_ylabel('(O-C)')
    res.axhline(0, ls='--', color='grey')
    ax.set_ylabel('Relative flux')

    for instrument, color in zip(instruments, colors):
        # Datapoints and model
        if interpolate is True and indv_models is True:
            model_times = utils.sample_model_times(results, samplingfreq, 'LC',
                                                   instrument)
        elif interpolate is True and indv_models is False:
            model_times = utils.sample_model_times(results, samplingfreq, 'LC')
        elif interpolate is False and indv_models is True:
            model_times = results.data.times_lc[instrument]
        else:
            model_times = results.data.t_lc
        model_times = np.sort(model_times)

        try:
            lc_offset = np.median(
                results.posteriors['posterior_samples'][f'mflux_{instrument}'])
        except IndexError:
            lc_offset = results.data.priors[f'mflux_{instruments}'][
                'hyperparameters']
        if show_binned:
            marker = '.'
            s = 8
            alpha = 0.8
            edgecolor = None

        else:
            marker = 'o'
            s = 15
            alpha = 1
            edgecolor = 'black'
        ax.scatter(results.data.times_lc[instrument] - jd_offset,
                   results.data.data_lc[instrument] + lc_offset,
                   marker=marker,
                   s=s,
                   alpha=alpha,
                   edgecolor=edgecolor,
                   color=color)

        if indv_models is True or np.logical_and(indv_models is False,
                                                 instrument == instruments[0]):
            model_lc, _, _ = tp.get_lc_model(results, nsamples, instrument,
                                             model_times)
            model_lc += lc_offset
            ax.plot(model_times - jd_offset, model_lc, color='black', zorder=5)

            # Confidence intervals
            model_quantiles = tp.get_lc_confidence_intervals(
                results, nsamples, instrument, model_times)
            model_quantiles += lc_offset
            for (upper, lower), color in zip(model_quantiles[::-1],
                                             ['grey', 'darkgrey', 'dimgrey']):
                ax.fill_between(model_times - jd_offset,
                                lower,
                                upper,
                                color=color,
                                zorder=0)

        # Residual axis
        instr_lc, _, _ = tp.get_lc_model(results, nsamples, instrument,
                                         results.data.data_lc[instrument])
        res.scatter(
            results.data.times_lc[instrument] - jd_offset,
            results.data.data_lc[instrument] - instr_lc + lc_offset,
            marker='.',
            s=8,  # s=0.6,
            alpha=0.8,
            color=color)

        if show_binned:
            binned_data = utils.bin_data(results.data.times_lc[instrument],
                                         results.data.data_lc[instrument],
                                         binlength=binlength)
            binned_model = utils.bin_data(results.data.times_lc[instrument],
                                          instr_lc,
                                          binlength=binlength)
            ax.errorbar(binned_data['jd'] - jd_offset,
                        binned_data['values'] + lc_offset,
                        yerr=binned_data['errors'],
                        fmt='o',
                        markersize=3.5,
                        markeredgecolor='black',
                        markerfacecolor="None",
                        ecolor='black')
            res.errorbar(binned_data['jd'] - jd_offset,
                         binned_data['values'] - binned_model['values'],
                         yerr=binned_data['errors'],
                         fmt='o',
                         markersize=3.5,
                         markeredgecolor='black',
                         markerfacecolor="None",
                         ecolor='black')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    fig.savefig(f'{results.data.out_folder}/full_photometry_plot.{saveformat}',
                dpi=400,
                bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
