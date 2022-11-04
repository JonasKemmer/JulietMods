# -*- coding: utf-8 -*-
from copy import deepcopy

import juliet
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

import juliet_mods.utils as utils

__author__ = "Jonas Kemmer @ ZAH, Landessternwarte Heidelberg"

sns.set(context='paper',
        font_scale=0.9,
        style='ticks',
        rc={
            "lines.linewidth": 1,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True
        })

colors = [
    '#AD4332',
    '#2d9a9a',
    '#DBA039',
    '#2C6A99',
    '#7f2d9a',
    '#394F82',
    '#0E9CA1',
]


def get_rv_confidence_intervals(results, nsamples, instrument, model_times):
    """Function to calculate the 0.68, 0.95, 0.99 quantiles to use as confidence
       intervals for a plot.

    Parameters
    ----------
    results : results from juliet.fit
    nsamples : int
        Number of samples drawn to determine the intervals
    instrument : str
        Name of the instrument that is evaluated
    model_times : array
        Array that contains the times at which the model is evaluated

    Returns
    -------
    array
        A 2-dim array that contains the upper and lower uncertainties
    """
    model_quantiles = []
    for quantile in [0.68, 0.95, 0.99]:
        resultsdummy = deepcopy(results)
        try:
            lm_args = resultsdummy.data.lm_rv_arguments[instrument]
        except (KeyError, TypeError):
            lm_args = None
        _, upper, lower = resultsdummy.rv.evaluate(instrument,
                                                   t=model_times,
                                                   GPregressors=model_times,
                                                   LMregressors=lm_args,
                                                   nsamples=nsamples,
                                                   return_err=True,
                                                   alpha=quantile)
        model_quantiles.append([upper, lower])
    model_quantiles = np.array(model_quantiles)
    return model_quantiles


def get_rv_model(results, nsamples, instrument, model_times):
    """Wrapper for evaluating a model in juliet.

    Parameters
    ----------
    results : results from juliet.fit
    nsamples : int
        Number of samples drawn from which the median model is determined.
    instrument : str
        Name of the instrument that is evaluated
    model_times : array
        Array that contains the times at which the model is evaluated

    Returns
    -------
    array
        An array that contains the median model.
    """
    resultsdummy = deepcopy(results)
    try:
        lm_args = resultsdummy.data.lm_rv_arguments[instrument]
        print('Linear model in RV detected, sampling model as data.')
        model_times = results.data.times_rv[instrument]
    except (KeyError, TypeError):
        lm_args = None
    model_rv, components = resultsdummy.rv.evaluate(instrument,
                                                    t=model_times,
                                                    GPregressors=model_times,
                                                    LMregressors=lm_args,
                                                    nsamples=nsamples,
                                                    return_components=True)

    return model_rv, components, resultsdummy


def plot_rv(results,
            jd_offset=2457000,
            samplingfreq='0.25D',
            nsamples=1000,
            external_legend=False,
            show=False,
            saveformat='pdf'):
    """Plots a broad RV curve of all data and the best model.
    CAUTION if a non-global GP model is used: The plotted model corresponds
    to the instrument whose initial letter appears first in the alphabet.


    Parameters
    ----------
    results : object
        Output from a juliet fit that contains all neccesary data.
    jd_offset : int, optional
        Offset for the x-axis, by default 2457000
    samplingfreq : str, optional
        Samplingfrequency of the model, by default '0.25D'
    nsamples : int, optional
        Number of models which are drawn from the posterior to derive
        the confidence intervals, by default 1000
    external_legend : bool, optional
        Wether the legend is plotted in the figure or in a
        separate plot, by default False
    show : bool, optional
        Whether the plot is shown after the function call, by default False
    saveformat : str, optional
        Saveformat passed to plt.savefig, by default 'pdf'

    """
    global colors
    instruments = sorted(list(results.data.data_rv.keys()))
    if len(instruments) > len(colors):
        colors = sns.color_palette("husl", len(instruments))
    model_times = utils.sample_model_times(results, samplingfreq, 'RV')

    model_rv, components, components_results = get_rv_model(
        results, nsamples, instruments[0], model_times)

    try:
        rv_offset = components['mu'][instruments[0]]
    except IndexError:
        rv_offset = components['mu']

    model_quantiles = get_rv_confidence_intervals(results, nsamples,
                                                  instruments[0], model_times)
    model_quantiles -= rv_offset

    # Figure initialised
    fig, (ax, res) = plt.subplots(2,
                                  1,
                                  figsize=(7, 2),
                                  sharex=True,
                                  gridspec_kw={'height_ratios': [3, 1]})
    res.set_xlabel(f'BJD -{jd_offset}')
    res.set_ylabel('(O-C) [m/s]')
    res.axhline(0, ls='--', color='grey')
    ax.set_ylabel('RV [m/s]')

    # Plot model
    ax.plot(model_times - jd_offset,
            model_rv - rv_offset,
            color='black',
            zorder=5)
    if 'GP' in components_results.rv.model:
        model_gp = components_results.rv.model['GP'] - rv_offset
        ax.plot(
            model_times - jd_offset,
            model_gp + rv_offset,
            color='#DBA039',
            # lw=1.5,
            zorder=6)

    # Plot uncertainty
    for (upper, lower), color in zip(model_quantiles[::-1],
                                     ['grey', 'darkgrey', 'dimgrey']):
        ax.fill_between(model_times - jd_offset,
                        lower,
                        upper,
                        color=color,
                        zorder=0)

    legend_elements = []
    for instrument, color in zip(instruments, colors):
        legend_elements.append(
            Line2D([0], [0],
                   marker='o',
                   color='w',
                   label=instrument,
                   markerfacecolor=color,
                   markeredgewidth=1,
                   markeredgecolor='black',
                   markersize=10))
        try:
            jitter = np.median(
                results.posteriors['posterior_samples']['sigma_w_' +
                                                        instrument])
        except KeyError:
            jitter = results.data.priors['sigma_w_' +
                                         instrument]['hyperparameters']

        try:
            rv_offset = components['mu'][instrument]
        except IndexError:
            rv_offset = components['mu']

        ax.errorbar(results.data.times_rv[instrument] - jd_offset,
                    results.data.data_rv[instrument] - rv_offset,
                    yerr=np.sqrt(results.data.errors_rv[instrument]**2 +
                                 jitter**2),
                    fmt='o',
                    color=color,
                    markeredgecolor='black',
                    zorder=7)

        instr_model, _, _ = get_rv_model(results, nsamples, instrument,
                                         results.data.times_rv[instrument])

        res.errorbar(results.data.times_rv[instrument] - jd_offset,
                     results.data.data_rv[instrument] - instr_model,
                     yerr=np.sqrt(results.data.errors_rv[instrument]**2 +
                                  jitter**2),
                     fmt='o',
                     color=color,
                     markeredgecolor='black',
                     zorder=7)
    ax.set_ylim(-3 * np.std(results.data.y_rv), 3 * np.std(results.data.y_rv))
    # np.mean(results.data.y_rv) - 3 * np.std(results.data.y_rv),
    # np.mean(results.data.y_rv) + 3 * np.std(results.data.y_rv))

    if not external_legend:
        ax.legend(handles=legend_elements, loc='best', ncol=3)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    fig.savefig(f'{results.data.out_folder}/rvs_vs_time_broad.{saveformat}',
                dpi=400,
                bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)

    if external_legend:
        fig, ax = plt.subplots(figsize=(6, 0.4))
        ax.legend(handles=legend_elements, loc='center', ncol=3)
        plt.axis('off')
        plt.savefig(f'{results.data.out_folder}/legend_rv_plots.{saveformat}',
                    dpi=400,
                    bbox_inches='tight')
        plt.close(fig)


def plot_phased_rvs(results,
                    nsamples=1000,
                    hide_uncertainty=False,
                    only_planet_uncertainty=False,
                    show_binned=False,
                    binlength=0.075,
                    external_legend=False,
                    show=False,
                    saveformat='pdf'):
    """Plot the phase-folded individual planet models.

    Parameters
    ----------
    results : object
        Output from a juliet fit that contains all neccesary data.
    nsamples : int, optional
        Number of models which are drawn from the posterior to derive
        the confidence intervals, by default 1000
    hide_uncertainty: bool, optional
        If true, no confidence interval will be plotted, by default False
    only_planet_uncertainty : bool, optional
        If True, the uncertainties will be derived only from the individual
        planet components, ignoring the overall model uncertainties,
        by default False
    show_binned : bool, optional
        Wether to show binned data points, by default False
    binlength : float
        Length of the bins in phase, by default 0.075
    external_legend : bool, optional
        Wether the legend is plotted in the figure or in a
        separate plot, by default False
    show : bool, optional
        Whether the plot is shown after the function call, by default False
    saveformat : str, optional
        Saveformat passed to plt.savefig, by default 'pdf'
    """
    global colors
    instruments = sorted(list(results.data.data_rv.keys()))
    if show_binned:
        marker = '.'
        alpha = 0.8
        edgecolor = None

    else:
        marker = 'o'
        alpha = 1
        edgecolor = 'black'

    for pnum in range(1, results.data.n_rv_planets + 1):
        P, t0 = utils.get_period_and_t0(results, pnum)

        model_phases = np.linspace(0.5, -0.5, 1000)
        model_times = model_phases * P + t0

        model_rv, components, _ = get_rv_model(results, nsamples,
                                               instruments[0], model_times)
        try:
            rv_offset = components['mu'][instruments[0]]
        except IndexError:
            rv_offset = components['mu']

        if only_planet_uncertainty:
            resultsdummy = deepcopy(results)
            try:
                lm_args = results.data.lm_rv_arguments[instruments[0]]
            except (KeyError, TypeError):
                lm_args = None
            _, _, samples_components = resultsdummy.rv.evaluate(
                instruments[0],
                t=model_times,
                GPregressors=model_times,
                LMregressors=lm_args,
                nsamples=nsamples,
                return_samples=True,
                return_components=True)
            model_quantiles = []
            for quantile in [0.68, 0.95, 0.99]:
                model_quantiles.append(
                    np.quantile(samples_components[f'p{pnum}'],
                                [0.5 - quantile / 2, 0.5 + quantile / 2],
                                axis=0))

        else:
            model_other_p = model_rv - components[f'p{pnum}']
            model_quantiles = get_rv_confidence_intervals(
                results, nsamples, instruments[0], model_times)
            model_quantiles -= model_other_p

        # Figure initialised
        fig, (ax, res) = plt.subplots(2,
                                      1,
                                      figsize=(3.45, 3),
                                      sharex=True,
                                      gridspec_kw={'height_ratios': [3, 1]})
        ax.set_title(f'$P_{pnum}={P:.3f}$ d')
        res.set_xlabel('Phase')
        res.set_ylabel('(O-C) [m/s]')
        res.axhline(0, ls='--', color='grey')
        ax.set_ylabel('RV [m/s]')

        # Plot model
        ax.plot(model_phases, components[f'p{pnum}'], color='black', zorder=10)
        if not hide_uncertainty:
            for (upper, lower), color in zip(model_quantiles[::-1],
                                             ['grey', 'darkgrey', 'dimgrey']):
                ax.fill_between(model_phases,
                                lower,
                                upper,
                                color=color,
                                zorder=0)
        legend_elements = []
        for instrument, color in zip(instruments, colors):
            legend_elements.append(
                Line2D([0], [0],
                       marker='o',
                       color='w',
                       label=instrument,
                       markerfacecolor=color,
                       markeredgewidth=1,
                       markeredgecolor='black',
                       markersize=10))

            if results.rv.global_model:
                model_rv, components = results.rv.evaluate(
                    instrument,
                    t=results.data.t_rv,
                    GPregressors=results.data.GP_rv_arguments['rv'],
                    return_components=True)
                model_other_p = model_rv - components[f'p{pnum}']
                mask = np.isin(results.data.t_rv,
                               results.data.times_rv[instrument])
                model_other_p = model_other_p[mask]
                model_rv = model_rv[mask]
            else:
                model_rv, components = results.rv.evaluate(
                    instrument,
                    t=results.data.times_rv[instrument],
                    GPregressors=results.data.times_rv[instrument],
                    return_components=True)
                model_other_p = model_rv - components[f'p{pnum}']

            data_phases = juliet.get_phases(results.data.times_rv[instrument],
                                            P, t0)
            try:
                rv_offset = components['mu'][instruments[0]]
            except IndexError:
                rv_offset = components['mu']
            try:
                jitter = np.median(
                    results.posteriors['posterior_samples']['sigma_w_' +
                                                            instrument])
            except KeyError:
                jitter = results.data.priors['sigma_w_' +
                                             instrument]['hyperparameters']
            ax.errorbar(data_phases,
                        results.data.data_rv[instrument] - model_other_p,
                        yerr=np.sqrt(results.data.errors_rv[instrument]**2 +
                                     jitter**2),
                        fmt=marker,
                        color=color,
                        alpha=alpha,
                        markeredgecolor=edgecolor,
                        zorder=5)
            res.errorbar(data_phases,
                         results.data.data_rv[instrument] - model_rv,
                         yerr=np.sqrt(results.data.errors_rv[instrument]**2 +
                                      jitter**2),
                         fmt=marker,
                         color=color,
                         alpha=alpha,
                         markeredgecolor=edgecolor,
                         zorder=5)

            if show_binned:
                binned_data = utils.bin_phased_data(
                    data_phases,
                    results.data.data_rv[instrument] - model_other_p, binlength)
                binned_model = utils.bin_phased_data(data_phases,
                                                     model_rv - model_other_p,
                                                     binlength)
                ax.errorbar(binned_data['phases'],
                            binned_data['values'],
                            yerr=binned_data['errors'],
                            fmt='s',
                            markersize=6,
                            markeredgecolor='black',
                            markerfacecolor='None',
                            ecolor='black',
                            zorder=6)
                res.errorbar(binned_data['phases'],
                             binned_data['values'] - binned_model['values'],
                             yerr=binned_data['errors'],
                             fmt='s',
                             markersize=6,
                             markeredgecolor='black',
                             markerfacecolor='None',
                             ecolor='black',
                             zorder=7)

        ax.set_ylim(-3 * np.std(results.data.y_rv),
                    3 * np.std(results.data.y_rv))
        # ax.set_ylim(-3 * np.std(results.data.y_rv - rv_offset),
        #             +3 * np.std(results.data.y_rv - rv_offset))
        ax.set_xlim(-0.52, 0.52)
        if not external_legend:
            ax.legend(handles=legend_elements, loc='best', ncol=3)
        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        fig.savefig(
            f'{results.data.out_folder}/Phased_RV_P{pnum}_{P:.0f}d.{saveformat}',
            dpi=400,
            bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)

    if external_legend:
        fig, ax = plt.subplots(figsize=(6, 0.4))
        ax.legend(handles=legend_elements, loc='center', ncol=3)
        plt.axis('off')
        plt.savefig(f'{results.data.out_folder}/legend_rv_plots.{saveformat}',
                    dpi=400,
                    bbox_inches='tight')
        plt.close(fig)


def plot_rv_indv_panels(results,
                        jd_offset=2457000,
                        samplingfreq='0.25D',
                        nsamples=1000,
                        show=False,
                        saveformat='pdf'):
    """Plots a non-phased RV curve with all instruments in seperate panels.

    Parameters
    ----------
    results : object
        Output from a juliet fit that contains all neccesary data.
    jd_offset : int, optional
        Offset for the x-axis, by default 2457000
    nsamples : int, optional
        Number of models which are drawn from the posterior to derive
        the confidence intervals, by default 1000
    show : bool, optional
        Whether the plot is shown after the function call, by default False
    saveformat : str, optional
        Saveformat passed to plt.savefig, by default 'pdf'
    """
    global colors
    instruments = sorted(list(results.data.data_rv.keys()))
    if len(instruments) > len(colors):
        colors = sns.color_palette("husl", len(instruments))

    for instrument, color in zip(instruments, colors):
        fig, (ax, res) = plt.subplots(2,
                                      1,
                                      figsize=(7, 2),
                                      sharex=True,
                                      gridspec_kw={'height_ratios': [3, 1]})
        res.set_xlabel(f'BJD -{jd_offset}')
        res.set_ylabel('(O-C)')
        res.axhline(0, ls='--', color='grey')
        # res.set_xlim(min_t - jd_offset, max_t - jd_offset)
        ax.set_ylabel('RV [m/s]')

        model_times = utils.sample_model_times(results, samplingfreq, "RV",
                                               instrument)

        model_rv, model_components, _ = get_rv_model(results, nsamples,
                                                     instrument, model_times)
        try:
            rv_offset = model_components['mu'][instrument]
        except IndexError:
            rv_offset = model_components['mu']
        try:
            jitter = np.median(
                results.posteriors['posterior_samples']['sigma_w_' +
                                                        instrument])
        except KeyError:
            jitter = results.data.priors['sigma_w_' +
                                         instrument]['hyperparameters']
        ax.errorbar(results.data.times_rv[instrument] - jd_offset,
                    results.data.data_rv[instrument] - rv_offset,
                    yerr=np.sqrt(results.data.errors_rv[instrument]**2 +
                                 jitter**2),
                    fmt='o',
                    color=color,
                    markeredgecolor='black',
                    zorder=5)
        ax.plot(model_times - jd_offset,
                model_rv - rv_offset,
                color='black',
                zorder=5)
        if any("GP" in param for param in results.model_parameters):
            try:
                model_gp = model_rv - model_components['keplerian'] - rv_offset
                ax.plot(model_times - jd_offset,
                        model_gp,
                        color='#DBA039',
                        lw=1.5,
                        zorder=6)
            except KeyError:
                pass

        instr_model, _, _ = get_rv_model(results, nsamples, instrument,
                                         results.data.times_rv[instrument])

        res.errorbar(results.data.times_rv[instrument] - jd_offset,
                     results.data.data_rv[instrument] - instr_model,
                     yerr=np.sqrt(results.data.errors_rv[instrument]**2 +
                                  jitter**2),
                     fmt='o',
                     color=color,
                     markeredgecolor='black',
                     zorder=5)
        # Confidence intervals
        model_quantiles = get_rv_confidence_intervals(results, nsamples,
                                                      instrument, model_times)
        model_quantiles -= rv_offset
        for (upper, lower), color in zip(model_quantiles[::-1],
                                         ['grey', 'darkgrey', 'dimgrey']):
            ax.fill_between(model_times - jd_offset,
                            lower,
                            upper,
                            color=color,
                            zorder=0)
        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        fig.savefig(
            f'{results.data.out_folder}/full_rv_plot_{instrument}.{saveformat}',
            dpi=400,
            bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)
