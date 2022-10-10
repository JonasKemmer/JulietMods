# -*- coding: utf-8 -*-
from copy import deepcopy

import juliet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

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


def get_lc_model(results, nsamples, instrument, model_times):
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
        lm_args = resultsdummy.data.lm_lc_arguments[instrument]
    except (KeyError, TypeError):
        lm_args = None
    model_lc, components = resultsdummy.lc.evaluate(instrument,
                                                    t=model_times,
                                                    GPregressors=model_times,
                                                    LMregressors=lm_args,
                                                    nsamples=nsamples,
                                                    return_components=True)

    return model_lc, components, resultsdummy


def get_lc_confidence_intervals(results, nsamples, instrument, model_times):
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
            lm_args = resultsdummy.data.lm_lc_arguments[instrument]
        except (KeyError, TypeError):
            lm_args = None
        _, upper, lower = resultsdummy.lc.evaluate(instrument,
                                                   t=model_times,
                                                   GPregressors=model_times,
                                                   LMregressors=lm_args,
                                                   nsamples=nsamples,
                                                   return_err=True,
                                                   alpha=quantile)
        model_quantiles.append([upper, lower])
    model_quantiles = np.array(model_quantiles)
    return model_quantiles


def plot_lc_indv_panels(results,
                        jd_offset=2457000,
                        samplingfreq='2min',
                        nsamples=1000,
                        show_binned=True,
                        binlength='10min',
                        show=False,
                        saveformat='pdf'):
    """Plots a non-phased light curve of a all instruments using individual panels.

    Parameters
    ----------
    results : object
        Output from a juliet fit that contains all neccesary data.
    jd_offset : int, optional
        Offset for the x-axis, by default 2457000
    samplingfreq : str, optional
        Samplingfrequency of the model, by default '2min'
    nsamples : int, optional
        Number of models which are drawn from the posterior to derive
        the confidence intervals, by default 1000
    show_binned : bool, optional
        Whether to overlay binned datapoints.
        Errorbars are the standart deviation of the data bins, by default True
    binlength : str, optional
        Timespan for the data bins, by default '10min'
    show : bool, optional
        Whether the plot is shown after the function call, by default False
    saveformat : str, optional
        Saveformat passed to plt.savefig, by default 'pdf'
    """
    for instrument in sorted(list(results.data.data_lc.keys())):
        print(instrument)
        fig, (ax, res) = plt.subplots(2,
                                      1,
                                      figsize=(7, 2),
                                      sharex=True,
                                      gridspec_kw={'height_ratios': [3, 1]})
        res.set_xlabel(f'BJD -{jd_offset}')
        res.set_ylabel('(O-C)')
        res.axhline(0, ls='--', color='grey')
        ax.set_ylabel('Relative flux')

        if any("theta" in param and instrument in param
               for param in results.model_parameters):
            print('Linear model in LC detected, sampling model as data.')
            model_times = results.data.times_lc[instrument]
        else:
            model_times = utils.sample_model_times(results, samplingfreq, 'LC',
                                                   instrument)

        model_lc, model_components, _ = get_lc_model(results, nsamples,
                                                     instrument, model_times)

        # try:
        #     lc_offset = np.median(
        #         results.posteriors['posterior_samples'][f'mflux_{instrument}'])
        # except IndexError:
        #     lc_offset = results.data.priors[f'mflux_{instruments}'][
        #         'hyperparameters']

        ax.scatter(
            results.data.times_lc[instrument] - jd_offset,
            results.data.data_lc[instrument],  # + lc_offset,
            marker='.',
            s=8,  # s=0.6,
            alpha=0.8,
            color='#5D8CC2')

        ax.plot(
            model_times - jd_offset,
            model_lc,  # + lc_offset,
            color='black',
            zorder=5)
        if any("GP" in param for param in results.model_parameters):
            try:
                model_gp = model_lc - model_components[
                    'keplerian']  # + lc_offset
                ax.plot(model_times - jd_offset,
                        model_gp,
                        color='#DBA039',
                        lw=1.5)
            except KeyError:
                pass
        instr_model, _, _ = get_lc_model(results, nsamples, instrument,
                                         results.data.times_lc[instrument])

        res.scatter(results.data.times_lc[instrument] - jd_offset,
                    results.data.data_lc[instrument] - instr_model,
                    marker='.',
                    s=0.75,
                    alpha=0.8,
                    color='#5D8CC2')

        # Confidence intervals
        model_quantiles = get_lc_confidence_intervals(results, nsamples,
                                                      instrument, model_times)
        # model_quantiles += lc_offset
        for (upper, lower), color in zip(model_quantiles[::-1],
                                         ['grey', 'darkgrey', 'dimgrey']):
            ax.fill_between(model_times - jd_offset,
                            lower,
                            upper,
                            color=color,
                            zorder=0)

        if show_binned:
            binned_data = utils.bin_data(results.data.times_lc[instrument],
                                         results.data.data_lc[instrument],
                                         binlength=binlength)
            binned_model = utils.bin_data(results.data.times_lc[instrument],
                                          instr_model,
                                          binlength=binlength)
            ax.errorbar(binned_data['jd'] - jd_offset,
                        binned_data['values'],
                        yerr=binned_data['errors'],
                        fmt='o',
                        markersize=2.5,
                        markeredgecolor='black',
                        markerfacecolor="None",
                        ecolor='black')
            res.errorbar(binned_data['jd'] - jd_offset,
                         binned_data['values'] - binned_model['values'],
                         yerr=binned_data['errors'],
                         fmt='o',
                         markersize=2.5,
                         markeredgecolor='black',
                         markerfacecolor="None",
                         ecolor='black')

        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        fig.savefig(
            f'{results.data.out_folder}/full_lc_plot_{instrument}.{saveformat}',
            dpi=400,
            bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)


def _plot_phased_lc_(fig,
                     axes,
                     results,
                     pnum,
                     instrument,
                     minphase=-0.5,
                     maxphase=0.5,
                     nsamples=1000,
                     show_binned=True,
                     binlength=0.005):
    """Plots the phased light curves for all planets, given an instrument.

    Parameters
    ----------
    results : object
        Output from a juliet fit that contains all neccesary data.
    pnum: int
        Planet-ID for which the light curve is generated.
    instrument : str
        Name of the instrument for which the light curve is plotted.
    minphase : float, optional
        Lower limit of x-axis, by default -0.5
    maxphase : float, optional
        Upper limit of a-axis, by default 0.5
    nsamples : int, optional
        Number of models which are drawn from the posterior to derive
        the confidence intervals, by default 1000
    show_binned : bool, optional
        Whether to overlay binned datapoints.
        Errorbars are the standart deviation of the data bins, by default True
    binlength : str, optional
        Span of the phase-bins, by default 0.005
    show : bool, optional
        Whether the plot is shown after the function call, by default False
    saveformat : str, optional
        Saveformat passed to plt.savefig, by default 'pdf'
    """
    ax, res = axes
    P, t0 = utils.get_period_and_t0(results, pnum)
    try:
        lmreg = results.data.lm_lc_arguments[instrument]
        model_times = results.data.times_lc[instrument]
        model_phases = juliet.get_phases(results.data.times_lc[instrument], P,
                                         t0)
    except (KeyError, TypeError):
        lmreg = None
        model_phases = np.linspace(minphase, maxphase, 1000)
        model_times = model_phases * P + t0
        model_times = model_times[model_times.argsort()]

    model_lc, components, _, = get_lc_model(results, nsamples, instrument,
                                            model_times)

    model_other_p = model_lc - components[f'p{pnum}']
    model_quantiles = get_lc_confidence_intervals(results, nsamples, instrument,
                                                  model_times)
    model_quantiles -= model_other_p

    # Plot model
    ax.plot(model_phases, components[f'p{pnum}'], color='black', zorder=10)
    for (upper, lower), color in zip(model_quantiles[::-1],
                                     ['grey', 'darkgrey', 'dimgrey']):
        ax.fill_between(model_phases, lower, upper, color=color, zorder=0)
    resultsdummy = deepcopy(results)
    model_lc, components = resultsdummy.lc.evaluate(
        instrument,
        t=results.data.times_lc[instrument],
        GPregressors=results.data.times_lc[instrument],
        LMregressors=lmreg,
        nsamples=nsamples,
        return_components=True)
    model_other_p = model_lc - components[f'p{pnum}']
    data_phases = juliet.get_phases(results.data.times_lc[instrument], P, t0)
    plotmask = np.logical_and(data_phases > minphase, data_phases < maxphase)
    data_phases = data_phases[plotmask]
    data_lc = results.data.data_lc[instrument][plotmask]
    model_lc = model_lc[plotmask]
    model_other_p = model_other_p[plotmask]
    ax.scatter(data_phases,
               data_lc - model_other_p,
               marker='.',
               s=0.6,
               alpha=0.8,
               color='#5D8CC2')
    res.scatter(data_phases,
                data_lc - model_lc,
                marker='.',
                s=0.75,
                alpha=0.8,
                color='#5D8CC2')
    if show_binned:
        binned_data = utils.bin_phased_data(data_phases, data_lc, binlength)
        binned_model = utils.bin_phased_data(data_phases, model_lc, binlength)
        binned_other_p = utils.bin_phased_data(data_phases, model_other_p,
                                               binlength)
        ax.errorbar(binned_data['phases'],
                    binned_data['values'] - binned_other_p['values'],
                    yerr=binned_data['errors'],
                    fmt='o',
                    markersize=2.5,
                    markeredgecolor='black',
                    markerfacecolor="None",
                    ecolor='black')
        res.errorbar(binned_data['phases'],
                     binned_data['values'] - binned_model['values'],
                     yerr=binned_data['errors'],
                     fmt='o',
                     markersize=2.5,
                     markeredgecolor='black',
                     markerfacecolor="None",
                     ecolor='black')
        res.set_xlim(data_phases.min(), data_phases.max())
    return fig, axes


def plot_phased_lcs(results,
                    minphase=-0.5,
                    maxphase=0.5,
                    yrange_quantile=[0.05, 0.95],
                    nsamples=1000,
                    show_binned=True,
                    binlength=0.005,
                    show=False,
                    saveformat='pdf'):
    """Plots the phased light curves.

    Parameters
    ----------
    results : object
        Output from a juliet fit that contains all neccesary data.
    minphase : float, optional
        Lower limit of x-axis, by default -0.5
    maxphase : float, optional
        Upper limit of a-axis, by default 0.5
    yrange_quantile : array_like
        Quantiles of the data to show on the yrange.
    nsamples : int, optional
        Number of models which are drawn from the posterior to derive
        the confidence intervals, by default 1000
    show_binned : bool, optional
        Whether to overlay binned datapoints.
        Errorbars are the standart deviation of the data bins, by default True
    binlength : str, optional
        Span of the phase-bins, by default 0.005
    show : bool, optional
        Whether the plot is shown after the function call, by default False
    saveformat : str, optional
        Saveformat passed to plt.savefig, by default 'pdf'
    """
    for instrument in sorted(list(results.data.data_lc.keys())):
        for pnum in range(1, results.data.n_transiting_planets + 1):
            fig, (ax, res) = plt.subplots(2,
                                          1,
                                          figsize=(3.45, 3),
                                          sharex=True,
                                          gridspec_kw={'height_ratios': [3, 1]})

            res.set_xlabel('Phase')
            res.set_ylabel('(O-C)')
            res.axhline(0, ls='--', color='grey')
            ax.set_ylabel('Relative flux')
            fig, (ax, res) = _plot_phased_lc_(
                fig,
                (ax, res),
                results,
                pnum,
                instrument,
                minphase=minphase,
                maxphase=maxphase,
                nsamples=nsamples,
                show_binned=show_binned,
                binlength=binlength,
            )
            lcmin, lcmax = np.quantile(results.data.y_lc, yrange_quantile)
            ax.set_ylim(lcmin, lcmax)
            res.set_ylim(1 - lcmax, 1 - lcmin)
            ax.set_title(f'{instrument}')
            fig.tight_layout()
            fig.subplots_adjust(hspace=0)
            fig.savefig(
                f'{results.data.out_folder}/{instrument}_p{pnum}_phased_lcplot.'
                f'{saveformat}',
                dpi=400,
                bbox_inches='tight')
            if show:
                plt.show()
            else:
                plt.close(fig)


def plot_transit_overview(results,
                          pnum,
                          ncols=3,
                          minphase=-0.5,
                          maxphase=0.5,
                          yrange_quantile=[0.05, 0.95],
                          nsamples=1000,
                          show_binned=True,
                          binlength=0.005,
                          labelpos='title',
                          show=False,
                          saveformat='png'):
    """Plot an overview of the light curves from all instruments,
       given a planet number.
       Important: TESS data should be named: "TESS-{sector}",
       other instruments like: {instrument}-{filt}-{date}.

    Parameters
    ----------
    results : object
        Output from a juliet fit that contains all neccesary data.
    pnum : int
        Planet-ID for which the overview is generated.
    ncols : int, optional
        Number of columns of the plot, by default 3
    minphase : float, optional
        Lower limit of x-axis, by default -0.5
    maxphase : float, optional
        Upper limit of a-axis, by default 0.5
    yrange_quantile : array_like
        Quantiles of the data to show on the yrange.
    nsamples : int, optional
        Number of models which are drawn from the posterior to derive
        the confidence intervals, by default 1000
    show_binned : bool, optional
        Whether to overlay binned datapoints.
        Errorbars are the standart deviation of the data bins, by default True
    binlength : str, optional
        Span of the phase-bins, by default 0.005
    labelpos : str
        Position where the instrument label is plotted. Can be either
        as "title" or as an "annotation" in the plot panel, by deault 'title'
    show : bool, optional
        Whether the plot is shown after the function call, by default False
    saveformat : str, optional
        Saveformat passed to plt.savefig, by default 'png' to
        optimise the load time in a pdf by rasterizing.
    """

    instruments = []
    dates = []
    for instrument in np.unique(list(results.data.data_lc.keys())):
        dates.append(np.median(results.data.times_lc[instrument]))
        instruments.append(instrument)
    sort_idx = np.argsort(dates)
    instruments = np.array(instruments)[sort_idx]

    num_plots = results.data.ninstruments_lc
    nrows = int(np.ceil(num_plots / ncols))

    fig = plt.figure(constrained_layout=False, figsize=(7, nrows * 1.8))
    outer_grid = fig.add_gridspec(ncols=ncols, nrows=nrows)

    for idx, instrument in enumerate(instruments):
        inner_grid = outer_grid[idx].subgridspec(ncols=1,
                                                 nrows=2,
                                                 height_ratios=[3, 1],
                                                 hspace=0)
        ax = fig.add_subplot(inner_grid[0])
        res = fig.add_subplot(inner_grid[1], sharex=ax)

        fig, (ax, res) = _plot_phased_lc_(
            fig,
            (ax, res),
            results,
            pnum,
            instrument,
            minphase=minphase,
            maxphase=maxphase,
            nsamples=nsamples,
            show_binned=show_binned,
            binlength=binlength,
        )
        midpoint = np.median(results.data.times_lc[instrument])
        date = pd.to_datetime(midpoint, origin='julian', unit='D')
        if 'TESS' in instrument:
            try:
                sector = instrument.split('-')[1][1:]
                instr_str = f'$TESS$ \n Sector {sector}'
                label = f'{instr_str}'
            except IndexError:
                label = 'TESS'

        elif 'OSN' in instrument:
            instr_str = '-'.join(instrument.split('-')[:-2])
            label = f'{instr_str} \n {date:%d %b, %Y}'

        elif len(instrument.split('-')) == 3:
            instr_str = instrument.split('-')[0]
            filt = instrument.split('-')[-2]
            label = f'{instr_str}$_{{{filt}}}$ \n {date:%d %b, %Y}'

        else:
            try:
                instr_str = '-'.join(instrument.split('-')[:-2])
                filt = instrument.split('-')[-2]
                label = f'{instr_str}$_{{{filt}}}$ \n {date:%d %b, %Y}'
            except IndexError:
                print('Cannot read instrument information. '
                      'TESS data should be named: "TESS-{sector}", '
                      'other instruments like: {instrument}-{filt}-{date}. '
                      'Falling back to the plain instrument name.')
                label = f'{instrument} \n {date:%d %b, %Y}'

        if labelpos.lower() == "title":
            ax.set_title(label, fontsize='small')
        elif labelpos.lower() == "annotation":
            boxprops = dict(facecolor='grey',
                            alpha=0.5,
                            edgecolor='grey',
                            boxstyle="round")
            ax.annotate(label, (0.1, 0.8),
                        xycoords='axes fraction',
                        va='center',
                        bbox=boxprops,
                        zorder=11)
        else:
            raise KeyError('"labelpos" must be either "title" or "annotation"')
        lcmin, lcmax = np.quantile(results.data.y_lc, yrange_quantile)
        ax.set_ylim(lcmin, lcmax)
        res.set_ylim(1 - lcmax, 1 - lcmin)
        res.set_xlim(minphase, maxphase)

    def get_idx_first_col(ncols):
        idxes = np.arange(0, num_plots * 2, ncols * 2)
        return np.append(idxes, np.arange(0, num_plots * 2, ncols * 2) + 1)

    def get_idx_last_row(num_plots, ncols):
        idxes = []
        for sub in range(ncols):
            idxes.append(num_plots * 2 - (2 * sub) - 1)
        return idxes

    first_cols = get_idx_first_col(ncols)
    last_rows = get_idx_last_row(num_plots, ncols)
    for idx, ax in enumerate(fig.get_axes()):
        ax.xaxis.set_major_locator(
            MaxNLocator(nbins=4, prune='both', symmetric=True))
        if idx not in first_cols:
            ax.tick_params(labelleft=False)
        elif idx % ncols == 0:
            ax.set_ylabel('Relative flux')
        else:
            ax.set_ylabel('(O-C)')
        if idx not in last_rows:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel('Phase')
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.06, hspace=0.4)
    fig.savefig(
        f'{results.data.out_folder}/transit_overview_p{pnum}.{saveformat}',
        dpi=500,
        bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
