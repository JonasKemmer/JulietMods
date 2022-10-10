import corner
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path

__author__ = "Jonas Kemmer @ ZAH, Landessternwarte Heidelberg"

sns.set(context='paper',
        font_scale=0.9,
        style='ticks',
        rc={
            "lines.linewidth": 1,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True
        })


def plot_loghist(ax, values, orientation):
    """Function to draw a log histogram given an figure axes and values."""
    hist, bins, _ = plt.hist(values, bins=40)
    logbins = np.logspace(
        np.log10(bins[0]),
        np.log10(bins[-1]),
        len(bins),
    )
    ax.hist(values,
            bins=logbins,
            orientation=orientation,
            log=False,
            edgecolor='black',
            facecolor='none')
    return ax


def plot_GPcorrelation(results,
                       correlation='alpha_vs_Prot',
                       model='RV',
                       label_model=False,
                       cbarpos=(0.5, 0.85, 1, 1),
                       colormap='coolwarm',
                       show=False,
                       saveformat='pdf'):
    """Plot two GP paramters against each other.


    Parameters
    ----------
    results : object
        Output from a juliet fit that contains all neccesary data.
    correlation : str, optional
        The correlation which is plotted. Implemented are:
        ['alpha_vs_Prot', 'sigma_vs_rho', 'omega0_vs_Q', 'Q0_vs_period',
         'dQ_vs_period', 'rho_vs_tau'], by default 'alpha_vs_Prot'
    model : str, optional
        Whether the GP from RV or Photometry is plotted, by default 'RV'
    label_model : bool, optional
        If True, the model is displayed in the figure, by default False
    cbarpos : tuple, optional
        Can be used to adjust the postition of the colorbar,
        by default (0.5, 0.85, 1, 1)
    colormap : str, optional
        Colormap passed to matplotlib, by default 'coolwarm'
    show : bool, optional
        Whether the plot is shown after the function call, by default False
    saveformat : str, optional
        Saveformat passed to plt.savefig, by default 'pdf'
    """
    # dictionary that contains the name as key and a list with
    # ["param1", "param2", logx, logy],
    correlations = {
        'alpha_vs_Prot': ['GP_Prot', 'GP_alpha', False, True],
        'sigma_vs_rho': ['GP_sigma', 'GP_rho', True, True],
        'omega0_vs_Q': ['GP_omega0', 'GP_Q', True, True],
        'Q0_vs_period': ['GP_period', 'GP_Q0', False, True],
        'dQ_vs_period': ['GP_period', 'GP_dQ', False, True],
        'rho_vs_tau': ['GP_rho', 'GP_tau', False, True],
    }
    # dictionary with the name of a correlation and the x and y labels to plot
    labels = {
        'alpha_vs_Prot': [
            r'$P_\mathrm{rot;GP}\,[\mathrm{d}]$',
            r'$\alpha_\mathrm{GP}\,[\mathrm{d}^{-2}]$'
        ],
        'sigma_vs_rho': [r'$\sigma_\mathrm{GP}$', r'$\rho_\mathrm{GP}$'],
        'omega0_vs_Q': [r'$\omega_\mathrm{0;GP}$', r'$Q_\mathrm{GP}$'],
        'Q0_vs_period': [
            r'$Period_\mathrm{GP}\,[\mathrm{d}]$', r'$Q_\mathrm{GP}$'
        ],
        'dQ_vs_period': [
            r'$Period_\mathrm{GP}\,[\mathrm{d}]$', r'$dQ_\mathrm{GP}$'
        ],
        'rho_vs_tau': [
            r'$Secondary Period_\mathrm{GP}\,[\mathrm{d}]$',
            r'$\tau_\mathrm{GP}$'
        ]
    }
    try:
        firstkey, secondkey, logx, logy = correlations[correlation]
    except KeyError as e:
        raise Exception(
            f'{e} is not a defined correlation. Possible options are: '
            f'{list(correlations.keys())}')
    if type(model) == list:
        firstkey += f'_{model[0]}'
        secondkey += f'_{model[1]}'
    else:
        if model.lower() == 'rv':
            firstkey += '_rv'
            secondkey += '_rv'
        elif model.lower() == 'photometry':
            firstkey += '_' + '_'.join(results.data.GP_lc_arguments.keys())
            secondkey += '_' + '_'.join(results.data.GP_lc_arguments.keys())
        else:
            firstkey += f'_{model}'
            secondkey += f'_{model}'

    maxloglike = np.amax(results.posteriors['posterior_samples']['loglike'])
    df = pd.DataFrame(np.array([
        results.posteriors['posterior_samples'][firstkey],
        results.posteriors['posterior_samples'][secondkey],
        results.posteriors['posterior_samples']['loglike']
    ]).T,
                      columns=['xval', 'yval', 'loglike'])
    df = df.sort_values(by='loglike')
    fig = plt.figure(figsize=(3.45, 3.45))
    gs = fig.add_gridspec(2,
                          2,
                          width_ratios=(7, 2),
                          height_ratios=(2, 7),
                          wspace=0.00,
                          hspace=0.00)

    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histx.set_yticks([])
    ax_histx.set(frame_on=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histy.set_xticks([])
    ax_histy.set(frame_on=False)

    ax.scatter(df.xval[df.loglike < maxloglike - 10],
               df.yval[df.loglike < maxloglike - 10],
               s=10,
               rasterized=True,
               color="grey")

    inner = ax.scatter(df.xval[df.loglike > maxloglike - 10],
                       df.yval[df.loglike > maxloglike - 10],
                       s=10,
                       c=df.loglike[df.loglike > maxloglike - 10],
                       cmap=plt.get_cmap(colormap),
                       vmin=maxloglike - 10,
                       vmax=maxloglike,
                       rasterized=True)
    if logx:
        ax_histx = plot_loghist(ax_histx, df.xval, 'vertical')
        ax.set_xscale('log')

    else:
        ax_histx.hist(df.xval,
                      bins=40,
                      histtype='bar',
                      edgecolor='black',
                      facecolor='none')

    if logy:
        ax_histy = plot_loghist(ax_histy, df.yval, 'horizontal')
        ax.set_yscale('log')
    else:
        ax_histy.hist(df.yval,
                      bins=40,
                      histtype='bar',
                      orientation='horizontal',
                      edgecolor='black',
                      facecolor='none')
    ax.set_ylim(df.yval.min(), df.yval.max())
    ax.set_xlim(df.xval.min(), df.xval.max())
    xlabel, ylabel = labels[correlation]
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='both', which='major')
    if label_model:
        ax.text(0.03,
                0.07,
                f"{model}",
                transform=ax.transAxes,
                horizontalalignment='left',
                zorder=5)
    axins = inset_axes(
        ax,
        width="45%",
        height="5%",
        loc='lower left',
        bbox_to_anchor=cbarpos,
        bbox_transform=ax.transAxes,
        borderpad=0,
    )

    cb = fig.colorbar(inner, cax=axins, orientation="horizontal")
    cb.set_label(r'$\ln\; \mathcal{L}$')
    if type(model) == list:
        name_post_script = '_vs_'.join(model)
    else:
        name_post_script = model
    fig.savefig(Path(results.data.out_folder,
                     f'GP-{correlation}_{name_post_script}.{saveformat}'),
                bbox_inches='tight',
                dpi=400)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_corner(results,
                subset=None,
                mark_values=None,
                plot_datapoints=True,
                plot_density=True,
                plot_contours=True,
                show=False,
                saveformat='pdf'):
    """Creates the cornerplot from the posterior of the sampling.
       A subset can be defined by specifying a unique string that is matched
       to the parameter names.

    Parameters
    ----------
    results : object
        Output from a juliet fit that contains all neccesary data.
    subset : str, optional
        If only a subset of the posterior should be shown, a unique string,
        or list of strings, can be passed that will be compared to the
        parameter names, by default None.
    mark_values : str, optional
        If 'maxlike' the maximum-likelihood samples will be marked.
        If 'median', the median samples will be marked, by default None.
    plot_datapoints: bool, optional
        Draw the individual data points, by default True
    plot_density: bool, optional
        Draw the density colormap, by default True
    plot_contours: bool, optional
        Draw the contours, by default True
    show : bool, optional
        Whether the plot is shown after the function call, by default False
    saveformat : str, optional
        Saveformat passed to plt.savefig, by default 'pdf'
    """
    if subset:
        keys = []
        for key in results.posteriors['posterior_samples'].keys():
            if isinstance(subset, str):
                if subset in key:
                    keys.append(key)
            elif isinstance(subset, list):
                for sub in subset:
                    if sub in key:
                        keys.append(key)
            else:
                raise TypeError('subset must be either string or list')
        if isinstance(subset, list):
            subset = subset[0]
    else:
        subset = '_all'
        keys = list(results.posteriors['posterior_samples'].keys())
        keys.remove('unnamed')
        keys.remove('loglike')
    data = np.array([results.posteriors['posterior_samples'][x] for x in keys])
    if mark_values == 'maxlike':
        idx_maxlike = np.argmax(
            results.posteriors['posterior_samples']['loglike'])
        truths = data[:, idx_maxlike]
    elif mark_values == 'median':
        truths = np.median(data, axis=1)
    else:
        truths = None
    corner_plot = corner.corner(data.T,
                                labels=keys,
                                quantiles=[0.16, 0.5, 0.84],
                                truths=truths,
                                show_titles=True,
                                title_fmt='.3f',
                                title_kwargs={"fontsize": 8},
                                label_kwargs={"fontsize": 8},
                                plot_datapoints=plot_datapoints,
                                plot_density=plot_density,
                                plot_contours=plot_contours)
    plt.savefig(Path(results.data.out_folder,
                     f'cornerplot{subset}.{saveformat}'),
                bbox_inches='tight',
                rasterized=True,
                dpi=400)
    if show:
        plt.show()
    else:
        plt.close()


def plot_parameter_correlation(results,
                               correlation='P_vs_K',
                               planets='p1',
                               label_model=False,
                               cbarpos=(0.5, 0.85, 1, 1),
                               colormap='coolwarm',
                               show=False,
                               saveformat='pdf'):
    """A function to plot two planetary posteriors against each other.

    Parameters
    ----------
    results : object
        Output from a juliet fit that contains all neccesary data.
    correlation : str, optional
        The correlation which is plotted. Implemented are:
        ['P_vs_K'], by default 'P_vs_K'
    planets : str/list, optional
        str or list of strings specifying the planets for which the
        correlation is plotted, by default 'p1'
    label_model : bool, optional
        If True, the model is displayed in the figure, by default False
    cbarpos : tuple, optional
        Can be used to adjust the postition of the colorbar,
        by default (0.5, 0.85, 1, 1)
    colormap : str, optional
        Colormap passed to matplotlib, by default 'coolwarm'
    show : bool, optional
        Whether the plot is shown after the function call, by default False
    saveformat : str, optional
        Saveformat passed to plt.savefig, by default 'pdf'
    """
    # dictionary that contains the name as key and a list with
    # ["param1", "param2", logx, logy],
    correlations = {
        'P_vs_K': ['P', 'K', False, False],
    }
    # dictionary with the name of a correlation and the x and y labels to plot
    labels = {
        'P_vs_K': [r'$Period\,[\mathrm{d}]$', r'$K\,[\mathrm{ms}^{-1}]$'],
    }
    try:
        firstkey, secondkey, logx, logy = correlations[correlation]
    except KeyError as e:
        raise Exception(
            f'{e} is not a defined correlation. Possible options are: '
            f'{list(correlations.keys())}')
    if type(planets) == list:
        firstkey += f'_{planets[0]}'
        secondkey += f'_{planets[1]}'
    else:
        firstkey += f'_{planets}'
        secondkey += f'_{planets}'

    maxloglike = np.amax(results.posteriors['posterior_samples']['loglike'])
    df = pd.DataFrame(np.array([
        results.posteriors['posterior_samples'][firstkey],
        results.posteriors['posterior_samples'][secondkey],
        results.posteriors['posterior_samples']['loglike']
    ]).T,
                      columns=['xval', 'yval', 'loglike'])
    df = df.sort_values(by='loglike')
    fig = plt.figure(figsize=(3.45, 3.45))
    gs = fig.add_gridspec(2,
                          2,
                          width_ratios=(7, 2),
                          height_ratios=(2, 7),
                          wspace=0.00,
                          hspace=0.00)

    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histx.set_yticks([])
    ax_histx.set(frame_on=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histy.set_xticks([])
    ax_histy.set(frame_on=False)

    ax.scatter(df.xval[df.loglike < maxloglike - 10],
               df.yval[df.loglike < maxloglike - 10],
               s=10,
               rasterized=True,
               color="grey")

    inner = ax.scatter(df.xval[df.loglike > maxloglike - 10],
                       df.yval[df.loglike > maxloglike - 10],
                       s=10,
                       c=df.loglike[df.loglike > maxloglike - 10],
                       cmap=plt.get_cmap(colormap),
                       vmin=maxloglike - 10,
                       vmax=maxloglike,
                       rasterized=True)
    if logx:
        ax_histx = plot_loghist(ax_histx, df.xval, 'vertical')
        ax.set_xscale('log')

    else:
        ax_histx.hist(df.xval,
                      bins=40,
                      histtype='bar',
                      edgecolor='black',
                      facecolor='none')

    if logy:
        ax_histy = plot_loghist(ax_histy, df.yval, 'horizontal')
        ax.set_yscale('log')
    else:
        ax_histy.hist(df.yval,
                      bins=40,
                      histtype='bar',
                      orientation='horizontal',
                      edgecolor='black',
                      facecolor='none')
    ax.set_ylim(df.yval.min(), df.yval.max())
    ax.set_xlim(df.xval.min(), df.xval.max())
    xlabel, ylabel = labels[correlation]
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='both', which='major')
    if label_model:
        ax.text(0.03,
                0.07,
                f"{planets}",
                transform=ax.transAxes,
                horizontalalignment='left',
                zorder=5)
    axins = inset_axes(
        ax,
        width="45%",
        height="5%",
        loc='lower left',
        bbox_to_anchor=cbarpos,
        bbox_transform=ax.transAxes,
        borderpad=0,
    )

    cb = fig.colorbar(inner, cax=axins, orientation="horizontal")
    cb.set_label(r'$\ln\; \mathcal{L}$')
    fig.savefig(Path(results.data.out_folder, f'{correlation}.{saveformat}'),
                bbox_inches='tight',
                dpi=400)
    if show:
        plt.show()
    else:
        plt.close(fig)
