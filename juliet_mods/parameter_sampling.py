# -*- coding: utf-8 -*-
import warnings

import astropy.units as u
import juliet
import numpy as np
from astropy.constants import G as grav_const
from astropy.modeling.models import BlackBody
from astropy.table import Table

import juliet_mods.utils as utils

__author__ = "Jonas Kemmer @ ZAH, Landessternwarte Heidelberg"
"""All used formulas are implemented using astropy units and quantities."""


def calc_msin_i(K=1 * u.m / u.s, e=0, P=1 * u.d, M_star=1 * u.Msun):
    """Calculates the minimum mass from radial velocities.

    Parameters
    ----------
    K : array_like
        Radial velocity semi-amplitude, by default 1*u.m/u.s
    e : array_like
        Eccentricity, by default 0
    P : array_like
        Period, by default 1*u.d
    M_star : array_like, optional
        Stellar mass, by default 1*u.Msun

    Returns
    -------
    array_like
        Planet Msin(i) in Earth masses (u.Mearth).
    """
    return (K * np.sqrt(1 - e**2) * (M_star)**(2 / 3) *
            (P / (2 * np.pi * grav_const))**(1 / 3)).to(u.Mearth)


def calc_au(P, M_star=1 * u.Msun):
    """Calculates the semi-major axis in astronomical units.

    Parameters
    ----------
    P : array_like
        Period
    M_star : array_like, optional
        Stellar mass, by default 1*u.Msun

    Returns
    -------
    float,array
        Planet semi-major axis in AU
    """
    return ((((P / u.yr)**2 * M_star / u.Msun)**(1 / 3)) * u.au).to(u.au)


def calc_a_from_rho(rho_star, P):
    """Calculates the scaled semi-major axis a/R_star from a transit,
       if the stellar density was used as a parametrisation.

    Parameters
    ----------
    rho_star : array_like
        Stellar density
    P : array_like
        Period

    Returns
    -------
    array_like
        Planet scaled semi-major axis in a/R_star
    """
    return ((rho_star * grav_const * P**2 / (3. * np.pi))**(1. /
                                                            3.)).decompose()


def calc_i(b, a, ecc, omega):
    """Calculates the inclination from a transit

    Parameters
    ----------
    b : array_like
        Impact parameter
    a : array_like
        Scaled semi-major axis
    ecc : array_like
        Eccentricity
    omega : array_like
        Argument of periastron

    Returns
    -------
    array_like
        Planet inclination in degree.
    """
    ecc_factor = (1. + ecc * np.sin(omega)) / (1. - ecc**2)
    inc_inv_factor = (b / a) * ecc_factor
    return (np.arccos(inc_inv_factor) * u.rad).to(u.deg)


def calc_Teq(a, T_star):
    """Calculates the planetary equilibrium temperature assuming zero Bond-albedo.

    Parameters
    ----------
    a : array_like
        Scaled semi-major axis
    T_star : array_like
        Stellar temperature in [K]

    Returns
    -------
    array_like
        Planetary equilibrium temperature for zero Bond-albedo.
    """
    return T_star / np.sqrt(2 * a)


def calc_S(au, L_star=1 * u.solLum):
    """Calculates the planetary instellation.

    Parameters
    ----------
    au : array_like
        Semi-major axis
    L_star : float, optional
        Stellar luminosity, by default 1

    Returns
    -------
    array_like
        Planetary instellation in multiples of the Earth's instellation
    """
    return (L_star / (u.solLum * (au / (1 * u.au))**2)).decompose()


def calc_ecc_omega(sesinomega, secosomega):
    """Conversion from the sqrt(ecc)*sin(omega*pi/180) and
       sqrt(ecc)*cos(omega*pi/180) parametrisation to eccentricity and omega.

    Parameters
    ----------
    sesinomega : array_like
        The juliet sesinomega parameter
    secosomega : array_like
        The juliet secosomega parameter

    Returns
    -------
    array_like
        Planetary eccentricity and argument of periastron.
    """
    omega = np.arctan2(sesinomega, secosomega) * 180 / np.pi
    ecc = sesinomega**2 + secosomega**2
    return ecc, omega


def calc_rho(r, m):
    """Calculates the density from radius and mass assuming spherical symmetry.

    Parameters
    ----------
    r : array_like
        Radius.
    m : array_like
        Mass.

    Returns
    -------
    array_like
        Density in g/cm³.
    """
    return (m / (4 / 3 * np.pi * r**3)).to(u.g / u.centimeter**3)


def calc_esm(Teq, T_star, Kmag, p):
    """Calculates the emission spectroscopy metric of Kempton et al. 2018
       https://ui.adsabs.harvard.edu/abs/2018PASP..130k4401K/abstract
       CAUTION: only meaningful for R_p < 1.5 R_earth.

    Parameters
    ----------
    Teq : array_like
        Planetary equilibrium temperature for zero albedo.
    T_star : array_like
        Effektive stellar temperature.
    Kmag : array_like
        Stellar K magnitude
    p : array_like
        Planet to star radius ratio.

    Returns
    -------
    array_like
        Emission spectroscopy metric.
    """
    return 4.29e6 * BlackBody(temperature=1.10 * Teq)(7.5e-6 * u.m) / BlackBody(
        temperature=T_star)(7.5e-6 * u.m) * p**2 * 10**(-Kmag / 5)


def calc_tsm(Teq, R_star, R_p, M_p, Jmag):
    """Calculates the transmission spectroscopy metric of Kempton et al. 2018
       https://ui.adsabs.harvard.edu/abs/2018PASP..130k4401K/abstract
       CAUTION: only meaningful for R_p < 10 R_earth.


    Parameters
    ----------
    Teq : array_like
        Planetary equilibrium temperature for zero albedo
    R_star : array_like
        Stellar radius
    R_p : array_like
        Planetary radius
    M_p : array_like
        Planetary mass
    Jmag : array_like
        Stellar J magnitude.

    Returns
    -------
    array_like
        Transmission spectroscopy metric
    """
    if np.median(R_p) < 1.5 * u.earthRad:
        S = 0.190
    elif np.logical_and(1.5 * u.earthRad < np.median(R_p),
                        np.median(R_p) < 2.75 * u.earthRad):
        S = 1.26
    elif np.logical_and(2.75 * u.earthRad < np.median(R_p),
                        np.median(R_p) < 4.0 * u.earthRad):
        S = 1.28
    elif np.logical_and(4 * u.earthRad < np.median(R_p),
                        np.median(R_p) < 10 * u.earthRad):
        S = 1.15
    else:
        S = np.nan
    return S * (
        (R_p.value**3) * Teq.value) / (M_p.value *
                                       (R_star.value**2)) * 10**(-Jmag / 5)


def calc_tt(P, a, p, b):
    """Calculates the transit time assuming a circular orbit.

    Parameters
    ----------
    P : array_like
        Planet period
    a : array_like
        Scaled planet semi-major axis
    p : array_like
        Planet-to-star radius ratio.
    b : array_like
        Impact parameter

    Returns
    -------
    array_like
        Planetary transit time.
    """
    return (np.pi * a)**(-1) * np.sqrt((1 + p)**2 - b**2) * P


def calc_g(m, r):
    """Calculates the gravitational acceleration.

    Parameters
    ----------
    m : array_like
        Mass
    r : array_like
        Radius

    Returns
    -------
    array_like
        Gravitational acceleration in [m/s²]
    """
    return (grav_const * m / r**2).to(u.m / u.second**2)


def _get_parameter(results, key, pnum):
    """Helper function to read out a posterior or get the fixed prior of a
    parameter."""
    try:
        param = results.posteriors['posterior_samples'][f'{key}{pnum}']
    except KeyError:
        try:
            param = results.data.priors[f'{key}{pnum}']['hyperparameters'][0]
        except IndexError:
            param = results.data.priors[f'{key}{pnum}']['hyperparameters']
    return param


def calculate_planet_parameters(results,
                                pnum,
                                R_star=None,
                                M_star=None,
                                L_star=None,
                                T_star=None,
                                Kmag=None,
                                Jmag=None):
    """A function to get a dictionary filled with the derived planet parameters
    from the posterior. The stellar parameters and their uncertainties are used
    to create a normal distribution with the same size as the posteriors.
    Note: table needs siunitx package.

    Parameters
    ----------
    results : object
        results from juliet.fit
    pnum : int
        Planet ID for which the parameters are determined.
    R_star : list, optional
        A tuple that contains the stellar radius and uncertainty in
        Solar radii, by default None
    M_star : list, optional
        A tuple that contains the stellar mass and uncertainty in
        Solar masses, by default None
    L_star : list, optional
        A tuple that contains the stellar luminosity and uncertainty
        in Solar luminosity, by default None
    T_star : list, optional
        A tuple that contains the stellar effective temperature and
        uncertaity in Kelvin, by default None
    Kmag : list, optional
        A tuple that contains the stellar K magnitude and uncertainty,
        by default None
    Jmag : list, optional
        A tuple that contains the stellar J magnitude and uncertainty,
        by default None

    Returns
    -------
    dict
        A dictionary with the derived planet parameters.
    """
    nsamples = len(results.posteriors['posterior_samples'][f'loglike'])
    if R_star:
        R_star = np.random.normal(R_star[0], R_star[1], size=nsamples) * u.Rsun
    if M_star:
        M_star = np.random.normal(M_star[0], M_star[1], size=nsamples) * u.Msun
    if L_star:
        L_star = np.random.normal(L_star[0], L_star[1],
                                  size=nsamples) * u.solLum
    if T_star:
        T_star = np.random.normal(T_star[0], T_star[1], size=nsamples) * u.K
    if Kmag:
        Kmag = np.random.normal(Kmag[0], Kmag[1], size=nsamples)
    if Jmag:
        Jmag = np.random.normal(Jmag[0], Jmag[1], size=nsamples)

    msini = None
    ecc = None
    omega = None
    a = None
    au = None
    Teq = None
    S = None
    b = None
    p = None
    r = None
    i = None
    m = None
    rho = None
    g = None
    esm = None
    tsm = None

    if pnum in results.data.numbering_rv_planets:
        P = _get_parameter(results, 'P_p', pnum) * u.day
        K = _get_parameter(results, 'K_p', pnum) * u.m / u.s
        if f'sesinomega_p{pnum}' in results.data.priors.keys():
            sesinomega = _get_parameter(results, 'sesinomega_p', pnum)
            secosomega = _get_parameter(results, 'secosomega_p', pnum)
            ecc, omega = calc_ecc_omega(sesinomega, secosomega)
        else:
            ecc = _get_parameter(results, 'ecc_p', pnum)
            omega = _get_parameter(results, 'omega_p', pnum)
        msini = calc_msin_i(K, ecc, P, M_star)
        au = calc_au(P, M_star)
        Teq = calc_Teq(au.to(u.Rsun) / R_star, T_star)
        S = calc_S(au, L_star)
    if pnum in results.data.numbering_transiting_planets:
        P = _get_parameter(results, 'P_p', pnum) * u.day
        if f'r1_p{pnum}' in results.data.priors.keys():
            r1 = _get_parameter(results, 'r1_p', pnum)
            r2 = _get_parameter(results, 'r2_p', pnum)
            b, p = juliet.utils.reverse_bp(r1, r2, results.posteriors['pl'],
                                           results.posteriors['pu'])
        else:
            b = _get_parameter(results, 'b_p', pnum)
            p = _get_parameter(results, 'p_p', pnum)
        if f'sesinomega_p{pnum}' in results.data.priors.keys():
            sesinomega = _get_parameter(results, 'sesinomega_p', pnum)
            secosomega = _get_parameter(results, 'secosomega_p', pnum)
            ecc, omega = calc_ecc_omega(sesinomega, secosomega)
        elif f'ecc_p{pnum}' in results.data.priors.keys():
            ecc = _get_parameter(results, 'ecc_p', pnum)
            omega = _get_parameter(results, 'omega_p', pnum)
        else:
            warnings.warn(
                'Warning: esinomega/ecosomega parametrisation not yet '
                'implemented. Parameters based on eccentricity not '
                'calculated')
        if "rho" in results.data.priors.keys():
            rho_star = _get_parameter(results, 'rho', '') * u.kg / u.m**3
            a = calc_a_from_rho(rho_star, P)
        else:
            a = _get_parameter(results, 'a_p', pnum)
        r = (p * R_star).to(u.earthRad)
        i = calc_i(b, a.value, ecc, omega)
        if msini:
            m = msini / np.sin(i)
            rho = calc_rho(r, m)
            g = calc_g(m, r)
        au = (a * R_star).to(u.au)
        Teq = calc_Teq(a, T_star)
        S = calc_S(au, L_star)
        if T_star is not None and Kmag is not None:
            esm = calc_esm(Teq, T_star, Kmag, p)
        if T_star is not None and Jmag is not None:
            tsm = calc_tsm(Teq, R_star, r, m, Jmag)

    datadict = {
        'p': utils.get_posterior_string(p),
        'b': utils.get_posterior_string(b),
        'a': utils.get_posterior_string(a),
        'i': utils.get_posterior_string(i),
        'M': utils.get_posterior_string(m),
        'Msini': utils.get_posterior_string(msini),
        'R': utils.get_posterior_string(r),
        'rho': utils.get_posterior_string(rho),
        'g': utils.get_posterior_string(g),
        'au': utils.get_posterior_string(au),
        'Teq': utils.get_posterior_string(Teq),
        'S': utils.get_posterior_string(S),
        'ESM': utils.get_posterior_string(esm),
        'TSM': utils.get_posterior_string(tsm)
    }

    return datadict


def create_planet_param_table(results,
                              R_star=None,
                              M_star=None,
                              L_star=None,
                              T_star=None,
                              Kmag=None,
                              Jmag=None,
                              paramdict=None,
                              saveformat='LaTeX'):
    """Function to create a table with the planet parameters derived from
    the posterior. Note: table needs siunitx package.

    Parameters
    ----------
    results : object
        juliet.fit results.
    R_star : list, optional
        A tuple that contains the stellar radius and uncertainty in
        Solar radii, by default None
    M_star : list, optional
        A tuple that contains the stellar mass and uncertainty in
        Solar masses, by default None
    L_star : list, optional
        A tuple that contains the stellar luminosity and uncertainty
        in Solar luminosity, by default None
    T_star : list, optional
        A tuple that contains the stellar effective temperature and
        uncertaity in Kelvin, by default None
    Kmag : list, optional
        A tuple that contains the stellar K magnitude and uncertainty,
        by default None
    Jmag : list, optional
        A tuple that contains the stellar J magnitude and uncertainty,
        by default None
    paramdict : dict, optional
        Can be used to set an own list of parameters that are shown in
        the table. Note that the parameters must in the dictionary
        created by 'calculate_planet_parameters', by default None
    saveformat : str, optional
        Saveformat must be either "LaTeX" or "html", by default 'LaTeX'

    Returns
    ------
    textfile
        A text file in the juliet "out_folder" with the name "pparam"
        either in LaTeX or html format.
    """
    if paramdict is None:
        paramdict = {
            'p': [r'$p = R_{\rm p}/R_\star$', r'\dots'],
            'b': [r'$b = (a_{\rm p}/R_\star)\cos i_{\rm p}$', r'\dots'],
            'a': [r'$a_{\rm p}/R_\star$', r'\dots'],
            'i': [r'$i_{\rm p}$', r'deg'],
            'M': [r'$M_{\rm p}$', r'$M_\oplus$'],
            'Msini': [r'$M_{\rm p}\sin i$', r'$M_\oplus$'],
            'R': [r'$R_{\rm p}$', r'$R_\oplus$'],
            'rho': [r'$\rho_{\rm p}$', r'\si{\gram\per\centi\meter\cubed}'],
            'g': [r'$g_{\rm p}$', r'\si{\meter\per\second\squared}'],
            'au': [r'$a_{\rm p}$', r'\si{\astronomicalunit}'],
            'Teq': [
                r'$T_\textnormal{eq, p}$\tablefootmark{({c})}', r'\si{\kelvin}'
            ],
            'S': [r'$S$', r'$S_\oplus$'],
            'ESM': [r'{ESM\tablefootmark{(d)}}', r'\dots'],
            'TSM': [r'TSM', r'\dots']
        }
    planets = []
    if results.data.n_rv_planets:
        maxnumrv = np.nanmax(results.data.numbering_rv_planets)
    else:
        maxnumrv = 0
    if results.data.n_transiting_planets:
        maxnumtran = np.nanmax(results.data.numbering_transiting_planets)
    else:
        maxnumtran = 0
    numplanets = int(max(maxnumrv, maxnumtran))
    names = []
    names.append("Parameter")
    for pnum in range(1, numplanets + 1):
        planets.append(
            calculate_planet_parameters(results, pnum, R_star, M_star, L_star,
                                        T_star, Kmag, Jmag))
        a = '(a)'
        names.append(f'Posterior P$_{pnum}$\\tablefootmark{({a})}')
    names.append("Units")
    tab = []
    for param in paramdict:
        row = []
        row.append(paramdict[param][0])
        for planet in planets:
            row.append(planet[param])
        row.append(paramdict[param][1])
        tab.append(row)
    tab = Table(np.array(tab), names=names)
    if saveformat == 'LaTeX':
        tab.write(
            f'{results.data.out_folder}/pparameter.tex',
            format='ascii.latex',
            latexdict={
                'tabletype': 'table',
                'header_start': '\\hline \\hline',
                'header_end': '\\hline',
                'data_end': '\\hline',
                'tablefoot':
                    r'\tablefoot{\tablefoottext{a}{Error bars denote the $68\%$ '
                    r'posterior credibility intervals.}}'
            },
            overwrite=True)
        tab.to_pandas()
    elif saveformat == 'html':
        tab = tab.to_pandas()
        tab.to_html(f'{results.data.out_folder}/pparameter.html')
    else:
        raise KeyError('Saveformat must be either "LaTeX" or "html"')
    return tab


def create_planet_posterior_table(results):
    """Function to create a LaTeX table with the planet and GP posterior parameters
    called "pposterior.tex" in the juliet "out_folder".
    """
    conversion_dict = {
        'rho': [r'$\rho_\star$', r'\si{\kilo\gram\per\meter\cubed}'],
        'P_p1': [r'$P_b$', 'd'],
        't0_p1': [r'$t_{0,b}$', r'd'],
        'r1_p1': [r'$r_{1,b}$', r'\dots'],
        'r2_p1': [r'$r_{2,b}$', r'\dots'],
        'secosomega_p1': [r'$\sqrt{e_{b}}\cos \omega_{b}$', r'\dots'],
        'sesinomega_p1': [r'$\sqrt{e_{b}}\sin \omega_{b}$', r'\dots'],
        'K_p1': [r'$K_{b}$', r'$\mathrm{m\,s^{-1}}$'],
        'P_p2': [r'$P_c$', r'd'],
        't0_p2': [r'$t_{0,c}$', r'd'],
        'r1_p2': [r'$r_{1,c}$', r'\dots'],
        'r2_p2': [r'$r_{2,c}$', r'\dots'],
        'secosomega_p2': [r'$\sqrt{e_{c}}\cos \omega_{c}$', r'\dots'],
        'sesinomega_p2': [r'$\sqrt{e_{c}}\sin \omega_{c}$', r'\dots'],
        'K_p2': [r'$K_{c}$', r'$\mathrm{m\,s^{-1}}$'],
        'P_p3': [r'$P_d$', 'd'],
        't0_p3': [r'$t_{0,d}$', r'd'],
        'r1_p3': [r'$r_{1,d}$', r'\dots'],
        'r2_p3': [r'$r_{2,d}$', r'\dots'],
        'secosomega_p3': [r'$\sqrt{e_{d}}\cos \omega_{d}$', r'\dots'],
        'sesinomega_p3': [r'$\sqrt{e_{d}}\sin \omega_{d}$', r'\dots'],
        'K_p3': [r'$K_{d}$', r'$\mathrm{m\,s^{-1}}$'],
        'GP_period_rv': [r'$P_\text{GP, rv}$', 'd'],
        'GP_sigma_rv': [r'$\sigma_\text{GP, rv}$', r'$\mathrm{m\,s^{-1}}$'],
        'GP_Q0_rv': [r'$Q_{0, \text{GP, rv}}$', r'\dots'],
        'GP_f_rv': [r'$f_\text{GP, rv}$', r'\dots'],
        'GP_dQ_rv': [r'$dQ_\text{GP, rv}$', r'\dots'],
    }
    posterior = results.posteriors['posterior_samples']
    tab = []
    for key in conversion_dict:
        try:
            posterior_string = utils.get_posterior_string(posterior[key])
        except KeyError:
            continue
        tab.append([
            conversion_dict[key][0], posterior_string, conversion_dict[key][1]
        ])
    tab = Table(np.array(tab),
                names=["Parameter", r"Posterior\tablefootmark{(a)}", "Units"])
    tab.write(
        f'{results.data.out_folder}/pposterior.tex',
        format='ascii.latex',
        latexdict={
            'tabletype': 'table',
            'header_start': '\\hline \\hline',
            'header_end': '\\hline',
            'data_end': '\\hline',
            'tablefoot':
                r'\tablefoot{\tablefoottext{a}{Error bars denote the $68\%$ '
                r'posterior credibility intervals.}}'
        },
        overwrite=True)
