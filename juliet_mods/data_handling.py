# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import juliet
import juliet_mods.utils as utils

__author__ = "Jonas Kemmer @ ZAH, Landessternwarte Heidelberg"


class Data(object):
    """A data class that handels all the needed input data for juliet.

    The main functions are "add_photometry" and "add_rv", which convert the
    the given files to the dictionaries needed by juliets "load" function.


    Parameters
    ----------
    phot_instr : list
        A list of the names of the photometric instruments.
    fluxes : dict
        Contains the fluxes. Input for "y_lc".
    fluxes_err: dict
        Contains the flux uncertainties. Input for "yerr_lc".
    t_fluxes = dict
        Contains the timestamps of the fluxes. Input for "t_lc".
    linear_regressors_lc = dict
        Contains linear regressors for the light curve.
        Input for "linear_regressors_lc".
    gp_reg_fluxes = dict
        Contains GP regressors for the light curve.
        Input for "GP_regressors_lc".
    rv_instr = list
        A list of the names of the photometric instruments.
    rvs = dict
        Contains the RVs. Input for "y_rv".
    rvs_err = dict
        Contains the RV uncertainties. Input for "yerr_rv".
    t_rvs = dict
        Contains the timestamps of the RVs. Input for "t_rv".
    linear_regressors_rv = dict
        Contains linear regressors for the RVs.
        Input for "linear_regressors_rv".
    gp_reg_rvs = dict
        Contains GP regressors for the Rvs.
        Input for "GP_regressors_rv".
    """

    def __init__(self):
        self.phot_instr = None
        self.fluxes = {}
        self.fluxes_err = {}
        self.t_fluxes = {}
        self.linear_regressors_lc = {}
        self.gp_reg_fluxes = None
        self._global_gp_fluxes = False
        self.rv_instr = None
        self.rvs = {}
        self.rvs_err = {}
        self.t_rvs = {}
        self.linear_regressors_rv = {}
        self.gp_reg_rvs = None
        self._global_gp_rvs = True

    def _read_tab(self, path, instrument, columns):
        if "http" in path or "www" in path:
            times, fluxes, flux_err = juliet.get_TESS_data(path)
            tab = pd.DataFrame(np.array([times + 2457000, fluxes, flux_err]).T,
                               columns=['times', 'yval', 'yval_err'])
            tab = tab.round(6)
            if instrument is not None:
                tab['instr'] = instrument
            else:
                tab['instr'] = 'TESS'
        else:
            try:
                tab = pd.read_table(path, sep="\s+|,|;", header=None)
            except Exception:
                tab = pd.read_fwf(path, header=None)
            tab = tab.rename(columns=columns)
            if instrument is not None:
                tab['instr'] = instrument
        return tab

    def add_photometry(self,
                       path,
                       instrument=None,
                       columns={
                           0: 'times',
                           1: 'yval',
                           2: 'yval_err',
                           3: 'instr'
                       },
                       global_gp=False,
                       offset=0):
        """Function to read photometry from a file.
        The file can contain multiple instruments, which need to be specified in
        a column.

        Parameters
        ----------
        path : str
            Path to the file.
        instrument : str, optional
            Name of the instrument, by default None.
            If "None", the name needs to be given in a instrument column
            in the file. This allows to read multiple instruments from one file.
        columns : dict, optional
            A dictionary that links the table columns with the input keywords,
            by default { 0: 'times', 1: 'yval', 2: 'yval_err', 3: 'instr' }.
            "times", "yval" and "yval_err" are mandatory.
            Linear regressors must contain "linreg" in the column name.
        global_gp : bool, optional
            Defines if the created GP regressors are global or instrument
            specific, by default False
        offset : int, optional
            An offset that can be added to the photometry, by default 0
        """
        tab = self._read_tab(path, instrument, columns)
        for instr in np.unique(tab['instr']):
            mask = tab['instr'] == instr
            self.fluxes[f'{instr}'] = tab[mask]['yval'].values
            self.fluxes_err[f'{instr}'] = tab[mask]['yval_err'].values
            self.t_fluxes[f'{instr}'] = tab[mask]['times'].values + offset
            linregs = []
            for colname in tab:
                if type(colname) == str:
                    if 'linreg' in colname:
                        linregs.append(colname)
            if len(linregs) > 0:
                self.linear_regressors_lc[f'{instr}'] = np.atleast_2d(
                    tab[linregs].values)
            self.phot_instr = list(self.fluxes.keys())
        if global_gp:
            self._global_gp_fluxes = True
            self.gp_reg_fluxes = self._create_gp_reg('lc')
        else:
            self._global_gp_fluxes = False
            self.gp_reg_fluxes = self.t_fluxes

    def remove_photometry(self, instr):
        """A helper function to remove photometry from the data class.

        Parameters
        ----------
        instr : str
            Name of the instrument that should be removed.
        """
        del self.fluxes[instr]
        del self.fluxes_err[instr]
        del self.t_fluxes[instr]
        try:
            del self.linear_regressors_lc[instr]
        except KeyError:
            pass
        self.phot_instr = list(self.fluxes.keys())
        if self._global_gp_fluxes:
            self.gp_reg_fluxes = self._create_gp_reg('lc')
        else:
            self.gp_reg_fluxes = self.t_fluxes

    def add_rv(self,
               path,
               instrument=None,
               columns={
                   0: 'times',
                   1: 'yval',
                   2: 'yval_err',
                   3: 'instr'
               },
               global_gp=True,
               offset=0):
        """Function to read RVs from a file.
        The file can contain multiple instruments, which need to be specified in
        a column.


        Parameters
        ----------
        path : str
            Path to the file.
        instrument : str, optional
            Name of the instrument, by default None.
            If "None", the name needs to be given in a instrument column
            in the file. This allows to read multiple instruments from one file.
        columns : dict, optional
            A dictionary that links the table columns with the input keywords,
            by default { 0: 'times', 1: 'yval', 2: 'yval_err', 3: 'instr' }.
            "times", "yval" and "yval_err" are mandatory.
            Linear regressors must contain "linreg" in the column name.
        global_gp : bool, optional
            Defines if the created GP regressors are global or instrument
            specific, by default False
        offset : int, optional
            An offset that can be added to the photometry, by default 0
        """
        tab = self._read_tab(path, instrument, columns)
        for instr in np.unique(tab['instr']):
            mask = tab['instr'] == instr
            self.rvs[instr] = tab[mask]['yval'].values
            self.rvs_err[instr] = tab[mask]['yval_err'].values
            self.t_rvs[instr] = tab[mask]['times'].values + offset
            linregs = []
            for colname in tab:
                if type(colname) == str:
                    if 'linreg' in colname:
                        linregs.append(colname)
            if len(linregs) > 0:
                self.linear_regressors_rv[f'{instr}'] = np.atleast_2d(
                    tab[linregs].values)
        self.rv_instr = list(self.rvs.keys())
        if global_gp:
            self._global_gp_rvs = True
            self.gp_reg_rvs = self._create_gp_reg('rv')
        else:
            self._global_gp_rvs = False
            self.gp_reg_rvs = self.t_rvs

    def remove_rv(self, instr):
        """A helper function to remove photometry from the data class.

        Parameters:
        -----------
            instr : str
             Name of the instrument that should be removed.
        """
        del self.rvs[instr]
        del self.rvs_err[instr]
        del self.t_rvs[instr]
        self.rv_instr = list(self.rvs.keys())
        if self.global_gp_rv:
            self.gp_reg_rvs = self._create_gp_reg('rv')
        else:
            self.gp_reg_rvs = self.t_rvs

    def _create_gp_reg(self, model):
        """Function to create the GP regressors"""
        flat = []
        if model == 'rv':
            for key in self.rvs.keys():
                flat.append(np.atleast_1d(self.t_rvs[key]))
        elif model == 'lc':
            for key in self.fluxes.keys():
                flat.append(np.atleast_1d(self.t_fluxes[key]))
        flat = np.hstack(flat)
        return {model: flat[flat.argsort()]}


class Prior(object):
    """A simple object to store prior information

    Parameters
    ----------
        pparam: str
            The parameter name of the prior.
        ptype: str
            The prior type.
        prange: float, list
            The prior range.
    """

    def __init__(self, pparam, ptype, prange):
        self.pparam = pparam
        self.ptype = ptype.lower()
        self.prange = prange

    def show(self):
        print(self.pparam, self.ptype, self.prange)


class PriorList(object):
    """A class that creates the prior dictionaries needed for juliet
       and provides some convenience functions.

       The main functions are "add", which inputs a Prior object and adds
       it to the PriorList and "read_prior_file", which can be used to read in
       a prior textfile in the old juliet format.

    Parameters
    ----------
    priordict : dict
        The prior dictionary that is needed for the juliet input.
    num_pl : int
        Returns the number of planets from the prior list.
    """

    def __init__(self, plist=[]):
        self.priordict = {}
        for prior in plist:
            self.add(prior)
        self.num_pl = self._count_planets_()

    def add(self, prior):
        """Function to add a prior to the list. Needs a Prior object as input."""
        self.priordict[prior.pparam] = {}
        self.priordict[prior.pparam]['distribution'] = prior.ptype
        self.priordict[prior.pparam]['hyperparameters'] = np.array(prior.prange)
        self.pnames = list(self.priordict.keys())
        self.nplanets = self._count_planets_()

    def remove(self, pparam):
        """Removes a prior from the list, given a prior name."""
        del self.priordict[pparam]
        self.pnames = list(self.priordict.keys())
        self.nplanets = self._count_planets_()

    def show(self):
        """Prints the priorlist"""
        print(self.priordict)

    def _count_planets_(self):
        num_pl = 0
        for key in np.unique(list(map(lambda x: x.split('_')[-1],
                                      self.pnames))):
            if 'p' in key:
                num_pl += 1
        return num_pl

    def read_prior_file(path):
        """Reads in a prior file in the old juliet format. Takes
            a path string as input."""
        file = pd.read_table(path, sep='\s+', header=None, comment='#')
        plist = []
        for _, row in file.iterrows():
            plist.append(
                Prior(row[0], row[1].lower(),
                      np.array(row[2].split(','), dtype=np.float)))
        return PriorList(plist)

    def read_posterior_as_prior(path, dist='fixed', sigma_fact=3):
        """convenience function to read in the posterior from another fit
            as a prior. Prior distribution can be "fixed" or "normal", which
            uses a normal distribution around the median with a standard
            deviation of three times the unertainty from the posterior."""
        file = pd.read_table(path,
                             delim_whitespace=True,
                             header=None,
                             comment='#')
        plist = []
        for _, row in file.iterrows():
            if dist == 'fixed':
                plist.append(Prior(row[0], 'fixed', row[1]))
            if dist == 'normal':
                error = sigma_fact * np.sqrt(row[2]**2 + row[3]**2)
                sig_decimals = utils.get_significant_decimals(error)
                plist.append(
                    Prior(
                        row[0], 'normal',
                        np.array([
                            utils.float_round(row[1], sig_decimals),
                            utils.float_upround(error, sig_decimals)
                        ])))
        return PriorList(plist)
