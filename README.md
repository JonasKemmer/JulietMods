# *Juliet Mods*

A collection of convenience functions for [*juliet*](http://juliet.readthedocs.io).

It provides:
* Classes for handling priors and instrument data
* Plotting functions for RV, transits and long term photometry
* Plotting functions for correlations (corner plots, GP alpha-vs-Prot, planet parameter correlations)
* Automatic creation of LaTeX tables containing planet posteriors (including derived parameters like minimum mass and planet radius)

### Documentation
 The functions of the package have become very complex over the years. Given that I don't know how much it will be utilized by others, I refrain from providing a detailed documentation (would be just to time consuming). However, I tried to provide an comprehensive documentation of the code itself. If you have any questions, don't hesitate to approach me (either per DM or opening issues here on github.)

### Usage
You can simply download the package and import it, if the download directory is in your Python path. Another option is to install the package from source using pip:

```bash
pip install git+https://github.com/JonasKemmer/JulietMods.git
```
or if you're planning to modify it
If you want to download the directory to a local path:
```bash
git clone https://github.com/JonasKemmer/JulietMods.git
cd JulietMods
python setup.py develop
```

### Disclaimer:
The package was originally created for my own workflow, so there is no guarantee for correctness. I am happy for any bug reports, however it is unlikely that I will actively develop the package in the near future.

**BUT the package is deliberately published with an open source licence. If you're an active juliet user, fork it, modify it to your liking and share it :)**

## Example file for an existing juliet fit:
```python
import juliet
import juliet_mods as jm

julobj = juliet.load(input_folder='/path/to/juliet_out/')
results = julobj.fit(
        sampler='dynamic_dynesty',
        nthreads=5,
    )
jm.append_lnZ(results)
jm.plot_rv(results)
jm.plot_phased_rvs(results, show=True, saveformat='png')
jm.plot_parameter_correlation(results, 'P_vs_K', 'p1')
```


## Example file of a joint RV and transit fit:
```python
import juliet
import juliet_mods as jm

priors = jm.PriorList.read_prior_file(f'./dummy_priors.priors')
data = jm.Data()

## RVs ##
data.add_rv('./RVinstr1.vels', 'RV1')
data.add_rv('./RVinstr2.vels', 'RV2')

## Photometry ##
data.add_photometry('./Photinstr1.lc', 'LC1')
data.add_photometry('./Photinstr2.lc', 'LC2')


julobj = juliet.load(priors=priors.priordict,
                     t_rv=data.t_rvs,
                     y_rv=data.rvs,
                     yerr_rv=data.rvs_err,
                     GP_regressors_rv=data.gp_reg_rvs,
                     t_lc=data.t_fluxes,
                     y_lc=data.fluxes,
                     yerr_lc=data.fluxes_err,
                     linear_regressors_lc=data.linear_regressors_lc,
                     GP_regressors_lc=data.gp_reg_lc,
                     out_folder=f'./dummyfit')
results = julobj.fit(
        sampler='dynamic_dynesty',
        nthreads=5,
    )
jm.append_lnZ(results)
jm.get_lc_residuals(results)


for pnum in range(1, results.data.n_transiting_planets + 1):
    jm.plot_transit_overview(results,
                             minphase=-0.02,
                             maxphase=0.02,
                             binlength=0.001,
                             ncols=4,
                             pnum=pnum,
                             show=True,
                             nsamples=10000)
for pnum in range(1, results.data.n_rv_planets + 1):
    jm.plot_corner(results, [f'_p{pnum}', 'rho'],
                    mark_values=None,
                    show=False)

jm.get_rv_residuals(results)
jm.plot_phased_rvs(results)
jm.plot_rv(results)
jm.plot_corner(results, 'GP')

jm.create_planet_posterior_table(results)
jm.create_planet_param_table(results,
                                R_star=(1, 0.2),
                                M_star=(1, 0.2),
                                L_star=(1, 0.2),
                                T_star=(6000, 100),
                                Kmag=(7.869, 0.02),
                                Jmag=(8.694, 0.024),
                                saveformat='LaTeX')

jm.plot_GPcorrelation(results, 'alpha_vs_Prot')
for instrument in np.unique(julobj.instruments_lc):
    jm.plot_corner(results, f'_{instrument}', mark_values=None, show=False)
```