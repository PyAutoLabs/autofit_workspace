The `searches` folder contains example scripts which fit a model using the different non-linear searches supported by **PyAutoFit**:

# Files

- `nest.py`: Fit a model using nested sampling algorithms.
- `mcmc.py`: Fit a model using MCMC algorithms.
- `mle.py`: Fit a model using maximum likelihood algorithms.
- `start_point.py`: Set the start-point of certain parameters in a model-fit.

# Searches

Every search **PyAutoFit** ships, by family, generated from the search manifest of the installed release
(`python -m autofit search-manifest --json`); the capability matrix compares them:
https://pyautofit.readthedocs.io/en/latest/searches/index.html

<!-- search-roster:begin style=list -->
<!-- search-manifest@1 autofit 2026.10.12.1 -->
- **Markov chain Monte Carlo** (`mcmc`): `af.Emcee`, `af.Zeus`, `af.BlackJAXNUTS`, `af.SMC` (experimental) — example: `autofit_workspace:scripts/searches/mcmc.py`
- **Nested sampling** (`nest`): `af.DynestyStatic`, `af.DynestyDynamic`, `af.Nautilus`, `af.NSS` (experimental) — example: `autofit_workspace:scripts/searches/nest.py`
- **Maximum likelihood / maximum a posteriori** (`mle`): `af.Drawer`, `af.BFGS`, `af.LBFGS`, `af.MultiStartAdam`, `af.MultiStartADABelief` (experimental), `af.MultiStartLion` (experimental), `af.MultiStartProdigy` — example: `autofit_workspace:scripts/searches/mle.py`
<!-- search-roster:end -->
