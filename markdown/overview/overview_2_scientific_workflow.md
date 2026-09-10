> ✏️ **This page is auto-generated from [`scripts/overview/overview_2_scientific_workflow.py`](../../scripts/overview/overview_2_scientific_workflow.py) — do not edit it directly.**
> It shows the example fully executed, with its real output images.
> Run it yourself via the [Python script](../../scripts/overview/overview_2_scientific_workflow.py) or the [Jupyter notebook](../../notebooks/overview/overview_2_scientific_workflow.ipynb).

Overview: Scientific Workflow
=============================

A scientific workflow connects fitting a model to interpreting, comparing and revisiting its results. Start with one
completed fit, then build a study in which many datasets, models and searches remain easy to inspect.

This Python overview implements the workflow. The
[Read the Docs guide](https://pyautofit.readthedocs.io/en/latest/overview/scientific_workflow.html)
describes the same tasks using natural language prompts for an assistant.

We cover hard disk output, visualization, on-the-fly output, loading results, result customization, model composition,
searches, configs, the database and scaling up. Only one short Gaussian fit runs in this overview.


```python

from autofit import setup_notebook; setup_notebook()

import json
from os import path

import matplotlib.pyplot as plt
import numpy as np

import autofit as af
```

    Working Directory has been set to `autofit_workspace`


__Data__

Load the Gaussian dataset used in the first overview. If necessary, create the example datasets first.


```python
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")
if not path.exists(dataset_path):
    import subprocess
    import sys

    subprocess.run([sys.executable, "scripts/simulators/simulators.py"], check=True)

data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)
```

__Hard Disk Output__

Saving results makes inference part of a scientific workflow: each fit retains its assumptions, parameter estimates,
diagnostics and scientific interpretation. You can inspect multiple datasets, resume supported searches after an
interruption, and revisit results without rerunning inference, including runs performed on a remote computer.

Give the search a `path_prefix` and `name` to enable persistent output. The configured output root is normally `output`;
do not repeat `output` inside `path_prefix`. Here the path identifies the dataset, model and search. PyAutoFit appends
a deterministic identifier derived from the model and search configuration, rather than a random directory name.

Inside a run, `model.info` describes the model in readable form, `model.results` summarizes inferred parameters and
`search.summary` reports search information such as runtime. The `image` folder holds visual diagnostics, and
search-specific internal files support resuming the search.

The `files` folder contains the machine-readable record:

- `model.json`: model classes, parameter names, fixed values and priors.
- `search.json`: the search class and configuration used for this fit.
- `samples_summary.json`: compact parameter estimates and available evidence information.
- `samples.csv`: sampled parameter values, likelihoods, priors and weights.
- `samples_info.json`: metadata needed to interpret the samples.
- `info.json`: optional metadata supplied to `fit`, such as a dataset label.
- `covariance.csv`: parameter covariance, when available and enabled.

Additional files depend on the search and output configuration. Later in this example we also save
`science_summary.json`: the Gaussian width and residual statistics have meaning for this particular fitting problem.
These summaries make a collection of fits useful scientifically, beyond simply keeping their sample arrays.

We use explicit priors so that the assumptions recorded in `model.json` are easy to recognize.


```python
model = af.Model(af.ex.Gaussian)
model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.sigma = af.UniformPrior(lower_limit=0.1, upper_limit=30.0)
```

For example, the `centre` entry in the saved `model.json` includes these fields (other fields omitted):

```json
"centre": {
    "type": "Uniform",
    "lower_limit": 0.0,
    "upper_limit": 100.0
}
```

It records the prior, not the inferred centre; estimates belong in the samples and their summary. This distinction
lets us recover what was assumed and what was learned for every fit in a larger study.

__Visualization__

Specify visualization before fitting separately from visualization during fitting. Before inference, plot the data
and uncertainties once. During inference, show the current maximum likelihood fit and residuals. Keeping a consistent
set of plots across runs makes different models and datasets much easier to assess.

The `Visualizer` saves `data.png`, `model_fit.png` and `residuals.png` separately, so no plot overwrites another.
It also saves a combined `fit.png` for the live notebook display used in the next section.


```python


class Visualizer(af.Visualizer):
    @staticmethod
    def visualize_before_fit(analysis, paths, model):
        figure, axis = plt.subplots()
        axis.errorbar(
            analysis.xvalues, analysis.data, yerr=analysis.noise_map, fmt="k."
        )
        axis.set(
            xlabel="x", ylabel="Profile normalization", title="Data and uncertainties"
        )
        figure.tight_layout()
        figure.savefig(path.join(paths.image_path, "data.png"))
        plt.close(figure)

    @staticmethod
    def visualize(analysis, paths, instance, during_analysis):
        model_data = instance.model_data_from(xvalues=analysis.xvalues)
        residuals = analysis.data - model_data

        figure, axis = plt.subplots()
        axis.errorbar(
            analysis.xvalues, analysis.data, yerr=analysis.noise_map, fmt="k."
        )
        axis.plot(analysis.xvalues, model_data, color="r")
        axis.set(
            xlabel="x", ylabel="Profile normalization", title="Maximum likelihood fit"
        )
        figure.tight_layout()
        figure.savefig(path.join(paths.image_path, "model_fit.png"))
        plt.close(figure)

        figure, axis = plt.subplots()
        axis.errorbar(analysis.xvalues, residuals, yerr=analysis.noise_map, fmt="k.")
        axis.axhline(0.0, color="r")
        axis.set(
            xlabel="x", ylabel="Residual", title="Residuals of maximum likelihood fit"
        )
        figure.tight_layout()
        figure.savefig(path.join(paths.image_path, "residuals.png"))
        plt.close(figure)

        figure, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 6))
        axes[0].errorbar(
            analysis.xvalues, analysis.data, yerr=analysis.noise_map, fmt="k."
        )
        axes[0].plot(analysis.xvalues, model_data, color="r")
        axes[0].set(
            ylabel="Profile normalization", title="Current maximum likelihood fit"
        )
        axes[1].errorbar(analysis.xvalues, residuals, yerr=analysis.noise_map, fmt="k.")
        axes[1].axhline(0.0, color="r")
        axes[1].set(xlabel="x", ylabel="Residual")
        figure.tight_layout()
        figure.savefig(path.join(paths.image_path, "fit.png"))
        plt.close(figure)

```

__On The Fly__

During inference, quick updates report progress and can refresh a model-fit image in a Jupyter notebook. The live
display reads `fit.png`; our `perform_quick_update` hook refreshes that image using the visualizer above. The same
plots remain available on disk when running this file in a terminal.

Quick updates run more frequently than full updates, which save samples and run more expensive diagnostics.
`iterations_per_quick_update` counts likelihood evaluations; `iterations_per_full_update` uses the search's own
update cadence. Neither is a wall-clock timer. Choose a cadence that gives useful feedback without spending most
of the run plotting. In a desktop script, live updates open a separate viewer; set `live_visual_update=False` for
a headless or cluster run, while keeping the saved plots.

Live output builds intuition: is the fit improving, are residual structures persisting, and is exploration progressing?
A good-looking curve alone does not establish convergence. Use posterior and search diagnostics too before deciding
whether inference is performing reliably. A short demonstration run is a starting point for tuning those settings.

Our analysis also saves the best-fit Gaussian full width at half maximum (FWHM), residual RMS and chi-squared in
`science_summary.json`. FWHM uses the x-axis units; residual RMS uses the data units. These are point-estimate
diagnostics, not posterior uncertainties or a calibrated goodness-of-fit probability.


```python


class Analysis(af.Analysis):
    Visualizer = Visualizer

    def __init__(self, data, noise_map):
        super().__init__()
        self.data = data
        self.noise_map = noise_map
        self.xvalues = np.arange(data.shape[0])

    def log_likelihood_function(self, instance):
        residuals = self.data - instance.model_data_from(xvalues=self.xvalues)
        chi_squared = np.sum((residuals / self.noise_map) ** 2)
        noise_normalization = np.sum(np.log(2.0 * np.pi * self.noise_map**2))
        return -0.5 * (chi_squared + noise_normalization)

    def science_summary(self, instance):
        residuals = self.data - instance.model_data_from(xvalues=self.xvalues)
        return {
            "dataset": "gaussian_x1",
            "estimate": "maximum_likelihood",
            "units": {
                "gaussian_fwhm": "x coordinate (pixel index)",
                "residual_rms": "data units",
                "chi_squared": "dimensionless",
            },
            "gaussian_fwhm": float(2.0 * np.sqrt(2.0 * np.log(2.0)) * instance.sigma),
            "residual_rms": float(np.sqrt(np.mean(residuals**2))),
            "chi_squared": float(np.sum((residuals / self.noise_map) ** 2)),
            "number_of_data_points": int(self.data.size),
        }

    def perform_quick_update(self, paths, instance):
        self.Visualizer.visualize(self, paths, instance, during_analysis=True)

    def save_results(self, paths, result):
        paths.save_json("science_summary", self.science_summary(result.instance))


analysis = Analysis(data=data, noise_map=noise_map)
search = af.Nautilus(
    path_prefix=path.join("scientific_workflow", "gaussian_x1", "gaussian"),
    name="nautilus",
    n_live=50,
    n_eff=100,
    number_of_cores=1,
    iterations_per_quick_update=500,
    iterations_per_full_update=1000,
    live_visual_update=True,
)
result = search.fit(
    model=model,
    analysis=analysis,
    info={
        "dataset": "gaussian_x1",
        "model_label": "gaussian",
        "search_label": "nautilus",
    },
)
print("Saved run:", search.paths.output_path)
print(json.dumps(analysis.science_summary(result.instance), indent=2))
```

    2026-09-09 22:49:36,411 - autofit.non_linear.search.abstract_search - INFO - Starting non-linear search with 1 cores.
    2026-09-09 22:49:36,412 - autofit.non_linear.search.abstract_search - INFO - On-the-fly updates of the maximum likelihood model every 500 iterations.
    2026-09-09 22:49:36,476 - nautilus - INFO - The output path of this fit is autofit_workspace/output/scientific_workflow/gaussian_x1/gaussian/nautilus/818c57b1e4997b7f481c57974e0c60e3
    2026-09-09 22:49:36,484 - nautilus - INFO - Fit Already Completed: skipping non-linear search.
    2026-09-09 22:49:37,304 - nautilus - INFO - Removing search internal folder.
    2026-09-09 22:49:37,305 - nautilus - INFO - Removing all files except for .zip file
    2026-09-09 22:49:37,368 - nautilus - INFO - Search complete, returning result
    Saved run: autofit_workspace/output/scientific_workflow/gaussian_x1/gaussian/nautilus/818c57b1e4997b7f481c57974e0c60e3
    {
      "dataset": "gaussian_x1",
      "estimate": "maximum_likelihood",
      "units": {
        "gaussian_fwhm": "x coordinate (pixel index)",
        "residual_rms": "data units",
        "chi_squared": "dimensionless"
      },
      "gaussian_fwhm": 23.103830262188456,
      "residual_rms": 0.035430537591837626,
      "chi_squared": 78.4576871279137,
      "number_of_data_points": 100
    }


__Loading Results__

The directory aggregator discovers saved fits without rerunning inference. Point it at the actual output path used
by the search; this respects the configured output root. A broader study folder discovers many runs at once.

Below, load saved samples and print a labelled parameter row. The generator loads one result at a time, avoiding
holding every run in memory. The custom JSON can also be read independently of the Python analysis class.


```python
from autofit.aggregator.aggregator import Aggregator

agg = Aggregator.from_directory(directory=search.paths.output_path)
for samples, info in zip(agg.values("samples"), agg.values("info")):
    instance = samples.max_log_likelihood()
    median = samples.median_pdf()
    lower = samples.values_at_lower_sigma(sigma=1.0)
    upper = samples.values_at_upper_sigma(sigma=1.0)
    print(
        info["dataset"],
        info["model_label"],
        info["search_label"],
        "centre =",
        instance.centre,
        "normalization =",
        instance.normalization,
        "sigma =",
        instance.sigma,
    )
    print(
        "Median sigma and 68.3% credible interval:",
        median.sigma,
        (lower.sigma, upper.sigma),
    )

print(search.paths.load_json("science_summary"))
```

    Aggregator loading search_outputs... could take some time.

     A total of 1 search_outputs and results were found.
    gaussian_x1 gaussian nautilus centre = 49.827785689899564 normalization = 24.86713647414844 sigma = 9.811293355915357
    Median sigma and 68.3% credible interval: 9.815629057216263 (9.694724547604153, 9.94675644048989)
    {'dataset': 'gaussian_x1', 'estimate': 'maximum_likelihood', 'gaussian_fwhm': 23.103830262188456, 'residual_rms': 0.035430537591837626, 'chi_squared': 78.4576871279137, 'number_of_data_points': 100}


__Result Customization__

Give the returned result properties that make scientific interpretation convenient: for example, the best-fit model
evaluated on the data grid, or the same derived width stored in our JSON summary. This keeps the scientific definition
in one place and makes interactive inspection agree with saved output.

We can construct a custom result from the completed fit without sampling again. For future searches, assign
`Result = ResultExample` on the analysis and override `make_result` to pass `analysis=self`, as illustrated in the
[result cookbook](https://pyautofit.readthedocs.io/en/latest/cookbooks/result.html).


```python


class ResultExample(af.Result):
    @property
    def max_log_likelihood_model_data_1d(self):
        return self.instance.model_data_from(xvalues=self.analysis.xvalues)

    @property
    def science_summary(self):
        return self.analysis.science_summary(self.instance)


scientific_result = ResultExample(
    samples_summary=result.samples_summary,
    paths=search.paths,
    samples=result.samples,
    analysis=analysis,
)
print(
    "Best-fit profile shape:", scientific_result.max_log_likelihood_model_data_1d.shape
)
print("Gaussian FWHM:", scientific_result.science_summary["gaussian_fwhm"])
```

    Best-fit profile shape: (100,)
    Gaussian FWHM: 23.103830262188456


Derived quantities can also be evaluated for posterior samples to obtain uncertainties. See the result cookbook for
latent variables; saving a maximum likelihood FWHM alone does not provide its credible interval.

__Model Composition__

The first guide already covers composing models, fixing and linking parameters, assertions and arithmetic. Here the
scientific question is how to compare competing assumptions for the same dataset in a way we can interpret later.

For example, compare a freely varying Gaussian width with a fixed width motivated by an external measurement.
Use meaningful model labels, save each model's priors and fixed values, and generate matching diagnostic plots.
Inspect changes in parameter constraints and residuals; compare Bayesian evidence when both searches provide it,
remembering that evidence depends on the priors. More models are useful only when their differences remain traceable.

These definitions illustrate two alternatives; they do not launch additional fits.


```python
free_width_model = model.copy()
fixed_width_model = model.copy()
fixed_width_model.sigma = 10.0
models = {
    "gaussian_free_width": free_width_model,
    "gaussian_fixed_width": fixed_width_model,
}
for label, candidate in models.items():
    print(label, "free parameters:", candidate.prior_count)
```

    gaussian_free_width free parameters: 3
    gaussian_fixed_width free parameters: 2


The [model cookbook](https://pyautofit.readthedocs.io/en/latest/cookbooks/model.html) explains larger compositions.

__Searches__

Try different searches on the same model and dataset while developing the workflow. Compare runtime, posterior
agreement and search-specific diagnostics; a faster run is useful only if it gives reliable inference. Nested
sampling, MCMC and optimization have different outputs: do not treat an optimizer as a posterior sampler or expect
an MCMC run to supply nested-sampling evidence.

Model dimension, degeneracy, likelihood cost and the availability of gradients all affect search choice. Save the
search settings alongside each model so an apparent scientific difference can be investigated as a numerical one.

Here are two nested-sampling search definitions for a later comparison. We do not run them in this short overview.


```python
searches = {"nautilus": af.Nautilus, "dynesty": af.DynestyStatic}
for label, search_class in searches.items():
    print(label, search_class.__name__)
```

    nautilus Nautilus
    dynesty DynestyStatic


See the [search cookbook](https://pyautofit.readthedocs.io/en/latest/cookbooks/search.html) for settings and diagnostics.

__Configs__

Once the workflow works for a few datasets, put shared defaults for priors, search settings and output in configuration
files. Record deliberate per-fit overrides, such as the priors in this example, so comparisons remain understandable.
Reusable defaults reduce repetitive scripts; each saved model and search still records what that fit actually used.

See the [configs cookbook](https://pyautofit.readthedocs.io/en/latest/cookbooks/configs.html).

__Database__

A directory collection is convenient for browsing individual fits. As the study grows, a SQLite database makes it
easier to select runs and compare subsets by their models or saved metadata. Import completed outputs into the
database and query them without repeating inference. This complements the plots and files used for individual checks.

Import our saved run into a SQLite database next to its output folder and select completed Nautilus runs. Repeating
the import updates existing entries rather than requiring another inference run.


```python
database = af.Aggregator.from_database(
    filename=path.join(path.dirname(search.paths.output_path), "results.sqlite"),
    completed_only=True,
)
database.add_directory(directory=search.paths.output_path)
nautilus_results = database(database.search.name == "nautilus")
for saved_info in nautilus_results.values("info"):
    print("Database result:", saved_info)
database.session.close()
```

    2026-09-09 22:49:37,689 - autofit.database.migration.migration - INFO - Performing migration from None to ec3f9efe138fc09873fc23b0d4061939 in 11 steps
    2026-09-09 22:49:37,694 - autofit.database.migration.migration - INFO - revision_id updated to ec3f9efe138fc09873fc23b0d4061939
    Aggregator loading search_outputs... could take some time.

     A total of 1 search_outputs and results were found.
    2026-09-09 22:49:37,696 - autofit.database.aggregator.scrape - INFO - Scraping directory autofit_workspace/output/scientific_workflow/gaussian_x1/gaussian/nautilus/818c57b1e4997b7f481c57974e0c60e3
    2026-09-09 22:49:37,697 - autofit.database.aggregator.scrape - INFO - 1 searches found
    2026-09-09 22:49:37,700 - autofit.database.aggregator.scrape - INFO - Creating fit for: scientific_workflow/gaussian_x1/gaussian None nautilus 818c57b1e4997b7f481c57974e0c60e3
    2026-09-09 22:49:37,755 - autofit.database.aggregator.scrape - WARNING - Fit already existed with identifier 818c57b1e4997b7f481c57974e0c60e3
    2026-09-09 22:49:37,765 - autofit.database.aggregator.scrape - WARNING - Failed to load latent variables for 818c57b1e4997b7f481c57974e0c60e3
    2026-09-09 22:49:37,817 - autofit.database.aggregator.aggregator - INFO - 1 fit(s) found matching query
    Database result: {'dataset': 'gaussian_x1', 'model_label': 'gaussian', 'search_label': 'nautilus'}


See the [multiple datasets cookbook](https://pyautofit.readthedocs.io/en/latest/cookbooks/multiple_datasets.html)
for creating a database, adding directories and querying results.

__Scaling Up__

Begin with a few datasets and iterate on visualization, diagnostics and summaries. Once you can reliably interpret
one fit, use the same structure across a study. Below is an illustrative layout, not twenty fits run by this script.
Each of five datasets has two model choices and two searches per model. Expand any search's identifier folder to
find the same readable summaries, machine-readable files and images described above.

```text
output/scientific_workflow/
├── dataset_01/
│   ├── gaussian_free_width/
│   │   ├── nautilus/<identifier>/
│   │   │   ├── model.info, model.results, search.summary
│   │   │   ├── files/  (model.json, samples.csv, science_summary.json, ...)
│   │   │   └── image/  (data.png, model_fit.png, residuals.png, fit.png)
│   │   └── dynesty/<identifier>/
│   └── gaussian_fixed_width/{nautilus,dynesty}/<identifier>/
├── dataset_02/{gaussian_free_width,gaussian_fixed_width}/{nautilus,dynesty}/<identifier>/
├── dataset_03/{gaussian_free_width,gaussian_fixed_width}/{nautilus,dynesty}/<identifier>/
├── dataset_04/{gaussian_free_width,gaussian_fixed_width}/{nautilus,dynesty}/<identifier>/
└── dataset_05/{gaussian_free_width,gaussian_fixed_width}/{nautilus,dynesty}/<identifier>/
```

Braces abbreviate separate folders. Dataset identifiers should refer to known inputs; keep those inputs and their
provenance with the study rather than assuming parameter samples alone reproduce a scientific analysis.

You can browse this collection on disk, load it with the aggregator, or ask an assistant:

> Inspect all five datasets and compare the models and searches fitted to each. Summarize parameter constraints,
> fit quality and runtime, compare Bayesian evidence where available, and link each assessment to its saved output.
> Flag results that need closer inspection.

That is the purpose of a scientific workflow: many fits remain navigable, interpretable and comparable as the study
grows. The next overview introduces further inference methods, including graphical and hierarchical models.
