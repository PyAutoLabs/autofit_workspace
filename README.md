# PyAutoFit Workspace

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PyAutoLabs/autofit_workspace/blob/2026.9.19.1/notebooks/overview/overview_1_the_basics.ipynb)
[![JOSS](https://joss.theoj.org/papers/10.21105/joss.02550/status.svg)](https://doi.org/10.21105/joss.02550)

[Installation Guide](https://pyautofit.readthedocs.io/en/latest/installation/overview.html) |
[readthedocs](https://pyautofit.readthedocs.io/en/latest/index.html) |
[Introduction on Colab](https://colab.research.google.com/github/PyAutoLabs/autofit_workspace/blob/2026.9.19.1/notebooks/overview/overview_1_the_basics.ipynb) |
[Browse Examples With Images](markdown/README.md) |
[HowToFit](https://github.com/PyAutoLabs/HowToFit)

Welcome to the **PyAutoFit** Workspace!

## Getting Started

You can get set up on your personal computer by following the installation guide on
our [readthedocs](https://pyautofit.readthedocs.io/).

Alternatively, you can try **PyAutoFit** out in a web browser by going to the [autofit workspace
Colab](https://colab.research.google.com/github/PyAutoLabs/autofit_workspace/blob/2026.9.19.1/notebooks/overview/overview_1_the_basics.ipynb).

## Where To Go?

We recommend that you start with the `autofit_workspace/notebooks/overview/overview_1_the_basics.ipynb`
notebook, which will give you a concise overview of **PyAutoFit**'s core features and API.

You can also [browse the overview examples fully executed, with their output images](markdown/README.md),
directly on GitHub — no installation required.

Next, read through the overview example notebooks of features you are interested in, in the folder: `autofit_workspace/notebooks/overview`.

Then, you may wish to implement your own model in **PyAutoFit**, using the `cookbooks` for help with the API. Alternative,
you may want to checkout the `features` package for a list of advanced statistical modeling features.

## HowToFit

For users less familiar with Bayesian inference and scientific analysis you may wish to work through
the **HowToFit** lectures. These teach you the basic principles of Bayesian inference, with the content
pitched at undergraduate level and above.

**HowToFit** now lives in its own standalone repository at
[PyAutoLabs/HowToFit](https://github.com/PyAutoLabs/HowToFit).

Use the [Jupyter notebooks](https://github.com/PyAutoLabs/HowToFit/tree/main/notebooks) if you want to run
the code (recommended), or read the available [Markdown lectures](https://github.com/PyAutoLabs/HowToFit/tree/main/markdown)
directly on GitHub.

For help alongside the lectures, open the [autofit_assistant](https://github.com/PyAutoLabs/autofit_assistant)
repository in your AI coding agent, following its setup instructions, and paste:

```text
Enter HowToFit mode.

I want to work through the HowToFit lectures. Show me where to find them
and how to use Jupyter Notebook or Markdown, then help me with questions
as I go.
```

The assistant will answer questions about concepts, equations, code and results as you study, and help with
notebook errors. Share the lecture link and section or the cell you are working on; you choose when to move on.

## Workspace Structure

The workspace includes the following main directories:

- `notebooks`: **PyAutoFit** examples written as Jupyter notebooks.
- `scripts`: **PyAutoFit** examples written as Python scripts.
- `config`: Configuration files which customize **PyAutoFit**'s behaviour.
- `dataset`: Where data is stored, including example datasets distributed with **PyAutoFit**.
- `output`: Where the **PyAutoFit** analysis and visualization are output.

The **examples** in the notebooks and scripts folders are structured as follows:

- `overview`: Examples using **PyAutoFit** to compose and fit a model to data via a non-linear search.
- `cookbooks`: Concise API reference guides for **PyAutoFit**'s core features.
- `features`: Examples of **PyAutoFit**'s advanced modeling features.
- `searches`: Example scripts of every non-linear search supported by **PyAutoFit**.
- `plot`: An API reference guide for **PyAutoFits**'s plotting tools.

The following **projects** are available in the project folder:

- `astro`: An Astronomy project which fits images of gravitationally lensed galaxies.

## Workspace Version

This workspace is built and tested against the **latest PyAutoFit release** — install it with
`pip install --upgrade autofit`.

The oldest release the scripts here are compatible with is recorded as
`version.minimum_library_version` in `config/general.yaml`, and is checked when the workspace
runs. That floor is the authoritative compatibility signal; this README no longer names an
exact version, which could go stale (or name a yanked release) between releases.

## Community & Support

Questions, help with your code or your analysis, and ideas: the
[PyAutoLabs Discussions](https://github.com/orgs/PyAutoLabs/discussions).
Bug reports with a reproducer (a snippet, the traceback, your versions):
an issue on the library's tracker. The Slack is for collaborators, by
invitation.

Collaborators receive the latest **PyAutoFit** updates in the [Slack channel](https://pyautofit.slack.com/).
Contact [James Nightingale](https://github.com/Jammy2211) about collaborator access.

## Build Configuration

The `config/` directory contains two files used by the automated build and test system
(CI, smoke tests, and pre-release checks). These are not relevant to normal workspace usage.

- `config/build/no_run.yaml` — scripts to skip during automated runs. Each entry is a filename stem
  or path pattern with an inline comment explaining why it is skipped.
- `config/build/profile_smoke.yaml` — environment variables applied to each script during automated runs.
  Defines default values (e.g. test mode, small datasets) and per-script overrides for scripts
  that need different settings.
