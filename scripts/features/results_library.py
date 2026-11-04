"""
Feature: Results Library
========================

When you fit many datasets — hundreds of objects on a cluster, say — the hard question at the end is not "what are
the results?" but "which results are *missing*?". A fit that crashed, ran out of time, or completed without a product
you need leaves no error behind: it is simply not there, and a catalogue built from the output quietly loses it.

The results library answers that question. You declare which results you expect (datasets × searches, and which
products each must hold), and **PyAutoFit** compares that declaration with what is on disk, giving every gap a reason.
It can then add a new product to results that already exist, without rerunning any fit.

__Contents__

This script is split into the following sections:

- **Fits**: Fit a few datasets, outputting the results to hard-disk.
- **Index**: Build an index of every result on disk.
- **Declare**: Declare which results the library is expected to hold.
- **Check**: Compare the declaration with the index and list every gap.
- **Enrich**: Add a new product to every completed result.
- **Command Line**: The same steps from the shell, e.g. on a cluster.
"""

# from autofit import setup_notebook; setup_notebook()

from os import path
from pathlib import Path

import autofit as af
from autofit.aggregator import Aggregator
from autofit.results import ResultsIndex

"""
__Fits__

We fit two of the three `gaussian_x1` datasets, each written to hard-disk with its dataset name as `unique_tag`. The
third dataset is deliberately not fitted, so the library has a hole to find.
"""
results_path = (
    af.with_test_mode_segment(Path("output")) / "features" / "results_library"
)

model = af.Collection(gaussian=af.ex.Gaussian)

for dataset_name in ["gaussian_x1_0", "gaussian_x1_1"]:
    dataset_path = path.join("dataset", "example_1d", dataset_name)

    if not path.exists(dataset_path):
        import subprocess
        import sys

        subprocess.run(
            [sys.executable, "scripts/simulators/simulators.py"],
            check=True,
        )

    data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
    noise_map = af.util.numpy_array_from_json(
        file_path=path.join(dataset_path, "noise_map.json")
    )

    search = af.DynestyStatic(
        name="fit",
        path_prefix=path.join("features", "results_library"),
        unique_tag=dataset_name,
        nlive=50,
    )
    search.fit(model=model, analysis=af.ex.Analysis(data=data, noise_map=noise_map))

"""
__Index__

The results index finds every result under a folder and classifies it: completed, running, interrupted with a valid
checkpoint, failed, corrupt, or superseded by a later fit of the same search. Only new or changed results are opened,
so re-indexing thousands of results after a few new ones is quick.
"""
index = ResultsIndex(results_path)
index.update()

print(index.counts())

"""
__Declare__

We expect each of the three datasets to have a completed `fit`, holding its samples summary and the `model.json` of
the fit. A required product is either a name the index records (`samples`, `samples_summary`, `latent`, `image`,
`dataset_identity`, `search_internal`) or a path inside the result.

The same declaration can live in a `library.yaml` file (`af.ResultsLibrary.from_yaml`), and on a cluster it can be
taken straight from an hpc-campaign file with `from_campaign: campaign.yaml`, so the campaign stays the single
declaration of what was run.
"""
library = af.ResultsLibrary.grid(
    datasets=["gaussian_x1_0", "gaussian_x1_1", "gaussian_x1_2"],
    searches={"fit": ["samples_summary", "files/model.json"]},
    roots=results_path,
    name="results_library",
)

"""
__Check__

`check` gives every expected result a status — `done`, `missing`, `incomplete`, `failed` or `duplicate` — and every
gap a reason. Here `gaussian_x1_2` is missing, with the reason `no_result`.

`write_missing_csv` writes one row per gap to `results_library.missing.csv`, the list to rerun or investigate.
"""
report = library.check()

print(report.summary())

for result in report.results:
    print(result.expected.dataset, result.status, result.reason)

report.write_missing_csv(results_path)

"""
__Enrich__

Suppose that after the fits finished you need a new product in every result: here, a small JSON file with the
maximum likelihood centre of the Gaussian. `enrich` computes it from each saved result and writes it into the result
as `files/centre.json`, atomically: if the job is killed part-way, every result is either untouched or complete.

Results that already hold the product are skipped, so an interrupted pass can simply be run again. On a cluster,
`shard="i/N"` splits the results into N parts, one per array task.
"""
agg = Aggregator.from_index(results_path)


def centre(search_output):
    return {"centre": search_output.instance.gaussian.centre}


enriched = agg.enrich(centre, name="centre")

print(enriched)

"""
__Command Line__

Every step above also runs from the shell, which is how it is used on a cluster:

    python -m autofit.results index output/features/results_library
    python -m autofit.results check library.yaml
    python -m autofit.results enrich output/features/results_library --fn products:centre --name centre --shard 0/4

`check` exits with status 1 while anything expected is not done, so a script can wait on it.
"""
