# cneuromod.train

Reproducible training, evaluation, and benchmarking of machine-learning models on [Courtois NeuroMod](https://www.cneuromod.ca/) datasets.

## Overview

`cneuromod.train` aims to make it straightforward to move from CNeuroMod data to reproducible machine-learning experiments:

```text
CNeuroMod data
      ↓
data preparation / loading 
      ↓
canonical experiment interface
      ↓
model → training → evaluation → benchmark
```

Training and evaluation code will depend on a small canonical data contract rather than on any particular loader implementation. The canonical contract will be derived from real usage rather than designed up front.

## Data access

CNeuroMod data is accessed through the existing NeuralSet `Study` classes developed for CNeuroMod datasets (see [neuralfetch-cneuromod](https://github.com/courtois-neuromod/neuralfetch-cneuromod)). These classes are already tested and working, and using them here serves two goals at once:

1. benchmarking models on CNeuroMod data;
2. exercising the Study classes against real training workloads.

If an experiment ever needs data for which no Study class exists yet, a lightweight dataloader may be used as a temporary bypass — this is expected to be rare, and the gap should be reported to the data-layer maintainers rather than worked around permanently.

## Roadmap

All planned:

- canonical training interface derived from real CNeuroMod experiments;
- baseline encoding models;
- experiment configuration and provenance;
- evaluation conventions and metrics;
- benchmark suite;
- GPU and HPC/SLURM execution;
- upstream contributions where fixes or functionality belong in the data layer.

The first milestone is intentionally narrow: **one clean, reproducible, tested CNeuroMod brain-encoding experiment from data loading to an evaluated baseline result.**

The first experiment (in design) is:

- **dataset:** [movie10](https://www.cneuromod.ca/) BOLD fMRI;
- **model inputs:** existing stimulus annotations;
- **design:** within-subject, evaluated across multiple datasets/tasks;
- **data access:** existing NeuralSet Study classes for CNeuroMod.

## Installation

Requires Python 3.11 or later.

```bash
pip install git+https://github.com/courtois-neuromod/cneuromod.train.git
```

The package installs as the `cneuromod.train` namespace package:

```python
import cneuromod.train
```

## Development

```bash
git clone https://github.com/courtois-neuromod/cneuromod.train.git
cd cneuromod.train
pip install -e ".[dev]"
```

Run the same checks CI runs:

```bash
ruff check . && ruff format --check . && mypy && pytest
```

## License

[MIT](LICENSE)
