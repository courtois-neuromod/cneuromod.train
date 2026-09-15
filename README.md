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

Training and evaluation code will depend on a small canonical data contract rather than on any particular loader implementation, so that direct CNeuroMod workflows and NeuroAI-compatible workflows can coexist. The first supported data path is CNeuroMod data already available in NeuralSet format, so the canonical contract will be derived from real usage rather than designed up front.

## Development tracks

### Track A — CNeuroMod training and benchmarking (primary)

All planned:

- canonical training interface derived from real CNeuroMod experiments;
- baseline encoding models;
- experiment configuration and provenance;
- evaluation conventions and metrics;
- benchmark suite;
- GPU and HPC/SLURM execution.

The first milestone is intentionally narrow: **one clean, reproducible, tested CNeuroMod brain-encoding experiment from data loading to an evaluated baseline result.**

The first experiment (in design) is:

- **dataset:** [movie10](https://www.cneuromod.ca/) BOLD fMRI;
- **model inputs:** existing stimulus annotations;
- **design:** within-subject, evaluated across multiple datasets/tasks;
- **data access:** existing CNeuroMod data in NeuralSet format, reusing the CNeuroMod data layer rather than re-implementing loading here.

### Track B — NeuroAI interoperability (parallel)

All planned:

- [NeuralFetch](https://github.com/courtois-neuromod/neuralfetch-cneuromod) adapter;
- NeuralSet compatibility;
- upstream NeuroAI contributions where functionality belongs there;
- lightweight compatibility tests.

Track B integrates through explicit adapters and does not block Track A.

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

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow.

## License

[MIT](LICENSE)
