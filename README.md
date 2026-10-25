# PRIME: Fast Primal-Dual Feedback Optimization for Markets

[![Paper](https://img.shields.io/badge/DOI-10.1109%2FCDC57313.2025.11312508-blue)](https://doi.org/10.1109/CDC57313.2025.11312508)
[![arXiv](https://img.shields.io/badge/arXiv-2504.16048-b31b1b.svg)](https://arxiv.org/abs/2504.16048)
[![Python](https://img.shields.io/badge/Python-%E2%89%A53.12-3776AB.svg)](https://www.python.org/)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC_BY--SA_4.0-lightgrey.svg)](LICENSE)

This repository contains the reference implementation and numerical experiments for
**PRIME (PRoximal Iterative MarkEts)**, an online feedback optimization method for
coordinating self-interested actors connected through a physical system.

PRIME combines a primal-dual formulation with proximal updates. It uses measurements of
the system output and input-output sensitivities, while allowing actors' costs and
constraints to remain private. The paper introduces two variants:

- **PRIME-Y** dualizes the output constraints and supports a market of input actors.
- **PRIME-H** dualizes the input-output relation and supports both input and output actors.

The experiments compare these methods with projected primal-gradient and standard
primal-dual feedback optimization on nonlinear toy problems and an AC optimal power flow
case based on the UNICORN 56-bus network, including a measurement-noise experiment.

## Quick start

The project requires Python 3.12 or newer and uses
[`uv`](https://docs.astral.sh/uv/) for dependency management. A LaTeX installation is
also required to render the figure labels.

```console
git clone https://github.com/NicholasBehr/prime.git
cd prime
uv sync --dev
uv run create_figures
```

Run commands from the repository root so that the scenario data paths resolve correctly.
Generated PDFs are written to [`figures/output`](figures/output).

To explore the experiments interactively, start JupyterLab and open one of the notebooks:

```console
uv run jupyter lab
```

The most useful starting points are
[`test_unicorn_proximal.ipynb`](notebooks/test_unicorn_proximal.ipynb) for the power-grid
case and [`test_toy_nonconvex.ipynb`](notebooks/test_toy_nonconvex.ipynb) for the nonlinear
toy example.

## Run a simulation directly

The following example runs PRIME-Y on the UNICORN scenario and returns the iteration
history as a pandas `DataFrame`:

```python
from feedback_opt.optimizers import OptimizerDualYProximal
from feedback_opt.simulation import Simulation
from feedback_opt.systems import SystemElectrical
from scenarios.scenario_unicorn import Unicorn

params = Unicorn()
system = SystemElectrical(params.sys)
optimizer = OptimizerDualYProximal(params.opt_dualyprox_dist, system)
result = Simulation(params.sim, system, optimizer).run()

print(result.tail())
```

Scenario classes collect the simulation, system, objective, constraint, and algorithm
parameters. To change an experiment, copy or edit a scenario under [`scenarios`](scenarios)
and pass its parameter groups to the corresponding system, optimizer, and simulation
classes.

## Implementation map

| Paper method | Implementation |
| --- | --- |
| Algorithm 1: Projected Primal Gradient OFO | `OptimizerPrimal` |
| Algorithm 2: Primal-Dual OFO | `OptimizerDualY` |
| Algorithm 3: PRIME-Y | `OptimizerDualYProximal` |
| Algorithm 4: PRIME-H | `OptimizerDualHProximal` |

## Repository structure

| Path | Contents |
| --- | --- |
| [`feedback_opt`](feedback_opt) | Simulation loop, system models, optimizers, constraints, and plotting utilities |
| [`scenarios`](scenarios) | Parameters for the toy and UNICORN experiments |
| [`figures`](figures) | Scripts used to generate the paper's numerical figures |
| [`notebooks`](notebooks) | Interactive experiments and UNICORN data import |
| [`data`](data) | Toy grid and UNICORN 56-bus network data |

## Citation

If you use this repository in your work, please cite the published paper:

```bibtex
@inproceedings{behr2025prime,
  author    = {Behr, Nicholas J. and Bianchi, Mattia and Moffat, Keith and
               Bolognani, Saverio and D{\"o}rfler, Florian},
  title     = {{PRIME}: Fast Primal-Dual Feedback Optimization for Markets with
               Application to Optimal Power Flow},
  booktitle = {2025 IEEE 64th Conference on Decision and Control (CDC)},
  year      = {2025},
  pages     = {5240--5245},
  doi       = {10.1109/CDC57313.2025.11312508},
  eprint    = {2504.16048},
  archivePrefix = {arXiv}
}
```

## License

This repository is licensed under the
[Creative Commons Attribution-ShareAlike 4.0 International License](LICENSE).
