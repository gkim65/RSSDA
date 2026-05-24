# RSSDA: Recursive Small-Step Semi-Decentralized A*

RSSDA solves multiagent planning problems where agents have partial
observability and can share information only at synchronization points or
triggered events. It bridges fully centralized planning, semi-decentralized
planning, and fully decentralized Dec-POMDP planning.

## Installation

```powershell
git clone <repository-url>
cd RSSDA
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Requirements are listed in `requirements.txt`.

## Quick Start

Run a small tiger instance:

```powershell
.venv\Scripts\python.exe benchmarks\sdec_tiger.py 4
```

Run the spacecraft conjunction-assessment comparison:

```powershell
.venv\Scripts\python.exe benchmarks\spacecraftCA\compare_variants.py --variants centralized,sdec,dec --solver-modes fixed --eval-mode expected --tag refined_drift_main
```

Run the experiment-performance runner. This checks paper-value anchors and
smoke-tests larger workflows; it is not a full recreation of every experiment
table entry.

```powershell
.venv\Scripts\python.exe experiments_runner.py
```

## Benchmark Domains

| Domain | Description | Agents | States |
|--------|-------------|--------|--------|
| Box Pushing | Coordinate to push boxes on a grid | 2 | 100 |
| Tiger | Classic door-opening coordination problem | 2 | 2 |
| Fire Fighting | Extinguish fires across three houses | 2 | 432 |
| Mars Rovers | Coordinate rock sampling and data transmission | 2 | 256 |
| Maritime MEDEVAC | Helicopter-ship patient rescue coordination | 2 | 512 |
| Labyrinth | Graph search with configurable sync triggers | 2 | Variable |
| SpacecraftCA | Spacecraft conjunction assessment with centralized, SDec, and Dec information structures | 2 | 1441 |

## Solver Configuration

`RSSDAConfig` controls exact and approximate search:

```python
config = RSSDAConfig(
    maxh=6,
    algorithm="approximate",
    iter_limit=2000,
    rec_limit=2,
    heuristic_type="HYBRID",
    TI1=True,
    TI2=True,
    TI3=True,
    TI4=True,
)
```

Approximation techniques:

- `TI1`: interleaved planning/execution.
- `TI2`: progress-based pruning.
- `TI3`: recursive tail approximation.
- `TI4`: memory-bounded clustering.

## Project Structure

```text
RSSDA/
|-- RSSDA.py                 # Core SDec-POMDP solver
|-- experiments_runner.py
|-- requirements.txt
|-- baselines/
|   |-- decPOMDP.py          # RS-MAA* baseline
|   `-- parser_decPOMDP.py
|-- benchmarks/
|   |-- sdec_box.py
|   |-- sdec_tiger.py
|   |-- sdec_mars.py
|   |-- sdec_fireFight3houses.py
|   |-- sdec_labyrinth.py
|   |-- labyrinth_benchmarks/    # Labyrinth domain data files
|   |-- maritimemedevac.py
|   `-- spacecraftCA/        # Spacecraft conjunction-assessment benchmark
```

## SpacecraftCA

The cleaned spacecraft benchmark is documented in
`benchmarks/spacecraftCA/README.md`. It compares centralized, semi-decentralized,
and fully decentralized policies under shared dynamics and reward models.

## Citation

```bibtex
@inproceedings{alhusseini2026,
  title={A Semi-Decentralized Approach to Multiagent Control},
  author={Al-Husseini, Mahdi and Wray, Kyle H. and Kochenderfer, Mykel J.},
  booktitle={International Conference on Autonomous Agents and Multiagent Systems},
  year={2026}
}
```

## License

MIT License. See source files for details.
