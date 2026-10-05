# Ambulance location and simulation, La Serena-Coquimbo

Code, aggregated data and simulation outputs for *An Optimization-Simulation Framework for Reducing Ambulance Response Time: The Case of La Serena-Coquimbo, Chile*.

Three facility location models (p-median, p-center, MCLP) and two dispersal benchmarks (weighted k-means, demand-weighted random allocation) locate 8 ambulances over 68 candidate bases and 4,819 demand nodes. Each configuration is then evaluated in a discrete-event simulation. The simulation model (AnyLogic) is not distributed; its specification and its complete per-call outputs are.

## Contents

```
data/                      model inputs (aggregated; no coordinates, no individual records)
  demand_nodes.csv           4,819 census blocks: id (N0001-N4819), comuna, weight (records + 1)
  candidate_bases.csv        68 feasible sites: id (B01-B68), comuna, demand within 1.5 km
  travel_times.csv           OSRM free-flow travel time and distance, base to node (327,692 pairs)
  configurations.csv         the 6 simulated configurations, 8 units each, by base id
  arrival_rates.csv          hourly arrival rate of the non-homogeneous Poisson process
  service_time_distributions.csv  fitted service-time distributions (AnyLogic parametrisation)
  speed_regimes.csv          effective travel speed by time of day
  observed_summary.csv       aggregate statistics of the observed current deployment
  observed_cdf.csv           observed cumulative distributions (0.5 min grid)
simulation_output/         one CSV per configuration, one row per simulated call
results/                   tables produced by the scripts
figures/                   figures produced by the scripts
```

Scripts, in execution order:

| Script | Purpose | Needs |
|---|---|---|
| `01_solve_location_models.py` | p-median, p-center, MCLP (tau = 5 min) | Gurobi |
| `02_benchmarks.py` | 500 random draws, static evaluation of all configurations; weighted k-means if coordinates are available | |
| `03_sensitivity.py` | MCLP for tau in {3, 5, 8, 10}; p-median and MCLP with raw counts instead of +1 | |
| `04_simulation_results.py` | per-replicate metrics, confidence intervals, Welch ANOVA, Games-Howell, figures | |
| `05_validation.py` | simulated current deployment against observed aggregates | |
| `06_warmup_welch.py` | warm-up check | |

`optimization_models.py` holds the three Gurobi formulations. `03_sensitivity.py` solves the same models with HiGHS (SciPy) and returns the same sites; use it if Gurobi is not available.

The p-median solution is recovered exactly by both solvers. The MCLP and p-center optima are not unique: at tau = 5 min at least two site sets cover 96.72 % of the demand (they differ in one site), and any set of at most 8 sites with a maximum travel time of 10.525 min is p-center optimal. Different solvers or versions may return any of them. The configurations evaluated in the paper are fixed in `data/configurations.csv`.

## Setup

```
pip install -r requirements.txt
python 02_benchmarks.py      # then 03, 04, 05, 06
```

Scripts run from the repository root and write to `results/` and `figures/`.

## Data

Demand weights are the number of pre-pandemic emergency records (January 2017 to 17 March 2020) assigned to each census block, plus one. Travel times are OSRM routes at free-flow speed, in seconds. Bases and nodes are identified by sequential ids; their coordinates are not distributed and are available from the corresponding author on request. Everything except the k-means clustering step runs on the travel-time matrix alone. Re-running the random draws reproduces their distribution (5th percentile 3.15 min, median 3.64 min of weighted mean travel time) but not the particular median draw that was simulated, which was selected in a run that included the clustering step; that configuration is recorded in `configurations.csv`. In `configurations.csv`, the current deployment (four units at each hospital) is represented by the candidate base nearest to each hospital; the simulation used the hospital locations. The observed baseline (`observed_*.csv`) is reported only as aggregates: means, quantiles and cumulative distributions over 8,684 records with a complete dispatch sequence. Individual emergency records belong to the EMS operator and are not included.

## Simulation model

Discrete-event simulation in AnyLogic. For each replicate:

1. All 8 ambulances start idle at their sites; no calls in the system.
2. Calls arrive as a non-homogeneous Poisson process with the hourly rates in `arrival_rates.csv`; the call location is a demand node drawn with probability proportional to its weight, and the destination hospital is that of the node's comuna.
3. A call seizes the idle ambulance with the shortest network travel time to the scene. If none is idle, it waits in a first-come, first-served queue.
4. The unit draws a setup time, travels to the scene on the road network at the speed of the regime in force at departure (`speed_regimes.csv`), draws an on-scene time, travels to the hospital, draws patient-handling and cleaning times, returns to its site, and becomes idle on arrival.
5. Horizon 720 h; the first 24 h are discarded; 12 independent replicates per configuration.

`simulation_output/<configuration>.csv` columns: `replicate`, `call`, `t_call_min` (call time from the start of the replicate), `queue_wait`, `travel_time`, `response_time` (call to arrival, includes waiting), `response_time_from_assignment`, `total_service_time` (call to unit release). All times in minutes.

Service-time distributions (minutes): setup Burr XII(c = 2.86594, k = 0.74544, scale = 4.39886) truncated at 60; on-scene Gamma(shape 3.81963, scale 5.36807); patient handling Lognormal(mu 2.31991, sigma 0.75984); cleaning Lognormal(mu 1.65067, sigma 0.88819). Effective speeds: 27 km/h (00-08 h), 22 km/h (08-16 h), 25 km/h (16-24 h).

## Citation

Olivos C., Arrey E., Caceres H. An Optimization-Simulation Framework for Reducing Ambulance Response Time: The Case of La Serena-Coquimbo, Chile. *International Journal of Health Geographics* (under review).

## License

All content of this repository (code, data, results and figures) is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Commercial use requires written permission from the authors.
