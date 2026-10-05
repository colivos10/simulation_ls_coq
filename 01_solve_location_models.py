"""Solve the p-median, p-center and MCLP models for 8 ambulances (Gurobi)."""
import json
import pandas as pd
from optimization_models import pmedian, pcenter, mclp

# %% Inputs
df_bases = pd.read_csv('data/candidate_bases.csv')
df_nodes = pd.read_csv('data/demand_nodes.csv')
df_tt = pd.read_csv('data/travel_times.csv')

n_ambulances = 8
tau = 5          # minutes of routed travel time (MCLP threshold)
time_limit = 600

# %% Sets and parameters
bases = list(df_bases['base'])                # I: candidate ambulance locations
nodes = list(df_nodes['node'])                # J: emergency nodes
edges = list(zip(df_tt['base'], df_tt['node']))

response_time = dict(zip(edges, df_tt['duration_s'] / 60))   # t_ij in minutes
weight = dict(zip(df_nodes['node'], df_nodes['weight']))      # a_j = records + 1

# %% Solve
output_pmedian = pmedian(bases=bases, emergency_nodes=nodes, edges=edges, weight=weight,
                         time=response_time, n_ambulances=n_ambulances,
                         optimizer_time_limit=time_limit)
print(output_pmedian['ambulance_locations'])

output_pcenter = pcenter(bases=bases, emergency_nodes=nodes, edges=edges, weight=weight,
                         time=response_time, n_ambulances=n_ambulances,
                         optimizer_time_limit=time_limit)
print(output_pcenter['ambulance_locations'])

output_mclp = mclp(bases=bases, emergency_nodes=nodes, edges=edges, weight=weight,
                   time=response_time, n_ambulances=n_ambulances, time_max=tau,
                   optimizer_time_limit=time_limit)
print(output_mclp['ambulance_locations'])

# %% Export
solutions = {}
for name, out in [('pmedian', output_pmedian), ('pcenter', output_pcenter), ('mclp', output_mclp)]:
    solutions[name] = sorted(out['ambulance_locations'])
    solutions[name + '_objective'] = out['optimizer_details']['objective_value']

with open('results/locations_optimized.json', 'w') as f:
    json.dump(solutions, f, indent=2)
print('Exported to results/locations_optimized.json')

# %% Check against the configurations that were simulated
conf = pd.read_csv('data/configurations.csv')
for name in ['pmedian', 'pcenter', 'mclp']:
    solved = set(solutions[name])
    simulated = set(conf[conf.configuration == name.upper()].base)
    print(f'{name}: {len(solved & simulated)} of {len(simulated)} simulated sites recovered, {len(solved)} sites selected')
# p-median is recovered exactly. The MCLP and p-center optima are not unique (alternative site sets
# with identical objective values exist), so a solver may return a different set, or fewer than 8
# sites for p-center. The configurations simulated in the paper are the ones in data/configurations.csv.
