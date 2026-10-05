"""Validation of the simulated current deployment against the observed system (aggregate level).
Observed values are the pre-pandemic aggregates in data/observed_summary.csv and data/observed_cdf.csv."""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

WARMUP_MIN = 24 * 60
plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'pdf.fonttype': 42})

# %% Inputs
sim = pd.read_csv('simulation_output/current.csv')
sim = sim[sim.t_call_min >= WARMUP_MIN].copy()
sim['setup_time'] = sim.response_time_from_assignment - sim.travel_time
obs = pd.read_csv('data/observed_summary.csv').set_index('quantity')
cdf = pd.read_csv('data/observed_cdf.csv')

# %% Comparison table
rep = sim.groupby('replicate')
rows = []
for name, col in [('response_time', 'response_time'), ('travel_time', 'travel_time'), ('setup_time', 'setup_time'),
                  ('total_service_time', 'total_service_time')]:
    means = rep[col].mean()
    hw = stats.t.ppf(0.975, len(means) - 1) * means.std(ddof=1) / np.sqrt(len(means))
    rows.append({'quantity': name, 'observed_mean': obs.loc[name, 'mean'], 'simulated_mean': sim[col].mean(),
                 'simulated_hw': hw, 'observed_median': obs.loc[name, 'median'], 'simulated_median': sim[col].median(),
                 'observed_p90': obs.loc[name, 'p90'], 'simulated_p90': sim[col].quantile(0.9)})
for name, col in [('response_time', 'response_time'), ('travel_time', 'travel_time')]:
    for t in [5, 8, 10, 15, 20, 30]:
        c = rep[col].apply(lambda x: 100 * (x <= t).mean())
        hw = stats.t.ppf(0.975, len(c) - 1) * c.std(ddof=1) / np.sqrt(len(c))
        rows.append({'quantity': f'{name}_within_{t}_pct',
                     'observed_mean': 100 * cdf.loc[cdf.minute == t, name + '_cdf'].iloc[0],
                     'simulated_mean': 100 * (sim[col] <= t).mean(), 'simulated_hw': hw})
table = pd.DataFrame(rows)
print(table.round(2).to_string(index=False))
table.to_csv('results/validation_table.csv', index=False)

# %% Kolmogorov-Smirnov distance against the observed CDF (observed records are in whole minutes)
grid = cdf.minute.values
for name, col in [('response_time', 'response_time'), ('travel_time', 'travel_time')]:
    sim_cdf = np.array([(np.round(sim[col]) <= g).mean() for g in grid])
    print(f'KS distance, {name}: {np.abs(sim_cdf - cdf[name + "_cdf"].values).max():.3f}')

# %% Figure: cumulative distributions
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
a.step(grid, 100 * cdf.response_time_cdf, where='post', color='#0072B2', lw=1.6, label='Observed records')
x = np.sort(sim.response_time.values)
a.plot(x, 100 * np.arange(1, len(x) + 1) / len(x), color='#D55E00', lw=1.8, ls='--', label='Simulated, calibrated speed')
a.axvline(8, color='grey', lw=0.8); a.text(8.4, 92, '8 min target', color='grey', fontsize=8)
a.set_xlim(0, 40); a.set_ylim(0, 100); a.set_xlabel('Response time (min)'); a.set_ylabel('Emergencies attended (%)')
a.set_title('(a) Response time', fontsize=10); a.legend(loc='lower right', frameon=False, fontsize=8.5)
b.step(grid, 100 * cdf.travel_time_cdf, where='post', color='#0072B2', lw=1.6, label='Observed records')
x = np.sort(sim.travel_time.values)
b.plot(x, 100 * np.arange(1, len(x) + 1) / len(x), color='#D55E00', lw=1.8, ls='--', label='Simulated, calibrated speed')
b.plot(grid, 100 * cdf.freeflow_travel_cdf, color='#CC79A7', lw=1.4, ls=':', label='OSRM free-flow routing')
b.set_xlim(0, 30); b.set_ylim(0, 100); b.set_xlabel('Base to scene travel time (min)'); b.set_ylabel('Trips (%)')
b.set_title('(b) Travel time', fontsize=10); b.legend(loc='lower right', frameon=False, fontsize=8.5)
for ax in (a, b):
    ax.grid(alpha=0.3, lw=0.5)
fig.tight_layout(); fig.savefig('figures/fig_validation_cdf.pdf'); plt.close(fig)
