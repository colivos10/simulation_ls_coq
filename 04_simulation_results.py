"""Operational results of the six simulated configurations.
Input: simulation_output/<configuration>.csv, one row per call, 12 replicates of 720 h.
Warm-up: first 24 h of each replicate discarded. Statistical unit: the replicate."""
import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg
import matplotlib.pyplot as plt

ORDER = ['current', 'pmedian', 'pcenter', 'mclp', 'kmeans', 'random']
LABEL = {'current': 'Current', 'pmedian': 'p-median', 'pcenter': 'p-center',
         'mclp': 'MCLP', 'kmeans': 'k-means', 'random': 'Random'}
PAL = {'current': '#7F7F7F', 'pmedian': '#0072B2', 'pcenter': '#D55E00',
       'mclp': '#009E73', 'kmeans': '#CC79A7', 'random': '#E69F00'}
WARMUP_MIN = 24 * 60
plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'pdf.fonttype': 42})

# %% Load and discard warm-up
data = pd.concat([pd.read_csv(f'simulation_output/{k}.csv').assign(configuration=k) for k in ORDER])
data = data[data.t_call_min >= WARMUP_MIN].copy()
data['setup_time'] = data.response_time_from_assignment - data.travel_time

# %% Per-replicate metrics
g = data.groupby(['configuration', 'replicate'])
rep = pd.DataFrame({
    'mean_rt': g.response_time.mean(),
    'p90_rt': g.response_time.quantile(0.90),
    'max_rt': g.response_time.max(),
    'cov8': g.response_time.apply(lambda x: 100 * (x <= 8).mean()),
    'cov15': g.response_time.apply(lambda x: 100 * (x <= 15).mean()),
    'mean_travel': g.travel_time.mean(),
    'mean_setup': g.setup_time.mean(),
    'mean_queue': g.queue_wait.mean(),
    'pct_queued': g.queue_wait.apply(lambda x: 100 * (x > 0).mean()),
    'mean_total': g.total_service_time.mean(),
    'n_calls': g.size(),
}).reset_index()
rep.to_csv('results/per_replicate_metrics.csv', index=False)

# %% Mean and 95% confidence half-width per configuration
metrics = [c for c in rep.columns if c not in ('configuration', 'replicate')]
summary = rep.groupby('configuration')[metrics].agg(['mean', 'sem', 'count'])
rows = []
for k in ORDER:
    row = {'configuration': k}
    for m in metrics:
        n = summary.loc[k, (m, 'count')]
        row[m + '_mean'] = summary.loc[k, (m, 'mean')]
        row[m + '_hw'] = stats.t.ppf(0.975, n - 1) * summary.loc[k, (m, 'sem')]
    rows.append(row)
summary = pd.DataFrame(rows).set_index('configuration')
summary.to_csv('results/summary_by_configuration.csv')
print(summary[['mean_rt_mean', 'mean_rt_hw', 'cov8_mean', 'cov8_hw', 'cov15_mean', 'cov15_hw',
               'mean_travel_mean', 'pct_queued_mean']].round(2).to_string())

# %% Welch ANOVA and Games-Howell
tests = []
for m in ['mean_rt', 'cov8', 'cov15', 'mean_travel', 'p90_rt', 'max_rt', 'mean_total']:
    aov = pg.welch_anova(dv=m, between='configuration', data=rep)
    gh = pg.pairwise_gameshowell(dv=m, between='configuration', data=rep)
    gh['metric'] = m
    gh['welch_F'] = aov['F'].iloc[0]
    gh['welch_ddof2'] = aov['ddof2'].iloc[0]
    gh['welch_p'] = aov['p_unc'].iloc[0]
    tests.append(gh)
    print(f"\n{m}: Welch F = {aov['F'].iloc[0]:.2f}, p = {aov['p_unc'].iloc[0]:.3g}")
    print(gh[['A', 'B', 'diff', 'pval']].round(3).to_string(index=False))
pd.concat(tests).to_csv('results/games_howell.csv', index=False)

# %% Differences against the current deployment (Welch t confidence intervals)
rows = []
cur = rep[rep.configuration == 'current']
for k in ORDER[1:]:
    r = rep[rep.configuration == k]
    for m in ['mean_rt', 'cov8', 'cov15', 'mean_travel']:
        t = stats.ttest_ind(r[m], cur[m], equal_var=False)
        ci = t.confidence_interval(0.95)
        rows.append({'configuration': k, 'metric': m, 'difference': r[m].mean() - cur[m].mean(),
                     'ci_low': ci.low, 'ci_high': ci.high, 'p': t.pvalue})
diffs = pd.DataFrame(rows)
diffs.to_csv('results/differences_vs_current.csv', index=False)
print(diffs.round(3).to_string(index=False))

# %% Figure: coverage curve
thr = np.arange(0, 15.5, 0.5)
fig, ax = plt.subplots(figsize=(6.2, 4))
for k in ORDER:
    d = data[data.configuration == k]
    M = np.array([[100 * (d[d.replicate == r].response_time <= t).mean() for t in thr]
                  for r in sorted(d.replicate.unique())])
    mean = M.mean(0)
    hw = stats.t.ppf(0.975, len(M) - 1) * M.std(0, ddof=1) / np.sqrt(len(M))
    ax.plot(thr, mean, color=PAL[k], lw=1.6, marker='o', ms=2.5, markevery=2, label=LABEL[k],
            ls='--' if k in ('kmeans', 'random') else '-')
    ax.fill_between(thr, mean - hw, mean + hw, color=PAL[k], alpha=0.18, lw=0)
ax.set_xlim(0, 15.5); ax.set_ylim(0, 100); ax.set_xticks(range(0, 16)); ax.grid(alpha=0.3, lw=0.5)
ax.set_xlabel('Response time threshold (min)'); ax.set_ylabel('Emergencies covered (%)')
ax.legend(title='Configuration', loc='upper left', fontsize=8, title_fontsize=8)
fig.tight_layout(); fig.savefig('figures/s_curve_replicate_ci.pdf'); plt.close(fig)

# %% Figures: replicate dot plots
for metric, ylabel, fname in [('mean_rt', 'Average response time per replicate (min)', 'avg_response_time_replicate_dotplot.pdf'),
                              ('mean_total', 'Average total service time per replicate (min)', 'total_time_replicate_dotplot.pdf')]:
    fig, ax = plt.subplots(figsize=(6.6, 4))
    jitter = np.random.default_rng(1)
    for i, k in enumerate(ORDER):
        v = rep[rep.configuration == k][metric].values
        hw = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        ax.scatter(i + jitter.uniform(-0.12, 0.12, len(v)), v, color=PAL[k], alpha=0.45, s=18, zorder=2)
        ax.errorbar(i, v.mean(), yerr=hw, fmt='o', color=PAL[k], mec='black', ms=8, capsize=3, lw=1.6, zorder=3)
    ax.set_xticks(range(6)); ax.set_xticklabels([LABEL[k] for k in ORDER]); ax.set_ylabel(ylabel)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, lw=0.5)
    fig.tight_layout(); fig.savefig(f'figures/{fname}'); plt.close(fig)
