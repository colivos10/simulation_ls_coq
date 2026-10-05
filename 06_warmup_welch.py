"""Welch warm-up check on the current deployment: daily mean response time across replicates."""
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'pdf.fonttype': 42})

# %% Daily means
sim = pd.read_csv('simulation_output/current.csv')
sim['day'] = (sim.t_call_min // 1440).astype(int) + 1
daily = sim.groupby(['day', 'replicate']).response_time.mean().groupby('day').mean()
moving = daily.rolling(3, center=True, min_periods=1).mean()
after = sim[sim.t_call_min >= 1440].response_time.mean()
print(f'day 1 mean {daily.iloc[0]:.2f}, mean after 24 h {after:.2f}')

# %% Figure
fig, ax = plt.subplots(figsize=(6.2, 3.4))
ax.plot(daily.index, daily.values, color='#9A9A9A', lw=0.9, marker='o', ms=3, label='Daily mean across 12 replicates')
ax.plot(moving.index, moving.values, color='#0072B2', lw=1.8, label='Welch moving average (window 3 days)')
ax.axhline(after, color='#D55E00', lw=1, ls='--', label='Mean after 24 h warm-up')
ax.axvline(1.5, color='black', lw=0.8, ls=':')
ax.set_xlabel('Simulated day'); ax.set_ylabel('Mean response time (min)')
ax.set_xlim(0.5, 30.5); ax.set_ylim(14, 19); ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
ax.legend(frameon=False, fontsize=8, loc='upper right'); ax.grid(alpha=0.3, lw=0.5)
fig.tight_layout(); fig.savefig('figures/fig_welch_warmup.pdf'); plt.close(fig)
