"""Dispersal benchmarks and static evaluation of all configurations.
B1  demand-weighted k-means, centroids snapped to the nearest unused candidate base (needs coordinates).
B2  demand-weighted random draws of 8 candidate bases (500 draws).
Static evaluation on the travel-time matrix: weighted mean travel time, maximum, coverage."""
import os
import json
import numpy as np
import pandas as pd

P, N_DRAWS, SEED, RESTARTS = 8, 500, 20260917, 50
TAU = [5.0, 8.0]
rng = np.random.default_rng(SEED)

# %% Inputs
bases = pd.read_csv('data/candidate_bases.csv')
nodes = pd.read_csv('data/demand_nodes.csv')
tt = pd.read_csv('data/travel_times.csv')
conf = pd.read_csv('data/configurations.csv')
bc = bases.base.values
nc = nodes.node.values
w = nodes.weight.values.astype(float)
W = w.sum()

bi = {c: i for i, c in enumerate(bc)}
ni = {c: i for i, c in enumerate(nc)}
T = np.full((len(bc), len(nc)), np.nan)
T[tt.base.map(bi).values, tt.node.map(ni).values] = tt.duration_s.values / 60.0
T = np.where(np.isfinite(T), T, np.nanmax(T))


def ev(S):
    t = T[list(S)].min(axis=0)
    o = {'mean_travel': float((t * w).sum() / W), 'max_travel': float(t.max())}
    for tau in TAU:
        o[f'cov_{tau:g}'] = float(w[t < tau].sum() / W)
    return o


# %% B2 demand-weighted random draws
# selection probability proportional to the demand within 1.5 km of each candidate base
p_smart = bases.demand_within_1500m.values.astype(float)
p_smart = p_smart / p_smart.sum()
rows = []
for r in range(N_DRAWS):
    S = sorted(rng.choice(len(bc), size=P, replace=False, p=p_smart))
    e = ev(S)
    e['draw'] = r
    e['S'] = '|'.join(map(str, S))
    rows.append(e)
draws = pd.DataFrame(rows)
med = draws.mean_travel.median()
S_rnd = [int(x) for x in draws.iloc[(draws.mean_travel - med).abs().argmin()].S.split('|')]
draws.to_csv('results/random_draws.csv', index=False)

# %% Configurations as simulated
# CURRENT is the two hospitals, represented here by the nearest candidate base to each
cfg = {k: sorted({bi[c] for c in conf[conf.configuration == k.upper()].base})
       for k in ['current', 'pmedian', 'pcenter', 'mclp', 'kmeans', 'random']}
print(f"random: {len(set(S_rnd) & set(cfg['random']))} of 8 sites equal to the simulated configuration")

# %% B1 weighted k-means (only if the coordinate file is available; see README)
if os.path.exists('data/coordinates.csv'):
    xy = pd.read_csv('data/coordinates.csv')          # columns: id, lat, lon (bases and nodes)
    nxy = xy.set_index('id').loc[nc]
    bxy = xy.set_index('id').loc[bc]
    lat0 = nxy.lat.mean()
    X = np.c_[(nxy.lon.values - nxy.lon.mean()) * 111.32 * np.cos(np.radians(lat0)),
              (nxy.lat.values - nxy.lat.mean()) * 111.32]

    def hav(la1, lo1, la2, lo2):
        R = 6371.0088
        p = np.radians
        a = np.sin(p(la2 - la1) / 2) ** 2 + np.cos(p(la1)) * np.cos(p(la2)) * np.sin(p(lo2 - lo1) / 2) ** 2
        return 2 * R * np.arcsin(np.sqrt(a))

    def nearest(lat, lon, excl=()):
        d = hav(lat, lon, bxy.lat.values, bxy.lon.values).copy()
        for i in excl:
            d[i] = np.inf
        return int(d.argmin())

    def wkm(X, w, k, rng, iters=300):
        idx = [rng.choice(len(X), p=w / w.sum())]
        for _ in range(k - 1):
            d2 = ((X - X[idx][:, None, :]) ** 2).sum(2).min(0)
            p = d2 * w
            idx.append(rng.choice(len(X), p=p / p.sum()))
        C = X[idx].copy()
        for _ in range(iters):
            lab = ((X[:, None, :] - C[None, :, :]) ** 2).sum(2).argmin(1)
            nC = C.copy()
            for j in range(k):
                m = lab == j
                if m.sum():
                    nC[j] = (X[m] * w[m, None]).sum(0) / w[m].sum()
            if np.allclose(nC, C):
                break
            C = nC
        lab = ((X[:, None, :] - C[None, :, :]) ** 2).sum(2).argmin(1)
        return C, lab, (w * ((X - C[lab]) ** 2).sum(1)).sum()

    best, blab, best_inertia = None, None, np.inf
    for _ in range(RESTARTS):
        C, lab, inertia = wkm(X, w, P, rng)
        if inertia < best_inertia:
            best, blab, best_inertia = C, lab, inertia
    lon_c = best[:, 0] / (111.32 * np.cos(np.radians(lat0))) + nxy.lon.mean()
    lat_c = best[:, 1] / 111.32 + nxy.lat.mean()
    cw = np.array([w[blab == j].sum() for j in range(P)])
    S_km = []
    for j in np.argsort(-cw):                   # heaviest clusters choose first
        S_km.append(nearest(lat_c[j], lon_c[j], excl=S_km))
    print(f"kmeans: {len(set(S_km) & set(cfg['kmeans']))} of 8 sites equal to the simulated configuration")

# %% Static evaluation
res = []
for k, S in cfg.items():
    e = ev(S)
    e['configuration'] = k
    e['n_sites'] = len(set(S))
    res.append(e)
out = pd.DataFrame(res)[['configuration', 'n_sites', 'mean_travel', 'max_travel'] + [f'cov_{t:g}' for t in TAU]]
print(out.round(4).to_string(index=False))
print('random draws, mean travel: p05 %.2f  median %.2f  p95 %.2f'
      % (draws.mean_travel.quantile(.05), med, draws.mean_travel.quantile(.95)))
out.to_csv('results/static_evaluation.csv', index=False)
json.dump({k: [bc[i] for i in S] for k, S in cfg.items()}, open('results/configurations_by_base.json', 'w'), indent=2)
