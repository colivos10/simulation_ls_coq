"""Sensitivity of the MCLP to tau and of p-median and MCLP to the +1 in the demand weights.
Solved with HiGHS (scipy.optimize.milp); reproduces the Gurobi solutions."""
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix, csr_matrix, vstack

P = 8

# %% Inputs
bases = pd.read_csv('data/candidate_bases.csv')
nodes = pd.read_csv('data/demand_nodes.csv')
tt = pd.read_csv('data/travel_times.csv')
conf = pd.read_csv('data/configurations.csv')

bc = bases.base.values
nc = nodes.node.values
bi = {c: i for i, c in enumerate(bc)}
ni = {c: i for i, c in enumerate(nc)}
nb, nn = len(bc), len(nc)
T = np.full((nb, nn), np.inf)
T[tt.base.map(bi).values, tt.node.map(ni).values] = tt.duration_s.values / 60.0
W1 = nodes.weight.values.astype(float)       # records + 1 (as in the paper)
W0 = W1 - 1                                  # raw record counts
paper = {m: set(conf[conf.configuration == m.upper()].base) for m in ['pmedian', 'mclp']}

# %% MCLP for tau in {3, 5, 8, 10}, weights +1
rows = []
sols = {}
for tau in [3, 5, 8, 10]:
    Wn = W1 / W1.sum()
    cov = T < tau
    r, c, v = [], [], []
    for j in range(nn):
        for i in np.where(cov[:, j])[0]:
            r.append(j); c.append(i); v.append(-1)
        r.append(j); c.append(nb + j); v.append(1)
    A = vstack([coo_matrix((v, (r, c)), shape=(nn, nb + nn)).tocsr(),
                csr_matrix(np.concatenate([np.ones(nb), np.zeros(nn)]))])
    lb = np.concatenate([-np.inf * np.ones(nn), [P]])
    ub = np.concatenate([np.zeros(nn), [P]])
    res = milp(np.concatenate([np.zeros(nb), -Wn]), constraints=LinearConstraint(A, lb, ub),
               integrality=np.concatenate([np.ones(nb), np.zeros(nn)]), bounds=Bounds(0, 1))
    S = {bc[i] for i in np.where(np.round(res.x[:nb]) > 0.5)[0]}
    sols[tau] = S
    t = T[[bi[s] for s in S]].min(0)
    rows.append({'tau': tau, 'coverage_pct': -res.fun * 100, 'mean_travel': (Wn * t).sum(),
                 'shared_with_tau5_paper': len(S & paper['mclp']), 'sites': ' '.join(sorted(S))})
tau_table = pd.DataFrame(rows)
print(tau_table.drop(columns='sites').round(2).to_string(index=False))
tau_table.to_csv('results/sensitivity_tau.csv', index=False)

# %% MCLP (tau = 5) and p-median with raw counts instead of +1
rows = []
for W, label in [(W1, 'weights +1'), (W0, 'raw counts')]:
    # MCLP
    Wn = W / W.sum()
    cov = T < 5
    r, c, v = [], [], []
    for j in range(nn):
        for i in np.where(cov[:, j])[0]:
            r.append(j); c.append(i); v.append(-1)
        r.append(j); c.append(nb + j); v.append(1)
    A = vstack([coo_matrix((v, (r, c)), shape=(nn, nb + nn)).tocsr(),
                csr_matrix(np.concatenate([np.ones(nb), np.zeros(nn)]))])
    lb = np.concatenate([-np.inf * np.ones(nn), [P]])
    ub = np.concatenate([np.zeros(nn), [P]])
    res = milp(np.concatenate([np.zeros(nb), -Wn]), constraints=LinearConstraint(A, lb, ub),
               integrality=np.concatenate([np.ones(nb), np.zeros(nn)]), bounds=Bounds(0, 1))
    S = {bc[i] for i in np.where(np.round(res.x[:nb]) > 0.5)[0]}
    rows.append({'model': 'mclp', 'weights': label, 'objective': -res.fun * 100,
                 'shared_with_paper': len(S & paper['mclp']), 'sites': ' '.join(sorted(S))})
    # p-median (x continuous, y binary; nodes with zero weight dropped)
    keep = np.where(W > 0)[0]
    Wk = W[keep] / W[keep].sum()
    Tk = T[:, keep]
    m = len(keep)
    nx = nb * m
    cost = np.concatenate([np.zeros(nb), (Tk * Wk[None, :]).ravel()])
    A1 = coo_matrix((np.ones(nx), (np.repeat(np.arange(m), nb),
                                   nb + np.tile(np.arange(nb), m) * m + np.repeat(np.arange(m), nb))),
                    shape=(m, nb + nx))
    rows_x = np.arange(nx)
    A2 = coo_matrix((np.concatenate([np.ones(nx), -np.ones(nx)]),
                     (np.concatenate([rows_x, rows_x]), np.concatenate([nb + rows_x, rows_x // m]))),
                    shape=(nx, nb + nx))
    A3 = csr_matrix(np.concatenate([np.ones(nb), np.zeros(nx)]))
    A = vstack([A1.tocsr(), A2.tocsr(), A3])
    lb = np.concatenate([np.ones(m), -np.inf * np.ones(nx), [P]])
    ub = np.concatenate([np.ones(m), np.zeros(nx), [P]])
    res = milp(cost, constraints=LinearConstraint(A, lb, ub),
               integrality=np.concatenate([np.ones(nb), np.zeros(nx)]), bounds=Bounds(0, 1))
    S = {bc[i] for i in np.where(np.round(res.x[:nb]) > 0.5)[0]}
    rows.append({'model': 'pmedian', 'weights': label, 'objective': res.fun,
                 'shared_with_paper': len(S & paper['pmedian']), 'sites': ' '.join(sorted(S))})
weight_table = pd.DataFrame(rows)
print(weight_table.drop(columns='sites').round(3).to_string(index=False))
weight_table.to_csv('results/sensitivity_weights.csv', index=False)
