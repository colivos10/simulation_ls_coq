from gurobipy import Model, GRB, quicksum


def pmedian(bases, emergency_nodes, edges, weight, time, n_ambulances, optimizer_time_limit=None):
    """p-median: minimize demand-weighted average travel time."""
    mdl = Model("pmedian")
    x = mdl.addVars(edges, vtype=GRB.BINARY, name='x')
    y = mdl.addVars(bases, vtype=GRB.BINARY, name='y')

    sum_weight = sum(weight.values())
    mdl.setObjective(
        quicksum((weight[j] / sum_weight) * time[(i, j)] * x[(i, j)] for i, j in edges),
        GRB.MINIMIZE)

    mdl.addConstr(quicksum(y[i] for i in bases) <= n_ambulances)
    mdl.addConstrs(x[(i, j)] <= y[i] for i, j in edges)
    mdl.addConstrs(quicksum(x[(i, j)] for i, j2 in edges if j2 == j) == 1 for j in emergency_nodes)

    if optimizer_time_limit is not None:
        mdl.Params.timeLimit = optimizer_time_limit
    mdl.setParam("OutputFlag", True)
    mdl.optimize()

    return {
        'model': mdl.getAttr('ModelName'),
        'ambulance_locations': [i for i in bases if y[i].x >= 0.9],
        'node_assignments': {i: [j for (i2, j) in edges if i2 == i and x[i, j].x > 0.9]
                             for i in bases if y[i].x > 0.9},
        'optimizer': 'gurobi',
        'optimizer_details': {'gap': mdl.MIPGap, 'objective_value': mdl.objVal,
                              'running_time': mdl.Runtime}
    }


def pcenter(bases, emergency_nodes, edges, weight, time, n_ambulances, optimizer_time_limit=None):
    """p-center: minimize the maximum assigned travel time."""
    mdl = Model("pcenter")
    x = mdl.addVars(edges, vtype=GRB.BINARY, name='x')
    y = mdl.addVars(bases, vtype=GRB.BINARY, name='y')
    z = mdl.addVar(vtype=GRB.CONTINUOUS, name='z')

    mdl.setObjective(z, GRB.MINIMIZE)
    mdl.addConstrs(z >= time[(i, j)] * x[(i, j)] for i, j in edges)

    mdl.addConstr(quicksum(y[i] for i in bases) <= n_ambulances)
    mdl.addConstrs(x[(i, j)] <= y[i] for i, j in edges)
    mdl.addConstrs(quicksum(x[(i, j)] for i, j2 in edges if j2 == j) == 1 for j in emergency_nodes)

    if optimizer_time_limit is not None:
        mdl.Params.timeLimit = optimizer_time_limit
    mdl.setParam("OutputFlag", True)
    mdl.optimize()

    return {
        'model': mdl.getAttr('ModelName'),
        'ambulance_locations': [i for i in bases if y[i].x >= 0.9],
        'node_assignments': {i: [j for (i2, j) in edges if i2 == i and x[i, j].x > 0.9]
                             for i in bases if y[i].x > 0.9},
        'optimizer': 'gurobi',
        'optimizer_details': {'gap': mdl.MIPGap, 'objective_value': mdl.objVal,
                              'running_time': mdl.Runtime}
    }


def mclp(bases, emergency_nodes, edges, weight, time, n_ambulances, time_max, optimizer_time_limit=None):
    """Coverage criterion: minimize the fraction of demand whose routed travel
    time from the assigned base is at least time_max (tau)."""
    mdl = Model("mclp")
    x = mdl.addVars(edges, vtype=GRB.BINARY, name='x')
    y = mdl.addVars(bases, vtype=GRB.BINARY, name='y')

    sum_weight = sum(weight.values())
    mdl.setObjective(
        quicksum((weight[j] / sum_weight) * x[(i, j)]
                 for i, j in edges if time[(i, j)] >= time_max),
        GRB.MINIMIZE)

    mdl.addConstr(quicksum(y[i] for i in bases) <= n_ambulances)
    mdl.addConstrs(x[(i, j)] <= y[i] for i, j in edges)
    mdl.addConstrs(quicksum(x[(i, j)] for i, j2 in edges if j2 == j) == 1 for j in emergency_nodes)

    if optimizer_time_limit is not None:
        mdl.Params.timeLimit = optimizer_time_limit
    mdl.setParam("OutputFlag", True)
    mdl.optimize()

    return {
        'model': mdl.getAttr('ModelName'),
        'ambulance_locations': [i for i in bases if y[i].x >= 0.9],
        'node_assignments': {i: [j for (i2, j) in edges if i2 == i and x[i, j].x > 0.9]
                             for i in bases if y[i].x > 0.9},
        'optimizer': 'gurobi',
        'optimizer_details': {'gap': mdl.MIPGap, 'objective_value': mdl.objVal,
                              'running_time': mdl.Runtime}
    }
