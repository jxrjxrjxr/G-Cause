import pandas as pd
import numpy as np
from algo.m_get_matrix_by_pc import buildGraphPC
from utils.m_pearson import calc_pearson
from utils.load import aggregate, fault_range_extract


def firstorder_randomwalk(
    P,
    epochs,
    start_node,
    teleportation_prob,
    label=[],
    walk_step=1000,
    print_trace=False,
):
    n = P.shape[0]
    score = np.zeros([n])
    current = start_node - 1
    for epoch in range(epochs):
        if print_trace:
            print("\n{:2d}".format(current + 1), end="->")
        for step in range(walk_step):
            if np.sum(P[current]) == 0:
                current = np.random.choice(range(n), p=teleportation_prob)
                break
            else:
                next_node = np.random.choice(range(n), p=P[current])
                if print_trace:
                    print("{:2d}".format(current + 1), end="->")
                score[next_node] += 1
                current = next_node
    score_list = list(zip(label, score))
    score_list.sort(key=lambda x: x[1], reverse=True)
    return score_list


def normalize(p):
    """Normalize the matrix in each row
    """
    p = p.copy()
    for i in range(p.shape[0]):
        row_sum = np.sum(p[i])
        if row_sum > 0:
            p[i] /= row_sum
    return p


def relaToRank(rela, access, rankPaces, frontend, rho=0.3, print_trace=False):
    n = len(access)
    S = [abs(_) for _ in rela[frontend - 1]]
    P = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            # forward edge
            if access[i][j] != 0:
                P[i, j] = abs(S[j])
            # backward edge
            elif access[j][i] != 0:
                P[i, j] = rho * abs(S[i])
    # Add self edges
    for i in range(n):
        if i != frontend - 1:
            P[i][i] = max(0, S[i] - max(P[i]))
    P = normalize(P)

    if np.sum(S) > 0:
        teleportation_prob = np.array(S) / np.sum(S).tolist()
    else:
        teleportation_prob = [1. / len(rela[frontend - 1]) for _ in rela[frontend - 1]]
    label = [i for i in range(1, n + 1)]
    ll = firstorder_randomwalk(
        P, rankPaces, frontend, teleportation_prob, label, print_trace=print_trace
    )
    # print(l)
    return ll, P


def RCA(
    df: pd.DataFrame,
    ftp: int,
    entry: str = "load1",
    prk: int = 10,
    pc_aggregate: int = 6,
    pc_alpha: float = 0.1
) -> list:
    fdf = fault_range_extract(df, ftp, zero=False)
    mlist = list(df.columns)
    frontend = mlist.index(entry)
    data = np.array([aggregate(row, pc_aggregate) for row in fdf.values.T])
    rela = calc_pearson(data, method="numpy", zero_diag=False)
    dep_graph = buildGraphPC(data, alpha=pc_alpha)
    nodesMR, P = relaToRank(rela, dep_graph, 10, frontend + 1, rho=0.2, print_trace=False)
    return list(map(lambda x: (mlist[x[0] - 1], x[1]), nodesMR[:prk]))
