import pandas as pd

import numpy as np
from algo.m_get_matrix_by_pc import buildGraphPC
from utils.m_pearson import calc_pearson
from utils.load import fault_range_extract, aggregate


def secondorder_randomwalk(
    M, epochs, start_node, label=[], walk_step=1000, print_trace=False
):
    n = M.shape[0]
    score = np.zeros([n])
    for epoch in range(epochs):
        previous = start_node - 1
        current = start_node - 1
        if print_trace:
            print("\n{:2d}".format(current + 1), end="->")
        for step in range(walk_step):
            if np.sum(M[previous, current]) == 0:
                break
            next_node = np.random.choice(range(n), p=M[previous, current])
            if print_trace:
                print("{:2d}".format(current + 1), end="->")
            score[next_node] += 1
            previous = current
            current = next_node
    score_list = list(zip(label, score))
    score_list.sort(key=lambda x: x[1], reverse=True)
    return score_list


def guiyi(p):
    """矩阵按照行归一化
    """
    nextp = [[0 for i in range(len(p[0]))] for j in range(len(p))]
    for i in range(len(p)):
        for j in range(len(p[0])):
            lineSum = (np.sum(p, axis=1))[i]
            if lineSum == 0:
                break
            nextp[i][j] = p[i][j] / lineSum
    return nextp


def relaToRank(rela, access, rankPaces, frontend, beta=0.1, rho=0.3, print_trace=False):
    n = len(access)
    S = rela[frontend - 1]
    P = [[0 for col in range(n)] for row in range(n)]
    for i in range(n):
        for j in range(n):
            if access[i][j] != 0:
                P[i][j] = abs(S[j])
    P = guiyi(P)
    M = np.zeros([n, n, n])
    # Forward probability
    for i in range(n):
        for j in range(n):
            if access[i][j] > 0:
                for k in range(n):
                    M[k, i, j] = (1 - beta) * P[k][i] + beta * P[i][j]
    # Normalize w.r.t. out nodes
    for k in range(n):
        for i in range(n):
            if np.sum(M[k, i]) > 0:
                M[k, i] = M[k, i] / np.sum(M[k, i])
    # Add backward edges
    for k in range(n):
        for i in range(n):
            in_inds = []
            for j in range(n):
                if access[i][j] == 0 and access[j][i] != 0:
                    M[k, i, j] = rho * ((1 - beta) * P[k][i] + beta * P[j][i])
                    in_inds.append(j)
            # Normalize wrt in nodes
            if np.sum(M[k, i, in_inds]) > 0:
                M[k, i, in_inds] /= np.sum(M[k, i, in_inds])
    # Add self edges
    for k in range(n):
        for i in range(n):
            if M[k, i, i] == 0:
                in_out_node = list(range(n))
                in_out_node.remove(i)
                M[k, i, i] = max(0, S[i] - max(M[k, i, in_out_node]))
    # Normalize all
    for k in range(n):
        for i in range(n):
            if np.sum(M[k, i]) > 0:
                M[k, i] /= np.sum(M[k, i])

    label = [i for i in range(1, n + 1)]
    # l = monitorrange(road, rankPaces, fronted, label)  # relaToRank = 16
    ll = secondorder_randomwalk(M, rankPaces, frontend, label, print_trace=print_trace)
    # print(l)
    return ll, P, M


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
    nodesCR, P, M = relaToRank(rela, dep_graph, 10, frontend + 1, beta=0.1, rho=0.2, print_trace=False)
    return list(map(lambda x: (mlist[x[0] - 1], x[1]), nodesCR[:prk]))
