from typing import Tuple
import numpy as np
import networkx as nx
from sknetwork.ranking import PageRank
from scipy.sparse import csr_matrix
import pandas as pd


def thres_cut(data: pd.DataFrame, thres: float = 0) -> pd.DataFrame:
    if thres < 0:
        levels = 3
        thres = levels + 2
        M = data.shape[0]
        for t in range((levels + 2) * 2, 0, -1):
            if (data > t / 2).sum().sum() > 0.03 * M * M:
                thres = t / 2
                break
    res = data.copy()
    res -= thres
    res[res < 0] = 0
    return res


def MSE(x: pd.DataFrame, y: pd.DataFrame) -> float:
    return ((x - y) ** 2).mean().mean()


def drop_iso(data: np.ndarray, mlist: list) -> Tuple[np.ndarray, list]:
    """data: numpy, columns: x, index: y
    """
    niso: list = []
    idx: list = []

    for i, mname in enumerate(mlist):
        if data.sum(0)[i] + data.sum(1)[i] > 0:
            idx.append(i)
            niso.append(mname)
    return data[np.ix_(idx, idx)], niso


def pagerank_sknetwork(adj: np.ndarray, nodes: list, entry: str = "load1", k: int = -1) -> list:
    adj_sp = csr_matrix(adj)
    pagerank = PageRank()
    if entry in nodes:
        seeds = {nodes.index(entry): 1}
        scores = pagerank.fit_predict(adj_sp, seeds)
    else:
        scores = pagerank.fit_predict(adj_sp)
    scores[scores == 0] = 0.001
    res = list(zip(nodes, scores))
    res.sort(key=lambda x: x[1], reverse=True)
    if k == -1:
        k = len(nodes)
    rclist: list = []
    rcsum = 0
    for u in res:
        if u[0] == entry:
            continue
        if len(rclist) == k or u[1] == 0.001:
            break
        rclist.append(list(u))
        rcsum += u[1]
    for i in range(len(rclist)):
        rclist[i][1] /= rcsum
    return rclist


def gen_nx_graph(adj: np.ndarray, nodes: list) -> nx.classes.graph.Graph:
    G = nx.from_numpy_array(adj)
    G.nodes = nodes
    return G


def pagerank(G: nx.classes.graph.Graph, entry: str = "load1", k: int = -1) -> list:
    nodes = G.nodes
    pr = nx.pagerank(G, personalization={nodes.index(entry): 1})
    scores: list = []
    for i in range(len(nodes)):
        scores.append(pr[i])
    scores[scores == 0] = 0.001
    res = list(zip(nodes, scores))
    res.sort(key=lambda x: x[1], reverse=True)
    rclist: list = []
    rcsum = 0
    if k == -1:
        k = len(nodes)
    for u in res:
        if u[0] == entry:
            continue
        if len(rclist) == k or u[1] == 0.001:
            break
        rclist.append(list(u))
        rcsum += u[1]
    for i in range(len(rclist)):
        rclist[i][1] /= rcsum
    return rclist
