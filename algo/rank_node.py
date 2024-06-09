# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python3.6-0522
#     language: python
#     name: python3.6-0522
# ---

import numpy as np
from collections import defaultdict


def corr(x, y):
    return np.corrcoef(np.concatenate([x.reshape(1, -1), y.reshape(1, -1)], axis=0))


def rankNode(matrix, data, outPath, config):
    nodeCount = defaultdict(float)
    nodeCorr = {}
    rankList = []
    topk_path = config["rw params"]["topk_path"]
    num_node = config["rw params"]["num_sel_node"]
    frontEnd = config["rw params"]["frontend"]
    for u in outPath[:topk_path]:
        for node in u[1][-num_node:]:
            nodeCount[node] = nodeCount[node] + 1
    if frontEnd - 1 in nodeCount:
        nodeCount.pop(frontEnd - 1)
    
    for node in nodeCount:
        nodeCount[node] = nodeCount[node] / (topk_path * num_node)
    for node in nodeCount:
        nodeCorr[node] = abs(corr(data[:, frontEnd - 1], data[:, node])[0, 1])
    for node in nodeCount:
        rankList.append([node, nodeCount[node], nodeCorr[node]])
    r = config["rw params"]["rate"]
    rankList.sort(key=lambda x: x[1] * r + x[2], reverse=True)
    return rankList

def scoreNorm(nodes):
    r = np.array(nodes)
    for col in range(1, 3):
        r[:, col] /= np.linalg.norm(r[:, col])
    lst = r.tolist()
    for i in range(len(lst)):
        lst[i][0] = int(lst[i][0])
    return lst

def printNode(nodes):
    num = len(nodes)
    print("Total :", num, "nodes")
    if len(nodes[0]) == 3:
        print("node  count  corr")
        print("----  -----  ----")
        for i in range(num):
            r = nodes[i]
            print("{:^4d}  {:.3f}  {:.2f}".format(r[0], r[1], r[2]))
    else:
        print("node   sore")
        print("----  -----")
        for i in range(num):
            r = nodes[i]
            print("{:^4d}  {:.3f}".format(r[0], r[1]))
            