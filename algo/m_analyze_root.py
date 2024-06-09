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

from .rand_walk import randWalk
from .rank_node import rankNode
from .ranknode import analyze_root_dycause
 

def filt(outPath):
    def correct(path):
        event = [1, 6, 12, 13, 28, 30, 31, 33]
        xforce = [2, 5, 8, 15, 17, 20, 21, 22]
        other = [3, 7, 9, 10, 19, 24, 27, 32]
        test = [4, 11, 14, 26]
        prod = [16, 18, 23, 25, 29]
        dash = test + prod
        for i in range(len(path) - 1):
            x, y = path[i] + 1, path[i + 1] + 1
            if y in event:
                return False
            if y in xforce + other and x in dash:
                return False
            if y in dash and x in dash:
                return False
        return True
    newPath = list(filter(lambda u: correct(u[1]), outPath))
    print(newPath)
    return newPath
                

def analyzeRootABC(matrix, peak, data, config):
    outPath = randWalk(matrix, peak, config)
    # outPath = filt(outPath)
    nodes = rankNode(matrix, data, outPath, config)
    for u in range(len(nodes)):
        nodes[u][0] += 1
    return nodes, (outPath)


def analyzeRootDyCause(matrix, data, frontend):
    nodes = analyze_root_dycause(
        matrix,
        frontend,  # config["rw params"]["frontend"],
        data,
        mean_method="harmonic",
        max_path_length=1000,
        topk_path=50,
        prob_thres=0.2,
        num_sel_node=3,
        verbose=False
    )
    return nodes
