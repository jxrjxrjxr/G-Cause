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

from scipy import stats
import numpy as np


def printQ(queue):
    copyQueue = queue
    for item in copyQueue:
        print(tuple(zip(*item))[0])


def bfs(matrix, peak, config):
    path_list = set()
    entry_point = config["rw params"]["frontend"]
    queue = [[(entry_point - 1, 999)]]
    max_path_length = config["rw params"]["max_path_length"]
    varNum = config["Data params"]['varNum']
    sortPeak = config["rw params"]["sortPeak"]
    reach_end = True
#     import pdb; pdb.set_trace()
    while len(queue) > 0:
        # Limit output path list size to 10000 in case of infinite bfs
        if len(path_list) > 10000:
            break
        # Limit bfs queue size to 10000 in case of infinite bfs, and flush paths to path_list
        if len(queue) > 10000:
            while len(queue) > 0:
                path_list.add(tuple(zip(*queue.pop(0)))[0])
            break
        path, peakList = list(zip(*queue.pop(0)))
        if np.sum(matrix[:, path[-1]]) == 0:
            # if there is no previous node, the path ends and we add it to output.
            path_list.add(tuple(path))
        else:
            # if there is at least one previous node
            if max_path_length is not None and len(path) >= max_path_length:
                # if path length exceeds limit
                path_list.add(tuple(path))
            else:
                # Try extending the path with every possible node
                curPeak = peakList[-1]
                endFlag = True
                for prev_node in range(varNum):
                    if matrix[prev_node, path[-1]] > 0.0 and (prev_node not in path):
                        prePeak = peak["%s->%s" % (prev_node, path[-1])]
                        if sortPeak and prePeak != -1 and curPeak < prePeak:
                            continue
                        # extend the path
                        endFlag = False
                        new_path = path + (prev_node, )
                        new_peak = peakList + ((curPeak,) if prePeak == -1 else (prePeak,))
                        queue.append(zip(new_path, new_peak))
                if endFlag:
                    path_list.add(tuple(path))
    return path_list


def randWalk(matrix, peak, config):
    pathList = bfs(matrix, peak, config)
    pathProb = []
    for path in pathList:
        p, end = [], path[0]
        for start in path[1:]:
            p.append(matrix[start, end])
            end = start
        if len(p) == 0:
            pathProb.append(0)
        else:
            p = [_ for _ in p if _ != 1]
            pathProb.append(stats.hmean(p))
    out = [item for item in zip(pathProb, pathList)]
    out.sort(key=lambda x: x[0], reverse=True)
    return out
