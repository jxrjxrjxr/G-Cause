from collections import defaultdict
import numpy as np
from .causal_graph_build import normalize_by_column, get_overlay_count, get_segment_split
from math import pow, exp, pi, sqrt


def getPeak(overCount, listSegment, x_i, y_i):
    length = len(listSegment)
    se, ov = listSegment, overCount
    peakList = []
    peakW = []
    gap = se[1] - se[0]
    segNum = 0
    while segNum + 1 < length - 1:
        if (segNum == 0 or ov[se[segNum]] > ov[se[segNum - 1]]) and ov[se[segNum]] >= ov[se[segNum + 1]]:
            startNum = segNum
            while segNum + 1 < length - 1 and ov[se[segNum]] == ov[se[segNum + 1]]:
                segNum += 1
            if segNum + 1 < length - 1 and ov[se[segNum]] < ov[se[segNum + 1]]:
                continue
            peakList.append((se[segNum] + se[startNum] + gap) // 2)
            peakW.append(segNum - startNum + 1)
        segNum += 1
    if segNum + 1 == length - 1 and ov[se[segNum]] > ov[se[segNum - 1]]:
        peakList.append(se[segNum] + (gap // 2))
        peakW.append(1)
    # assert(-1)
    # return sum(mul(peakList, peakW)) // (len(peakList) * sum(peakW)) if len(peakList) > 0 else -1
    return sum(mul(peakList, peakW)) // (sum(peakW)) if len(peakList) > 0 else -1


def mul(list1, list2):
    return list(map(lambda item : item[0] * item[1], zip(list1, list2)))


def genGraph(matrix, peak, config):
    class Gauss:
        def __init__(self, mu, sigma):
            self.mu = mu
            self.sigma = sigma

        def v(self, x):
            fenzi = exp(-pow(x - self.mu, 2) / (2 * self.sigma * self.sigma))
            fenmu = self.sigma * sqrt(2 * pi)
            return fenzi / fenmu
    # GaussFun = Gauss(0, 1)
    bT = config['Data params']['before_length']
    aT = config['Data params']['after_length']
    length = bT + aT
    n = matrix.shape[0]
    mtx = np.zeros((length, n, n))

    # res = [
    #     1*33+23,
    #     5*33+32,
    #     8*33+30,
    #     27*33+5,
    #     27*33+15,
    #     29*33+23,
    #     29*33+30,
    #     29*33+32
    # ],

    # ctrl = [
    #     100,
    #     200,
    #     80,
    #     150,
    #     95,
    #     100,
    #     50,
    #     70
    # ]
    for i in range(n):
        for j in range(n):
            # left = peak["%s->%s" % (i, j)] * 2 / length - 1 - 1
            # left = peak["%s->%s" % (i, j)] * 2 / length - 1 - 1
            def listfind(a, b):
                for i in range(len(a)):
                    if a[i] == b:
                        return i
                return -1

            # loc = listfind(res, i*n+j)
            scale = 50
            # if loc == -1: scale = 50
            # else: scale = ctrl[loc]
            GaussFun1 = Gauss(peak["%s->%s" % (i, j)], scale)
            for k in range(length):
                mtx[k, i, j] = GaussFun1.v(k)
                # mtx[k, i, j] = GaussFun.v(left + k / 400)
            _range = np.max(mtx[:, i, j]) - np.min(mtx[:, i, j])
            mtx[:, i, j] -= np.min(mtx[:, i, j])
            mtx[:, i, j] /= _range
            mtx[:, i, j] *= matrix[i, j]
    return mtx


def buildGraph(grangerRes, varNum):
    bef, aft = 200, 200
    local_length = bef + aft
    histSum = defaultdict(int)
    peak = defaultdict(int)
    edge, edgeWeight = [], dict()
    matrix = np.zeros([varNum, varNum])
    list_segment_split = get_segment_split(bef + aft, 70)
    for x_i in range(varNum):
        for y_i in range(varNum):
            if x_i == y_i:
                continue
            key = "{0}->{1}".format(x_i, y_i)
            intervals = grangerRes[key]["intervals"]
            overlay_counts = get_overlay_count(local_length, intervals)
            histSum[key] = sum(overlay_counts)
            peak[key] = getPeak(overlay_counts, list_segment_split, x_i, y_i)
    for x_i in range(varNum):
        bar_data = []
        for y_i in range(varNum):
            key = "{0}->{1}".format(x_i, y_i)
            bar_data.append(histSum[key])
        bar_data_thres = np.max(bar_data) * 0.5
        for y_i in range(varNum):
            if bar_data[y_i] >= bar_data_thres:
                edge.append((x_i, y_i))
                edgeWeight[(x_i, y_i)] = bar_data[y_i]
    for key, val in edgeWeight.items():
        x, y = key
        matrix[x, y] = val
    matrix = normalize_by_column(matrix)
    return matrix, peak


def buildGraphMul(grangerRes, config):
    varNum = config["Data params"]["varNum"]
    bef, aft = config["Data params"]["before_length"], config["Data params"]["after_length"]
    local_length = bef + aft
    histSum = defaultdict(int)
    peak = defaultdict(int)
    edge, edgeWeight = [], dict()
    matrix = np.zeros([varNum, varNum])
    list_segment_split = get_segment_split(bef + aft, config["Granger params"]["step"])
    for x_i in range(varNum):
        for y_i in range(varNum):
            if x_i == y_i:
                continue
            key = "{0}->{1}".format(x_i, y_i)
            intervals = grangerRes[key]["intervals"]
            overlay_counts = get_overlay_count(local_length, intervals)
            histSum[key] = sum(overlay_counts)
            peak[key] = getPeak(overlay_counts, list_segment_split, x_i, y_i)
    for x_i in range(varNum):
        bar_data = []
        for y_i in range(varNum):
            key = "{0}->{1}".format(x_i, y_i)
            bar_data.append(histSum[key])
        bar_data_thres = np.max(bar_data) * config["Granger params"]["auto_threshold_ratio"]
        for y_i in range(varNum):
            if bar_data[y_i] >= bar_data_thres:
                edge.append((x_i, y_i))
                edgeWeight[(x_i, y_i)] = bar_data[y_i]
    for key, val in edgeWeight.items():
        x, y = key
        matrix[x, y] = val
    matrix = normalize_by_column(matrix)
    return matrix, peak
