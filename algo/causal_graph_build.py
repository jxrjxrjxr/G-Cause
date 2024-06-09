# -*- coding: utf-8 -*-
import numpy as np
from scipy import interpolate


def normalize_by_column(transition_matrix):
    for col_index in range(transition_matrix.shape[1]):
        if np.sum(transition_matrix[:, col_index]) == 0:
            continue
        transition_matrix[:, col_index] = transition_matrix[:, col_index] / np.sum(
            transition_matrix[:, col_index]
        )
    return transition_matrix


def normalize_by_row(transition_matrix):
    for row_index in range(transition_matrix.shape[0]):
        if np.sum(transition_matrix[row_index, :]) == 0:
            continue
        transition_matrix[row_index, :] = transition_matrix[row_index, :] / np.sum(
            transition_matrix[row_index, :]
        )
    return transition_matrix


def get_overlay_count(n_sample, ordered_intervals):
    overlay_counts = np.zeros([n_sample, 1], dtype=np.int)
    for interval in ordered_intervals:
        overlay_counts[interval[1][0] : interval[1][1], 0] += 1
    return overlay_counts


def getData(overCount, t):
    length = overCount.shape[0]
    if t < 0:
        return overCount[length - 1, 0]
    elif t >= length:
        return overCount[0, 0]
    else:
        return overCount[t, 0]
#     return 0 if t < 0 or t >= length else overCount[t, 0]


def curveInter(overCount, listSegment, mask):
    '''
    overCount: np.array, [local_length, 1]
    listSegment: list, i.e. [0, 70, 140, 210, 280, 350, 420, 490, 560, 600]
    mask: np.array, [local_length, ]
    '''
    length = len(listSegment)
    maxSeg, maxSegZero = 0, 0
    for seg in range(length - 1):
        segZero = (mask[listSegment[seg] : listSegment[seg + 1]] == 0).sum()
        if segZero > maxSegZero:
            maxSeg, maxSegZero = seg, segZero
    left, right = listSegment[maxSeg] - 1, listSegment[maxSeg + 1]
    lData, rData = getData(overCount, left), getData(overCount, right)
    f = interpolate.interp1d(np.array([left, right]), np.array([lData, rData]), kind='linear')
    fillSeg = np.arange(listSegment[maxSeg], listSegment[maxSeg + 1])
    overCount[fillSeg, 0] = f(fillSeg)
    return overCount


def curveInter1(overCount, listSegment, mask):
    '''
    overCount: np.array, [local_length, 1]
    listSegment: list, i.e. [0, 70, 140, 210, 280, 350, 420, 490, 560, 600]
    mask: np.array, [local_length, ]
    '''
    length = len(listSegment)
    maxSeg, maxSegZero = 0, 0
    segNum = 1
    segList = []
    while segNum < length - 1:
        st, ed = listSegment[segNum], listSegment[segNum + 1]
        if overCount[st - 1, 0] > overCount[st, 0] and overCount[ed, 0] <= getData(overCount, ed + 1):
            segLen = 1
            flag = True
            while segNum + 1 < length - 1:
                st, ed = listSegment[segNum + 1], listSegment[segNum + 2]
                if overCount[ed, 0] == getData(overCount, ed + 1):
                    segLen += 1
                    segNum += 1
                    continue
                elif overCount[ed, 0] > getData(overCount, ed + 1):
                    flag = False
                break
            if flag:
                segList.append((segNum, segLen))
        segNum += 1
    for segItem in segList:
        st, last = segItem
        left, right = listSegment[st] - 1, listSegment[st + last]
        lData, rData = getData(overCount, left), getData(overCount, right)
        f = interpolate.interp1d(np.array([left, right]), np.array([lData, rData]), kind='linear')
        fillSeg = np.arange(listSegment[st], listSegment[st + last])
        overCount[fillSeg, 0] = f(fillSeg)
    return overCount


def curveInter2(overCount, listSegment, mask):
    '''
    overCount: np.array, [local_length, 1]
    listSegment: list, i.e. [0, 70, 140, 210, 280, 350, 420, 490, 560, 600]
    mask: np.array, [local_length, ]
    '''
    length = len(listSegment)
    segNum = 0
    se = listSegment
    while segNum + 1 < length - 1:
        downNum = segNum
        while downNum + 1 < length - 1 and overCount[se[downNum], 0] >= overCount[se[downNum + 1], 0]:
            downNum += 1
        if downNum + 1 == length - 1:
            break
        endNum = downNum
        while endNum + 1 < length - 1 and overCount[se[endNum], 0] < overCount[se[endNum + 1], 0]:
            endNum += 1
        left, right = listSegment[segNum], listSegment[endNum]
        lData, rData = overCount[left, 0], overCount[right, 0]
        f = interpolate.interp1d(np.array([left, right]), np.array([lData, rData]), kind='linear')
        fillSeg = np.arange(left, right)
        overCount[fillSeg, 0] = f(fillSeg)
        segNum = endNum
    return overCount


def get_ordered_intervals(matrics, significant_thres, list_segment_split):
    array_results_YX, array_results_XY = matrics
    array_results_YX = np.abs(array_results_YX)
    array_results_XY = np.abs(array_results_XY)
    nrows, ncols = array_results_YX.shape
    intervals = []
    pvalues = []
    for i in range(nrows):
        for j in range(i + 1, ncols):
            if (abs(array_results_YX[i, j]) < significant_thres) and (
                array_results_XY[i, j] >= significant_thres
                or array_results_XY[i, j] == -1
            ):
                intervals.append((list_segment_split[i], list_segment_split[j]))
                pvalues.append((array_results_YX[i, j], array_results_XY[i, j]))
    ordered_intervals = list(zip(pvalues, intervals))
    ordered_intervals.sort(key=lambda x: (x[0][0], -x[0][1]))
    # 按照YX从小到大，XY从大到小的顺序排序
    return ordered_intervals


def get_segment_split(n_sample, step):
    n_step = int(n_sample / step)
    list_segment_split = [step * i for i in range(n_step)]
    if n_sample > step * (n_step):
        list_segment_split.append(n_sample)
    else:
        list_segment_split.append(step * n_step)
    return list_segment_split


# Following are not used functions


def get_intervals_over_overlaythres(counts, overlay_thres):
    mask = counts > overlay_thres
    if not np.any(mask):
        return []
    indices = np.where(mask)[0]
    starts = [indices[0]]
    ends = []
    for i in np.where(np.diff(indices, axis=0) > 1)[0]:
        ends.append(indices[i] + 1)
        starts.append(indices[i + 1])
    ends.append(indices[-1] + 1)
    return list(zip(starts, ends))


def get_max_overlay_intervals(counts):
    if np.max(counts) == 0:
        return []
    sample_indices_max = np.where(np.max(counts) == counts)[0]
    starts = [sample_indices_max[0]]
    ends = []
    for i in np.where(np.diff(sample_indices_max, axis=0) > 1)[0]:
        ends.append(sample_indices_max[i] + 1)
        starts.append(sample_indices_max[i + 1])
    ends.append(sample_indices_max[-1] + 1)
    return list(zip(starts, ends))


def get_max_proportion(n_sample, ordered_intervals):
    x = np.zeros([n_sample])
    for interval in ordered_intervals:
        x[interval[1][0] : interval[1][1]] = 1
    return np.sum(x) / (0.0 + n_sample)
