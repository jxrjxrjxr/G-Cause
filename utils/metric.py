# from utils.load import root_causes, case_list
from utils.load import case_list
import pandas as pd
import numpy as np


def prCal(scoreList, prk, rightOne, weight=False):
    """计算scoreList的prk值

    Params:
        scoreList: list of tuple (node, score)
        prk: the top n nodes to consider
        rightOne: ground truth nodes
    """
    denominator = min(len(rightOne), prk)
    cumsum = 0
    for k in range(min(prk, len(scoreList))):
        cumsum += scoreList[k][1]
    prkSum = 0
    for k in range(min(prk, len(scoreList))):
        if scoreList[k][0] in rightOne:
            score = scoreList[k][1] / cumsum if weight else 1. / denominator
            prkSum += score
    return prkSum


def pr_stat(scoreList, rightOne, k=5, weight=False):
    topk_list = range(1, k + 1)
    prkS = [0] * len(topk_list)
    for j, k in enumerate(topk_list):
        prkS[j] += prCal(scoreList, k, rightOne, weight=weight)
    return prkS


def my_acc(scoreList, rightOne, n=None):
    """Accuracy for Root Cause Analysis with multiple causes.
    Refined from the Acc metric in TBAC paper.
    """
    node_rank = [_[0] for _ in scoreList]
    if n is None:
        n = len(scoreList)
    s = 0.0
    for i in range(len(rightOne)):
        if rightOne[i] in node_rank:
            rank = node_rank.index(rightOne[i]) + 1
            s += (n - max(0, rank - len(rightOne))) / n
        else:
            s += 0
    s /= len(rightOne)
    return s


def evaluation(rcdict: dict, metrics: list = [], prk: int = 10, gt_IBM: list = []) -> tuple:
    rcIdx = []
    if 'Case0' in rcdict.keys():
        case_list = ['0']
    for case in case_list:
        rcIdx.append(f"Case{case} Groundtruth")
        rcIdx.append(f"Case{case} Metric")
        rcIdx.append(f"Case{case} Score")
    rcCol = [f"Rank{i + 1}" for i in range(prk)]
    prCol = [f"PR@{i + 1}" for i in range(prk)] + ["PR@Avg", "RankScore"]
    prIdx = [f"Case{case}" for case in case_list]
    root_cause_df = pd.DataFrame(columns=rcCol, index=rcIdx)
    # print(root_cause_df)
    pr_df = pd.DataFrame(np.zeros((len(prIdx), len(prCol))), columns=prCol, index=prIdx)
    for key in rcdict.keys():
        case = key[4:]
        rclist = rcdict[key]
        if int(case) > 0:
            gt = root_causes[f"case{case}"]
        else:
            gt = gt_IBM
        # print(gt)
        for i in range(min(len(rclist), prk)):
            root_cause_df.loc[f"Case{case} Metric", f"Rank{i + 1}"] = rclist[i][0]
            root_cause_df.loc[f"Case{case} Score", f"Rank{i + 1}"] = round(rclist[i][1], 3)
            if i < len(gt):
                root_cause_df.loc[f"Case{case} Groundtruth", f"Rank{i + 1}"] = gt[i]
        prlist = pr_stat(rclist, gt, k=prk)
        for i in range(prk):
            pr_df.loc[f"Case{case}", f"PR@{i + 1}"] = np.array(prlist[i])
            pr_df.loc[f"Case{case}", "PR@Avg"] = np.mean(prlist)
            pr_df.loc[f"Case{case}", "RankScore"] = my_acc(rclist, gt)
    pr_df.loc["Avg", :] = pr_df.mean(axis=0)
    if len(metrics) == 0:
        return pr_df, root_cause_df
    else:
        return pr_df[metrics], root_cause_df


def comb_eval(pr_dfs: list) -> pd.DataFrame:
    prCol = pr_dfs[0].columns
    prIdx = [f"Case{case}" for case in case_list]
    pr_all_df = pd.DataFrame(np.zeros((len(prIdx), len(prCol))), columns=prCol, index=prIdx)
    for pr_df in pr_dfs:
        pr_all_df += pr_df
    pr_all_df /= len(pr_dfs)
    pr_all_df.loc["Avg", :] = pr_all_df.mean(axis=0)
    return pr_all_df


def comb_band_eval(pr_dfs: list) -> pd.DataFrame:
    prCol = pr_dfs[0].columns
    prIdx = list(map(lambda x: f"Band{x}", range(len(pr_dfs))))
    pr_band_df = pd.DataFrame(np.zeros((len(prIdx), len(prCol))), columns=prCol, index=prIdx)
    for band, pr_df in enumerate(pr_dfs):
        pr_band_df.loc[f"Band{band}", :] = pr_df.loc["Avg", :]
    return pr_band_df
