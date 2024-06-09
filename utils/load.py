import pandas as pd
import json
from typing import Tuple, List
import numpy as np
import sys
from sklearn.preprocessing import MinMaxScaler, StandardScaler

rc_subfixes = ["jxr", "pyc", "btz"]
rc_subfix = rc_subfixes[0]

# fault_times = dict()
# with open(f"{sys.path[0]}/data/case8-25/fault_times.json", "r") as f:
#     fault_times = json.load(f)

# entry_points = dict()
# with open(f"{sys.path[0]}/data/case8-25/entry_points-{rc_subfix}.json", "r") as f:
#     entry_points = json.load(f)

# root_causes = dict()
# with open(f"{sys.path[0]}/data/case8-25/root_causes-{rc_subfix}.json", "r") as f:
#     root_causes = json.load(f)

case_list = [8, 9, 10, 11, 12, 13, 14, 15, 17, 21, 24, 25]
exp_list = ["0107", "0108-1", "0108-2", "0109-1", "0110-1", "0111-1", "0111-2", "0111-3", "0111-4", "0111-5"]

# runtimes = [1167.27, 985.81, 1193.76, 1130.70, 1290.26, 1159.87, 1089.02, 1127.53, 1141.66, 881.25, 844.76, 1203.29]
runtimes = [365.15414452552795, 306.04069805145264, 400.7954351902008, 370.05400562286377, 388.61150097846985, 357.5771312713623, 410.5610055923462, 333.16557598114014, 403.65099573135376, 249.73692393302917, 289.19305086135864, 453.4888184070587]
runtimes_IBM = [333.78]

def load_sdt(from_start: bool = True, diff: bool = True) -> Tuple[int, pd.Series, pd.DataFrame, tuple]:
    if from_start:
        df = pd.read_excel(f'{sys.path[0]}/data/sdt_micro_service/rawdata.t.xlsx', sheet_name='load')
        df.index = pd.date_range(start='2016-08-19 14:16:50', periods=7199, freq='1S')
        ori_df = df.copy()
        mlist = df.columns

        # filldf = ori_df.copy()
        # filldf = filldf.replace(0, np.nan)
        # filldf = filldf.interpolate(limit_direction="both")
        # ori_df = filldf.copy()
        mmscaler = MinMaxScaler()

        ori_diff_df = ori_df.diff().interpolate(limit_direction='both')
        ori_mlist_df = ori_df[mlist]
        ori_diff_mlist_df = ori_diff_df[mlist]

        ori_mlist_norm_df = pd.DataFrame(mmscaler.fit_transform(ori_mlist_df),
                                    columns=mlist,
                                    index=ori_mlist_df.index)
        ori_diff_mlist_norm_df = pd.DataFrame(mmscaler.fit_transform(ori_diff_mlist_df),
                                        columns=mlist,
                                        index=ori_diff_mlist_df.index)
        # ori_diff_mlist_norm_df.to_csv(f'{sys.path[0]}/data/case8-25/caseIBM_ori_diff_mlist_norm_df.csv')
        # ori_mlist_norm_df.to_csv(f'{sys.path[0]}/data/case8-25/caseIBM_ori_mlist_norm_df.csv')
        if diff:
            df = ori_diff_mlist_norm_df.copy()
        else:
            df = ori_mlist_norm_df.copy()
    else:
        if diff:
            df = pd.read_csv(f'{sys.path[0]}/data/case8-25/caseIBM_ori_diff_mlist_norm_df.csv', index_col=0)
        else:
            df = pd.read_csv(f'{sys.path[0]}/data/case8-25/caseIBM_ori_mlist_norm_df.csv', index_col=0)
        df.index = pd.to_datetime(df.index)
        mlist = df.columns
    fault_time_str = '2016-08-19 15:34:23'
    fault_time = pd.Timestamp(fault_time_str)
    ftp = np.abs(df.index - fault_time).argmin()
    entry = mlist[13]
    # rcnums = [5, 23, 27, 29, 30]
    rcnums = [5, 27, 29, 30]
    root_causes = list(map(lambda x: mlist[x], rcnums))
    return ftp, mlist, df, (entry, root_causes)


def load_case(case: int) -> Tuple[int, pd.Series, pd.DataFrame, tuple]:
    # if case not in case_list:
    #     raise Exception(f'Case {case} currently not supported!')
    fault_time_str = fault_times[f"case{case}"]
    fault_time = pd.Timestamp(fault_time_str, tz='Asia/Shanghai')
    df = pd.read_csv(f'{sys.path[0]}/data/case8-25/case{case}_ori_diff_mlist_norm_df.csv', index_col=0)
    df.index = pd.to_datetime(df.index)
    mlist = df.columns
    ftp = np.abs(df.index - fault_time).argmin()
    return ftp, mlist, df, (entry_points[f"case{case}"], root_causes[f"case{case}"])


def fault_range_extract(df: pd.DataFrame, ftp: int, ranges: list = [200, 200], zero: bool = True) -> pd.DataFrame:
    fr = [ftp - ranges[0], ftp + ranges[1]]
    fdf = df.iloc[fr[0]: fr[1], :].copy()
    if zero:
        fdf_zm = fdf - fdf.mean()
    else:
        fdf_zm = fdf
    return fdf_zm


def historical_range_extract(df: pd.DataFrame, ftp: int, ranges: list = [200, 200]) -> pd.DataFrame:
    length = ranges[0] + ranges[1]
    rdf = df.iloc[:ftp - length, :].copy()
    rdf_zm = rdf - rdf.mean()
    return rdf_zm


def historical_range_extract_list(df: pd.DataFrame,
                                  ftp: int,
                                  ranges: list = [200, 200],
                                  times: int = 30) -> Tuple[List[pd.DataFrame], List[int]]:
    length = ranges[0] + ranges[1]
    hislist: List[pd.DataFrame] = []
    histp: List[int] = []
    st, ed = 0, ftp - length
    _sum = df.sum().sum()
    while(st + 1 < ed):
        mid = (st + ed) // 2
        if df.iloc[mid:, :].sum().sum() > _sum * 0.90:
            st = mid
        else:
            ed = mid
    st = 3800
    for t in range(times):
        # randtp = np.random.randint(st + ranges[0], ftp - ranges[1])
        randtp = np.random.randint(ftp + length, df.shape[0] - ranges[1])
        randfr = [randtp - ranges[0], randtp + ranges[1]]
        randdf = df.iloc[randfr[0]: randfr[1], :].copy()
        randdf_zm = randdf - randdf.mean()
        hislist.append(randdf_zm)
        histp.append(randtp)
    return hislist, histp


def aggregate(a, n=3):
    """From MicroCU
    """
    cumsum = np.cumsum(a, dtype=float)
    ret = []
    for i in range(-1, len(a) - n, n):
        low_index = i if i >= 0 else 0
        ret.append(cumsum[low_index + n] - cumsum[low_index])
    return ret
