from statsmodels.tsa.stattools import grangercausalitytests
from tqdm.auto import tqdm
import os
import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InfeasibleTestError
from concurrent.futures import as_completed, ProcessPoolExecutor
from typing import Union, Tuple, List


def task(dat: Union[np.ndarray, pd.DataFrame],
         lag: int,
         test: str,
         r: str,
         c: str) -> Tuple[float, str, str]:
    affinity_mask = set(range(72))
    os.sched_setaffinity(0, affinity_mask)
    min_p_value = 1.0
    try:
        test_result = grangercausalitytests(dat, maxlag=lag, verbose=False)
        p_values = [round(test_result[i + 1][0][test][1], 4) for i in range(lag)]
        min_p_value = np.min(p_values)
    except InfeasibleTestError:
        min_p_value = 1.0
    return min_p_value, r, c


def grangers_causation_matrix_mp(data: pd.DataFrame,
                                 var_x: Union[list, pd.Series],
                                 var_y: Union[list, pd.Series],
                                 test: str = 'ssr_chi2test',
                                 maxlag: int = 5,
                                 verbose: bool = False,
                                 show: bool = False,
                                 thres: float = 0.05,
                                 origin: Union[None, pd.DataFrame] = None,
                                 max_workers: int = 40,
                                 theta: float = 0.05):
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table
    are the P-Values. P-Values lesser than the significance level (0.05), implies
    the Null Hypothesis that the coefficients of the corresponding past values is
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    executor = ProcessPoolExecutor(max_workers=max_workers)
    futures = []
    rdf = pd.DataFrame(np.zeros((len(var_y), len(var_x))), columns=var_x, index=var_y)
    for c in rdf.columns:
        for r in rdf.index:
            if verbose:
                print(f'Y = {r}, X = {c}', flush=True)

            def judge(dat):
                if dat.max() - dat.min() < thres:
                    return True
                if (dat[dat == dat.mode()[0]]).count() > len(dat) * 0.99:
                    return True
                return False

            if judge(data[r]) or judge(data[c]) \
               or (origin is not None and judge(origin[r])) \
               or (origin is not None and judge(origin[c])):
                rdf.loc[r, c] = 1.0
                continue
            if verbose:
                print(f'Y = {r}, X = {c}', flush=True)
            futures.append(executor.submit(task, data[[r, c]], maxlag, test, r, c))
    threadNum = len(futures)
    if show:
        pbar = tqdm(total=threadNum, ascii=True)
    for future in as_completed(futures):
        p, r, c = future.result()
        rdf.loc[r, c] = p
        if show:
            pbar.update(1)
    if show:
        pbar.close()
    executor.shutdown(wait=True)
    del futures
    rdf.columns = [var + '_x' for var in var_x]
    rdf.index = [var + '_y' for var in var_y]
    return rdf < theta


def gen_graph(data: pd.DataFrame, ci: str = "GC", **kwargs) -> pd.DataFrame:
    res = pd.DataFrame()
    if ci == "GC":
        mlist = data.columns
        res = grangers_causation_matrix_mp(data, mlist, mlist, **kwargs)
        res.replace({True: 1, False: 0}, inplace=True)
    else:
        pass
    return res


def gen_graphs(data: List[pd.DataFrame], ci: str = "GC", **kwargs) -> List[pd.DataFrame]:
    res: list = []
    for item in data:
        res.append(gen_graph(item, ci=ci, **kwargs))
    return res


def merge_graph(graphs: List[pd.DataFrame]) -> pd.DataFrame:
    length = len(graphs)
    res = graphs[0].copy()
    for i in range(1, length):
        res += graphs[i]
    return res


def his_merge_by_freq(graphs: List[List[pd.DataFrame]]) -> List[pd.DataFrame]:
    def get_graphs(freq: int):
        tmp = graphs[0][freq]
        for i in range(1, len(graphs)):
            tmp += graphs[i][freq]
        tmp /= len(graphs)
        tmp /= tmp.max().max()
        return tmp
    return list(map(lambda x: get_graphs(x), range(len(graphs[0]))))


def abnormal_causality_extraction(fgraphs, rgraphs, latter=False):
    gnum = len(fgraphs)
    abgraphs = list(map(lambda x: x.copy(), fgraphs))
    if latter:
        for i in range(gnum):
            abgraphs[i] += rgraphs[i]
            abgraphs[i][abgraphs[i] < 0] = 0
    else:
        for i in range(gnum):
            abgraphs[i] -= rgraphs[i]
            abgraphs[i][abgraphs[i] < 0] = 0
    return abgraphs
