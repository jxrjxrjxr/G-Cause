from algo.m_granger_causal import grangerCausalmp
from algo.m_build_graph import buildGraph
from algo.m_analyze_root import analyzeRootDyCause
from utils.load import fault_range_extract
import pandas as pd


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
    grangerRes = grangerCausalmp(fdf.values, mlist)
    matrix, peak = buildGraph(grangerRes, len(mlist))
    nodesDyCause = analyzeRootDyCause(matrix, fdf.values, frontend + 1)
    return list(map(lambda x: (mlist[x[0] - 1], x[1]), nodesDyCause[:prk]))
