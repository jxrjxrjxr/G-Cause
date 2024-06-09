import pandas as pd
from typing import Union
from utils.exp import create_exp
import sys
from utils.load import fault_range_extract, historical_range_extract_list
from algo.msdec import wavedec
from algo.causality import gen_graphs, merge_graph, his_merge_by_freq, abnormal_causality_extraction
from algo.graph import drop_iso, pagerank_sknetwork, thres_cut
# from algo.graph import gen_nx_graph, pagerank
import numpy as np


def RCA(
    df: pd.DataFrame,
    ftp: int,
    entry: str = "load1",
    prk: int = 10,
    from_start: bool = True,
    use_history: bool = False,
    exp: Union[None, str] = None,
    case: int = 99,
    levels: int = 3,
    thres: float = 0,
    new: bool = False,
    length: int = 200
) -> list:
    if exp:
        create_exp(exp)
    mlist = list(df.columns)
    abgraph: Union[None, pd.DataFrame] = None
    if from_start:
        fdf = fault_range_extract(df, ftp, ranges=[length, length])
        fdecres = wavedec(fdf, levels)
        fault_freq_graphs = gen_graphs(fdecres)
        fgraph = merge_graph(fault_freq_graphs)
        if exp:
            for i, graph in enumerate(fault_freq_graphs):
                graph.to_csv(f"{sys.path[0]}/results/{exp}/fgraph_{i}.csv")
            fgraph.to_csv(f"{sys.path[0]}/results/{exp}/fgraph_all.csv")
        if use_history:
            rdfs, rtp = historical_range_extract_list(df, ftp, ranges=[length, length], times=30)
            rfreq_single_graphs = list(map(lambda x: gen_graphs(wavedec(x, levels)), rdfs))
            rfreq_graphs = his_merge_by_freq(rfreq_single_graphs)
            abfreq_graphs = abnormal_causality_extraction(fault_freq_graphs, rfreq_graphs, latter=True)
            abgraph = merge_graph(abfreq_graphs)
            if exp:
                for i, graphs in enumerate(rfreq_single_graphs):
                    for j, graph in enumerate(graphs):
                        graph.to_csv(f"{sys.path[0]}/results/{exp}/rgraph_single_freq{j}_time{i}.csv")
                for i, graph in enumerate(rfreq_graphs):
                    graph.to_csv(f"{sys.path[0]}/results/{exp}/rgraph_freq{i}.csv")
                for i, graph in enumerate(abfreq_graphs):
                    graph.to_csv(f"{sys.path[0]}/results/{exp}/abgraph_freq{i}.csv")
                abgraph.to_csv(f"{sys.path[0]}/results/{exp}/abgraph_all.csv")
                rtp_numpy = np.array(rtp)
                with open(f"{sys.path[0]}/results/{exp}/rtp.npy", 'wb') as f:
                    np.save(f, rtp_numpy)
        else:
            abgraph = fgraph
    else:
        if use_history:
            if new:
                # abgraph = pd.read_csv(f"{sys.path[0]}/results/{exp}/abgraph_all.csv", index_col=0)
                fault_freq_graphs = []
                for i in range(levels + 2):
                    fault_freq_graphs.append(pd.read_csv(f"{sys.path[0]}/results/{exp}/fgraph_{i}.csv", index_col=0))
                rfreq_graphs = []
                for i in range(levels + 2):
                    rfreq_graphs.append(pd.read_csv(f"{sys.path[0]}/results/{exp}/rgraph_freq{i}.csv", index_col=0))
                abfreq_graphs = abnormal_causality_extraction(fault_freq_graphs, rfreq_graphs, latter=True)
                abgraph = merge_graph(abfreq_graphs)
            else:
                abgraph = pd.read_csv(f"{sys.path[0]}/results/{exp}/Case{case}/minus_sum_df.csv", index_col=0)
        else:
            # print("arrived")
            if new:
                abgraph = pd.read_csv(f"{sys.path[0]}/results/{exp}/fgraph_all.csv", index_col=0)
            else:
                abgraph = pd.read_csv(f"{sys.path[0]}/results/{exp}/Case{case}/fault_causal_df.csv", index_col=0)

    abgraph = thres_cut(abgraph, thres=thres)
    adj, nodes = drop_iso(abgraph.values.T, mlist)
    # G = gen_nx_graph(adj.T, nodes)
    # return pagerank(G, entry=entry, k=prk)
    return pagerank_sknetwork(adj.T, nodes, entry=entry, k=prk)


def get_graph(case: int, band: int, exp: str, use_history: bool) -> pd.DataFrame:
    path = f"{sys.path[0]}/results/{exp}/Case{case}"
    if band == 5:
        if use_history:
            return pd.read_csv(f"{path}/minus_sum_df.csv", index_col=0)
        else:
            return pd.read_csv(f"{path}/fault_causal_df.csv", index_col=0)
    else:
        fgraph = pd.read_csv(f"{path}/fault_range_{band}_df.csv", index_col=0)
        if use_history:
            hgraph = pd.read_csv(f"{path}/historical_30_sum_pf_{band}_df.csv", index_col=0)
            fgraph -= hgraph
            fgraph[fgraph < 0] = 0
        return fgraph
