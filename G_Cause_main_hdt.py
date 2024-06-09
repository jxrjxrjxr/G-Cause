import argparse
from utils.load import load_case, case_list
from methods.G_Cause import RCA as gcRCA
from utils.metric import pr_stat, my_acc
from tqdm.auto import tqdm
from time import time

parser = argparse.ArgumentParser(prog='G-Cause',
                                 description='Global Root Cause Analysis For Both SDT and HDT')
parser.add_argument('-c', '--case', dest='case', type=int, default=8)
parser.add_argument('-k', '--prk', dest='prk', type=int, default=-1)
parser.add_argument('-l', '--levels', dest='levels', type=int, default=3)
parser.add_argument('-e', '--exp', dest='exp', type=str, default="cur")
parser.add_argument('-t', '--thres', dest='thres', type=float, default=0)
args = parser.parse_args()
# case = args.case
prk = args.prk
levels = args.levels
thres = args.thres
exp = args.exp
rcdict = {}
runtimes = []
for case in [7]:
    ftp, mlist, df, _ = load_case(case=case)
    st = time()
    rclist = gcRCA(df, ftp, from_start=True, use_history=True, exp="0211", case=case, entry=_[0], prk=prk, new=True)
    rcdict[f"Case{case}"] = rclist
    runtime = time() - st
    print(f"Case {case}: {runtime}s")
    runtimes.append(runtime)
print(rcdict)
print(runtimes)
# print(pr_stat(rclist, _[1], k=10))
# print(my_acc(rclist, _[1]))