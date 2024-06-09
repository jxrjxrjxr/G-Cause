import argparse
from utils.load import load_case, case_list, load_sdt
from methods.G_Cause import RCA as gcRCA
from utils.metric import pr_stat, my_acc
from tqdm.auto import tqdm
from time import time
import pprint
from utils.metric import evaluation

parser = argparse.ArgumentParser(prog='G-Cause',
                                 description='Global Root Cause Analysis For Both SDT and HDT')
parser.add_argument('-k', '--prk', dest='prk', type=int, default=-1)
parser.add_argument('-l', '--levels', dest='levels', type=int, default=3)
parser.add_argument('-r', '--range', dest='range', type=int, default=200)
parser.add_argument('-e', '--exp', dest='exp', type=str, default="None")
parser.add_argument('-t', '--thres', dest='thres', type=float, default=0)
parser.add_argument('-f', '--from_start', action='store_true')
parser.add_argument('-u', '--use_history', action='store_true')
args = parser.parse_args()
# case = args.case
prk = args.prk
levels = args.levels
range = args.range
thres = args.thres
exp = args.exp
if exp == "None":
    exp = None
rcdict = {}
runtimes = []
for case in [0]:
    ftp, mlist, df, _ = load_sdt(from_start=True)
    st = time()
    rclist = gcRCA(df, ftp, from_start=args.from_start, use_history=args.use_history, exp=exp, case=case, entry=_[0], prk=prk, new=True, thres=thres, length=range)
    rcdict[f"Case{case}"] = rclist
    runtime = time() - st
    print(f"Case {case}: {runtime}s")
    runtimes.append(runtime)
# print(rcdict)
pprint.pprint(rcdict)
print(runtimes)
pr_df, rc_df = evaluation(rcdict, prk=prk, gt_IBM=_[1])
print(pr_df)
# print(pr_stat(rclist, _[1], k=10))
# print(my_acc(rclist, _[1]))