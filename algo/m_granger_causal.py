from .Granger_all_code import loop_granger
from .causal_graph_build import get_ordered_intervals, get_segment_split
from collections import defaultdict
# import os
# import json
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from multiprocessing import Manager
from tqdm import tqdm


def granger_process(
        shared_params_dict,
        specific_params,
        shared_result_dict):
    try:
        # with open(common_params_filename, 'rb') as f:
        #     common_params = pickle.load(f)
        common_params = shared_params_dict
        ret = loop_granger(
            common_params['local_data'],
            common_params['data_head'],
            common_params['dir_output'],
            common_params['data_head'][specific_params['x_i']],
            common_params['data_head'][specific_params['y_i']],
            common_params['significant_thres'],
            common_params['method'],
            common_params['trip'],
            common_params['lag'],
            common_params['step'],
            common_params['simu_real'],
            common_params['max_segment_len'],
            common_params['min_segment_len'],
            verbose=False,
            return_result=True,
        )
    except Exception as e:
        print("Exception occurred at {} -> {}!".format(
            specific_params['x_i'], specific_params['y_i']), e)
        # logging.error("Exception occurred at {} -> {}!".format(
        #     specific_params['x_i'], specific_params['y_i']))
        ret = (None, None, None, None, None)
    shared_result_dict['{}->{}'.format(specific_params['x_i'], specific_params['y_i'])] = ret
    return ret


def grangerCausalmp(data, data_head):
    # print("In granger causal mp")
    varNum = data.shape[1]
    useNewMask = True
    # granger = config["Granger params"]
    bef, aft = 200, 200
    significant_thres = 0.1
    step = 70
    max_segment_len = bef + aft
    min_segment_len = step
    list_segment_split = get_segment_split(bef + aft, step)
    local_results = defaultdict(dict)
    if useNewMask:
        threadNum = [varNum * (varNum - 1)]
        # threadRes = [0 for i in range(threadNum[0])]
        pbar = tqdm(total=threadNum[0], ascii=True)
        common_params = {
            'local_data': data,
            'data_head': data_head,
            'dir_output': "/workspace/code/MultiFreqGranger/trash",
            'significant_thres': significant_thres,
            'method': "fast_version_3",
            'trip': -1,
            'lag': 5,
            'step': step,
            'simu_real': "simu",
            'max_segment_len': max_segment_len,
            'min_segment_len': min_segment_len,
            'verbose': False,
            'return_result': True
        }
        manager = Manager()
        shared_params_dict = manager.dict()
        shared_result_dict = manager.dict()
        for key, value in common_params.items():
            shared_params_dict[key] = value
        executor = ProcessPoolExecutor(max_workers=70)
        futures = []
        for x_i in range(varNum):
            for y_i in range(varNum):
                if x_i == y_i:
                    continue
                futures.append(executor.submit(
                    granger_process,
                    shared_params_dict,
                    {'x_i': x_i, 'y_i': y_i},
                    shared_result_dict)
                )
        for future in as_completed(futures):
            pbar.update(1)
        pbar.close()
        executor.shutdown(wait=True)
        for x_i in range(varNum):
            for y_i in range(varNum):
                if x_i == y_i:
                    continue
                # (
                #     total_time,
                #     time_granger,
                #     time_adf,
                #     array_results_YX,
                #     array_results_XY,
                # ) = futures[i].result()
                (
                    total_time,
                    time_granger,
                    time_adf,
                    array_results_YX,
                    array_results_XY
                ) = shared_result_dict['{}->{}'.format(x_i, y_i)]

                matrics = [array_results_YX, array_results_XY]
                ordered_intervals = get_ordered_intervals(
                    matrics, significant_thres, list_segment_split
                )
                local_results["%s->%s" %
                              (x_i, y_i)]["intervals"] = ordered_intervals
    else:
        local_results = None
    return local_results


def grangerCausal(data, data_head, config):
    varNum = data.shape[1]
    useNewMask = config["Data params"]["useNewMask"]
    dirOutput = config["Debug params"]["dirOutput"]
    granger = config["Granger params"]
    bef, aft = config["Data params"]["before_length"], config["Data params"]["after_length"]
    significant_thres = granger["significant_thres"]
    step = granger["step"]
    max_segment_len = bef + aft
    min_segment_len = step
    list_segment_split = get_segment_split(bef + aft, step)
    local_results = defaultdict(dict)
    if useNewMask:
        for x_i in range(varNum):
            for y_i in range(varNum):
                if x_i == y_i:
                    continue
                (total_time, time_granger, time_adf, array_results_YX,
                 array_results_XY) = loop_granger(
                    data,  # np.array L*N
                    data_head,  # N
                    dirOutput,
                    data_head[x_i],
                    data_head[y_i],
                    significant_thres,
                    granger["method"],
                    granger["trip"],
                    granger["lag"],
                    granger["step"],
                    granger["simu_real"],
                    max_segment_len,
                    min_segment_len,
                    verbose=False,
                    return_result=True
                )
                matrics = [array_results_YX, array_results_XY]
                ordered_intervals = get_ordered_intervals(
                    matrics, significant_thres, list_segment_split)
                # number = number of interval pairs i to j
                # shape of ordered_intervals : number(list)*2(tuple)*2(tuple)
                # every pair includes ((YX, XY), (start, end))
                # sort by (YX, -XY)
                # shape like [((1, 2), (7, 8)), ((3, 4), (9, 10)), ((5, 6), (11, 12))]
                local_results['%s->%s' % (x_i, y_i)]['intervals'] = ordered_intervals
    else:
        # with open(os.path.join(lastDirOut, "local.json"), "r") as localF:
        #     local_results = json.load(localF)
        pass
    return local_results
