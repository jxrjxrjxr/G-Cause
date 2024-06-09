# **G-Cause**

## **Descriptions**

This is the code repository for "G-Cause: Parameter-free Global Diagnosis for Hyperscale Web Service Infrastructures".

## **Requirements**

Run the following command to prepare the runnning environment.

```shell
pip install -r requirements.txt 
```

It only requires some basic Python libraries, and requires no GPUs.

### **Usage**

#### **SDT**

To validate the effectiveness of the SDT proposed in our paper, the fastest method is to use the pre-generated causal graphs that we have stored. To do this, please use the following command:
```shell
python G_Cause_main_sdt.py -k 5 -r 400 -e sdt_exp_save
```

where the parameters are defined as follows:
- k: provides the top k recommended root causes.
- r: duration of the abnormal interval.
- e: the folder location of the pre-stored causal graph.

If you wish to use the historical and post-abnormal causal graphs mentioned in Section III D, please add the `-u` parameter, as follows:
```shell
python G_Cause_main_sdt.py -k 5 -r 400 -e sdt_exp_save -u
```

If you want to train from scratch without using the pre-generated causal graphs, use the `-f` parameter, as follows:
```shell
python G_Cause_main_sdt.py -k 5 -r 400 -e sdt_exp_save -fu
```

Note: This code is intended to be used in conjunction with SDT proposed in our paper. Please refer to the paper for more information.

#### **HDT**

Currently we can not provide data due to the confidentiality of the cooperation with the top-tier IT company. However, users can conduct experiments on their own data. To make our code easier to use, we have modularized the functions and placed them under each folder. All the functions related to our algorithms are placed in the `./algo` folder. `./methods` folder provides example usages about how to use modularized functions. Our main function locates at `./methods/G_Cause.py`

To Use your own data, you need to pass two data inputs:

- df: `pd.DataFrame` type data, each column is a host-level metric, and each row is a time point (with/without timestamp).
    
- ftp: Fault time point, `int` type, representing the time point that anomaly happens.
    

By running `gcRCA(df, ftp)`, you will get a list of culprits of root causes like:

```
[('memSwapUsed', 0.601),
 ('ssEstab', 0.257),
 ...]
```

These are the root causes recommended by G-Cause.