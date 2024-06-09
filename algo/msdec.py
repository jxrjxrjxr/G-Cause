import pywt
from typing import Union
import numpy as np
import pandas as pd


def wavedec(data: Union[np.ndarray, pd.DataFrame],
            levels: int = 3,
            wvname: str = 'sym5') -> list:
    res: list = []
    if type(data) == pd.DataFrame:
        wavedf = [pd.DataFrame() for i in range(levels + 2)]
        wavedf[0] = data.copy().reset_index(drop=True)
        decres = pywt.wavedec(wavedf[0].values.T, wvname, level=levels)[::-1]
        for i in range(levels + 1):
            wavedf[i + 1] = pd.DataFrame(decres[i].T, columns=wavedf[0].columns)
        res = wavedf
    elif type(data) == np.ndarray:
        wavear = [np.zeros(data.shape) for i in range(levels + 2)]
        wavear[0] = data.copy()
        decres = pywt.wavedec(wavear[0].T, wvname, level=levels)[::-1]
        for i in range(levels + 1):
            wavear[i + 1] = decres[i].T
        res = wavear
    return res
