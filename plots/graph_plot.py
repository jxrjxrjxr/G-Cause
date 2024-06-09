import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Union


def adjplot(data: pd.DataFrame,
            save: Union[None, str] = None,
            figx: int = 12,
            figy: int = 10,
            dpi: int = 200,
            square: bool = True,
            lw: float = 1.5) -> None:
    plt.figure(figsize=(figx, figy), dpi=dpi)
    sns.heatmap(data, linewidths=lw, square=square, cmap=plt.get_cmap('YlGnBu'))
    if save is None:
        plt.show()
    else:
        plt.savefig(save)
