import os
import sys


def create_exp(exp: str) -> None:
    os.makedirs(f'{sys.path[0]}/results/{exp}', exist_ok=True)
