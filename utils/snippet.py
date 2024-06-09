from typing import List


def subfix(data: List[str], sub: str) -> List[str]:
    return list(map(lambda x: x + sub, data))


def nosubfix(data: List[str], subint: int = 2) -> List[str]:
    return list(map(lambda x: x[:-subint], data))
