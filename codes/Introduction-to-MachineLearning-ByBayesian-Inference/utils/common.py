from typing import List
from enum import Enum


class ApproximateMethod(str, Enum):
    gibbs = "gibbs"
    variational = "variational"
    collapsed_gibbs = "collapsed_gibbs"

    @staticmethod
    def members() -> List[str]:
        return [v.value for v in ApproximateMethod.__members__.values()]

    @staticmethod
    def valid(name: str) -> bool:
        return name in ApproximateMethod.members()
