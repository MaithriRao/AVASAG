from enum import Enum, verify, UNIQUE


@verify(UNIQUE)
class Tokenization(Enum):
    SOURCE_ONLY = 0
    SOURCE_TARGET = 1
