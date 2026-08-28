from enum import Enum


class Treatment(str, Enum):
    FAST = "FAST"
    SLOW = "SLOW"
    RESTART = "RESTART"
