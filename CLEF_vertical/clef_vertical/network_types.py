"""Network type definitions for clef-vertical."""

from enum import Enum


class NetworkType(Enum):
    """Enum representing the different types of networks in the system."""

    PERSONAL = 0
    CLINICAL = 1


class WeightType(Enum):
    """Enum representing the different types of weights in the system."""

    MODEL = 0
    OPTIMIZER = 1
