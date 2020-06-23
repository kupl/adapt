# Import strategies.
from adapt.strategy.adapt import AdaptiveParameterizedStrategy
from adapt.strategy.adapt import ParameterizedStrategy
from adapt.strategy.deepxplore import UncoveredRandomStrategy
from adapt.strategy.dlfuzz import DLFuzzRoundRobin
from adapt.strategy.dlfuzz import MostCoveredStrategy
from adapt.strategy.random import RandomStrategy

# Aliases for some strategies.
Adapt = AdaptiveParameterizedStrategy
DeepXplore = UncoveredRandomStrategy
DLFuzzFirst = MostCoveredStrategy
