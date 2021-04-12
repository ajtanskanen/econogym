"""
Gym module implementing the Finnish social security including earnings-related components,
e.g., the unemployment benefit
"""

import math
import gym
from gym import spaces, logger, utils, error
from gym.utils import seeding
import numpy as np
from fin_benefits import BenefitsEK
from .unemployment_rev_v0 import UnemploymentRevEnv_v0

class UnemploymentLongEnv_v0(UnemploymentRevEnv_v0):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.ansiopvraha_kesto=2.0
