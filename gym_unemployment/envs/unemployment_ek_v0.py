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
from .unemployment_v0 import UnemploymentEnv_v0

class UnemploymentEKEnv(UnemploymentEnv_v0):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.ben = BenefitsEK()
