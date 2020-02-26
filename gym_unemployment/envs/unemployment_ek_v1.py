"""
Gym module implementing the Finnish social security including earnings-related components,
e.g., the unemployment benefit
"""

import math
import gym
from gym import spaces, logger, utils, error
from gym.utils import seeding
import numpy as np
#from .benefits import *
from fin_benefits import BenefitsEK
from .unemployment_v1 import UnemploymentLargeEnv

class UnemploymentEKLargeEnv(UnemploymentLargeEnv):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.ben = BenefitsEK()
        
        self.ansiopvraha_kesto400=350
        self.ansiopvraha_kesto300=250
        
        if 'kwargs' in kwargs:
            kwarg=kwargs['kwargs']
        else:
            kwarg={}
        for key, value in kwarg.items():
            if key=='toe_vaatimus':
                if value is not None:
                    self.toe_vaatimus=value
            elif key=='ansiopvraha_kesto400':
                if value is not None:
                    self.ansiopvraha_kesto400=value
            elif key=='ansiopvraha_kesto300':
                if value is not None:
                    self.ansiopvraha_kesto300=value
            elif key=='porrastus':
                if value is not None:
                    self.porrastus=value
            elif key=='pvhoito':
                if value is not None:
                    self.muuta_pvhoito=value
            elif key=='muuta_ansiopv_ylaraja':
                if value is not None:
                    self.muuta_ansiopv_ylaraja=value