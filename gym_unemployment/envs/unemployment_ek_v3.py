"""
Gym module implementing the Finnish social security including earnings-related components,
e.g., the unemployment benefit
"""

import math
import gym
from gym import spaces, logger, utils, error
from gym.utils import seeding
import numpy as np
from fin_benefits import BenefitsEK, Benefits
from .unemployment_v3 import UnemploymentLargeEnv_v3

class UnemploymentEKLargeEnv_v3(UnemploymentLargeEnv_v3):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
        self.ben = BenefitsEK(**kwargs)
        #self.ben = Benefits(**kwargs)
            
        self.ben.set_year(self.year)

        self.ansiopvraha_toe=1.0 # = 12kk
        
        self.ansiopvraha_kesto400=350
        self.ansiopvraha_kesto300=250
        scale=21.5*12
        self.apvkesto300=np.round(self.ansiopvraha_kesto300/scale/self.timestep)*self.timestep
        self.apvkesto400=np.round(self.ansiopvraha_kesto400/scale/self.timestep)*self.timestep
        self.apvkesto500=np.round(self.ansiopvraha_kesto500/scale/self.timestep)*self.timestep
        
        if 'kwargs' in kwargs:
            kwarg=kwargs['kwargs']
        else:
            kwarg=kwargs
            
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
                    
        self.explain()