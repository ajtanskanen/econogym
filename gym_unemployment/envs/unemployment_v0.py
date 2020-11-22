"""
Gym module implementing the Finnish social security including earnings-related components,
e.g., the unemployment benefit

This is the minimal v0 version

- earning-related unemployment compensation is available for one year, not longer
"""

import math
import gym
from gym import spaces, logger, utils, error
from gym.utils import seeding
import numpy as np
import fin_benefits
import random
from scipy.stats import lognorm

class UnemploymentEnv_v0(gym.Env):
    """
    Description:
        The Finnish Unemployment Pension Scheme 

    Source:
        This environment corresponds to the environment of the Finnish Social Security
employment_status,pension,old_wage,age,time_in_state,next_wage
    Observation: 
        Type: Box(12) FIXME!
        Num    Observation                Min           Max
        0    Employment status             0             3
        1    Old-age pension               0           inf
        2    Salary                        0           inf
        3    Age                          25            69
        4    time-in-state                 0             2
        5    Next Salary                   0           inf
        
    Actions:
        Type: Discrete(2)
        Num    Action
        0    Stay in the current state
        1    Switch to the other state (work -> unemployed; unemployed -> work)
        2    Retire if >=63, else Stay in the current state (Työttömyysputki?)
        
        Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it

    Reward:
        Reward is the sum of wage and benefit for every step taken, including the termination step
        
    To be done:
        Compare the results

    Starting State:
        Starting state in employed (in work)
        
    Step:
        Each step corresponds to one month in time

    Episode Termination:
        120 steps i.e. 10 years
    """
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self,**kwargs):
        super().__init__()
        
        self.version=0
        
        self.elinaikakerroin=0.95
        
        self.timestep=1.0
        gamma=0.92 # discounting
        #self.gamma=gamma**self.timestep # discounting
        reaalinen_palkkojenkasvu=1.016
        self.palkkakerroin=(0.8*1+0.2*1.0/reaalinen_palkkojenkasvu)**self.timestep
        self.elakeindeksi=(0.2*1+0.8*1.0/reaalinen_palkkojenkasvu)**self.timestep
        self.kelaindeksi=(1.0/reaalinen_palkkojenkasvu)**self.timestep

        # karttumaprosentit
        self.acc=0.015*self.timestep
        self.acc_over_52=0.019*self.timestep
        self.acc_family=1.15*self.acc
        self.acc_family_over_52=1.15*self.acc_over_52
        self.acc_unemp=0.75*self.acc
        self.acc_unemp_over_52=0.75*self.acc_over_52

        self.max_age=70
        self.min_age=20
        self.min_retirementage=65
        self.max_retirementage=70        
        
        self.ansiopvraha_kesto=1.0
        self.karenssi_kesto=0.24 # vuotta
        self.accbasis_tmtuki=1413.75*12        
        self.plotdebug=False
        self.wage_without_tis=True
        self.include_mort=False
        self.reset_exploration_go=True
        self.reset_exploration_ratio=0.4
        self.train=False
        self.zero_npv=False
        
        if 'kwargs' in kwargs:
            kwarg=kwargs['kwargs']
        else:
            kwarg={}
            
        for key, value in kwarg.items():
            if key=='step':
                if value is not None:
                    self.timestep==value
            elif key=='gamma':
                if value is not None:
                    gamma=value
            elif key=='train':
                if value is not None:
                    self.train=value
            elif key=='reset_exploration_go':
                if value is not None:
                    self.reset_exploration_go=value
            elif key=='reset_exploration_ratio':
                if value is not None:
                    self.reset_exploration_ratio=value
            elif key=='min_age':
                if value is not None:
                    self.min_age=value
            elif key=='max_age':
                if value is not None:
                    self.max_age=value
            elif key=='min_retirementage':
                if value is not None:
                    self.min_retirementage=value
            elif key=='max_retirementage':
                if value is not None:
                    self.max_retirementage=value
            elif key=='ansiopvraha_kesto400':
                if value is not None:
                    self.ansiopvraha_kesto400=value
            elif key=='mortality':
                if value is not None:
                    self.include_mort=value
            elif key=='randomness':
                if value is not None:
                    self.randomness=value
            elif key=='plotdebug':
                if value is not None:
                    self.plotdebug=value
            elif key=='wage_without_tis':
                if value is not None:
                    self.no_tis=value
                    
        print('Train ',self.train)
        print('plotdebug',self.plotdebug)
        
        #if self.train:
        #    self.partial_npv=True
        #    print('partial')
        #else:
        self.partial_npv=False
         
        # ei skaalata!
        #self.ansiopvraha_kesto400=self.ansiopvraha_kesto400/(12*21.5)

        self.salary_const=0.05*self.timestep
        self.gamma=gamma**self.timestep
        
        self.acc=0.015*self.timestep
        self.acc_unemp=0.75*self.acc
        
        print('minimal model')
        self.n_age=self.max_age-self.min_age+1
        self.n_empl=3 # state of employment, 0,1,2,3,4
            
        self.salary=np.zeros(self.max_age+1)
        
        self.pinkslip_intensity=0.05*self.timestep # todennäköisyys tulla irtisanotuksi vuodessa, skaalaa!
        self.mort_intensity=self.get_mort_rate()*self.timestep # todennäköisyys , skaalaa!
        self.npv=self.comp_npv()
        
        self.state_limits()        

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        
        if self.wage_without_tis:
            self.get_wage=self.get_wage_without_tis
        else:
            self.get_wage=self.get_wage_with_tis
        
        self.steps_beyond_done = None
        
        self.ben = fin_benefits.Benefits()
        self.setup_salaries()
        
        self.n_actions=3

        self.explain()
        
        #self.unit_test_code_decode()
        
    def get_n_states(self):
        '''
        Palauta parametrien arvoja
        '''
        return self.n_empl,self.n_actions
        
    def comp_benefits(self,wage,old_wage,pension,employment_status,time_in_state,ika=25):
        '''
        Laske etuuksien arvo, kun 
            wage on palkka
            old_wage on vanha palkka
            pension on eläkkeen määrä
            employment_status on töissä olo (0)/työttömyys (1)/eläkkeellä olo (2)
            prev_empl on työttömyyden kesto (0/1/2)
            ika on henkilön ikä
        '''
        p={}

        p['perustulo']=0
        p['toimeentulotuki_vahennys']=0
        p['ika']=ika
        p['lapsia']=0
        p['lapsia_paivahoidossa']=0
        p['aikuisia']=1
        p['veromalli']=0
        p['kuntaryhma']=3
        p['lapsia_kotihoidontuella']=0
        p['lapsia_alle_3v']=0
        p['tyottomyyden_kesto']=1
        p['puoliso_tyottomyyden_kesto']=10
        p['isyysvapaalla']=0
        p['aitiysvapaalla']=0
        p['kotihoidontuella']=0
        p['lapsia_alle_kouluikaisia']=0
        p['tyoelake']=0
        if employment_status==1: # työssä
            # voisi olla työtön siinä mielessä, että oikeutettu soviteltuun päivärahaan
            # oletus kuitenkin on, että ei saa soviteltua ansiopäivärahaa
            p['tyoton']=0 
            p['t']=wage/12
            p['vakiintunutpalkka']=0
            p['elakkeella']=0
            p['saa_ansiopaivarahaa']=0
        elif employment_status==0: # työtön, ansiopäiväraha alle 60 ja työmarkkinatuki
            if ika<65:
                p['tyoton']=1
                p['t']=0
                p['vakiintunutpalkka']=old_wage/12
                p['elakkeella']=0
                p['tyottomyyden_kesto']=time_in_state*12*21.5
                if time_in_state<self.ansiopvraha_kesto: # ansiosidonnaista vain vuosi
                    p['saa_ansiopaivarahaa']=1
                else:
                    p['saa_ansiopaivarahaa']=0
            else:
                p['tyoton']=0 # ei oikeutta työttömyysturvaan
                p['t']=0
                p['vakiintunutpalkka']=0
                p['elakkeella']=0
                p['saa_ansiopaivarahaa']=0
        elif employment_status==2:
            p['tyoton']=0
            p['saa_ansiopaivarahaa']=0
            p['t']=0
            p['vakiintunutpalkka']=0
            p['elakkeella']=1  
            p['tyoelake']=pension/12
        else:
            print('Unknown employment_status ',employment_status)
    
        # tarkastellaan yksinasuvia henkilöitä
        p['asumismenot_toimeentulo']=500
        p['asumismenot_asumistuki']=500

        p['ansiopvrahan_suojaosa']=1
        p['ansiopvraha_lapsikorotus']=1
        p['puoliso_tyoton']=0  
        p['puoliso_vakiintunutpalkka']=0  
        p['puoliso_saa_ansiopaivarahaa']=0
        p['puoliso_tulot']=0
        
        netto,benefitq=self.ben.laske_tulot(p)
        netto=netto*12
        
        return netto

    def seed(self, seed=None):
        '''
        Open AI interfacen mukainen seed-funktio, joka alustaa satunnaisluvut
        '''
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def env_seed(self, seed=None):
        '''
        Alustetaan numpy.random enviä varten
        '''
        np.random.seed(seed)
        #return [seed]
        
    def get_wage_with_tis(self,age,time_in_state):
        #if age==self.min_age-1:
        #    print(self.salary[int(np.floor(age))])
        if age<=self.max_age and age>=self.min_age-1:
            return self.salary[int(np.floor(age))]*(1-min(time_in_state,5)*self.salary_const)
        else:
            return 0
            
    def get_wage_without_tis(self,age,time_in_state):
        if age==self.min_age-1:
            print(self.salary[int(np.floor(age))])
        if age<=self.max_age and age>=self.min_age-1:
            return self.salary[int(np.floor(age))]
        else:
            return 0
            
    def get_mort_rate(self):
        # tilastokeskuksen kuolleisuusdata 2017 sukupuolittain
        mort=np.array([2.12,0.32,0.17,0.07,0.07,0.10,0.00,0.09,0.03,0.13,0.03,0.07,0.10,0.10,0.10,0.23,0.50,0.52,0.42,0.87,0.79,0.66,0.71,0.69,0.98,0.80,0.77,1.07,0.97,0.76,0.83,1.03,0.98,1.20,1.03,0.76,1.22,1.29,1.10,1.26,1.37,1.43,1.71,2.32,2.22,1.89,2.05,2.15,2.71,2.96,3.52,3.54,4.30,4.34,5.09,4.75,6.17,5.88,6.67,8.00,9.20,10.52,10.30,12.26,12.74,13.22,15.03,17.24,18.14,17.78,20.35,25.57,23.53,26.50,28.57,31.87,34.65,40.88,42.43,52.28,59.26,62.92,68.86,72.70,94.04,99.88,113.11,128.52,147.96,161.89,175.99,199.39,212.52,248.32,260.47,284.01,319.98,349.28,301.37,370.17,370.17])/1000.0
        
        return mort
        
    def pension_accrual(self,age,wage,pension,state=1,time_in_state=0):
        '''
        Eläkkeen karttumisrutiini
        '''
        if state in set([0]):
            #kesto=time_in_state*12*21.5
            #if kesto>self.ansiopvraha_kesto400:
            if time_in_state>=self.ansiopvraha_kesto:
                w=self.accbasis_tmtuki
            else:
                w=wage
        
            if age>=52 and age<63:
                acc=self.acc_unemp_over_52
            else:
                acc=self.acc_unemp
            
            if age<self.min_retirementage:
                pension=pension*self.palkkakerroin+acc*w
            else: # muuten ei karttumaa
                pension=pension*self.palkkakerroin
        elif state in set([1]):
            if age>=52 and age<63:
                acc=self.acc_over_52
            else:
                acc=self.acc

            if age<self.max_retirementage:
                pension=pension*self.palkkakerroin+acc*wage
            else:
                pension=pension*self.palkkakerroin
        else: # 2 - ei karttumaa
            pension=pension
            
        return pension        
        
    def comp_npv(self):
        '''
        lasketaan montako timestep:iä (diskontattuna) max_age:n jälkeen henkilö on vanhuuseläkkeellä 
        Hyvin yksinkertainen toteutus. Tulos on odotettu lukumäärä timestep:jä
        
        npv <- diskontattu
        npv0 <- ei ole diskontattu
        '''
        npv=0
        npv0=0

        cpsum=0
        cpsum0=0
        for x in np.arange(100,self.max_age,-self.timestep):
            intx=int(np.floor(x))
            m=self.mort_intensity[intx]
            cpsum=m*1+(1-m)*(1+self.gamma*cpsum)
            #cpsum0=m*1+(1-m)*(1+cpsum0)
            
        if self.zero_npv:
            npv=0*cpsum
        else:
            npv=cpsum
        #npv0[g]=cpsum0
            
        if self.plotdebug:
            print('npv: {}',format(npv))

        return npv #,npv0
        
    # from Hakola and Määttänen, 2005
    def log_utility(self,income,employment_state,age):
        # kappa tells how much person values free-time
        kappa_kokoaika=0.62
        kappa_retirement=0.10
        mu_age=60
        
        if age>mu_age:
            mu=0.18
            kappa_kokoaika *= (1+mu*max(0,min(4,age-mu_age)))
        
        if employment_state == 1:
            u=np.log(income)-kappa_kokoaika
        elif employment_state == 2:
            u=np.log(income)+kappa_retirement
        else:
            u=np.log(income)
            
        #if income<1.0:
        #    print(income,employment_state,age)
        #    self.render()
            
        return u/10
   
    # From Hakola & Määttänen, 2005
    def wage_process_hakola(self,w,age):
        eps=np.random.normal(loc=0,scale=0.2,size=1)[0]
        a0=1.158242+0.5
        a1=0.8227191 # 0.835 # 
        a2=0.0029402
        a3=-0.0000517
        a4=1.62e-6
        a5=-2.13e-8
        if w>0:
            wt=np.exp(a0+a1*np.log(w)+a2*age+a3*age**2+a4*age**3+a5*age**4+eps)
        else:
            wt=np.exp(a0+a2*age+a3*age**2+a4*age**3+a5*age**4+eps)
            
        # debug stuff
        #wt=w
        wt=max(20000,wt)
            
        return wt
        
    # wage process reparametrized
    def wage_process(self,w,age,a0=3300*12,a1=3300*12):
        '''
        Palkkaprosessi muokattu lähteestä Määttänen, 2013 
        palkka vuositasolla
        '''
        sigma=0.05
        eps=np.random.normal(loc=0,scale=sigma,size=1)[0]
        c1=0.89
        if w<self.min_salary:
            w=self.min_salary
            
         # pidetään keskiarvo/a1 samana kuin w/a0
        wt=a1*np.exp(c1*np.log(w/a0)+eps-0.5*sigma*sigma)
        
        #print('age {} wt {} a1 {} a0 {} w {}'.format(age,wt,a1,a0,w))

        # täysiaikainen vuositulo vähintään self.min_salary
        wt=np.maximum(self.min_salary,wt)

        return wt
                
        
    # wage process cumulative function
    def wage_process_cumulative(self,w_cum,w_old,age):
        '''
        Palkkaprosessin kertymäfunktio 
        palkka vuositasolla
        
        w_old  vanha palkka
        age     ika
        '''
        
        if age>self.min_age:        
            a0=self.palkat_ika_miehet[age-1-self.min_age]
            a1=self.palkat_ika_miehet[age-self.min_age]
        else:
            a0=self.palkat_ika_miehet[0]
            a1=self.palkat_ika_miehet[0]
                
        sigma=0.05
        c1=0.89
        if w_cum<self.min_salary:
            c=0
        else:
            # pidetään keskiarvo/a1 samana kuin w/a0
            q=max(w_cum,self.min_salary)/(a1*np.exp(c1*np.log(max(w_old,self.min_salary)/a0)-0.5*sigma*sigma))
            c=lognorm.cdf(q,sigma,loc=0,scale=1)
            #print(q,w_old,a0,a1,c1)

        return c
        
    def wage_process_simple(self,w,age,ave=3300*12):
        return w

    def setup_salaries(self):
        self.min_salary=1000
        self.palkat_ika_miehet=12.5*np.array([2339.01,2489.09,2571.40,2632.58,2718.03,2774.21,2884.89,2987.55,3072.40,3198.48,3283.81,3336.51,3437.30,3483.45,3576.67,3623.00,3731.27,3809.58,3853.66,3995.90,4006.16,4028.60,4104.72,4181.51,4134.13,4157.54,4217.15,4165.21,4141.23,4172.14,4121.26,4127.43,4134.00,4093.10,4065.53,4063.17,4085.31,4071.25,4026.50,4031.17,4047.32,4026.96,4028.39,4163.14,4266.42,4488.40,4201.40,4252.15,4443.96,3316.92,3536.03,3536.03])
        self.palkat_ika_naiset=12.5*np.array([2223.96,2257.10,2284.57,2365.57,2443.64,2548.35,2648.06,2712.89,2768.83,2831.99,2896.76,2946.37,2963.84,2993.79,3040.83,3090.43,3142.91,3159.91,3226.95,3272.29,3270.97,3297.32,3333.42,3362.99,3381.84,3342.78,3345.25,3360.21,3324.67,3322.28,3326.72,3326.06,3314.82,3303.73,3302.65,3246.03,3244.65,3248.04,3223.94,3211.96,3167.00,3156.29,3175.23,3228.67,3388.39,3457.17,3400.23,3293.52,2967.68,2702.05,2528.84,2528.84])

    def compute_salary(self,initial_salary=None,initial_age=None):
        if initial_salary is not None:
            loc=initial_salary
        else:
            loc=self.palkat_ika_miehet[0]

        if initial_age is not None:
            minage=initial_age
        else:
            minage=self.min_age
            
        s1=self.palkat_ika_miehet[minage-self.min_age]/5
        self.salary[minage-1]=np.maximum(self.min_salary,np.random.normal(loc=loc,scale=s1,size=1)[0]) # e/y
        a1=self.palkat_ika_miehet[0]
        #print('age {} sal {}'.format(minage-1,self.salary[minage-1]))
        for age in range(minage,self.max_age+1):
            a0=a1
            a1=self.palkat_ika_miehet[age-self.min_age]
            self.salary[age]=self.wage_process(self.salary[age-1],age,a0,a1)
            #print('age {} sal {}'.format(age,self.salary[age]))
        
    def scale_pension(self,pension,age):
        return self.elinaikakerroin*pension*self.elakeindeksi*(1+0.048*(age-self.min_retirementage))

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def step(self, action, randomness=False, dynprog=False):
        '''
        Open AI interfacen mukainen step-funktio, joka tekee askeleen eteenpäin
        toiminnon action mukaan
        
        Tässä versiossa vain kolme tilaa: töissä, työtön ja vanhuuseläkkeellä
        '''
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        employment_status,pension,old_wage,age,time_in_state,wage=self.state_decode(self.state)
        intage=int(np.floor(age))
        
        # here age is age at the beginning of the period
        
        action=int(action)
        
        if self.include_mort:
            sattuma = np.random.uniform(size=4)
            if sattuma[3]<self.mort_intensity[intage]:
                action=15

        if action==15: # kuolee
            employment_status,pension,old_wage,time_in_state,wage,netto=(-1,0,0,0,0,0)
            done=True
            self.state = self.state_encode(employment_status,pension,wage,age+self.timestep,time_in_state,wage)
            reward=0
            return np.array(self.state), reward, done, {}
        # ei siirretä automaattisesti eläkkeelle
        #elif age>self.max_retirementage and employment_status<2:
        #    pension=self.scale_pension(pension,age)
        #    employment_status=2
        #    wage=old_wage
        #    pension=pension+self.ben.laske_kansanelake(age,pension,1)
        #    netto=self.comp_benefits(0,0,pension,employment_status,0,age)
        elif employment_status == 0:
            if action == 0 or (action==2 and age<self.min_retirementage):
                if not dynprog:
                    wage=self.get_wage(intage,time_in_state) # tämä on väärin
                employment_status = 0 # unchanged
                pension=self.pension_accrual(age,old_wage,pension,state=0,time_in_state=time_in_state)
                netto=self.comp_benefits(0,old_wage,0,employment_status,time_in_state,age)
                time_in_state+=self.timestep
                if not dynprog:
                    #wage=old_wage
                    next_wage=self.get_wage(int(np.floor(age+self.timestep)),time_in_state)
                else:
                    #wage=old_wage
                    next_wage=0
            elif action == 1: # 
                employment_status  = 1 # switch
                if not dynprog:
                    wage=self.get_wage(intage,time_in_state)
                time_in_state=0
                pension=self.pension_accrual(age,wage,pension,state=employment_status,time_in_state=time_in_state)
                netto=self.comp_benefits(wage,0,0,employment_status,time_in_state,age)
                time_in_state+=self.timestep
                if not dynprog:
                    next_wage=self.get_wage(int(np.floor(age+self.timestep)),0)
                else:
                    next_wage=0
            elif action == 2:
                if age>=self.min_retirementage: # ve
                    employment_status  = 2 
                    time_in_state=0
                    wage=0 #old_wage
                    pension=self.scale_pension(pension,age)+self.ben.laske_kansanelake(age,pension/12,1)*12
                    netto=self.comp_benefits(0,0,pension,employment_status,0,age)
                    time_in_state+=self.timestep
                    next_wage=0
                else:
                    print('error 99')
            else:
                print('error 17')
        elif employment_status == 1:
            if action == 0 or (action==2 and age<self.min_retirementage):
                employment_status  = 1 # unchanged
                if not dynprog:
                    wage=self.get_wage(intage,0)
                pension=self.pension_accrual(age,wage,pension,state=1)
                netto=self.comp_benefits(wage,0,0,employment_status,time_in_state,age)
                time_in_state+=self.timestep
                if not dynprog:
                    next_wage=self.get_wage(int(np.floor(age+self.timestep)),0)
                else:
                    next_wage=0
            elif action == 1: # työttömäksi
                employment_status = 0 # switch
                time_in_state=0
                if not dynprog:
                    wage=self.get_wage(intage,time_in_state)
                pension=self.pension_accrual(age,old_wage,pension,state=employment_status)
                netto=self.comp_benefits(0,old_wage,0,employment_status,time_in_state,age)
                time_in_state+=self.timestep
                if not dynprog:
                    next_wage=self.get_wage(int(np.floor(age+self.timestep)),time_in_state)
                else:
                    next_wage=0
            elif action==2:
                if age>=self.min_retirementage: # ve
                    employment_status  = 2 # unchanged
                    wage=0
                    pension=self.scale_pension(pension,age)+self.ben.laske_kansanelake(age,pension/12,1)*12
                    time_in_state=0
                    netto=self.comp_benefits(0,0,pension,employment_status,0,age)
                    time_in_state+=self.timestep
                    next_wage=0
                else:
                    print('error 13')
            else:
                print('error 12')
        elif employment_status == 2: # eläkkeellä, ei voi palata töihin
            employment_status = 2 # unchanged
            pension=pension*self.elakeindeksi
            wage=0
            netto=self.comp_benefits(0,0,pension,employment_status,0,age)
            time_in_state+=self.timestep
            next_wage=0
        else:
            print('Unknown employment_status {s} of type {t}'.format(s=employment_status,t=type(employment_status)))
        
        done = age >= self.max_age
        done = bool(done)

        if dynprog:
            if not done:
                reward = self.log_utility(netto,int(employment_status),age)
                self.state = self.state_encode(employment_status,pension,wage,age+self.timestep,time_in_state,next_wage)
            else:
                if age>self.max_age+0.001:
                    self.steps_beyond_done = self.steps_beyond_done+1
                    reward = 0.0
                    self.state = self.state_encode(employment_status,pension,wage,age+self.timestep,time_in_state,next_wage)
                else:
                    self.steps_beyond_done = 0
                    if employment_status==2:
                        reward = self.npv*self.log_utility(netto,2,age)
                    else:
                        reward = 0.0
                    self.state = self.state_encode(employment_status,pension,wage,age+self.timestep,time_in_state,next_wage)
        else:
            if not done:
                if self.partial_npv:
                    basenetto=self.comp_benefits(0,0,0,0,0,18)
                    reward = self.npv*(self.log_utility(netto,int(employment_status),age)-self.log_utility(basenetto,0,18))
                else:            
                    reward = self.log_utility(netto,int(employment_status),age)
                self.state = self.state_encode(employment_status,pension,wage,age+self.timestep,time_in_state,next_wage)
            elif self.steps_beyond_done is None:
                self.steps_beyond_done = 0
                if employment_status == 2:
                    if self.partial_npv:
                        basenetto=self.comp_benefits(0,0,0,0,0,18)
                        reward = self.npv*(self.log_utility(netto,2,age)-self.log_utility(basenetto,0,18))
                        #print(reward,self.npv,self.log_utility(netto,2,age),self.log_utility(basenetto,0,18))
                    else:
                        reward = self.npv*self.log_utility(netto,2,age)
                else:
                    reward = 0.0
                self.state = self.state_encode(employment_status,pension,wage,age+self.timestep,time_in_state,next_wage)
            else:
                self.state = self.state_encode(employment_status,pension,wage,age+self.timestep,time_in_state,next_wage)
                if self.steps_beyond_done == 0:
                    logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
                self.steps_beyond_done += 1
                reward = 0.0
                
        if self.plotdebug:
            self.render(done=done,reward=reward, netto=netto)
            
        return np.array(self.state), reward, done, {'netto': netto, 'r': reward}

    def render(self, mode='human', close=False, done=False, reward=None, netto=None,state=None):
        '''
        Tulostus-rutiini
        '''
        if state is None:
            emp,pension,wage,age,time_in_state,nextwage=self.state_decode(self.state)
        else:
            emp,pension,wage,age,time_in_state,nextwage=self.state_decode(state)
            
        if reward is None:
            print('Tila {} palkka {:.2f} ikä {} t-i-s {} elake {:.2f} seur.palkka {:.2f}'.format(\
                emp,wage,age,time_in_state,pension,nextwage))
        elif netto is None:
            print('Tila {} palkka {:.2f} ikä {} t-i-s {} elake {:.2f} seur.palkka {:.2f} r {:.4f}'.format(\
                emp,wage,age,time_in_state,pension,nextwage,reward))
        else:
            print('Tila {} palkka {:.2f} ikä {} t-i-s {} elake {:.2f} seur.palkka {:.2f} r {:.4f} n {:.2f}'.format(\
                emp,wage,age,time_in_state,pension,nextwage,reward,netto))
        if done:
            print('-------------------------------------------------------------------------------------------------------')

    def state_encode(self,emp,pension,old_wage,age,time_in_state,nextwage):
        d=np.zeros(self.n_empl+8)
        states=self.n_empl
        if emp==1:
            d[0:states]=np.array([0,1,0])
        elif emp==0:
            d[0:states]=np.array([1,0,0])
        elif emp==2:
            d[0:states]=np.array([0,0,1])
        else:
            print('state error '+str(emp))
            
        d[states]=(pension-40_000)/40_000 # vastainen/alkanut eläke
        d[states+1]=(old_wage-40_000)/40_000
        d[states+2]=(age-(self.max_age+self.min_age)/2)/((self.max_age+self.min_age)/2)
        d[states+3]=(time_in_state-20)/20
        d[states+4]=(nextwage-40_000)/40_000
        if age>=self.min_retirementage:
            d[states+5]=1
        else:
            d[states+5]=0
            
        if time_in_state<self.ansiopvraha_kesto:
            d[states+6]=1
        else:
            d[states+6]=0
            
        d[states+7]=(self.min_retirementage-age)/max(self.max_age-self.min_retirementage,self.min_retirementage-self.min_age)
        
        return d

    def state_decode(self,vec):
        emp=-1
        for k in range(self.n_empl):
            if vec[k]>0:
                emp=k
                break
                
        if emp<0:
            print('state error '+str(vec))
            
        pension=vec[self.n_empl]*40_000+40_000
        wage=vec[self.n_empl+1]*40_000+40_000
        age=int(vec[self.n_empl+2]*((self.max_age+self.min_age)/2)+(self.max_age+self.min_age)/2)
        time_in_state=vec[self.n_empl+3]*20+20
        nextwage=vec[self.n_empl+4]*40_000+40_000
                
        return int(emp),pension,wage,age,time_in_state,nextwage

    def unit_test_code_decode(self):
        for k in range(100):
            emp=random.randint(0,2)
            pension=random.uniform(0,80_000)
            old_wage=random.uniform(0,80_000)
            age=np.random.randint(self.min_age,self.max_age)
            time_in_state=random.uniform(0,30)
            nextwage=random.uniform(0,80_000)
        
            vec=self.state_encode(emp,pension,old_wage,age,time_in_state,nextwage)
                                
            emp2,pension2,old_wage2,age2,time_in_state2,nextwage2\
                =self.state_decode(vec)
                
            self.check_state(emp,pension,old_wage,age,time_in_state,nextwage,
                             emp2,pension2,old_wage2,age2,time_in_state2,nextwage2)
                             
        return 'Unit test done'
        
    
    def check_state(self,emp,pension,old_wage,age,time_in_state,nextwage,
                             emp2,pension2,old_wage2,age2,time_in_state2,nextwage2):
        if not emp==emp2:  
            print('emp: {} vs {}'.format(emp,emp2))
        if not pension==pension2:  
            print('pension: {} vs {}'.format(pension,pension2))
        if not old_wage==old_wage2:  
            print('old_wage: {} vs {}'.format(old_wage,old_wage2))
        if not time_in_state==time_in_state2:  
            print('time_in_state: {} vs {}'.format(time_in_state,time_in_state2))
        if not age==age2:  
            print('age: {} vs {}'.format(age,age2))
        if not nextwage==nextwage2:  
            print('nextwage: {} vs {}'.format(nextwage,nextwage2))
                

    def state_limits(self):
        # Limits on states
        self.low = np.array([
            0,
            0,
            0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            0,
            0,
            -1.0])
        self.high = np.array([
            1,
            1,
            1,
            10.0,
            10.0,
            1.0,
            10.0,
            10.0,
            1,
            1,
            1.0])
                    
    def explain(self):
        '''
        Tulosta laskennan parametrit
        '''
        print('Parameters of lifecycle:\ntimestep {}\ngamma {} ({} per anno)\nmin_age {}\nmax_age {}\nmin_retirementage {}'.format(self.timestep,self.gamma,self.gamma**(1.0/self.timestep),self.min_age,self.max_age,self.min_retirementage))
        print('max_retirementage {} mortality {}'.format(self.max_retirementage,self.include_mort))
        print('ansiopvraha_kesto {}'.format(self.ansiopvraha_kesto))
        print('reset_exploration_go {} reset_exploration_ratio {} plotdebug {}\n'.format(self.reset_exploration_go,self.reset_exploration_ratio,self.plotdebug))

    def reset(self,init=None):
        '''
        Open AI-interfacen mukainen reset-funktio, joka nollaa laskennan alkutilaan
        '''
                    
        #employment_status=0

        employment_status=random.choices(np.array([0,1],dtype=int),weights=[0.626,0.374])[0]
        
        age=int(self.min_age)
        time_in_state=random.choices(np.array([0,1],dtype=int),weights=[0.40,0.60])[0] # 60% tm-tuella
        pension=0
        
        # set up salary for the entire career
        initial_salary=None
        initial_age=None
        if self.reset_exploration_go and self.train:
            if self.reset_exploration_ratio>np.random.uniform():
                #print('exploration')
                time_in_state=random.choices(np.array([0,1,2,3,4,5],dtype=int),weights=[0.25,0.25,0.20,0.10,0.10,0.10])[0] # 60% tm-tuella
                employment_status=random.choices(np.array([0,1],dtype=int),weights=[0.5,0.5])[0]
                if random.random()<0.5:
                    age=int(np.random.uniform(low=self.min_age,high=self.max_age-1))
                else:
                    age=int(np.random.uniform(low=58,high=self.max_age-1))
                initial_salary=np.random.uniform(low=1_000,high=110_000)
                pension=np.random.uniform(low=0,high=90_000)
                initial_age=age
                #print('Explore: age {} initial {} pension {}'.format(age,initial_salary,pension))

        self.compute_salary(initial_salary=initial_salary,initial_age=initial_age)
        old_wage=self.salary[age-1]
        wage=self.salary[age] # timestep == 1
        
        #print(initial_salary,old_wage,wage)

        self.state = self.state_encode(employment_status,pension,old_wage,age,time_in_state,wage)
        self.steps_beyond_done = None
        
        return np.array(self.state)
