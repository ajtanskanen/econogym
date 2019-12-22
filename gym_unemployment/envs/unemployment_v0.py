"""
Gym module implementing the Finnish social security including earnings-related components,
e.g., the unemployment benefit

This is the minimal v0 version
"""

import math
import gym
from gym import spaces, logger, utils, error
from gym.utils import seeding
import numpy as np
import fin_benefits

class UnemploymentEnv(gym.Env):
    """
    Description:
        The Finnish Unemployment Pension Scheme 

    Source:
        This environment corresponds to the environment of the Finnish Social Security

    Observation: 
        Type: Box(12) FIXME!
        Num    Observation                Min           Max
        0    Employment status             0             3
        1    time-in-state         0             2
        2    Old-age pension               0           inf
        3    Salary                        0           inf
        4    Age                          25            69
        5    Paid old-age pension
        
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
        self.elinaikakerroin=0.95
        self.palkkakerroin=0.8*1+0.2*1.0/1.016
        self.elakeindeksi=0.2*1+0.8*1.0/1.016
        
        self.timestep=1.0
        self.gamma=0.92 # discounting
        self.max_age=75
        self.min_age=25
        self.min_retirementage=65
        self.max_retirementage=70        
        
        self.ansiopvraha_kesto400=400
        self.karenssi_kesto=0.24 # vuotta
        self.accbasis_tmtuki=1413.75*12        
        
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
                    self.gamma==value
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
            elif key=='randomness':
                if value is not None:
                    self.randomness=value
                
         
        # ei skaalata!
        #self.ansiopvraha_kesto400=self.ansiopvraha_kesto400/(12*21.5)

        self.salary_const=0.05*self.timestep
        
        self.acc=0.015*self.timestep
        self.acc_unemp=0.75*self.acc
        
        print('minimal model')
        self.n_age=self.max_age-self.min_age+1
        self.n_empl=3 # state of employment, 0,1,2,3,4
            
        self.salary=np.zeros(self.max_age+1)
        
        self.pinkslip_intensity=0.05*self.timestep # todennäköisyys tulla irtisanotuksi vuodessa, skaalaa!
        self.mort_intensity=self.get_mort_rate()*self.timestep # todennäköisyys , skaalaa!
        self.npv=self.comp_npv()

        # Limits on states
        self.low = np.array([
            0,
            0,
            0,
            0,
            0,
            self.min_age,
            0])
        self.high = np.array([
            1,
            1,
            1,
            1000,
            1000,
            self.max_age+1,
            100])
        

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None
        
        self.ben = fin_benefits.Benefits()
        
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
        p['paivahoidossa']=0
        p['aikuisia']=1
        p['veromalli']=0
        p['kuntaryhma']=3
        p['lapsia_kotihoidontuella']=0
        p['alle3v']=0
        p['tyottomyyden_kesto']=1
        p['puolison_tyottomyyden_kesto']=10
        p['isyysvapaalla']=0
        p['aitiysvapaalla']=0
        p['kotihoidontuella']=0
        p['alle_kouluikaisia']=0
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
                if (p['tyottomyyden_kesto']<=self.ansiopvraha_kesto400): # and (p['tyottomyyden_kesto']>self.karenssi_kesto): # karenssi, jos ei irtisanottu
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
        p['puolison_tulot']=0
        p['puoliso_tyoton']=0  
        p['puoliso_vakiintunutpalkka']=0  
        p['puoliso_saa_ansiopaivarahaa']=0
        p['puolison_tulot']=0
        
        netto,benefitq=self.ben.laske_tulot(p)
        netto=max(0,netto-p['asumismenot_asumistuki']) # netotetaan asumismenot pois käteenjäävästä
        netto=netto*12
        
        #print('ben:',(wage,old_wage,pension,employment_status,time_in_state,ika),netto)
        
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
        
    def get_wage(self,age,time_in_state):
        if age<=self.max_age and age>=self.min_age-1:
            return self.salary[int(np.floor(age))]*(1-min(time_in_state,5)*self.salary_const)
        else:
            return 0
            
    def get_mort_rate(self):
        # tilastokeskuksen kuolleisuusdata 2017 sukupuolittain
        mort=np.array([2.12,0.32,0.17,0.07,0.07,0.10,0.00,0.09,0.03,0.13,0.03,0.07,0.10,0.10,0.10,0.23,0.50,0.52,0.42,0.87,0.79,0.66,0.71,0.69,0.98,0.80,0.77,1.07,0.97,0.76,0.83,1.03,0.98,1.20,1.03,0.76,1.22,1.29,1.10,1.26,1.37,1.43,1.71,2.32,2.22,1.89,2.05,2.15,2.71,2.96,3.52,3.54,4.30,4.34,5.09,4.75,6.17,5.88,6.67,8.00,9.20,10.52,10.30,12.26,12.74,13.22,15.03,17.24,18.14,17.78,20.35,25.57,23.53,26.50,28.57,31.87,34.65,40.88,42.43,52.28,59.26,62.92,68.86,72.70,94.04,99.88,113.11,128.52,147.96,161.89,175.99,199.39,212.52,248.32,260.47,284.01,319.98,349.28,301.37,370.17,370.17])/1000.0
        
        return mort
        
    def comp_npv(self):
        cpsum=1
        for x in range(100,self.max_age,-1):
            #print(x,cpsum)
            cpsum=self.mort_intensity[x]*1+(1-self.mort_intensity[x])*(1+self.gamma*cpsum)
            
        return cpsum
        
    # from Hakola and Määttänen, 2005
    def log_utility(self,income,employment_state,age):
        # kappa tells how much person values free-time
        kappa_kokoaika=0.60
        
        if age>58:
            mu=0.10
            kappa_kokoaika *= (1+mu*max(0,age-58))
        
        if employment_state == 1:
            u=np.log(income)-kappa_kokoaika
        elif employment_state == 2:
            u=np.log(income)+0.14
        else:
            u=np.log(income)
            
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
        
    # From Määttänen, 2013
    def wage_process(self,w,age,ave=3300*12):
        eps=np.random.normal(loc=0,scale=0.02,size=1)[0]
        a0=ave
        a1=0.89
        if w>0:
            wt=a0*np.exp(a1*np.log(w/a0)+eps)
        else:
            wt=a0*np.exp(eps)
            
        return wt
        
    def wage_process_simple(self,w,age,ave=3300*12):
        return w
        
    def compute_salary(self,group=1):
        group_ave=np.array([2000,3300,5000])*12
        a0=group_ave[group]

        self.salary[self.min_age]=np.maximum(5000,np.random.normal(loc=a0,scale=12*1000,size=1)[0]) # e/y
    
        # asetetaan palkat samoiksi kaikkina ikävuosina
        for age in range(self.min_age+1,self.max_age+1):
            #self.salary[age]=self.wage_process(self.salary[age-1],age,ave=a0)
            self.salary[age]=self.salary[age-1]
        
    def scale_pension(self,pension,age):
        return self.elinaikakerroin*pension*self.elakeindeksi*(1+0.004*(age-self.min_retirementage)*12)

    def render(self, mode='human'):
        emp,pension,wage,age,time_in_state=self.state_decode(self.state)
        print('Tila {} palkka {} ikä {} t-i-s {} eläke {}'.format(\
            emp,wage,age,time_in_state,pension))

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
        employment_status,pension,old_wage,age,time_in_state=self.state_decode(self.state)
        intage=int(np.floor(age))
        
        # here age is age at the beginning of the period
        
        action=int(action)
        
        if randomness:
            sattuma = np.random.uniform(size=4)
            if sattuma[3]<self.mort_intensity[intage]:
                action=15

        if action==15: # kuolee
            employment_status,pension,old_wage,time_in_state,wage,netto=(-1,0,0,0,0,0)
            done=True
            self.state = self.state_encode(employment_status,pension,wage,age+self.timestep,time_in_state)
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
                employment_status = 0 # unchanged
                wage=old_wage
                time_in_state+=self.timestep
                if age<65:                
                    if time_in_state<=1.5: # 1.5 years
                        pension=pension*self.palkkakerroin+self.acc_unemp*old_wage
                    else:
                        pension=pension*self.palkkakerroin+self.acc*self.accbasis_tmtuki
                else:
                    pension=pension*self.palkkakerroin
                netto=self.comp_benefits(0,old_wage,0,employment_status,time_in_state,age)
            elif action == 1: # 
                employment_status  = 1 # switch
                time_in_state+=self.timestep
                if dynprog:
                    wage=old_wage*(1-min(time_in_state,5)*self.salary_const)
                else:
                    wage=self.get_wage(intage,time_in_state)
                
                time_in_state=0
                pension=pension*self.palkkakerroin+self.acc*wage
                netto=self.comp_benefits(wage,0,0,employment_status,time_in_state,age)
            elif action == 2:
                if age>=self.min_retirementage: # ve
                    employment_status  = 2 
                    time_in_state=0
                    wage=old_wage
                    pension=self.scale_pension(pension,age)
                    pension=pension+self.ben.laske_kansanelake(age,pension,1)
                    netto=self.comp_benefits(0,0,pension,employment_status,0,age)
                else:
                    print('error 99')
            else:
                print('error 17')
        elif employment_status == 1:
            if action == 0 or (action==2 and age<self.min_retirementage):
                employment_status  = 1 # unchanged
                if dynprog:
                    wage=old_wage
                else:
                    wage=self.get_wage(intage,0)
                time_in_state+=self.timestep
                pension=pension*self.palkkakerroin+self.acc*wage
                netto=self.comp_benefits(wage,0,0,employment_status,time_in_state,age)
            elif action == 1: # työttömäksi
                employment_status  = 0 # switch
                time_in_state=0
                wage=old_wage
                if age<65:                
                    if time_in_state<=1.5: # 1.5 years
                        pension=pension*self.palkkakerroin+self.acc_unemp*old_wage
                    else:
                        pension=pension*self.palkkakerroin+self.acc*self.accbasis_tmtuki
                else:
                    pension=pension*self.palkkakerroin
                netto=self.comp_benefits(0,old_wage,0,employment_status,time_in_state,age)
            elif action==2:
                if age>=self.min_retirementage: # ve
                    employment_status  = 2 # unchanged
                    wage=old_wage
                    pension=self.scale_pension(pension,age)
                    pension=pension+self.ben.laske_kansanelake(age,pension,1)
                    time_in_state=0
                    netto=self.comp_benefits(0,0,pension,employment_status,0,age)
                else:
                    print('error 13')
            else:
                print('error 12')
        elif employment_status == 2: # eläkkeellä, ei voi palata töihin
            employment_status = 2 # unchanged
            time_in_state+=self.timestep
            pension=pension*self.elakeindeksi
            wage=old_wage
            netto=self.comp_benefits(0,0,pension,employment_status,0,age)
        else:
            print('Unknown employment_status {s} of type {t}'.format(s=employment_status,t=type(employment_status)))
        
        self.state = self.state_encode(employment_status,pension,wage,age+self.timestep,time_in_state)
        
        done = age >= self.max_age
        done = bool(done)

        if dynprog:
            if not done:
                reward = self.log_utility(netto,int(employment_status),age)
            else:
                if age>self.max_age+0.001:
                    self.steps_beyond_done = 0
                    reward = 0
                    self.state = self.state_encode(employment_status,pension,wage,age+self.timestep,0)
                else:
                    self.steps_beyond_done = 0
                    if employment_status==2:
                        reward = self.npv*self.log_utility(netto,2,age)
                    else:
                        reward = 0
                    self.state = self.state_encode(employment_status,pension,wage,age+self.timestep,0)
        else:
            if not done:
                reward = self.log_utility(netto,int(employment_status),age)
            elif self.steps_beyond_done is None:
                self.steps_beyond_done = 0
                #if employment_status == 2:
                reward = self.npv*self.log_utility(netto,2,age)
                #else:
                #    reward = 0.0
                self.state = self.state_encode(employment_status,0,wage,age+self.timestep,0)
            else:
                if self.steps_beyond_done == 0:
                    logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
                self.steps_beyond_done += 1
                reward = 0.0
            
        return np.array(self.state), reward, done, {'netto': netto, 'r': reward}

    def state_encode(self,emp,pension,old_wage,age,time_in_state):
        d=np.zeros(self.n_empl+4)
        states=self.n_empl
        if emp==1:
            d[0:states]=np.array([0,1,0])
        elif emp==0:
            d[0:states]=np.array([1,0,0])
        elif emp==2:
            d[0:states]=np.array([0,0,1])
        else:
            print('state error '+str(emp))
            
        d[states]=(pension-20_000)/10_000 # vastainen eläke
        d[states+1]=(old_wage-40_000)/15_000
        d[states+2]=(age-(69+25)/2)/10
        d[states+3]=time_in_state/10
        
        return d

    def state_decode(self,vec):
        emp=-1
        for k in range(self.n_empl):
            if vec[k]>0:
                emp=k
                break
                
        if emp<0:
            print('state error '+str(vec))
            
        pension=vec[self.n_empl]*10_000+20_000
        wage=vec[self.n_empl+1]*15_000+40_000
        age=vec[self.n_empl+2]*10+(69+25)/2
        time_in_state=vec[self.n_empl+3]*10
                
        return int(emp),pension,wage,age,time_in_state

    def reset(self,init=None):
        '''
        Open AI-interfacen mukainen reset-funktio, joka nollaa laskennan alkutilaan
        '''
                    
        employment_status=0
        age=int(self.min_age)
        time_in_state=0
        pension=0
        
        # set up salary for the entire career
        group=np.random.randint(3)
        self.compute_salary(group=group)
        old_wage=self.salary[self.min_age]

        self.state = self.state_encode(employment_status,pension,old_wage,self.min_age,0)
        self.steps_beyond_done = None
        
        return np.array(self.state)
