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

class UnemploymentRevEnv_v0(gym.Env):
    """
    Description:
        The Finnish Unemployment Pension Scheme 

    Source:
        This environment corresponds to the environment of the Finnish Social Security
    Observation: 
        Type: Box(6)
        Num    Observation                Min           Max
        0    Employment status             0             3
        1    Old-age pension               0           inf
        2    Salary                        0           inf
        3    Age                          25            69
        4    time-in-state                 0             2
        5    Next Salary                   0           inf
        
    Actions:
        Type: Discrete(4)
        Num    Action
        0    Stay in the current state
        1    Switch to the other state (work -> unemployed; unemployed -> work)
        2    Retire if >=63, else Stay in the current state (Työttömyysputki?)
        3    Switch to parttime/fulltime work
        
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
    
    def __init__(self,**kwargs):
        super().__init__()
        
        self.version=0
        
        self.elinaikakerroin=0.95
        
        if False: # DEBUG
            self.unemp_wageshock=0.95
            # stdev of wage process
            self.wage_sigma=0.00001 # DEBUG
        else:
            self.unemp_wageshock=0.95
            # stdev of wage process
            self.wage_sigma=0.07 # for all
            
        
        self.statedebug=False
        self.bentest=False
        if self.bentest:
            self.comp_benefits=self.comp_benefits_test
        else:
            self.comp_benefits=self.comp_benefits_real
        
        self.timestep=1.0
        gamma_discount=0.92 # discounting
        #self.gamma=gamma**self.timestep # discounting
        reaalinen_palkkojenkasvu=1.016
        self.palkkakerroin=(0.8*1+0.2*1.0/reaalinen_palkkojenkasvu)**self.timestep
        self.elakeindeksi=(0.2*1+0.8*1.0/reaalinen_palkkojenkasvu)**self.timestep
        self.kelaindeksi=(1.0/reaalinen_palkkojenkasvu)**self.timestep

        # karttumaprosentit
        self.acc=0.015*self.timestep
        self.acc_over_52=0.019*self.timestep
        #self.acc_family=1.15*self.acc
        #self.acc_family_over_52=1.15*self.acc_over_52
        self.acc_unemp=0.75*self.acc
        self.acc_unemp_over_52=0.75*self.acc_over_52

        self.max_age=70
        self.min_age=18
        self.startage=self.min_age
    
        self.min_retirementage=65
        self.max_retirementage=70       
        self.max_unemp_age=65 
        
        self.ansiopvraha_kesto=1.0
        self.karenssi_kesto=0.24 # vuotta
        self.accbasis_tmtuki=1413.75*12        
        self.plotdebug=False
        self.wage_without_tis=True
        self.include_mort=False
        self.reset_exploration_go=False
        self.reset_exploration_ratio=0.3
        self.train=False
        self.zero_npv=False
        self.additional_income_tax=0
        self.additional_income_tax_high=0
        self.additional_tyel_premium=0
        self.additional_kunnallisvero=0
        self.scale_tyel_accrual=True
        self.scale_additional_tyel_accrual=0
        self.scale_additional_unemp_benefit=0
        self.include_pt=False
        self.perustulo=False # onko Kelan perustulo laskelmissa
        
        if 'kwargs' in kwargs:
            kwarg=kwargs['kwargs']
        else:
            kwarg=kwargs
            
        for key, value in kwarg.items():
            if key=='step':
                if value is not None:
                    self.timestep==value
            elif key=='gamma':
                if value is not None:
                    gamma_discount=value
            elif key=='train':
                if value is not None:
                    self.train=value
            elif key=='perustulo':
                if value is not None:
                    self.perustulo=value
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
            elif key=='additional_income_tax':
                if value is not None:
                    self.additional_income_tax=value
            elif key=='additional_income_tax_high':
                if value is not None:
                    self.additional_income_tax_high=value
            elif key=='additional_tyel_premium':
                if value is not None:
                    self.additional_tyel_premium=value
            elif key=='additional_kunnallisvero':
                if value is not None:
                    self.additional_kunnallisvero=value
            elif key=='scale_tyel_accrual':
                if value is not None:
                    self.scale_tyel_accrual=value
            elif key=='year':
                if value is not None:
                    self.year=value
            elif key=='scale_additional_tyel_accrual':
                if value is not None:
                    self.scale_additional_tyel_accrual=value
            elif key=='startage': 
                if value is not None:
                    self.startage=value

        self.startage=max(self.min_age,self.startage)

        #if self.train:
        #    self.partial_npv=True
        #    print('partial')
        #else:
        self.partial_npv=False
         
        # ei skaalata!
        #self.ansiopvraha_kesto400=self.ansiopvraha_kesto400/(12*21.5)

        self.salary_const=0.05*self.timestep
        self.gamma=gamma_discount**self.timestep
        
        self.parttime_income=0.5
        
        print('minimal model')
        self.n_age=self.max_age-self.min_age+1
        self.n_empl=4 # states of employment, 0,1,2,3
        
        #if self.min_retirementage>self.max_unemp_age:
        self.max_unemp_age=self.min_retirementage 
    
        if self.include_pt:
            self.n_actions=4
        else:
            self.n_actions=3
            
        self.salary=np.zeros(self.max_age+1)
        
        self.mort_intensity=self.get_mort_rate()*self.timestep # todennäköisyys , skaalaa!
        self.npv,self.npv0,self.npv_pension=self.comp_npv()
        
        self.state_limits()        

        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        
        #if self.wage_without_tis:
        self.get_wage=self.get_wage_without_tis
        #else:
        #    self.get_wage=self.get_wage_with_tis
        
        self.steps_beyond_done = None
        
        if self.perustulo:
            self.ben = fin_benefits.BasicIncomeBenefits(**kwargs)
        else:
            self.ben = fin_benefits.Benefits(**kwargs)

        self.setup_salaries()
        
        self.explain()
        
        if self.plotdebug:
            self.unit_test_code_decode()
            
    def set_startage(self,startage):
        '''
        Aseta ikä, josta simulaatio aloitetaan
        '''
        self.startage=max(self.min_age,startage)
        
    def get_n_states(self):
        '''
        Palauta parametrien arvoja
        '''
        return self.n_empl,self.n_actions

    def get_lc_version(self):
        '''
        returns the version of life-cycle model's episodestate used
        '''
        return 0        
        
    def comp_benefits_real(self,wage,old_wage,pension,employment_status,time_in_state,ika):
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
        
        if self.perustulo:
            p['perustulo']=1
        else:
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
            if ika<self.max_unemp_age:
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
                p['toimeentulotuki_vahennys']=1
        elif employment_status==3: # osa-aikatyö
            p['tyoton']=0 
            p['t']=wage/12
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
        if ika<self.max_unemp_age or employment_status>0:
            p['asumismenot_toimeentulo']=500
            p['asumismenot_asumistuki']=500
        else:
            p['asumismenot_toimeentulo']=0
            p['asumismenot_asumistuki']=0

        p['ansiopvrahan_suojaosa']=1
        p['ansiopvraha_lapsikorotus']=1
        p['puoliso_tyoton']=0  
        p['puoliso_vakiintunutpalkka']=0  
        p['puoliso_saa_ansiopaivarahaa']=0
        p['puoliso_tulot']=0
        
        netto,benefitq=self.ben.laske_tulot(p)
        netto=netto*12
        #print(benefitq)
        
        return netto
        
    def comp_benefits_test(self,wage,old_wage,pension,employment_status,time_in_state,ika):
        '''
        Laske etuuksien arvo, kun TESTING
            wage on palkka
            old_wage on vanha palkka
            pension on eläkkeen määrä
            employment_status on töissä olo (0)/työttömyys (1)/eläkkeellä olo (2)
            prev_empl on työttömyyden kesto (0/1/2)
            ika on henkilön ikä
        '''
        p={}
        
        if self.perustulo:
            p['perustulo']=1
        else:
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
            netto=0.8*wage
        elif employment_status==0: # työtön, ansiopäiväraha alle 60 ja työmarkkinatuki
            if ika<self.max_unemp_age:
                p['tyoton']=1
                p['t']=0
                p['vakiintunutpalkka']=old_wage/12
                p['elakkeella']=0
                p['tyottomyyden_kesto']=time_in_state*12*21.5
                if time_in_state<self.ansiopvraha_kesto: # ansiosidonnaista vain vuosi
                    p['saa_ansiopaivarahaa']=1
                    netto=max(700*12,0.5*old_wage)
                else:
                    p['saa_ansiopaivarahaa']=0
                    netto=700*12
            else:
                p['tyoton']=0 # ei oikeutta työttömyysturvaan
                p['t']=0
                p['vakiintunutpalkka']=0
                p['elakkeella']=0
                p['saa_ansiopaivarahaa']=0
                p['toimeentulotuki_vahennys']=1
                netto=1000
        elif employment_status==3: # osa-aikatyö
            p['tyoton']=0 
            p['t']=wage/12
            p['vakiintunutpalkka']=0
            p['elakkeella']=0
            p['saa_ansiopaivarahaa']=0
            netto=0.5*wage
        elif employment_status==2:
            p['tyoton']=0
            p['saa_ansiopaivarahaa']=0
            p['t']=0
            p['vakiintunutpalkka']=0
            p['elakkeella']=1  
            p['tyoelake']=pension/12
            netto=pension
        else:
            print('Unknown employment_status ',employment_status)
    
        # tarkastellaan yksinasuvia henkilöitä
        if ika<self.max_unemp_age or employment_status>0:
            p['asumismenot_toimeentulo']=500
            p['asumismenot_asumistuki']=500
        else:
            p['asumismenot_toimeentulo']=0
            p['asumismenot_asumistuki']=0

        p['ansiopvrahan_suojaosa']=1
        p['ansiopvraha_lapsikorotus']=1
        p['puoliso_tyoton']=0  
        p['puoliso_vakiintunutpalkka']=0  
        p['puoliso_saa_ansiopaivarahaa']=0
        p['puoliso_tulot']=0
        
        #netto,benefitq=self.ben.laske_tulot(p)
        #netto=netto*12
        #print(benefitq)
        
        return netto        
        
    def scale_q(self,npv,npv0,npv_pension,benq):
        '''
        Scaling the incomes etc by a discounted nominal present value
        '''
        benq['verot']*=npv_pension
        benq['etuustulo_brutto']*=npv_pension
        benq['etuustulo_netto']*=npv_pension
        
        benq['valtionvero']*=npv_pension
        benq['kunnallisvero']*=npv_pension
        benq['asumistuki']*=npv_pension
        benq['elake_maksussa']*=npv_pension
        benq['kokoelake']*=npv_pension
        benq['perustulo']*=npv_pension
        benq['toimtuki']*=npv_pension
        #benq['palkkatulot']*=npv0
        benq['kateen']*=npv_pension
        benq['multiplier']=npv0

        return benq     

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
            return self.salary[int(np.floor(age))]
        if age<=self.max_age and age>=self.min_age-1:
            return self.salary[int(np.floor(age))]
        else:
            return 0
            
    def get_mort_rate(self):
        '''
        tilastokeskuksen kuolleisuusdata 2017 sukupuolittain
        ei käytössä
        '''
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
                w=0 #self.accbasis_tmtuki
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
        elif state in set([1,3]):
            if age>=52 and age<63:
                acc=self.acc_over_52
            else:
                acc=self.acc

            if age<self.max_retirementage:
                pension=pension*self.palkkakerroin+acc*wage
            else:
                pension=pension*self.palkkakerroin
        else: # 2 - ei karttumaa
            pension=pension*self.elakeindeksi
            
        return pension        
        
    def comp_npv(self):
        '''
        lasketaan montako timestep:iä (diskontattuna) max_age:n jälkeen henkilö on vanhuuseläkkeellä 
        hyvin yksinkertainen toteutus. Tulos on odotettu lukumäärä timestep:jä
        
        npv <- diskontattu
        npv0 <- ei ole diskontattu
        '''
        npv=0
        npv0=0
        npv_pension=0

        cpsum=1
        cpsum0=1
        cpsum_pension=1
        for x in np.arange(100,self.max_age,-self.timestep):
            intx=int(np.floor(x))
            m=self.mort_intensity[intx]
            cpsum=m*1+(1-m)*(1+self.gamma*cpsum)
            cpsum0=m*1+(1-m)*(1+cpsum0)
            cpsum_pension=m*1+(1-m)*(1+cpsum_pension*self.elakeindeksi)
        npv=cpsum
        npv0=cpsum0
        npv_pension=cpsum_pension
            
        if self.plotdebug:
            print('npv: {}'.format(npv))

        return npv,npv0,npv_pension
        
    def comp_npv_orig(self):
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
        '''
        logarithmic utility based on the employment state, age and net income
        '''
        # kappa tells how much person values free-time
        kappa_fulltime=0.75
        kappa_parttime=0.10
        kappa_retirement=0.40
        mu_age=58+(self.min_retirementage-63.5)
        
        if age>mu_age:
            mu=0.01 #45
            kappa_fulltime += mu*max(0,min(6,age-mu_age))
            kappa_parttime += mu*max(0,min(6,age-mu_age))
            #kappa_fulltime *= (1+mu*max(0,min(68,age)-mu_age))
            #kappa_parttime *= (1+mu*max(0,min(68,age)-mu_age))
        #elif age<40:
            #kappa_fulltime += 0.015*min(6,40-age)
            #kappa_parttime += 0.010*min(6,40-age)
        
        if employment_state == 1:
            u=np.log(income)-kappa_fulltime
            equ=income*np.exp(-kappa_fulltime)
        elif employment_state == 2:
            u=np.log(income)+kappa_retirement
            equ=income*np.exp(kappa_retirement)
        elif employment_state == 3:
            u=np.log(income)-kappa_parttime
            equ=income*np.exp(-kappa_parttime)
        else:
            u=np.log(income)
            equ=income
            
        return u/10,equ
   
    # wage process reparametrized
    def wage_process(self,w,age,a0=3300*12,a1=3300*12,r=1):
        '''
        Palkkaprosessi muokattu lähteestä Määttänen, 2013 
        palkka vuositasolla
        '''
        sigma=self.wage_sigma # 0.05
        eps=np.random.normal(loc=0,scale=sigma,size=1)[0]
        c1=0.89
        if w<self.min_salary:
            w=self.min_salary

         # pidetään keskiarvo/a1 samana kuin w/a0
        wt=a1*np.exp(c1*np.log(w/a0)+eps-0.5*sigma*sigma)

        # täysiaikainen vuositulo vähintään self.min_salary
        wt=np.maximum(self.min_salary,r*wt)

        return wt
                
    def get_wage_raw(self,age,wage,state):
        '''
        The used salary function
        '''
        a0=self.palkat_ika[max(0,age-1-self.min_age)]
        a1=self.palkat_ika[age-self.min_age]
        if state==0:
            r=self.unemp_wageshock
        else:
            r=1.0
            
        return self.wage_process(wage,age,a0,a1,r=r)
                    
    def wage_process_cumulative(self,w_cum,w_old,age,state=1):
        '''
        wage process cumulative function
        Palkkaprosessin kertymäfunktio 
        palkka vuositasolla
        
        w_cum  cumulative probability until w_cum
        w_old  vanha palkka
        age    ika
        state  employment state
        '''
        
        a0=self.palkat_ika[max(0,age-1-self.min_age)]
        a1=self.palkat_ika[age-self.min_age]
           
        if state==0:
           factor=self.unemp_wageshock
        else:
           factor=1.0
                
        sigma=self.wage_sigma
        c1=0.89
            
        wt=max(factor*a1*np.exp(c1*np.log(max(w_old,self.min_salary)/a0)-0.5*sigma*sigma),self.min_salary)
        c=lognorm.cdf(np.maximum(w_cum,self.min_salary)/wt,sigma,loc=0,scale=1)

        return c
        
    def wage_process_pdf(self,x,w_old,age,state=1):
        '''
        wage process pdf function
        Palkkaprosessin tiheysfunktio 
        palkka vuositasolla
        
        w_cum  cumulative probability until w_cum
        w_old  vanha palkka
        age    ika
        state  employment state
        '''
        
        a0=self.palkat_ika[max(0,age-1-self.min_age)]
        a1=self.palkat_ika[age-self.min_age]
           
        if state==0:
           factor=self.unemp_wageshock
        else:
           factor=1.0
                
        sigma=self.wage_sigma
        c1=0.89

        wt=np.maximum(factor*a1*np.exp(c1*np.log(max(w_old,self.min_salary)/a0)-0.5*sigma*sigma),self.min_salary) 
        cw=np.maximum(x,self.min_salary)/wt
        c=lognorm.pdf(cw,sigma,loc=0,scale=1)/wt

        return c
        
    def wage_process_map(self,pp,w_old,age,state=1):
        '''
        wage process map function
        Palkkaprosessin kertymäfunktio 
        palkka vuositasolla
        
        w_cum  cumulative probability until w_cum
        w_old  vanha palkka
        age    ika
        state  employment state
        '''
        
        a0=self.palkat_ika[max(0,age-1-self.min_age)]
        a1=self.palkat_ika[age-self.min_age]
           
        if state==0:
           factor=self.unemp_wageshock
        else:
           factor=1.0
                
        sigma=self.wage_sigma
        c1=0.89
        
        wt=max(factor*a1*np.exp(c1*np.log(max(w_old,self.min_salary)/a0)-0.5*sigma*sigma),self.min_salary)
        c=lognorm.ppf(pp,sigma,loc=0,scale=1)*wt

        return c

    def wage_process_mean(self,w_old,age,state=1):
        '''
        wage process mean function
        palkka vuositasolla
        
        w_cum  cumulative probability until w_cum
        w_old  vanha palkka
        age    ika
        state  employment state
        '''
        
        a0=self.palkat_ika[max(0,age-1-self.min_age)]
        a1=self.palkat_ika[age-self.min_age]
           
        if state==0:
           factor=self.unemp_wageshock
        else:
           factor=1.0
                
        sigma=self.wage_sigma
        c1=0.89
        
        wt=max(factor*a1*np.exp(c1*np.log(max(w_old,self.min_salary)/a0)-0.5*sigma*sigma),self.min_salary)
        c=lognorm.mean(sigma,loc=0,scale=1)*wt

        return c
    
    def wage_process_expect(self,lb,ub,w_old,age,state=1):
        '''
        wage process mean function
        palkka vuositasolla
        
        w_cum  cumulative probability until w_cum
        w_old  vanha palkka
        age    ika
        state  employment state
        '''
        
        a0=self.palkat_ika[max(0,age-1-self.min_age)]
        a1=self.palkat_ika[age-self.min_age]
           
        if state==0:
           factor=self.unemp_wageshock
        else:
           factor=1.0
                
        sigma=self.wage_sigma
        c1=0.89
        
        wt=max(factor*a1*np.exp(c1*np.log(max(w_old,self.min_salary)/a0)-0.5*sigma*sigma),self.min_salary)
        #c=lognorm(sigma,loc=0,scale=1).expect(lambda x: x, lb=lb/wt,ub=ub/wt)*wt
        c=lognorm(sigma,loc=0,scale=wt).expect(lambda x: x, lb=lb,ub=ub)

        return c
    
    def wage_process_simple(self,w,age,ave=3300*12):
        '''
        minimal wage process
        '''
        return w

    def setup_salaries(self):
        self.min_salary=1000
        self.palkat_ika_miehet=12.5*np.array([2339.01,2339.01,2339.01,2489.09,2571.40,2632.58,2718.03,2774.21,2884.89,2987.55,3072.40,3198.48,3283.81,3336.51,3437.30,3483.45,3576.67,3623.00,3731.27,3809.58,3853.66,3995.90,4006.16,4028.60,4104.72,4181.51,4134.13,4157.54,4217.15,4165.21,4141.23,4172.14,4121.26,4127.43,4134.00,4093.10,4065.53,4063.17,4085.31,4071.25,4026.50,4031.17,4047.32,4026.96,4028.39,4163.14,4266.42,4488.40,4201.40,4252.15,4443.96,3316.92,3536.03,3536.03,3536.03])
        self.palkat_ika_naiset=12.5*np.array([2223.96,2223.96,2223.96,2257.10,2284.57,2365.57,2443.64,2548.35,2648.06,2712.89,2768.83,2831.99,2896.76,2946.37,2963.84,2993.79,3040.83,3090.43,3142.91,3159.91,3226.95,3272.29,3270.97,3297.32,3333.42,3362.99,3381.84,3342.78,3345.25,3360.21,3324.67,3322.28,3326.72,3326.06,3314.82,3303.73,3302.65,3246.03,3244.65,3248.04,3223.94,3211.96,3167.00,3156.29,3175.23,3228.67,3388.39,3457.17,3400.23,3293.52,2967.68,2702.05,2528.84,2528.84,2528.84])
        if False:
            self.palkat_ika=0.5*(self.palkat_ika_naiset+self.palkat_ika_miehet)*0+3000*12
        else:
            self.palkat_ika=0.5*(self.palkat_ika_naiset+self.palkat_ika_miehet)

    def compute_salary(self,initial_salary=None,initial_age=None):
        if initial_salary is not None:
            if initial_age is not None:
                loc=initial_salary*self.palkat_ika[0]/self.palkat_ika[initial_age-self.min_age]
                #loc=initial_salary*self.palkat_ika_miehet[0]/self.palkat_ika_miehet[initial_age-self.min_age]
            else:
                loc=initial_salary
        else:
            #loc=self.palkat_ika_miehet[0]
            loc=self.palkat_ika[0]

        if initial_age is not None:
            minage=initial_age
        else:
            minage=self.min_age
            
        #s1=self.palkat_ika_miehet[minage-self.min_age]/5
        s1=self.palkat_ika[minage-self.min_age]/5
        self.salary[minage-1]=np.maximum(self.min_salary,np.random.normal(loc=loc,scale=s1,size=1)[0]) # e/y
        a1=self.palkat_ika[0]
        #a1=self.palkat_ika_miehet[0]
        #print('age {} sal {}'.format(minage-1,self.salary[minage-1]))
        for age in range(minage,self.max_age+1):
            a0=a1
            #a1=self.palkat_ika_miehet[age-self.min_age]
            a1=self.palkat_ika[age-self.min_age]
            self.salary[age]=self.wage_process(self.salary[age-1],age,a0,a1)
            #print('age {} sal {}'.format(age,self.salary[age]))
        
    def scale_pension(self,pension,age,debug=False,emp=1,time_in_state=0):
        '''
        scaling of pensions that takes into account elinaikakerroin and postponing pension
        '''
        if emp==0:
            return self.elinaikakerroin*pension*self.elakeindeksi #*(1+0.048*max(0,age-self.min_retirementage-time_in_state))
        else:
            return self.elinaikakerroin*pension*self.elakeindeksi*(1+0.048*max(0,age-self.min_retirementage))

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def step(self, action, randomness=True, dynprog=False):
        '''
        Open AI interfacen mukainen step-funktio, joka tekee askeleen eteenpäin
        toiminnon action mukaan
        
        Tässä versiossa vain kolme tilaa: töissä, työtön ja vanhuuseläkkeellä
        '''
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        employment_status,pension,old_wage,age,time_in_state,wage=self.state_decode(self.state)
        intage=int(np.round(age))
        
        # here age is age at the beginning of the period
        
        action=int(action)
        next_age=int(np.round(age+self.timestep))
        
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
        elif employment_status == 0:
            if action == 0 or (action==2 and age<self.min_retirementage):
                employment_status = 0 # unchanged
                pension=self.pension_accrual(age,old_wage,pension,state=employment_status,time_in_state=time_in_state)
                netto=self.comp_benefits(0,old_wage,0,employment_status,time_in_state,age)
                time_in_state+=self.timestep
                if not dynprog:
                    next_wage=self.get_wage_raw(next_age,wage,employment_status)
                else:
                    #wage=old_wage
                    next_wage=0
            elif action == 1: # 
                employment_status  = 1 # switch to fulltime work
                time_in_state=0
                pension=self.pension_accrual(age,wage,pension,state=employment_status,time_in_state=time_in_state)
                netto=self.comp_benefits(wage,0,0,employment_status,time_in_state,age)
                time_in_state+=self.timestep
                if not dynprog:
                    next_wage=self.get_wage_raw(next_age,wage,employment_status)
                else:
                    next_wage=0
            elif action == 3: # 
                employment_status  = 3 # switch to parttime work
                time_in_state=0
                pension=self.pension_accrual(age,self.parttime_income*wage,pension,state=employment_status,time_in_state=time_in_state)
                netto=self.comp_benefits(self.parttime_income*wage,0,0,employment_status,time_in_state,age)
                time_in_state+=self.timestep
                if not dynprog:
                    next_wage=self.get_wage_raw(next_age,wage,employment_status)
                else:
                    next_wage=0
            elif action == 2:
                if age>=self.min_retirementage: # ve
                    pension=self.scale_pension(pension,age,emp=0,time_in_state=time_in_state)
                    employment_status  = 2 
                    pension=self.ben.laske_kokonaiselake(age,pension/12,include_kansanelake=True,include_takuuelake=True)*12
                    time_in_state=0
                    netto=self.comp_benefits(0,0,pension,employment_status,0,age)
                    time_in_state+=self.timestep
                    next_wage=0
                else:
                    print('error 99')
            else:
                print('error 17')
        elif employment_status == 1: # töissä
            if action == 0 or (action==2 and age<self.min_retirementage):
                employment_status  = 1 # unchanged
                pension=self.pension_accrual(age,wage,pension,state=employment_status)
                netto=self.comp_benefits(wage,0,0,employment_status,time_in_state,age)
                time_in_state+=self.timestep
                if not dynprog:
                    next_wage=self.get_wage_raw(next_age,wage,employment_status)
                else:
                    next_wage=0
            elif action == 1: # työttömäksi
                employment_status = 0 # switch to unemployment
                time_in_state=0
                # pension accrual based on the old salary
                pension=self.pension_accrual(age,old_wage,pension,state=employment_status,time_in_state=time_in_state)
                # income is based on the old salary for 1 year, then independent of it
                netto=self.comp_benefits(0,old_wage,0,employment_status,time_in_state,age)
                time_in_state+=self.timestep
                if not dynprog:
                    next_wage=self.get_wage_raw(next_age,wage,employment_status)
                else:
                    next_wage=0
            elif action == 3: # 
                employment_status = 3 # switch to parttime work
                time_in_state=0
                pension=self.pension_accrual(age,self.parttime_income*wage,pension,state=employment_status,time_in_state=time_in_state)
                netto=self.comp_benefits(self.parttime_income*wage,0,0,employment_status,time_in_state,age)
                time_in_state+=self.timestep
                if not dynprog:
                    next_wage=self.get_wage_raw(next_age,wage,employment_status)
                else:
                    next_wage=0
            elif action==2:
                if age>=self.min_retirementage: # ve
                    employment_status = 2
                    pension=self.scale_pension(pension,age,emp=1)
                    pension=self.ben.laske_kokonaiselake(age,pension/12,include_kansanelake=True,include_takuuelake=True)*12
                    time_in_state=0
                    netto=self.comp_benefits(0,0,pension,employment_status,0,age)
                    time_in_state+=self.timestep
                    next_wage=0
                else:
                    print('error 13')
            else:
                print('error 12')
        elif employment_status == 3:
            if action == 0 or (action==2 and age<self.min_retirementage):
                employment_status  = 3 # unchanged
                pension=self.pension_accrual(age,self.parttime_income*wage,pension,state=3)
                netto=self.comp_benefits(self.parttime_income*wage,0,0,employment_status,time_in_state,age)
                time_in_state+=self.timestep
                if not dynprog:
                    next_wage=self.get_wage_raw(next_age,wage,employment_status)
                else:
                    next_wage=0
            elif action == 1: # työttömäksi
                employment_status = 0 # switch
                time_in_state=0
                pension=self.pension_accrual(age,self.parttime_income*old_wage,pension,state=employment_status,time_in_state=time_in_state)
                netto=self.comp_benefits(0,self.parttime_income*old_wage,0,employment_status,time_in_state,age)
                time_in_state+=self.timestep
                if not dynprog:
                    next_wage=self.get_wage_raw(next_age,wage,employment_status)
                else:
                    next_wage=0
            elif action == 3: # 
                employment_status  = 1 # switch to fulltime
                time_in_state=0
                pension=self.pension_accrual(age,wage,pension,state=employment_status)
                netto=self.comp_benefits(wage,0,0,employment_status,time_in_state,age)
                time_in_state+=self.timestep
                if not dynprog:
                    next_wage=self.get_wage_raw(next_age,wage,employment_status)
                else:
                    next_wage=0
            elif action==2:
                if age>=self.min_retirementage: # ve
                    employment_status  = 2 # unchanged
                    pension=self.scale_pension(pension,age,emp=3)
                    pension=self.ben.laske_kokonaiselake(age,pension/12,include_kansanelake=True,include_takuuelake=True)*12
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
            netto=self.comp_benefits(0,0,pension,employment_status,0,age)
            time_in_state+=self.timestep
            next_wage=0
        else:
            print('Unknown employment_status {s} of type {t}'.format(s=employment_status,t=type(employment_status)))
        
        done = age >= self.max_age
        done = bool(done)
        
        if dynprog: # only for fitting dynprog
            if not done:
                if self.statedebug:
                    if employment_status in set([1,2]):
                        reward,equivalent = self.log_utility(netto,int(employment_status),age)
                        self.state = self.state_encode(employment_status,pension,wage,next_age,time_in_state,next_wage)
                    else:
                        reward = 0.0
                        equivalent=0.0
                        netto = 1.0e-20
                        self.state = self.state_encode(employment_status,pension,wage,next_age,time_in_state,next_wage)
                else:
                    reward,equivalent = self.log_utility(netto,int(employment_status),age)
                    self.state = self.state_encode(employment_status,pension,wage,next_age,time_in_state,next_wage)
            else:
                if age>self.max_age+0.001:
                    self.steps_beyond_done = self.steps_beyond_done+1
                    reward = 0.0
                    equivalent=0.0
                    netto = 1.0e-20
                    self.state = self.state_encode(employment_status,pension,wage,next_age,time_in_state,next_wage)
                else:
                    self.steps_beyond_done = 0
                    if employment_status==2:
                        reward,equivalent = self.log_utility(netto,2,age)
                        reward = self.npv*reward
                        equivalent=self.npv*equivalent
                    else:
                        reward = 0
                        equivalent=0.0
                        netto=1.0e-20
                        
                    netto=netto*self.npv_pension
                    self.state = self.state_encode(employment_status,pension,wage,next_age,time_in_state,next_wage)
        else:
            if not done:
                if self.statedebug:
                    if employment_status in set([1,2]):
                        reward,equivalent = self.log_utility(netto,int(employment_status),age)
                        self.state = self.state_encode(employment_status,pension,wage,next_age,time_in_state,next_wage)
                    else:
                        reward = 0.0
                        equivalent=0.0
                        netto = 1.0e-20
                        self.state = self.state_encode(employment_status,pension,wage,next_age,time_in_state,next_wage)
                else:
                    reward,equivalent = self.log_utility(netto,int(employment_status),age)
                    self.state = self.state_encode(employment_status,pension,wage,next_age,time_in_state,next_wage)
            elif self.steps_beyond_done is None:
                self.steps_beyond_done = 0
                if employment_status == 2 and age<self.max_age+0.001: #self.timestep:
                    reward,equivalent = self.log_utility(netto,2,age)
                    reward = self.npv*reward
                    equivalent=self.npv*equivalent
                else:
                    reward = 0.0
                    equivalent =0.0
                    netto=1.0e-20

                #benq=self.scale_q(npv,npv0,npv_pension,benq)                
                netto=netto*self.npv_pension

                self.state = self.state_encode(employment_status,pension,wage,next_age,time_in_state,next_wage)
            else:
                self.state = self.state_encode(employment_status,pension,wage,next_age,time_in_state,next_wage)
                if self.steps_beyond_done == 0:
                    logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
                self.steps_beyond_done += 1
                reward = 0.0
                equivalent=0.0
                netto=1.0e-20
                
        if self.plotdebug:
            self.render(done=done,reward=reward, netto=netto)
            
        return np.array(self.state), reward, done, {'netto': netto, 'r': reward, 'eq': equivalent}

    def render(self, mode='human', close=False, done=False, reward=None, netto=None,state=None,pred_r=None,pred_V=None):
        '''
        Tulostus-rutiini
        '''
        if state is None:
            emp,pension,wage,age,time_in_state,nextwage=self.state_decode(self.state)
        else:
            emp,pension,wage,age,time_in_state,nextwage=self.state_decode(state)
            
        s='Tila {} palkka {:.2f} ikä {} t-i-s {} elake {:.2f} seur.palkka {:.2f}'.format(emp,wage,age,time_in_state,pension,nextwage)
        if reward is not None:
            s+=f' r {reward:.4f}'
            if pred_r is not None:
                de=reward-pred_r
                s+=f' pr {pred_r:.4f} d {de:.4f}'
        else:
            if pred_r is not None:
                s+=f' r {pred_r:.4f}'

        if pred_V is not None:
            s+=f' pV {pred_V:.4f}'

        if netto is not None:
            s+=f' n {netto:.2f}'

        print(s)
        
        if done:
            print('-------------------------------------------------------------------------------------------------------')

    def state_encode(self,emp,pension,old_wage,age,time_in_state,nextwage):
        '''
        Encode the state for neural nets
        '''
        d=np.zeros(self.n_empl+8)
        states=self.n_empl
        if emp==1:
            d[0:states]=np.array([0,1,0,0])
        elif emp==0:
            d[0:states]=np.array([1,0,0,0])
        elif emp==2:
            d[0:states]=np.array([0,0,1,0])
        elif emp==3:
            d[0:states]=np.array([0,0,0,1])
        else:
            print('state error '+str(emp))
            
        d[states]=(pension-20_000)/20_000 # vastainen/alkanut eläke
        #d[states]=(pension-40_000)/40_000 # vastainen/alkanut eläke
        d[states+1]=(old_wage-35_000)/35_000
        d[states+2]=(age-(self.max_age+self.min_age)/2)/((self.max_age+self.min_age)/2)
        d[states+3]=(time_in_state-20)/20
        d[states+4]=(nextwage-35_000)/35_000
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
        '''
        Decode the state
        '''
        emp=-1
        for k in range(self.n_empl):
            if vec[k]>0:
                emp=k
                break
                
        if emp<0:
            print('state error '+str(vec))
            
        pension=vec[self.n_empl]*20_000+20_000
        #pension=vec[self.n_empl]*40_000+40_000
        wage=vec[self.n_empl+1]*35_000+35_000
        age=int(np.round(vec[self.n_empl+2]*((self.max_age+self.min_age)/2)+(self.max_age+self.min_age)/2))
        time_in_state=vec[self.n_empl+3]*20+20
        nextwage=vec[self.n_empl+4]*35_000+35_000
                
        return int(emp),pension,wage,age,time_in_state,nextwage

    def unit_test_code_decode(self):
        '''
        Test the state encoding and decoding
        '''
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
        '''
        Check state
        '''
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
        '''
        Set limits on states
        '''
        self.low = np.array([
            0,
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
        print(f'max_retirementage {self.max_retirementage}\nmortality {self.include_mort}')
        print(f'ansiopvraha_kesto {self.ansiopvraha_kesto}\nTrain {self.train}')
        print(f'reset_exploration_go {self.reset_exploration_go}\nreset_exploration_ratio {self.reset_exploration_ratio}\nplotdebug {self.plotdebug}')
        print(f'basic income {self.perustulo}')

    def reset(self,init=None,debug=False,ini_age=None,pension=None,ini_wage=None,ini_old_wage=None):
        '''
        Open AI-interfacen mukainen reset-funktio, joka nollaa laskennan alkutilaan
        
        parametrit dynaamista ohjelmointia varten
        pl. debug
        '''
                    
        #employment_status=0

        #employment_status=random.choices(np.array([0,1,3],dtype=int),weights=[0.56*0.626,0.374,0.44*0.626])[0]
        employment_status=random.choices(np.array([0,1],dtype=int),weights=[0.626,0.374])[0]
        
        if debug:
            age=self.min_age
            initial_age=age
            time_in_state=random.choices(np.array([0,1],dtype=int),weights=[0.40,0.60])[0] # 60% tm-tuella
            initial_salary=np.random.uniform(low=1_000,high=70_000)
            pension=np.random.uniform(low=0,high=40_000)
        else:
            if ini_age is not None:
                age=ini_age
            else:
                #age=int(self.min_age)
                age=int(self.startage)
                
            time_in_state=random.choices(np.array([0,1],dtype=int),weights=[0.40,0.60])[0] # 60% tm-tuella
            if pension is None:
                pension=0
                
        # set up salary for the entire career
        initial_salary=None
        initial_age=None
        if self.reset_exploration_go and self.train and self.reset_exploration_ratio>np.random.uniform():
            #print('exploration')
            time_in_state=random.choices(np.array([0,1,2,3,4,5],dtype=int),weights=[0.10,0.35,0.20,0.15,0.10,0.05])[0] # 60% tm-tuella
            #employment_status=random.choices(np.array([0,1,3],dtype=int),weights=[0.4,0.4,0.2])[0]
            if random.random()<0.5:
                age=int(np.random.uniform(low=self.min_age,high=self.max_age-5))
            else:
                age=int(np.random.uniform(low=58,high=self.max_age-2))
            if age<self.min_retirementage:
                if self.include_pt:
                    employment_status=random.choices(np.array([0,1,3],dtype=int),weights=[0.4,0.4,0.2])[0]
                else:
                    employment_status=random.choices(np.array([0,1],dtype=int),weights=[0.4,0.4])[0]
            else:
                if self.include_pt:
                    employment_status=random.choices(np.array([0,1,2,3],dtype=int),weights=[0.3,0.3,0.3,0.1])[0]
                else:
                    employment_status=random.choices(np.array([0,1,2],dtype=int),weights=[0.4,0.3,0.3])[0]
            
            initial_salary=np.random.uniform(low=1_000,high=80_000)
            pension=np.random.uniform(low=0,high=50_000)
            initial_age=age
            #print('Explore: age {} initial {} pension {}'.format(age,initial_salary,pension))

        self.compute_salary(initial_salary=initial_salary,initial_age=initial_age)
        if ini_old_wage is None:
            old_wage=self.salary[age-1] # timestep == 1
        else:
            old_wage=ini_old_wage
            
        if ini_wage is None:
            wage=self.salary[age] # timestep == 1
        else:
            wage=ini_wage
        
        #print(initial_salary,old_wage,wage)

        self.state = self.state_encode(employment_status,pension,old_wage,age,time_in_state,wage)
        self.steps_beyond_done = None
        
        if self.plotdebug: # or debug:
            self.render()
        
        return np.array(self.state)

    def set_state(self,employment_status,pension,old_wage,age,time_in_state,wage):
        '''
        Sets the internal state
        '''
        s0=self.env.state_encode(employment_status,pension,old_wage,age,time_in_state,wage)
        self.state=s0
