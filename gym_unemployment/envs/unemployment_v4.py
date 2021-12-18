"""

    unemployment_v4


    Gym module implementing the Finnish social security including earnings-related components,
    e.g., the unemployment benefit
    
    - sisältää täyden tuen lapsille
    - mukana myös puolisot (as first-class citizens)
    - arvonlisävero toteutettu
    - full scale kotitaloudet mukana
    
    Spouse:
    - has same the group modified by gender
    - has the same age
    - has the same children
    - has independent wage and all other state parameters
    
"""

import math
import gym
from gym import spaces, logger, utils, error
from gym.utils import seeding   
import numpy as np
import fin_benefits
import random
from . rates import Rates
from scipy.interpolate import interp1d

# class StayDict(dict):
#     '''
#     Apuluokka, jonka avulla tehdään virheenkorjausta 
#     '''
#     def __missing__(self, key):
#         return 'Unknown state '+key


class UnemploymentLargeEnv_v4(gym.Env):
    """
    Description:
        The Finnish Unemployment Pension Scheme 

    Source:
        This environment corresponds to the environment of the Finnish Social Security

    Observation: 
        Type: Box(17)  UPDATE! Sama puolisolle
        Num    Observation                Min           Max
        0    Employment status             0             12
        1    Ryhmä                         0              6
        2    time-in-state                 0              2
        3    Accrued old-age pension       0            inf
        4    Paid old-age pension          0            inf
        5    Salary                        0            inf
        6    Age                          25             69
        7    Työssä-olo-Ehto               0          28/12
        8    Työuran kesto                 0             50
        9    Työstä pois (aika)            0            100 OK?
       10    Irtisanottu                   0              1
       11    Käytetty työttömyyskorvausaika
       12    Palkka-alennus
       13    Aika työttömänä ve-iässä
       14    TOE-ajan palkka (tuleva)
       15    Ansiosid. tt-etuuden perustepalkka
       16    OVE maksussa
       ..
       spouse ..
       ..
       5x    Preferenssikohina         

    Employment states:
        Type: Int
        Num     State
        0   Earnings-related Unemployed
        1   Employed
        2   Retired
        3   Disabled
        4   Extended earnings-related unemployment
        5   Mother's leave
        6   Father's leave
        7   Support for taking care of children at home
        8   Retired working part-time
        9   Retired worling full-time   
        10  Part-time work
        11  Outside of work-force
        12  Student or in the army 
        13  Earnings-unrelated Unemployment (Työmarkkinatuki)
        14  Dead

    Actions:
        These really depend on the state (see function step)
    
        Type: Discrete(4)
        Num    Action
        0    Stay in the current state
        1    Switch to the other state (work -> unemployed; unemployed -> work)
        2    Retire if >=min_retirementage
        3    Some other transition
        4     foo

    Reward:
        Reward is the sum of wage and benefit for every step taken, including the termination step

    Starting State:
        Starting state in unemployed at age 18

    Step:
        Each step corresponds to three months in time

    Episode Termination:
        Age 70
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self,**kwargs):
        '''
        Alkurutiinit
        '''
        super().__init__()

        self.dis_ratio=0
        self.nnn=0
    
        self.additive_logutil=True # if log utility is log(net1+net2) [False] or log(net1)+log(net2) [True]
    
        self.ansiopvraha_toe=0.5 # = 6kk
        self.karenssi_kesto=0.25 #0.25 # = 3kk
        self.isyysvapaa_kesto=0.25 # = 3kk
        self.aitiysvapaa_kesto=0.75 # = 9kk ml vanhempainvapaa
        self.aitiysvapaa_pois=0.06/0.25
        self.min_tyottputki_ika=61 # vuotta. Ikä, jonka täytyttyä pääsee putkeen
        self.tyohistoria_tyottputki=5 # vuotta. vähimmäistyöura putkeenpääsylle
        self.kht_kesto=2.0 # kotihoidontuen kesto 2 v
        self.tyohistoria_vaatimus=3.0 # 3 vuotta
        self.tyohistoria_vaatimus500=5.0 # 5 vuotta
        self.ansiopvraha_kesto400=400 # päivää
        self.ansiopvraha_kesto300=300 # päivää
        self.ansiopvraha_kesto500=500 # päivää
        self.minage_500=58 # minimi-ikä 500 päivälle
        #self.min_salary=1000 # julkaistut laskelmat olettavat tämän
        self.min_salary=1000 # julkaistujen laskelmien jälkeen
        
        self.map_stays={0: self.stay_unemployed,  1: self.stay_employed,         2: self.stay_retired,       3: self.stay_disabled,
                       4: self.stay_pipeline,    5: self.stay_motherleave,      6: self.stay_fatherleave,   7: self.stay_khh,
                       8: self.stay_oa_parttime, 9: self.stay_oa_fulltime,     10: self.stay_parttime,     11: self.stay_outsider,
                       12: self.stay_student,   13: self.stay_tyomarkkinatuki}


        self.timestep=0.25
        self.max_age=71
        self.min_age=18
        self.min_retirementage=63.5 #65
        self.max_retirementage=68 # 70

        #self.elinaikakerroin=0.925 # etk:n arvio 1962 syntyneille
        self.elinaikakerroin=0.96344 # vuoden 2017 kuolleisuutta käytettäessä myös elinaikakerroin on sieltä
        
        self.reaalinen_palkkojenkasvu=1.016
        
        # exploration does not really work here due to the use of history
        self.reset_exploration_go=False
        self.reset_exploration_ratio=0.4
        
        self.train=False

        self.include_spouses=True # Puolisot mukana?
        self.include_mort=True # onko kuolleisuus mukana laskelmissa
        self.include_npv_mort=False # onko kuolleisuus mukana laskelmissa
        self.include_preferencenoise=False # onko työllisyyspreferenssissä hajonta mukana 
        self.perustulo=False # onko Kelan perustulo laskelmissa
        self.universalcredit=False # Yleistuki
        self.randomness=True # onko stokastiikka mukana
        self.mortstop=True # pysäytä kuolleisuuden jälkeen
        self.include_putki=True # työttömyysputki mukana
        self.include_pinkslip=True # irtisanomiset mukana
        self.use_sigma_reduction=True # kumpi palkkareduktio
        self.include_kansanelake=True
        self.include_takuuelake=True
        self.preferencenoise_std=0.1
        
        self.additional_income_tax=0
        self.additional_income_tax_high=0
        self.additional_tyel_premium=0
        self.additional_kunnallisvero=0
        self.scale_tyel_accrual=False
        self.scale_additional_tyel_accrual=0
        self.scale_additional_unemp_benefit=0
        self.include_halftoe=True
        self.porrasta_toe=False
        self.porrastus500=False
        self.include_ove=False
        
        self.unemp_limit_reemp=True # työttömästä työlliseksi tn, jos hakee töitä
        prob_3m=0.5
        prob_1y=1-(1-prob_3m)**(1./0.25)
        self.unemp_reemp_prob=1-(1-prob_1y)**self.timestep # kolmessa kuukaudessa
        
        gamma=0.92
        
        # etuuksien laskentavuosi
        self.year=2018
        
        # OVE-parametrit
        self.ove_ratio=0.5
        self.min_ove_age=61

        self.plotdebug=False # tulostetaanko rivi riviltä tiloja

        # sets parameters based on kwargs
        self.set_parameters(**kwargs)
        
        if self.perustulo:
            self.ben = fin_benefits.BasicIncomeBenefits(**kwargs)
        elif self.universalcredit:
            self.ben = fin_benefits.UniversalCreditIncomeBenefits(**kwargs)
        else:
            #self.ben = fin_benefits.CyBenefits(**kwargs)
            self.ben = fin_benefits.Benefits(**kwargs)
             
        self.gamma=gamma**self.timestep # discounting
        self.palkkakerroin=(0.8*1+0.2*1.0/self.reaalinen_palkkojenkasvu)**self.timestep
        self.elakeindeksi=(0.2*1+0.8*1.0/self.reaalinen_palkkojenkasvu)**self.timestep
        self.kelaindeksi=(1.0/self.reaalinen_palkkojenkasvu)**self.timestep
        self.n_age = self.max_age-self.min_age+1
        self.inv_timestep=int(np.round(1/self.timestep)) # pitäisi olla kokonaisluku
        self.n_time = int(np.round((self.n_age-1)*self.inv_timestep))+2

        # karttumaprosentit
        if self.scale_tyel_accrual:
            acc_scaling=1+self.scale_additional_tyel_accrual
        else:
            acc_scaling=1
        
        self.acc=0.015*self.timestep*acc_scaling
        self.acc_over_52=0.019*self.timestep*acc_scaling
        #self.acc_over_52=self.acc
        self.acc_family=1.15*self.acc
        self.acc_family_over_52=1.15*self.acc_over_52
        self.acc_unemp=0.75*self.acc
        self.acc_unemp_over_52=0.75*self.acc_over_52
        #self.min_family_accwage=12*757

        self.log_utility_default_params()

        self.n_age=self.max_age-self.min_age+1

        # male low income, male mid, male high, female low, female mid, female high income
        self.n_groups=6

        # käytetäänkö exp/log-muunnosta tiloissa vai ei?
        self.log_transform=False
        self.eps=1e-20

        self.set_year(self.year)

        self.set_state_limits()
        if self.include_mort: # and not self.mortstop:
            if self.include_mort and self.mortstop:
                print('Mortality included, stopped')
            else:
                print('Mortality included, not stopped')

            self.n_empl=15 # state of employment, 0,1,2,3,4
        else:
            print('No mortality included')
            self.n_empl=15 # state of employment, 0,1,2,3,4
            
        self.n_spouseempl=self.n_empl
            
        self.setup_state_encoding()

        self.n_actions=5 # valittavien toimenpiteiden määrä, kasvata viiteen ja korjaa siirtymät 4 & 5
        self.n_spouse_actions=3 #5 # stay, switch, retire

        self.action_space = spaces.MultiDiscrete([self.n_actions,self.n_spouse_actions])
        #self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        
        if self.use_sigma_reduction:
            self.update_wage_reduction=self.update_wage_reduction_sigma
        else:
            self.update_wage_reduction=self.update_wage_reduction_baseline

        #self.seed()
        self.viewer = None
        self.state = None
        
        inflation_raw=np.array([1.0,1.011,1.010,1.009,1.01,1.01,1.01]) # 2018 2019 2020 2021 2022 2023
        self.inflation=np.cumprod(inflation_raw)
        
        self.steps_beyond_done = None
        
        # normitetaan työttömyysjaksot askelvälin monikerroiksi
        scale=21.5*12
        self.apvkesto300=np.round(self.ansiopvraha_kesto300/scale/self.timestep)*self.timestep
        self.apvkesto400=np.round(self.ansiopvraha_kesto400/scale/self.timestep)*self.timestep
        self.apvkesto500=np.round(self.ansiopvraha_kesto500/scale/self.timestep)*self.timestep
                
        self.init_inforate()
        
        self.explain()
        
        if self.plotdebug:
            self.unit_test_code_decode()
            
    def set_annual_params(self,year):
        inflation_raw=np.array([1.0,1.011,1.010,1.009,1.01,1.01,1.01]) # 2018 2019 2020 2021 2022 2023
        self.inflation=np.cumprod(inflation_raw)
        self.inflationfactor=self.inflation[year-2018]
        
        if self.porrasta_toe:
            self.max_toe=35/12
            self.min_toewage=844*12*self.inflationfactor # vuoden 2019 luku tilanteessa, jossa ei tessiä
            self.min_halftoewage=422*12*self.inflationfactor # vuoden 2019 luku tilanteessa, jossa ei tessiä
        else:
            self.max_toe=28/12
            self.min_toewage=1211*12*self.inflationfactor # vuoden 2019 luku tilanteessa, jossa ei tessiä
            self.min_halftoewage=800*12*self.inflationfactor # vuoden 2019 luku tilanteessa, jossa ei tessiä
        self.setup_unempdays_left(porrastus=self.porrasta_toe)

        self.accbasis_kht=719.0*12*self.inflationfactor
        self.accbasis_tmtuki=0 # 1413.75*12
        self.disabbasis_tmtuki=1413.75*12*self.inflationfactor
        
        self.opiskelija_asumismenot_toimeentulo=250*self.inflationfactor
        self.opiskelija_asumismenot_asumistuki=250*self.inflationfactor
        self.elakelainen_asumismenot_toimeentulo=200*self.inflationfactor
        self.elakelainen_asumismenot_asumistuki=200*self.inflationfactor
        self.elakelainen_asumismenot_puoliso=250*self.inflationfactor
        self.muu_asumismenot_toimeentulo=320*self.inflationfactor # per hlö, ehkä 500 e olisi realistisempi, mutta se tuottaa suuren asumistukimenon
        self.muu_asumismenot_asumistuki=320*self.inflationfactor
        self.muu_asumismenot_lapsi=100*self.inflationfactor
        self.muu_asumismenot_puoliso=250*self.inflationfactor
            
    def set_year(self,year):
        self.year=year
        self.set_annual_params(year)
        self.ben.set_year(year)
        self.rates=Rates(year=self.year,max_age=self.max_age,n_groups=self.n_groups,timestep=self.timestep,inv_timestep=self.inv_timestep)
        self.palkat_ika_miehet,self.palkat_ika_naiset,self.g_r=self.rates.setup_salaries_v3(self.min_retirementage)
        self.get_wage=self.get_wage_step        
        self.get_spousewage=self.get_spousewage_step
        
        #self.disability_intensity=self.rates.get_eff_disab_rate()
        self.disability_intensity=self.rates.get_eff_disab_rate_v2()
        self.pinkslip_intensity=self.rates.get_pinkslip_rate()*self.timestep
        self.birth_intensity=self.rates.get_birth_rate(symmetric=True)
        self.mort_intensity=self.rates.get_mort_rate()
        self.student_inrate,self.student_outrate=self.rates.get_student_rate_v2() # myös armeijassa olevat tässä
        self.outsider_inrate,self.outsider_outrate=self.rates.get_outsider_rate()
        self.divorce_rate=self.rates.get_divorce_rate()
        self.marriage_rate=self.rates.get_marriage_rate()

        self.npv,self.npv0,self.npv_pension=self.comp_npv()
        self.initial_weights=self.get_initial_weights()
        
    def get_n_states(self):
        '''
        returns number of the employment state & number of actions
        '''
        return self.n_empl,self.n_actions
        
    def get_lc_version(self):
        '''
        returns the version of life-cycle model's episodestate used
        '''
        return 4
        
    def test_comp_npv(self):
        npv,npv0,cpsum_pension=self.comp_npv()
        
        n=10000
        snpv=np.zeros((6,n))
        snpv0=np.zeros((6,n))
        scpsum_pension=np.zeros((6,n))
        
        for g in range(6):
            for k in range(n):
                snpv[g,k],snpv0[g,k],scpsum_pension[g,k]=self.comp_npv_simulation(g)
                
        print(npv)
        for g in range(6):
            print('{}: {}'.format(g,np.mean(snpv[g,:])))

    def comp_npv(self):
        '''
        lasketaan montako timestep:iä (diskontattuna) max_age:n jälkeen henkilö on vanhuuseläkkeellä 
        hyvin yksinkertainen toteutus. Tulos on odotettu lukumäärä timestep:jä
        
        npv <- diskontattu
        npv0 <- ei ole diskontattu
        '''
        npv=np.zeros(self.n_groups)
        npv0=np.zeros(self.n_groups)
        npv_pension=np.zeros(self.n_groups)

        for g in range(self.n_groups):
            cpsum=1
            cpsum0=1
            cpsum_pension=1
            for x in np.arange(100,self.max_age,-self.timestep):
                intx=int(np.floor(x))
                m=self.mort_intensity[intx,g]
                cpsum=m*1+(1-m)*(1+self.gamma*cpsum)
                cpsum0=m*1+(1-m)*(1+cpsum0)
                cpsum_pension=m*1+(1-m)*(1+cpsum_pension*self.gamma*self.elakeindeksi)  # self.gamma??
            npv[g]=cpsum
            npv0[g]=cpsum0
            npv_pension[g]=cpsum_pension
            #print(g,npv0)
            
        if self.plotdebug:
            print('npv: {}'.format(npv))

        return npv,npv0,npv_pension

    def comp_npv_simulation(self,g):
        '''
        simuloidaan npv jokaiselle erikseen montako timestep:iä (diskontattuna) max_age:n jälkeen henkilö on vanhuuseläkkeellä 
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
        alive=True
        num=int(np.ceil(100-self.max_age+2)/self.timestep)
        sattuma = np.random.uniform(size=num)
        x=self.max_age+self.timestep
        k=0
        while alive and x<100+self.timestep:
            intx=int(np.floor(x))
            if sattuma[k]>self.mort_intensity[intx,g]:
                cpsum=1+self.gamma*cpsum
                cpsum0=1+cpsum0
                cpsum_pension=1+cpsum_pension*self.elakeindeksi
            else:
                alive=False
            k=k+1
            x=x+self.timestep
                
        npv=cpsum
        npv0=cpsum0
        npv_pension=cpsum_pension
            
        if self.plotdebug:
            print('npv: {}'.format(npv))

        return npv,npv0,cpsum_pension

#     def comp_npv_simulation_v2(self,g,mortality=True):
#         '''
#         simuloidaan npv jokaiselle erikseen montako timestep:iä (diskontattuna) max_age:n jälkeen henkilö on vanhuuseläkkeellä 
#         hyvin yksinkertainen toteutus. Tulos on odotettu lukumäärä timestep:jä
#         
#         npv <- diskontattu
#         npv0 <- ei ole diskontattu
#         '''
#         npv=0
#         npv0=0
#         npv_pension=0
# 
#         if mortality:
#             cpsum=1
#             cpsum0=1
#             cpsum_pension=1
#             alive=True
#             num=int(np.ceil(100-self.max_age+2)/self.timestep)
#             sattuma = np.random.uniform(size=num)
#             x=self.max_age+self.timestep
#             k=0
#             while alive and x<100+self.timestep:
#                 intx=int(np.floor(x))
#                 if sattuma[k]>self.mort_intensity[intx,g]:
#                     cpsum=1+self.gamma*cpsum
#                     cpsum0=1+cpsum0
#                     cpsum_pension=1+cpsum_pension*self.gamma*self.elakeindeksi
#                 else:
#                     alive=False
#                 k=k+1
#                 x=x+self.timestep
#                 
#             npv=cpsum
#             npv0=cpsum0
#             npv_pension=cpsum_pension
#         else:
#             cpsum=1
#             cpsum0=1
#             cpsum_pension=1
#             alive=True
#             num=int(np.ceil(100-self.max_age+2)/self.timestep)
#             sattuma = np.random.uniform(size=num)
#             x=self.max_age+self.timestep
#             k=0
#                 if self.include_npv_mort:
#                     npv,npv0,npv_pension=self.comp_npv_simulation(g)
#                     reward,equivalent = self.log_utility(netto,employment_status,age,pinkslip=0)
#                     reward*=npv
#                 else:
#                     npv,npv0,npv_pension=self.npv[g],self.npv0[g],self.npv_pension[g]
#                     reward,equivalent = self.log_utility(netto,employment_status,age,pinkslip=0)
#                     reward*=self.npv[g]
#             
#             while alive and x<100+self.timestep:
#                 reward,equivalent = self.log_utility(netto,employment_status,age,pinkslip=0)
#                 intx=int(np.floor(x))
#                 m=self.mort_intensity[intx,g]
#                 cpsum=m*1+(1-m)*(1+self.gamma*cpsum)
#                 cpsum0=m*1+(1-m)*(1+cpsum0)
#                 cpsum_pension=m*1+(1-m)*(1+cpsum_pension*self.gamma*self.elakeindeksi)  # self.gamma??
# 
#                 intx=int(np.floor(x))
#                 if sattuma[k]>self.mort_intensity[intx,g]:
#                     cpsum=1+self.gamma*cpsum
#                     cpsum0=1+cpsum0
#                     cpsum_pension=1+cpsum_pension*self.gamma*self.elakeindeksi
#                 else:
#                     alive=False
#                 k=k+1
#                 x=x+self.timestep
#                 
#             npv=cpsum
#             npv0=cpsum0
#             npv_pension=cpsum_pension
#         
#             
#         if self.plotdebug:
#             print('npv: {}'.format(npv))
# 
#         return npv,npv0,cpsum_pension

    def setup_benefits(self,wage,old_wage,kansanelake,tyoelake,employment_state,time_in_state,ika,children_under3,children_under7,children_under18,puoliso=0,
                      irtisanottu=0,karenssia_jaljella=0,alku='omat_',p2=None,puolisoalku='puoliso_'):
        if p2 is not None:
            p=p2.copy()
        else:
            p={}
            
        if self.perustulo:
            p[alku+'perustulo']=1
        else:
            p[alku+'perustulo']=0
            
        p[alku+'saa_elatustukea']=1 # vain yksinhuoltaja
            
        # tässä ei alku+, koska lapset yhteisiä
        p['lapsia']=children_under18
        p['lapsia_paivahoidossa']=children_under7
        p['lapsia_alle_kouluikaisia']=children_under7
        p['lapsia_alle_3v']=children_under3
        
        p[alku+'opiskelija']=0
        p[alku+'elakkeella']=0
        p[alku+'toimeentulotuki_vahennys']=0
        p[alku+'ika']=ika
        p[alku+'tyoton']=0
        p[alku+'saa_ansiopaivarahaa']=0
        p[alku+'vakiintunutpalkka']=0
        
        p['veromalli']=0
        p['kuntaryhma']=3
        p['lapsia_kotihoidontuella']=0

        p[alku+'tyottomyyden_kesto']=0
        p[alku+'isyysvapaalla']=0
        p[alku+'aitiysvapaalla']=0
        p[alku+'kotihoidontuella']=0
        p[alku+'tyoelake']=0
        p[alku+'kansanelake']=0
        p[alku+'elakemaksussa']=0
        p[alku+'elakkeella']=0
        p[alku+'sairauspaivarahalla']=0
        p[alku+'disabled']=0
        
        
        if employment_state==1:
            p[alku+'tyoton']=0 # voisi olla työtön siinä mielessä, että oikeutettu soviteltuun päivärahaan
            p[alku+'t']=wage/12
            p[alku+'vakiintunutpalkka']=wage/12
            p[alku+'saa_ansiopaivarahaa']=0
            p[alku+'tyoelake']=tyoelake/12 # ove
        elif employment_state==0: # työtön, ansiopäivärahalla
            if ika<65:
                #self.render()
                p[alku+'tyoton']=1
                p[alku+'t']=0
                p[alku+'vakiintunutpalkka']=old_wage/12
                p[alku+'saa_ansiopaivarahaa']=1
                p[alku+'tyottomyyden_kesto']=12*21.5*time_in_state
                    
                if irtisanottu<1 and karenssia_jaljella>0:
                    p[alku+'saa_ansiopaivarahaa']=0
                    p[alku+'tyoton']=0
                    
                p[alku+'tyoelake']=tyoelake/12 # ove
            else:
                p[alku+'tyoton']=0 # ei oikeutta työttömyysturvaan
                p[alku+'t']=0
                p[alku+'vakiintunutpalkka']=0
                p[alku+'saa_ansiopaivarahaa']=0
                p[alku+'tyoelake']=tyoelake/12 # ove
        elif employment_state==13: # työmarkkinatuki
            if ika<65:
                p[alku+'tyoton']=1
                p[alku+'t']=0
                p[alku+'vakiintunutpalkka']=0
                p[alku+'tyottomyyden_kesto']=12*21.5*time_in_state
                p[alku+'saa_ansiopaivarahaa']=0
                p[alku+'tyoelake']=tyoelake/12 # ove
            else:
                p[alku+'tyoton']=0 # ei oikeutta työttömyysturvaan
                p[alku+'t']=0
                p[alku+'vakiintunutpalkka']=0
                p[alku+'saa_ansiopaivarahaa']=0
                p[alku+'tyoelake']=tyoelake/12 # ove
        elif employment_state==3: # tk
            p[alku+'t']=0
            p[alku+'elakkeella']=1 
            p[alku+'tyoelake']=tyoelake/12 # ove
            p[alku+'kansanelake']=kansanelake/12
            p[alku+'disabled']=1
        elif employment_state==4: # työttömyysputki
            if ika<65:
                p[alku+'tyoton']=1
                p[alku+'t']=0
                p[alku+'vakiintunutpalkka']=old_wage/12
                p[alku+'saa_ansiopaivarahaa']=1
                p[alku+'tyottomyyden_kesto']=12*21.5*time_in_state
                p[alku+'tyoelake']=tyoelake/12 # ove
            else:
                p[alku+'tyoton']=0 # ei oikeutta työttömyysturvaan
                p[alku+'t']=0
                p[alku+'vakiintunutpalkka']=0
                p[alku+'saa_ansiopaivarahaa']=0
                p[alku+'tyoelake']=tyoelake/12 # ove
        elif employment_state==5: # ansiosidonnainen vanhempainvapaa, äidit
            p[alku+'aitiysvapaalla']=1
            p[alku+'aitiysvapaa_kesto']=0
            p[alku+'t']=0
            p[alku+'vakiintunutpalkka']=old_wage/12
            p[alku+'saa_ansiopaivarahaa']=1
        elif employment_state==6: # ansiosidonnainen vanhempainvapaa, isät
            p[alku+'isyysvapaalla']=1
            p[alku+'t']=0
            p[alku+'vakiintunutpalkka']=old_wage/12
            p[alku+'saa_ansiopaivarahaa']=1
        elif employment_state==7: # hoitovapaa
            p[alku+'kotihoidontuella']=1
            p[alku+'lapsia_paivahoidossa']=0
            p[alku+'lapsia_kotihoidontuella']=children_under7
            p[alku+'kotihoidontuki_kesto']=time_in_state
            p[alku+'t']=0
            p[alku+'vakiintunutpalkka']=old_wage/12
        elif employment_state==2: # vanhuuseläke
            if ika>=self.min_retirementage:
                p[alku+'t']=0
                p[alku+'elakkeella']=1  
                p[alku+'tyoelake']=tyoelake/12
                p[alku+'kansanelake']=kansanelake/12
            else:
                p[alku+'t']=0
                p[alku+'elakkeella']=0
                p[alku+'tyoelake']=0
        elif employment_state in set([8,9]): # ve+osatyö
            p[alku+'t']=wage/12
            p[alku+'elakkeella']=1  
            p[alku+'tyoelake']=tyoelake/12
            p[alku+'kansanelake']=kansanelake/12
        elif employment_state==10: # osa-aikatyö
            p[alku+'t']=wage/12
            p[alku+'tyoelake']=tyoelake/12 # ove
        elif employment_state==11: # työelämän ulkopuolella
            p[alku+'toimeentulotuki_vahennys']=0 # oletetaan että ei kieltäytynyt työstä
            p[alku+'t']=0
            p[alku+'tyoelake']=tyoelake/12 # ove
        elif employment_state==12: # opiskelija
            p[alku+'opiskelija']=1
            p[alku+'t']=0
            p[alku+'tyoelake']=tyoelake/12 # ove
        elif employment_state==14: # kuollut
            p[alku+'t']=0
        else:
            print('Unknown employment_state ',employment_state)
        
        p[alku+'elake_maksussa']=p[alku+'tyoelake']+p[alku+'kansanelake']
            
        if employment_state==12: # opiskelija
            p['asumismenot_toimeentulo']=self.opiskelija_asumismenot_toimeentulo
            p['asumismenot_asumistuki']=self.opiskelija_asumismenot_asumistuki
        elif employment_state in set([2,8,9]): # eläkeläinen
            p['asumismenot_toimeentulo']=self.elakelainen_asumismenot_toimeentulo+puoliso*self.elakelainen_asumismenot_puoliso
            p['asumismenot_asumistuki']=self.elakelainen_asumismenot_asumistuki+puoliso*self.elakelainen_asumismenot_puoliso
        else: # muu
            p['asumismenot_toimeentulo']=self.muu_asumismenot_toimeentulo+puoliso*self.muu_asumismenot_puoliso+p['lapsia']*self.muu_asumismenot_lapsi
            p['asumismenot_asumistuki']=self.muu_asumismenot_asumistuki+puoliso*self.muu_asumismenot_puoliso+p['lapsia']*self.muu_asumismenot_lapsi

        p['ansiopvrahan_suojaosa']=1
        p['ansiopvraha_lapsikorotus']=1

        if puoliso>0:
            p['aikuisia']=2
        else:
            p['aikuisia']=1
            p[puolisoalku+'opiskelija']=0
            p[puolisoalku+'elakkeella']=0
            p[puolisoalku+'toimeentulotuki_vahennys']=0
            p[puolisoalku+'ika']=ika
            p[puolisoalku+'tyoton']=0
            p[puolisoalku+'saa_ansiopaivarahaa']=0
            p[puolisoalku+'vakiintunutpalkka']=0
            p[puolisoalku+'tyottomyyden_kesto']=0
            p[puolisoalku+'isyysvapaalla']=0
            p[puolisoalku+'aitiysvapaalla']=0
            p[puolisoalku+'kotihoidontuella']=0
            p[puolisoalku+'tyoelake']=0
            p[puolisoalku+'elakkeella']=0
            p[puolisoalku+'sairauspaivarahalla']=0
            p[puolisoalku+'disabled']=0
            p[puolisoalku+'t']=0
            
        return p    

    def comp_benefits(self,wage,old_wage,kansanelake,tyoelake,employment_state,time_in_state,children_under3,children_under7,children_under18,ika,
                      puoliso,puoliso_tila,puoliso_palkka,puoliso_kansanelake,puoliso_tyoelake,puoliso_old_wage,puoliso_time_in_state,
                      irtisanottu=0,puoliso_irtisanottu=0,karenssia_jaljella=0,puoliso_karenssia_jaljella=0,
                      retq=True,ove=False):
        '''
        Kutsuu fin_benefits-modulia, jonka avulla lasketaan etuudet ja huomioidaan verotus
        Laske etuuksien arvo, kun 
            wage on palkka
            old_wage on vanha palkka
            pension on maksettavan eläkkeen määrä
            employment_state on töissä olo (0)/työttömyys (1)/eläkkeellä olo (2) jne.
            time_in_state on kesto tilassa
            ika on henkilön ikä
        '''
            
        p=self.setup_benefits(wage,old_wage,kansanelake,tyoelake,employment_state,time_in_state,ika,
            children_under3,children_under7,children_under18,puoliso=0,
            irtisanottu=irtisanottu,karenssia_jaljella=karenssia_jaljella,alku='')
        
        #print(p)
        
        if puoliso>0: # pariskunta
            p=self.setup_benefits(puoliso_palkka,puoliso_old_wage,puoliso_kansanelake,puoliso_tyoelake,puoliso_tila,puoliso_time_in_state,ika,
                      children_under3,children_under7,children_under18,puoliso=1,
                      irtisanottu=puoliso_irtisanottu,karenssia_jaljella=puoliso_karenssia_jaljella,
                      alku='puoliso_',p2=p)

            p['aikuisia']=2
            
            netto,benefitq=self.ben.laske_tulot_v2(p,include_takuuelake=self.include_takuuelake)
            benefitq['netto']=netto
            netto=netto*12
            
            if benefitq['omat_netto']<1e-6 or benefitq['puoliso_netto']<1e-6:
                #print(f'omat netto {employment_state}: ',benefitq['omat_netto'])
                #print(f'puoliso netto {puoliso_tila}: ',benefitq['puoliso_netto'])
                if benefitq['omat_netto']<1e-6:
                    benefitq['omat_netto']=1.0
                    if benefitq['puoliso_netto']>2.0:
                        benefitq['puoliso_netto']-=1.0
                else:
                    benefitq['puoliso_netto']=1.0
                    if benefitq['omat_netto']>2.0:
                        benefitq['omat_netto']-=1.0
                        
            netto_omat=benefitq['omat_netto']*12
            netto_puoliso=benefitq['puoliso_netto']*12
        else: # ei pariskunta
            netto1,benefitq1=self.ben.laske_tulot_v2(p,include_takuuelake=self.include_takuuelake)
            
            # lapset 0 tässä, yksinkertaistus
            p['lapsia']=0
            p['lapsia_paivahoidossa']=0
            p['lapsia_alle_kouluikaisia']=0
            p['lapsia_alle_3v']=0
            p2=self.setup_benefits(puoliso_palkka,puoliso_old_wage,puoliso_kansanelake,puoliso_tyoelake,puoliso_tila,puoliso_time_in_state,ika,
                      0,0,0,puoliso=0,
                      irtisanottu=puoliso_irtisanottu,karenssia_jaljella=puoliso_karenssia_jaljella,alku='',p2=p)
            netto2,benefitq2=self.ben.laske_tulot_v2(p2,include_takuuelake=self.include_takuuelake,omat='puoliso_',puoliso='omat_') # switch order
            netto=netto1+netto2
            
            if netto1<1:
                print(f'netto<1, omat tila {employment_state}',wage,old_wage,kansanelake,tyoelake,time_in_state,ika,children_under3,children_under7,children_under18)
                print(benefitq1)
            if netto2<1 and puoliso:
                print(f'netto<1, spouse {puoliso_tila}',puoliso_palkka,puoliso_old_wage,puoliso_kansanelake,puoliso_tyoelake,puoliso_time_in_state,ika)
                print(benefitq2)

            if benefitq1['omat_netto']<1e-6 or benefitq2['puoliso_netto']<1e-6:
                print(f'omat netto {employment_state}: ',benefitq1['omat_netto'])
                print(f'puoliso netto {puoliso_tila}: ',benefitq2['puoliso_netto'])

            netto_omat=benefitq1['omat_netto']*12
            netto_puoliso=benefitq2['puoliso_netto']*12

            benefitq=self.ben.add_q(benefitq1,benefitq2)
            benefitq['netto']=netto
            netto=netto*12
            
            #print('netto',benefitq['netto'],netto,netto_omat,netto_puoliso)
            
            #print('omat',benefitq1)
            #print('puoliso',benefitq2)
            #print('yhteensä',benefitq)

        if retq:
            return netto,benefitq,netto_omat,netto_puoliso
        else:
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

    def get_initial_weights(self):
        '''
        Lähtöarvot 18-vuotiaana
        '''
        group_weights=np.zeros(3)
        group_weights=[0.3,0.5,0.2]

        def get_wees(w0,w1,intensity):
            return w0*intensity,w1*intensity,get_w(w0,w1)*intensity
        
        def get_w(a0,a1):
            return (1-a0*group_weights[0]-a1*group_weights[1])/group_weights[2]
        
        initial_weight=np.zeros((6,7))
        if self.year==2018:
            # tilat [13,0,1,10,3,11,12] siis [tmtuki,ansiosidonnainen,kokoaikatyö,osaaikatyö,työvoimanulkopuolella,opiskelija]
            # lasketaan painotetut kertoimet eri tulotasoille
            # työttömät miehet, osuus väestöstä
            tyoton=0.019
            m1,m2,m3=get_wees(1.27,1.0,tyoton)
            # työttömät naiset, osuus väestöstä
            tyoton=0.016
            w1,w2,w3=get_wees(1.2,1.0,tyoton)
            # opiskelijat+työlliset+muut työvoiman ulkop+työttömät, miehet
            om=0.686+0.257+0.029+0.019 # miehet töissä + opiskelija
            om1,om2,om3=get_wees(1.25,1.0,0.257)
            # opiskelijat+työlliset+muut työvoiman ulkop+työttömät, naiset
            ow=0.587+0.360+0.027+0.017
            ow1,ow2,ow3=get_wees(1.2,1.0,0.340)
            # työvoiman ulkopuolella, miehet
            tyovoimanulkop=0.029
            um1,um2,um3=get_wees(1.3,1.0,tyovoimanulkop)
            # työvoiman ulkopuolella, naiset
            tyovoimanulkop=0.0267
            uw1,uw2,uw3=get_wees(1.2,1.0,tyovoimanulkop)
            # työkyvyttömät, miehet
            md=0.009
            md1,md2,md3=get_wees(1.3,1.0,md)
            # työkyvyttömät, naiset
            wd=0.00730
            wd1,wd2,wd3=get_wees(1.2,1.0,wd)

            initial_weight[0,:]=[m1*4/5,m1*1/5,0.68*om1,0.32*om1,md1,um1,om-om1-um1-m1-md1]
            initial_weight[1,:]=[m2*4/5,m2*1/5,0.68*om2,0.32*om2,md2,um2,om-om2-um2-m2-md2]
            initial_weight[2,:]=[m3*4/5,m3*1/5,0.68*om3,0.32*om3,md3,um3,om-om3-um3-m3-md3]
            initial_weight[3,:]=[w1*4/5,w1*1/5,0.44*ow1,0.56*ow1,wd1,uw1,ow-ow1-uw1-w1-wd1]
            initial_weight[4,:]=[w2*4/5,w2*1/5,0.44*ow2,0.56*ow2,wd2,uw2,ow-ow2-uw2-w2-wd2]
            initial_weight[5,:]=[w3*4/5,w3*1/5,0.44*ow3,0.56*ow3,wd3,uw3,ow-ow3-uw3-w3-wd3]
        elif self.year==2019 or self.year==2020 or self.year==2021:
            # tilat [13,0,1,10,3,11,12]

            tyoton=0.021
            m1,m2,m3=get_wees(1.27,1.0,tyoton)
            tyoton=0.017
            w1,w2,w3=get_wees(1.2,1.0,tyoton)
            om=0.679+0.261+0.030+0.021
            om1,om2,om3=get_wees(1.25,1.0,0.261)
            ow=0.587+0.360+0.027+0.017
            ow1,ow2,ow3=get_wees(1.2,1.0,0.360)
            tyovoimanulkop=0.030
            um1,um2,um3=get_wees(1.3,1.0,tyovoimanulkop)
            tyovoimanulkop=0.027
            uw1,uw2,uw3=get_wees(1.2,1.0,tyovoimanulkop)
            md=0.009 # 0.010315
            md1,md2,md3=get_wees(1.3,1.0,md)
            wd=0.006 # 0.0071558
            wd1,wd2,wd3=get_wees(1.2,1.0,wd)

            initial_weight[0,:]=[m1*4/5,m1*1/5,0.68*om1,0.32*om1,md1,um1,om-om1-um1-m1-md1]
            initial_weight[1,:]=[m2*4/5,m2*1/5,0.68*om2,0.32*om2,md2,um2,om-om2-um2-m2-md2]
            initial_weight[2,:]=[m3*4/5,m3*1/5,0.68*om3,0.32*om3,md3,um3,om-om3-um3-m3-md3]
            initial_weight[3,:]=[w1*4/5,w1*1/5,0.44*ow1,0.56*ow1,wd1,uw1,ow-ow1-uw1-w1-wd1]
            initial_weight[4,:]=[w2*4/5,w2*1/5,0.44*ow2,0.56*ow2,wd2,uw2,ow-ow2-uw2-w2-wd2]
            initial_weight[5,:]=[w3*4/5,w3*1/5,0.44*ow3,0.56*ow3,wd3,uw3,ow-ow3-uw3-w3-wd3]
        
        else:
            error(999)
            
        for k in range(6):
            scale=np.sum(initial_weight[k,:])
            initial_weight[k,:] /= scale
            
        return initial_weight

    def scale_pension(self,pension,age,scale=True,unemp_after_ra=0,elinaikakerroin=True):
        '''
        Elinaikakertoimen ja lykkäyskorotuksen huomiointi
        '''
        if elinaikakerroin:
            eak=self.elinaikakerroin
        else:
            eak=1
        
        if scale:
            return eak*pension*self.elakeindeksi*(1+0.048*(age-self.min_retirementage-unemp_after_ra)) 
        else:
            return eak*pension*self.elakeindeksi
        
    def move_to_parttime(self,wage,pension,old_wage,age,tyoura,time_in_state):
        '''
        Siirtymä osa-aikaiseen työskentelyyn
        '''
        employment_status = 10 # switch to part-time work
        parttimewage=0.5*wage
        tyoura += self.timestep
        time_in_state=0
        old_wage=0
        pension=self.pension_accrual(age,parttimewage,pension,state=employment_status)
        pinkslip=0
        time_in_state=self.timestep

        return employment_status,pension,wage,time_in_state,tyoura,pinkslip

    def move_to_work(self,wage,pension,old_wage,age,time_in_state,tyoura,pinkslip):
        '''
        Siirtymä täysiaikaiseen työskentelyyn
        '''
        employment_status = 1 # töihin
        pinkslip=0
        time_in_state=0
        old_wage=0
        tyoura+=self.timestep
        pension=self.pension_accrual(age,wage,pension,state=employment_status)
        time_in_state=self.timestep

        return employment_status,pension,wage,time_in_state,tyoura,pinkslip

    def move_to_oa_fulltime(self,wage,pension,old_wage,age,kansanelake,tyoelake,employment_status,
            unemp_after_ra,all_acc=True,scale_acc=True):
        '''
        Siirtymä vanhuuseläkkeelle
        '''
        if age>=self.max_retirementage: # move to retirement state 2
            if employment_status in set([2,8,9]): # ve, ve+työ, ve+osatyö
                # ei lykkäyskorotusta
                tyoelake = self.elakeindeksi*tyoelake+self.scale_pension(pension,age,scale=False,unemp_after_ra=unemp_after_ra)
                if self.include_kansanelake:
                    kansanelake = kansanelake * self.kelaindeksi
                else:
                    kansanelake = 0
                paid_pension = tyoelake + kansanelake
                pension=0
                employment_status = 2
            elif employment_status==3: # tk
                # do nothing
                employment_status=3
                tyoelake = self.elakeindeksi*tyoelake+self.scale_pension(pension,age,scale=False,unemp_after_ra=unemp_after_ra)
                if self.include_kansanelake:
                    kansanelake = kansanelake * self.kelaindeksi
                else:
                    kansanelake = 0
                paid_pension = tyoelake + kansanelake
                pension=0
            else: # ei vielä eläkkeellä
                # lykkäyskorotus
                tyoelake = tyoelake*self.elakeindeksi + pension
                if self.include_kansanelake:
                    if spouse:
                        kansanelake = self.ben.laske_kansanelake(age,tyoelake*self.elinaikakerroin/12,0)*12 # ben-modulissa palkat kk-tasolla
                    else:
                        kansanelake = self.ben.laske_kansanelake(age,tyoelake*self.elinaikakerroin/12,1)*12 # ben-modulissa palkat kk-tasolla
                else:
                    kansanelake = 0
                # lykkäys ei vähennä kansaneläkettä
                tyoelake += (self.scale_pension(pension,age,scale=scale_acc,unemp_after_ra=unemp_after_ra) - pension)
                paid_pension = tyoelake + kansanelake
                pension=0
                employment_status = 2
                
            time_in_state=self.timestep
            alkanut_ansiosidonnainen=0
            ove_paid=0
        elif age>=self.min_retirementage:
            if all_acc:
                if employment_status in set([2,8,9]): # ve, ve+työ, ve+osatyö
                    tyoelake = self.tyoelake*paid_pension
                    kansanelake = kansanelake * self.kelaindeksi
                    employment_status = 9
                elif employment_status==3: # tk
                    # do nothing
                    employment_status=3
                    tyoelake = self.elakeindeksi*tyoelake
                    kansanelake = kansanelake * self.kelaindeksi
                    paid_pension = tyoelake + kansanelake
                else:
                    # lykkäyskorotus
                    tyoelake = tyoelake*self.elakeindeksi + pension
                    if self.include_kansanelake:
                        if spouse:
                            kansanelake = self.ben.laske_kansanelake(age,tyoelake*self.elinaikakerroin/12,0)*12 # ben-modulissa palkat kk-tasolla
                        else:
                            kansanelake = self.ben.laske_kansanelake(age,tyoelake*self.elinaikakerroin/12,1)*12 # ben-modulissa palkat kk-tasolla
                    else:
                        kansanelake = 0
                    # lykkäys ei vähennä kansaneläkettä
                    tyoelake += (self.scale_pension(pension,age,scale=scale_acc,unemp_after_ra=unemp_after_ra) - pension)
                    paid_pension = tyoelake + kansanelake
                    pension=0
                    employment_status = 9
            elif employment_status in set([2,8,9]): # ve, ve+työ, ve+osatyö
                tyoelake = self.elakeindeksi*tyoelake
                pension = pension*self.palkkakerroin
                employment_status = 9

            time_in_state=self.timestep
            alkanut_ansiosidonnainen=0
            pension=self.pension_accrual(age,wage,pension,state=employment_status)
            ove_paid=0
        else: # työvoiman ulkopuolella
            time_in_state=0
            employment_status = 2 
            wage=old_wage
            time_in_state+=self.timestep
            ove_paid=0

        return employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid

    def move_to_student(self,wage,pension,old_wage,age,time_in_state,tyoura,pinkslip):
        '''
        Siirtymä opiskelijaksi
        Tässä ei muuttujien päivityksiä, koska se tehdään jo muualla!
        '''
        employment_status = 12
        time_in_state=0
        time_in_state+=self.timestep
        pinkslip=0

        return employment_status,pension,wage,time_in_state,pinkslip
        
    def move_to_oa_parttime(self,wage,pension,old_wage,age,kansanelake,tyoelake,employment_status,
            unemp_after_ra,all_acc=True,scale_acc=True,spouse=False):
        '''
        Siirtymä vanhuuseläkkeelle
        '''
        if age>=self.max_retirementage: # move to retirement state 2
            if employment_status in set([2,8,9]): # ve, ve+työ, ve+osatyö
                # ei lykkäyskorotusta
                tyoelake = self.elakeindeksi*tyoelake+self.scale_pension(pension,age,scale=False,unemp_after_ra=unemp_after_ra)
                if self.include_kansanelake:
                    kansanelake = kansanelake * self.kelaindeksi
                else:
                    kansanelake = 0
                paid_pension = tyoelake + kansanelake
                pension=0
                employment_status = 2
            elif employment_status==3: # tk
                # do nothing
                employment_status=3
                tyoelake = self.elakeindeksi*tyoelake+self.scale_pension(pension,age,scale=False,unemp_after_ra=unemp_after_ra)
                if self.include_kansanelake:
                    kansanelake = kansanelake * self.kelaindeksi
                else:
                    kansanelake = 0
                paid_pension = tyoelake + kansanelake
                pension=0
            else: # ei vielä eläkkeellä
                # lykkäyskorotus
                tyoelake = tyoelake*self.elakeindeksi + pension
                if self.include_kansanelake:
                    if spouse:
                        kansanelake = self.ben.laske_kansanelake(age,tyoelake*self.elinaikakerroin/12,0)*12 # ben-modulissa palkat kk-tasolla
                    else:
                        kansanelake = self.ben.laske_kansanelake(age,tyoelake*self.elinaikakerroin/12,1)*12 # ben-modulissa palkat kk-tasolla
                else:
                    kansanelake = 0
                    
                # lykkäys ei vähennä kansaneläkettä
                tyoelake += (self.scale_pension(pension,age,scale=scale_acc,unemp_after_ra=unemp_after_ra) - pension)
                paid_pension = tyoelake + kansanelake
                pension=0
                employment_status = 2
                
            time_in_state=self.timestep
            alkanut_ansiosidonnainen=0
            ove_paid=0
        elif age>=self.min_retirementage:
            if all_acc:
                if employment_status in set([2,8,9]): # ve, ve+työ, ve+osatyö
                    tyoelake = self.elakeindeksi*tyoelake
                    kansanelake = kansanelake * self.kelaindeksi
                    paid_pension = tyoelake + kansanelake
                    pension=pension*self.palkkakerroin
                    employment_status = 8
                elif employment_status==3: # tk
                    # do nothing
                    employment_status=3
                    tyoelake = self.elakeindeksi*tyoelake
                    kansanelake = kansanelake * self.kelaindeksi
                    paid_pension = tyoelake + kansanelake
                    pension=pension*self.palkkakerroin
                else:
                    # lykkäyskorotus
                    tyoelake = tyoelake*self.elakeindeksi + pension
                    if self.include_kansanelake:
                        if spouse:
                            kansanelake = self.ben.laske_kansanelake(age,tyoelake*self.elinaikakerroin/12,0)*12 # ben-modulissa palkat kk-tasolla
                        else:
                            kansanelake = self.ben.laske_kansanelake(age,tyoelake*self.elinaikakerroin/12,1)*12 # ben-modulissa palkat kk-tasolla
                    else:
                        kansanelake = 0
                    # lykkäys ei vähennä kansaneläkettä
                    tyoelake += (self.scale_pension(pension,age,scale=scale_acc,unemp_after_ra=unemp_after_ra) - pension)
                    paid_pension = tyoelake + kansanelake
                    pension=0
                    employment_status = 8
            elif employment_status in set([2,8,9]): # ve, ve+työ, ve+osatyö
                tyoelake = self.elakeindeksi*tyoelake
                pension = pension*self.palkkakerroin
                employment_status = 8

            time_in_state=self.timestep
            alkanut_ansiosidonnainen=0
            ptwage=0.5*wage
            pension=self.pension_accrual(age,ptwage,pension,state=employment_status)
            ove_paid=0
        else: # työvoiman ulkopuolella
            time_in_state=0
            employment_status = 2 
            wage=old_wage
            time_in_state+=self.timestep

        return employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid

    def move_to_ove(self,employment_status,pension,tyoelake,ove_paid,age,unemp_after_ra):
        if not self.include_ove:
            return pension,tyoelake,0
    
        if ove_paid:
            print('Moving to OVE twice')
            error('failure')
            exit()
            
        if employment_status in set([2,3,8,9]): # ei eläkettä maksuun
            print('Incorrect state',employment_status)
            error('failure')
            exit()
        else:
            tyoelake = self.scale_pension(self.ove_ratio*pension,age,scale=True,unemp_after_ra=unemp_after_ra)
            kansanelake = 0
            pension=(1-self.ove_ratio)*pension # *self.palkkakerroin, tässä ei indeksoida, koska pension_accrual hoitaa tämän
            ove_paid=1

        return pension,tyoelake,ove_paid

    def move_to_retirement(self,wage,pension,old_wage,age,kansanelake,tyoelake,employment_status,unemp_after_ra,all_acc=True,scale_acc=True,spouse=0):
        '''
        Moving to retirement
        '''
        if age>=self.max_retirementage:
            if employment_status in set([2,8,9]): # ve, ve+työ, ve+osatyö
                # ei lykkäyskorotusta
                tyoelake = self.elakeindeksi*tyoelake+self.scale_pension(pension,age,scale=False,unemp_after_ra=unemp_after_ra)
                if self.include_kansanelake:
                    kansanelake = kansanelake * self.kelaindeksi
                else:
                    kansanelake = 0
                pension=0
                employment_status = 2 
            elif employment_status==3: # tk
                # do nothing
                employment_status=3
                tyoelake = self.elakeindeksi*tyoelake+self.scale_pension(pension,age,scale=False,unemp_after_ra=unemp_after_ra)
                if self.include_kansanelake:
                    kansanelake = kansanelake * self.kelaindeksi
                else:
                    kansanelake = 0
                pension=0
            else: # ei vielä eläkkeellä
                # lykkäyskorotus
                tyoelake = tyoelake*self.elakeindeksi + pension
                if self.include_kansanelake:
                    # puoliso??
                    kansanelake = self.ben.laske_kansanelake(age,tyoelake*self.elinaikakerroin/12,1-spouse)*12 # ben-modulissa palkat kk-tasolla
                else:
                    kansanelake = 0
                # lykkäys ei vähennä kansaneläkettä
                tyoelake += (self.scale_pension(pension,age,scale=scale_acc,unemp_after_ra=unemp_after_ra) - pension)
                pension=0
                employment_status = 2 
                
            time_in_state=self.timestep
            alkanut_ansiosidonnainen=0
            ove_paid=0
        elif age>=self.min_retirementage:
            if all_acc:
                if employment_status in set([2,8,9]): # ve, ve+työ, ve+osatyö
                    tyoelake = self.elakeindeksi*tyoelake
                    kansanelake = kansanelake * self.kelaindeksi
                    pension=pension*self.palkkakerroin
                    employment_status = 2 
                elif employment_status==3: # tk
                    # do nothing
                    employment_status = 3
                    pension=pension*self.palkkakerroin
                    tyoelake = self.elakeindeksi*tyoelake
                    kansanelake = kansanelake * self.kelaindeksi
                else:
                    # lykkäyskorotus
                    tyoelake = tyoelake*self.elakeindeksi + pension
                    if self.include_kansanelake:
                        kansanelake = self.ben.laske_kansanelake(age,tyoelake*self.elinaikakerroin/12,1-spouse)*12 # ben-modulissa palkat kk-tasolla
                    else:
                        kansanelake = 0
                    # lykkäys ei vähennä kansaneläkettä
                    tyoelake += (self.scale_pension(pension,age,scale=scale_acc,unemp_after_ra=unemp_after_ra) - pension)
                    pension=0
                    employment_status = 2 
            elif employment_status in set([8,9]): # ve, ve+työ, ve+osatyö
                tyoelake = self.elakeindeksi*tyoelake
                kansanelake = kansanelake * self.kelaindeksi
                pension=pension*self.palkkakerroin
                employment_status = 2 

            time_in_state=self.timestep
            alkanut_ansiosidonnainen=0
            ove_paid=0
        else: # työvoiman ulkopuolella
            time_in_state=0
            employment_status = 2 
            wage=old_wage
            time_in_state+=self.timestep

        return employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid

    def move_to_retdisab(self,wage,pension,old_wage,age,time_in_state,kansanelake,tyoelake,unemp_after_ra):   
        '''
        Siirtymä vanhuuseläkkeelle, jossa ei voi tehdä työtä
        '''
        
        if age>=self.max_retirementage:
            # ei mene täsmälleen oikein
            tyoelake = self.elakeindeksi*tyoelake+self.scale_pension(pension,age,scale=False,unemp_after_ra=unemp_after_ra)
            kansanelake = kansanelake * self.kelaindeksi
            pension=0                        
        else:
            tyoelake = self.elakeindeksi*tyoelake
            kansanelake = kansanelake * self.kelaindeksi
            pension = self.palkkakerroin*pension

        employment_status = 3
        wage=0
        time_in_state=self.timestep
        #wage_reduction=0.9
        alkanut_ansiosidonnainen=0

        return employment_status,kansanelake,tyoelake,pension,wage,time_in_state,kansanelake,tyoelake
        
    def tyossaoloehto(self,toe,tyoura,age):
        '''
        täyttyykö työssäoloehto
        '''
        if toe>=self.ansiopvraha_toe: 
            return True
        else:
            return False
                
    def setup_unempdays_left(self,porrastus=False):
        '''
        valmistaudu toen porrastukseen
        '''
        if porrastus:
            self.comp_unempdays_left=self.comp_unempdays_left_porrastus
            self.paivarahapaivia_jaljella=self.paivarahapaivia_jaljella_porrastus
            self.comp_toe_wage=self.comp_toe_wage_porrastus
        else:
            self.comp_unempdays_left=self.comp_unempdays_left_nykytila
            self.paivarahapaivia_jaljella=self.paivarahapaivia_jaljella_nykytila
            self.comp_toe_wage=self.comp_toe_wage_nykytila
    
    def comp_unempdays_left_nykytila(self,kesto,tyoura,age,toe,emp,alkanut_ansiosidonnainen,toe58,old_toe,printti=False):
        '''
        Nykytilan mukainen jäljellä olevien työttömyyspäivärahapäivien laskenta
        '''
        if emp in set([2,3,8,9]):
            return 0
    
        if emp==4:
            return min(0,65-age)

        toe_tayttyy=self.tyossaoloehto(toe,tyoura,age)

        if (not toe_tayttyy) and alkanut_ansiosidonnainen<1:
            return 0 

        if toe_tayttyy:
            kesto=0

        if tyoura>=self.tyohistoria_vaatimus500 and age>=self.minage_500 and toe58>0:
            toekesto=max(0,self.apvkesto500-kesto)
        elif tyoura>=self.tyohistoria_vaatimus:
            toekesto=max(0,self.apvkesto400-kesto)
        else:
            toekesto=max(0,self.apvkesto300-kesto)
         
        return max(0,min(toekesto,65-age))

    def paivarahapaivia_jaljella_nykytila(self,kesto,tyoura,age,toe58,toe):
        '''
        Onko työttömyyspäivärahapäiviä jäljellä?
        '''
        if age>=65:
            return False

        if ((tyoura>=self.tyohistoria_vaatimus500 and kesto>=self.apvkesto500 and age>=self.minage_500 and toe58>0) \
            or (tyoura>=self.tyohistoria_vaatimus and kesto>=self.apvkesto400 and (age<self.minage_500 or tyoura<self.tyohistoria_vaatimus500 or toe58<1)) \
            or (tyoura<self.tyohistoria_vaatimus and kesto>=self.apvkesto300)):    
            return False
        else:
            return True
            
    def comp_unempdays_left_porrastus(self,kesto,tyoura,age,toe,emp,alkanut_ansiosidonnainen,toe58,old_toe,printti=False):
        if emp in set([2,3,8,9]):
            return 0
    
        if emp==4:
            return min(0,65-age)
        
        if (not self.tyossaoloehto(toe,tyoura,age)) and alkanut_ansiosidonnainen<1:
            return 0 

        if self.tyossaoloehto(toe,tyoura,age):
            kesto=0
            
        if tyoura>=self.tyohistoria_vaatimus500 and age>=self.minage_500 and toe58>0 and not self.porrastus500:
            ret=max(0,self.apvkesto500-kesto)
        else:
            scale=21.5*12
            if self.porrastus500:
                t2=np.floor(old_toe*2)/2
                toekesto_raw=min(500,max(0,t2-0.5)*200+200)/scale
            else:
                if old_toe>=20/12:
                    toekesto_raw=400/scale
                elif old_toe>=15/12:
                    toekesto_raw=300/scale
                elif old_toe>=10/12:
                    toekesto_raw=200/scale
                elif old_toe>=5/12:
                    toekesto_raw=100/scale
                else:
                    toekesto_raw=0
                #toekesto_raw=max(0,min(toe,21/12)-0.5)*20/(21.5)+100/scale
                            
            toekesto=np.round(toekesto_raw/self.timestep)*self.timestep
                
            ret=max(0,toekesto-kesto)
            
        return max(0,min(ret,65-age))

    def toe_porrastus_kesto(self,kesto,toe,tyoura):
        if toe<0.5:
            return False

        scale=21.5*12
        if self.porrastus500:
            t2=np.floor(toe*2)/2
            toekesto_raw=min(500,max(0,t2-0.5)*200+200)/scale
        else:
            if toe>=20/12:
                toekesto_raw=400/scale
            elif toe>=15/12:
                toekesto_raw=300/scale
            elif toe>=10/12:
                toekesto_raw=200/scale
            elif toe>=5/12:
                toekesto_raw=100/scale
            else:
                toekesto_raw=0
            #toekesto_raw=max(0,min(toe,21/12)-0.5)*20/(21.5)+100/scale
            
        toekesto=np.round(toekesto_raw/self.timestep)*self.timestep
            
        if kesto<toekesto:
            return True
        else:
            return False
        
    def paivarahapaivia_jaljella_porrastus(self,kesto,tyoura,age,toe58,toe):
        if age>=65:
            return False

        if (tyoura>=self.tyohistoria_vaatimus500 and kesto>=self.apvkesto500 and age>=self.minage_500 and toe58>0 and not self.porrastus500) \
            or ((not self.toe_porrastus_kesto(kesto,toe,tyoura)) and (age<self.minage_500 or tyoura<self.tyohistoria_vaatimus500 or toe58<1)):
            return False
        else:
            return True
            
    def update_unempwage_basis(self,unempwage_basis,unempwage,use80percent):
        '''
        Tähän 80% sääntö (jos edellisestä uudelleenmäärittelystä alle vuosi, 80% suojaosa)
        '''
        if use80percent:
            return max(unempwage_basis*0.8,unempwage)
        else:
            return unempwage

    def move_to_unemp(self,wage,pension,old_wage,age,kansanelake,tyoelake,toe,toekesto,irtisanottu,tyoura,
                    used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,alkanut_ansiosidonnainen,toe58,spouse=False):
        '''
        Siirtymä työttömyysturvalle
        '''
        if age>=self.min_retirementage: # ei uusia työttömiä enää alimman ve-iän jälkeen, vanhat jatkavat
            pinkslip=0
            employment_status=0
            unempwage_basis=0
            alkanut_ansiosidonnainen = 0
            used_unemp_benefit = 0
            karenssia_jaljella=0
            
            employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                self.move_to_retirement(wage,pension,old_wage,age,kansanelake,tyoelake,employment_status,unemp_after_ra,all_acc=True,spouse=spouse)
                
            return employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
                   used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,\
                   alkanut_ansiosidonnainen,karenssia_jaljella
        else:
            #if toe>=self.ansiopvraha_toe: # täyttyykö työssäoloehto
            tehto=self.tyossaoloehto(toe,tyoura,age)
            if tehto or alkanut_ansiosidonnainen>0:
                if tehto:
                    kesto=0
                    used_unemp_benefit=0
                    self.infostate_set_enimmaisaika(age,spouse=spouse) # resetoidaan enimmäisaika
                    if self.infostat_check_aareset(age,spouse=spouse):
                        unempwage_basis=self.update_unempwage_basis(unempwage_basis,unempwage,True)
                    else:
                        unempwage_basis=self.update_unempwage_basis(unempwage_basis,unempwage,False)
                    
                    jaljella=self.paivarahapaivia_jaljella(kesto,tyoura,age,toe58,toekesto)
                else:
                    kesto=used_unemp_benefit
                    if self.porrasta_toe:
                        jaljella=self.paivarahapaivia_jaljella(kesto,tyoura,age,toe58,self.comp_oldtoe(spouse=spouse)) # FIXME! Ei toimi PUOLISO
                    else:
                        jaljella=self.paivarahapaivia_jaljella(kesto,tyoura,age,toe58,toekesto) # toe ei vaikuta
                    
                if jaljella:
                    employment_status  = 0 # siirto ansiosidonnaiselle
                    #if alkanut_ansiosidonnainen<1:
                    if irtisanottu: # or alkanut_ansiosidonnainen>0: # muuten ei oikeutta ansiopäivärahaan karenssi vuoksi
                        used_unemp_benefit+=self.timestep
                        karenssia_jaljella=0.0
                    else:
                        karenssia_jaljella=0.25 # 90 pv
                    #else:
                    #    karenssia_jaljella=0
                    #    used_unemp_benefit+=self.timestep
                    alkanut_ansiosidonnainen = 1
                else:
                    if self.include_putki and age>=self.min_tyottputki_ika and tyoura>=self.tyohistoria_tyottputki: 
                        employment_status = 4 # siirto lisäpäiville
                        used_unemp_benefit+=self.timestep
                        karenssia_jaljella=0.0
                        alkanut_ansiosidonnainen = 1
                    else:
                        employment_status = 13 # siirto työmarkkinatuelle
                        unempwage_basis=0
                        alkanut_ansiosidonnainen = 0
                        karenssia_jaljella=0
            else:
                employment_status = 13 # siirto työmarkkinatuelle
                alkanut_ansiosidonnainen = 0
                unempwage_basis=0
                if irtisanottu: # muuten ei oikeutta tm-tukeen karenssin vuoksi
                    karenssia_jaljella=0.0
                else:
                    karenssia_jaljella=0.25 # 90 pv

            time_in_state=0                
            #toe=0 #max(0,toe-self.timestep) # nollataan työssäoloehto
            
            if karenssia_jaljella>0:
                pension=pension*self.palkkakerroin
            else:
                if tyoelake>0: # ove maksussa, ei karttumaa
                    pension=self.pension_accrual(age,unempwage_basis,pension,state=employment_status,ove_paid=1)
                else:
                    pension=self.pension_accrual(age,unempwage_basis,pension,state=employment_status)

            time_in_state=self.timestep
            #karenssia_jaljella=max(0,karenssia_jaljella-self.timestep) # FIXME!
            #unemp_after_ra ei muutu
            
            # Tässä ei tehdä karenssia_jaljella -muuttujasta tilamuuttujaa, koska karenssin kesto on lyhyempi kuin aika-askeleen
            # Samoin karenssia ei ole tm-tuessa, koska toimeentulotuki on suurempi
            
            pinkslip=irtisanottu

        return employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
               used_unemp_benefit,pinkslip,unemp_after_ra,\
               unempwage_basis,alkanut_ansiosidonnainen,karenssia_jaljella

    def update_karenssi(self,karenssia_jaljella):
        karenssia_jaljella=max(0,karenssia_jaljella-self.timestep)
        return karenssia_jaljella
        

    def move_to_outsider(self,wage,pension,old_wage,age,irtisanottu):
        '''
        Siirtymä työvoiman ulkopuolelle
        '''
        employment_status = 11 # switch
        time_in_state=0
        pension=pension*self.palkkakerroin

        time_in_state+=self.timestep
        pinkslip=0

        return employment_status,pension,wage,time_in_state,pinkslip

    def move_to_disab(self,wage,pension,old_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,spouse):
        '''
        Siirtymä työkyvyttömyyseläkkeelle aktiivista, ei eläkkeeltä
        '''
        employment_status = 3 # tk
        if age<self.min_retirementage:
            wage5y=self.infostat_comp_5y_ave_wage(spouse=spouse) # spouse?? FIXME
            
            tyoelake=(tyoelake+self.elinaikakerroin*(pension+self.acc/self.timestep*wage5y*max(0,self.min_retirementage-age)))*self.elakeindeksi
            if self.include_kansanelake:
                kansanelake = self.ben.laske_kansanelake(age,tyoelake/12,1-spouse,disability=True)*12 # ben-modulissa palkat kk-tasolla
            else:
                kansanelake = 0
            
            pension=0
            alkanut_ansiosidonnainen=0
            time_in_state=0
            wage=0
            time_in_state+=self.timestep
            #wage_reduction=0.60 # vastaa määritelmää
        else:
            # siirtymä vanhuuseläkkeelle, lykkäyskorotus, ei tulevaa aikaa
            tyoelake = tyoelake*self.elakeindeksi + pension
            if self.include_kansanelake:
                kansanelake = self.ben.laske_kansanelake(age,tyoelake*self.elinaikakerroin/12,1-spouse)*12 # ben-modulissa palkat kk-tasolla
            else:
                kansanelake = 0
            # lykkäys ei vähennä kansaneläkettä
            tyoelake += (self.scale_pension(pension,age,scale=True,unemp_after_ra=unemp_after_ra) - pension)
            pension=0

            time_in_state=self.timestep
            alkanut_ansiosidonnainen=0
            employment_status = 3
            ove_paid=0
            #wage_reduction=0.60 # vastaa määritelmää

        return employment_status,pension,kansanelake,tyoelake,wage,time_in_state,ove_paid

    def move_to_deceiced(self,wage,pension,old_wage,age):
        '''
        Siirtymä tilaan kuollut
        '''
        employment_status = 18 # deceiced
        wage=old_wage
        pension=pension
        #netto=0
        time_in_state=0
        alkanut_ansiosidonnainen=0

        return employment_status,pension,wage,time_in_state

    def move_to_kht(self,wage,pension,old_wage,age):
        '''
        Siirtymä kotihoidontuelle
        '''
        employment_status = 7 # kotihoidontuelle
        pension=self.pension_accrual(age,old_wage,pension,state=7)
        
        time_in_state=0
        time_in_state+=self.timestep

        return employment_status,pension,wage,time_in_state

    def move_to_fatherleave(self,wage,pension,old_wage,age):
        '''
        Siirtymä isyysvapaalle
        '''
        
        #self.infostate_add_child(age) # only for the mother
        employment_status = 6 # isyysvapaa
        time_in_state=0
        pension=self.pension_accrual(age,old_wage,pension,state=6)
        time_in_state+=self.timestep        
        pinkslip=0
        
        return employment_status,pension,wage,time_in_state,pinkslip

    def move_to_motherleave(self,wage,pension,old_wage,age):
        '''
        Siirtymä äitiysvapaalle
        '''
        self.infostate_add_child(age)
        employment_status = 5 # äitiysvapaa
        time_in_state=0
        pension=self.pension_accrual(age,old_wage,pension,state=5)
        time_in_state+=self.timestep
        pinkslip=0

        return employment_status,pension,wage,time_in_state,pinkslip

    def stay_unemployed(self,wage,employment_status,kansanelake,tyoelake,pension,time_in_state,toe,toekesto,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,alkanut_ansiosidonnainen,
                        toe58,ove_paid,children_under3,spouse_value,spouse):
        '''
        Pysy tilassa työtön (0)
        '''
        time_in_state+=self.timestep
        karenssia_jaljella=0
        
        tyoelake=tyoelake*self.elakeindeksi
            
        if age>=65:
            employment_status,kansanelake,tyoelake,pension,wage,time_in_state,wove_paid\
                =self.move_to_retirement(wage,pension,0,age,kansanelake,tyoelake,employment_status,unemp_after_ra,all_acc=True,spouse=spouse)
        elif action == 0 or ((action == 2 or action==4) and age < self.min_retirementage) or (action == 5):
            employment_status = 0 # unchanged
                
            if self.porrasta_toe:
                oldtoe=self.comp_oldtoe(spouse=spouse_value)
            else:
                oldtoe=0
                
            if (action == 1 or action == 3) and self.unemp_limit_reemp:
                if np.random.uniform()>self.unemp_reemp_prob:
                    action = 0
                    
            
            if action == 5 and (not ove_paid) and (age>=self.min_ove_age):
                pension,tyoelake,ove_paid=self.move_to_ove(wage,employment_status,pension,tyoelake,ove_paid,age,unemp_after_ra)

            kesto=used_unemp_benefit
                
            if not self.paivarahapaivia_jaljella(kesto,tyoura,age,toe58,oldtoe):
                if self.include_putki and age>=self.min_tyottputki_ika and tyoura>=self.tyohistoria_tyottputki: 
                    employment_status = 4 # siirto lisäpäiville
                    pension=self.pension_accrual(age,unempwage_basis,pension,state=4,ove_paid=ove_paid)
                    used_unemp_benefit+=self.timestep
                else:
                    employment_status = 13 # siirto työmarkkinatuelle
                    alkanut_ansiosidonnainen=0
                    pension=self.pension_accrual(age,old_wage,pension,state=13)
            else:
                pension=self.pension_accrual(age,unempwage_basis,pension,state=0,ove_paid=ove_paid)                
                used_unemp_benefit+=self.timestep # sic!

            if age >= self.min_retirementage:
                unemp_after_ra+=self.timestep

        elif action == 1: # 
            employment_status,pension,wage,time_in_state,tyoura,pinkslip=\
                self.move_to_work(wage,pension,old_wage,age,time_in_state,tyoura,pinkslip)
        elif action==2:
            if age >= self.min_retirementage: # ve
                employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                    self.move_to_retirement(wage,pension,old_wage,age,kansanelake,tyoelake,employment_status,
                        unemp_after_ra,scale_acc=True,spouse=spouse)
        elif action == 3: # osatyö 50%
            employment_status,pension,wage,time_in_state,tyoura,pinkslip=\
                self.move_to_parttime(wage,pension,old_wage,age,tyoura,time_in_state)
        elif action == 4: # osatyö 50% + ve
            employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                self.move_to_oa_parttime(wage,pension,old_wage,age,kansanelake,tyoelake,employment_status,
                        0,all_acc=True,scale_acc=True)
        elif action==11: # tk
            employment_status,pension,kansanelake,tyoelake,wage,time_in_state,ove_paid=\
                self.move_to_disab(wage,pension,old_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,spouse)
            pinkslip=0
        else:
            print('error 17')  
            
        return employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
            pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
            alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella

    def stay_tyomarkkinatuki(self,wage,employment_status,kansanelake,tyoelake,pension,time_in_state,toe,toekesto,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,alkanut_ansiosidonnainen,
                        toe58,ove_paid,children_under3,spouse_value,spouse):
        '''
        Pysy tilassa työmarkkinatuki (13)
        '''
        time_in_state+=self.timestep
        karenssia_jaljella=0
        
        tyoelake=tyoelake*self.elakeindeksi
        
        if (action == 1 or action == 3) and self.unemp_limit_reemp:
            if np.random.uniform()>self.unemp_reemp_prob:
                action = 0

        if age>=65:
            employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                self.move_to_retirement(wage,pension,0,age,kansanelake,tyoelake,employment_status,unemp_after_ra,all_acc=True,spouse=spouse)
        elif action == 0 or ((action == 2 or action == 4) and age < self.min_retirementage) or (action == 5):
            employment_status = 13 # unchanged
                
            if action == 5 and (not ove_paid) and (age>=self.min_ove_age):
                pension,tyoelake,ove_paid=self.move_to_ove(wage,employment_status,pension,tyoelake,ove_paid,age,unemp_after_ra)

            pension=self.pension_accrual(age,wage,pension,state=13)

            if age >= self.min_retirementage:
                unemp_after_ra+=self.timestep
        
        elif action == 1: # 
            employment_status,pension,wage,time_in_state,tyoura,pinkslip=\
                self.move_to_work(wage,pension,old_wage,age,time_in_state,tyoura,pinkslip)
        elif action == 2:
            if age >= self.min_retirementage: # ve
                employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                    self.move_to_retirement(wage,pension,old_wage,age,kansanelake,tyoelake,employment_status,
                        unemp_after_ra,scale_acc=True,spouse=spouse)
        elif action == 3: # osatyö 50%
            employment_status,pension,wage,time_in_state,tyoura,pinkslip=\
                self.move_to_parttime(wage,pension,old_wage,age,tyoura,time_in_state)
        elif action == 4: # osatyö 50% + ve
            employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                self.move_to_oa_parttime(wage,pension,old_wage,age,kansanelake,tyoelake,employment_status,
                        unemp_after_ra,all_acc=True,scale_acc=False)
        elif action==11: # tk
            employment_status,pension,kansanelake,tyoelake,wage,time_in_state,ove_paid=\
                self.move_to_disab(wage,pension,old_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,spouse)
        else:
            print('error 17')        
                
        return employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
               pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella
                
    def stay_pipeline(self,wage,employment_status,kansanelake,tyoelake,pension,time_in_state,toe,toekesto,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,alkanut_ansiosidonnainen,
                        toe58,ove_paid,children_under3,spouse_value,spouse):
        '''
        Pysy tilassa työttömyysputki (4)
        '''
        time_in_state+=self.timestep
        karenssia_jaljella=0

        tyoelake=tyoelake*self.elakeindeksi

        if (action == 1 or action == 3) and self.unemp_limit_reemp:
            if np.random.uniform()>self.unemp_reemp_prob:
                action = 0
        
        if age>=65:
            employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                self.move_to_retirement(wage,pension,0,age,kansanelake,tyoelake,employment_status,
                    unemp_after_ra,all_acc=True,spouse=spouse)
        elif action == 0 or ((action == 2 or action == 4) and age < self.min_retirementage) or (action == 5):
            employment_status  = 4 # unchanged
            pension=self.pension_accrual(age,unempwage_basis,pension,state=4,ove_paid=ove_paid)
                
            if action == 5 and (not ove_paid) and (age>=self.min_ove_age):
                pension,tyoelake,ove_paid=self.move_to_ove(wage,employment_status,pension,tyoelake,ove_paid,age,unemp_after_ra)
                
            used_unemp_benefit+=self.timestep
            if age >= self.min_retirementage:
                unemp_after_ra+=self.timestep
                
        elif action == 1: # 
            employment_status,pension,wage,time_in_state,tyoura,pinkslip=\
                self.move_to_work(wage,pension,old_wage,age,time_in_state,tyoura,pinkslip)
        elif action==2:
            employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                self.move_to_retirement(wage,pension,old_wage,age,kansanelake,tyoelake,employment_status,
                    unemp_after_ra,all_acc=True,scale_acc=True,spouse=spouse)
            pinkslip=0
        elif action == 3: # 
            employment_status,pension,wage,time_in_state,tyoura,pinkslip=\
                self.move_to_parttime(wage,pension,old_wage,age,tyoura,time_in_state)
        elif action == 4: # osatyö 50% + ve
            employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                self.move_to_oa_parttime(wage,pension,old_wage,age,kansanelake,tyoelake,employment_status,
                        unemp_after_ra,all_acc=True,scale_acc=True)
        elif action==11: # tk
            employment_status,pension,kansanelake,tyoelake,wage,time_in_state,ove_paid=\
                self.move_to_disab(wage,pension,old_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,spouse)
        else:
            print('error 1: ',action)
            
        return employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
               pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella
        
    def stay_employed(self,wage,employment_status,kansanelake,tyoelake,pension,time_in_state,toe,toekesto,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,alkanut_ansiosidonnainen,
                        toe58,ove_paid,children_under3,spouse_value,spouse):
        '''
        Pysy tilassa töissä (1)
        '''
        time_in_state+=self.timestep
        karenssia_jaljella=0
        if sattuma[1]<self.pinkslip_intensity[g]:
            if age<self.min_retirementage:
                pinkslip=1
                action=1 # unemp
            else:
                pinkslip=0
                action=2 # ve
        else:
            pinkslip=0

        tyoelake=tyoelake*self.elakeindeksi

        if action == 0 or ((action == 2 or action == 4) and age < self.min_retirementage) or action == 5:
            employment_status = 1 # unchanged
            
            if action == 5 and (not ove_paid) and (age>=self.min_ove_age):
                pension,tyoelake,ove_paid=self.move_to_ove(wage,employment_status,pension,tyoelake,ove_paid,age,unemp_after_ra)
                            
            tyoura+=self.timestep
            pension=self.pension_accrual(age,wage,pension,state=1)
        elif action == 1: # työttömäksi
            employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
                used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,alkanut_ansiosidonnainen,karenssia_jaljella=\
                self.move_to_unemp(wage,pension,old_wage,age,kansanelake,tyoelake,toe,toekesto,pinkslip,tyoura,
                    used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,alkanut_ansiosidonnainen,toe58,spouse=spouse)
        elif action==2:
            if age >= self.min_retirementage: # ve
                employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                    self.move_to_retirement(wage,pension,old_wage,age,kansanelake,tyoelake,employment_status,unemp_after_ra,spouse=spouse) 
        elif action == 3: # osatyö 50%
            employment_status,pension,wage,time_in_state,tyoura,pinkslip=\
                self.move_to_parttime(wage,pension,old_wage,age,tyoura,0)
        elif action == 4: # osatyö 50% + ve
            employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                self.move_to_oa_parttime(wage,pension,old_wage,age,kansanelake,tyoelake,employment_status,
                        unemp_after_ra,all_acc=True,scale_acc=True)
        elif action==11: # tk
            employment_status,pension,kansanelake,tyoelake,wage,time_in_state,ove_paid=\
                self.move_to_disab(wage,pension,old_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,spouse)
        else:
            print('error 12')    
            
        return employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
              pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
              alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella
           
    def stay_disabled(self,wage,employment_status,kansanelake,tyoelake,pension,time_in_state,toe,toekesto,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,alkanut_ansiosidonnainen,
                        toe58,ove_paid,children_under3,spouse_value,spouse):
            
        '''
        Pysy tilassa työkyvytön (4)
        '''
        time_in_state+=self.timestep
        karenssia_jaljella=0
        if age >= self.min_retirementage:
            employment_status = 3 # ve
        else:
            employment_status = 3 # unchanged

        tyoelake=tyoelake*self.elakeindeksi
        if age>=self.max_retirementage:
            tyoelake = tyoelake+self.scale_pension(pension,age,scale=False)/self.elakeindeksi # hack
            pension=0           
        else:
            pension=pension*self.palkkakerroin
        wage=0

        return employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
               pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella

    def stay_retired(self,wage,employment_status,kansanelake,tyoelake,pension,time_in_state,toe,toekesto,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,alkanut_ansiosidonnainen,
                        toe58,ove_paid,children_under3,spouse_value,spouse):
        '''
        Pysy tilassa vanhuuseläke (2)
        '''
        karenssia_jaljella=0
        if age >= self.min_retirementage: # ve
            time_in_state+=self.timestep

            if age>=self.max_retirementage:
                tyoelake = tyoelake+self.scale_pension(pension,age,scale=False,unemp_after_ra=unemp_after_ra)/self.elakeindeksi
                pension=0           

            if action == 0 or action == 1 or ((action == 2 or action == 3) and age>=self.max_retirementage) or (action == 5) or (action == 4):
                employment_status = 2 # unchanged

                tyoelake = self.elakeindeksi*tyoelake
                kansanelake = kansanelake * self.kelaindeksi
                pension=pension*self.palkkakerroin
                
            elif action == 2 and age<self.max_retirementage:
                employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                    self.move_to_oa_parttime(wage,pension,wage,age,kansanelake,tyoelake,employment_status,0,
                            all_acc=False,scale_acc=False)
            elif action == 3 and age<self.max_retirementage:
                employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                    self.move_to_oa_fulltime(wage,pension,wage,age,kansanelake,tyoelake,employment_status,0,all_acc=False,scale_acc=False)
            elif action == 11:
                employment_status,kansanelake,tyoelake,pension,wage,time_in_state,kansanelake,tyoelake=\
                    self.move_to_retdisab(wage,pension,old_wage,age,time_in_state,kansanelake,tyoelake,unemp_after_ra)
            else:
                print('error 221, action {} age {}'.format(action,age))
        else:
            # työvoiman ulkopuolella
            time_in_state+=self.timestep
            if action == 0:
                employment_status = 2
                wage=old_wage
                tyoelake = self.elakeindeksi*tyoelake
                kansanelake = kansanelake * self.kelaindeksi
                pension=pension*self.palkkakerroin
            elif action == 1: # työttömäksi
                employment_status,kansanelake,tyoelake,pension,wage,time_in_state,netto,\
                    used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,alkanut_ansiosidonnainen,karenssia_jaljella=\
                    self.move_to_unemp(wage,pension,old_wage,age,kansanelake,tyoelake,toe,toekesto,0,tyoura,
                        used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,alkanut_ansiosidonnainen,toe58,spouse=spouse)
            elif action == 2: # töihin
                employment_status,pension,wage,time_in_state,tyoura,pinkslip=\
                    self.move_to_work(wage,pension,wage,age,time_in_state,tyoura,pinkslip)
            elif action == 3: # osatyö 50%
                employment_status,pension,wage,time_in_state,tyoura,pinkslip=\
                    self.move_to_parttime(wage,pension,wage,age,tyoura,time_in_state)
            elif action == 11: # tk
                employment_status,pension,kansanelake,tyoelake,wage,time_in_state,ove_paid=\
                    self.move_to_disab(wage,pension,old_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,spouse)
            else:
                print('error 12')
                
        return employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
               pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella

    def stay_motherleave(self,wage,employment_status,kansanelake,tyoelake,pension,time_in_state,toe,toekesto,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,alkanut_ansiosidonnainen,
                        toe58,ove_paid,children_under3,spouse_value,spouse):
        '''
        Pysy tilassa äitiysvapaa (5)
        '''
        #exit_prb=np.random.uniform(0,80_000)
        karenssia_jaljella=0
        if time_in_state>=self.aitiysvapaa_kesto or sattuma[5]<self.aitiysvapaa_pois*self.timestep:
            pinkslip=0
            if action == 0:
                employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
                    used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,alkanut_ansiosidonnainen,karenssia_jaljella=\
                    self.move_to_unemp(wage,pension,old_wage,age,kansanelake,tyoelake,toe,toekesto,pinkslip,tyoura,
                        used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,alkanut_ansiosidonnainen,toe58,spouse=spouse)
            elif action == 1 or action == 2: # 
                employment_status,pension,wage,time_in_state,tyoura,pinkslip=\
                    self.move_to_work(wage,pension,old_wage,age,time_in_state,tyoura,pinkslip)
            elif action == 3: # 
                employment_status,pension,wage,time_in_state=self.move_to_kht(wage,pension,old_wage,age)
            elif action == 4 or action == 5:
                employment_status,pension,wage,time_in_state,tyoura,pinkslip=\
                    self.move_to_parttime(wage,pension,old_wage,age,tyoura,time_in_state)
            elif action==11: # tk
                employment_status,pension,kansanelake,tyoelake,wage,time_in_state,ove_paid=\
                    self.move_to_disab(wage,pension,old_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,spouse)
            else:
                print('Error 21')
        else:
            pension=self.pension_accrual(age,old_wage,pension,state=5)
            time_in_state+=self.timestep
                
        return employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
               pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella

    def stay_fatherleave(self,wage,employment_status,kansanelake,tyoelake,pension,time_in_state,toe,toekesto,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,alkanut_ansiosidonnainen,
                        toe58,ove_paid,children_under3,spouse_value,spouse):
        '''
        Pysy tilassa isyysvapaa (6)
        '''
        karenssia_jaljella=0
        if time_in_state>=self.isyysvapaa_kesto:
            pinkslip=0
            if action == 0:
                employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
                    used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,alkanut_ansiosidonnainen,karenssia_jaljella=\
                    self.move_to_unemp(wage,pension,old_wage,age,kansanelake,tyoelake,toe,toekesto,pinkslip,tyoura,
                        used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,alkanut_ansiosidonnainen,toe58,spouse=spouse)
            elif action == 1 or action == 2: # 
                # ei vaikutusta palkkaan
                employment_status,pension,wage,time_in_state,tyoura,pinkslip=\
                    self.move_to_work(wage,pension,old_wage,age,0,tyoura,pinkslip)
            elif action == 3: # 
                employment_status,pension,wage,time_in_state=self.move_to_kht(wage,pension,old_wage,age)
            elif action == 4 or action == 5:
                employment_status,pension,wage,time_in_state,tyoura,pinkslip=\
                    self.move_to_parttime(wage,pension,old_wage,age,tyoura,0)
            elif action==11: # tk
                employment_status,pension,kansanelake,tyoelake,wage,time_in_state,ove_paid=\
                    self.move_to_disab(wage,pension,old_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,spouse)
            else:
                print('Error 23')
        else:
            pension=self.pension_accrual(age,old_wage,pension,state=6)
            time_in_state+=self.timestep

        return employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
               pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella

    def stay_khh(self,wage,employment_status,kansanelake,tyoelake,pension,time_in_state,toe,toekesto,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,alkanut_ansiosidonnainen,
                        toe58,ove_paid,children_under3,spouse_value,spouse):
        '''
        Pysy tilassa kotihoidontuki (0)
        '''
        time_in_state+=self.timestep
        karenssia_jaljella=0

        if (action == 0) and (time_in_state>self.kht_kesto or children_under3<1): # jos etuus loppuu, siirtymä satunnaisesti
            s=np.random.uniform()
            if s<1/3:
                action=1
            elif s<2/3:
                action=2
            else:
                action=3

        if age >= self.min_retirementage: # ve
            employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                self.move_to_retirement(wage,pension,old_wage,age,kansanelake,tyoelake,employment_status,unemp_after_ra,all_acc=True,spouse=spouse)
        elif (action == 0) and ((time_in_state<=self.kht_kesto and children_under3>0) or self.perustulo): # jos perustulo, ei aikarajoitetta
            employment_status  = 7 # stay
            pension=self.pension_accrual(age,wage,pension,state=7)
        elif action == 1 or action == 4: # 
            pinkslip=0
            employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
                used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,alkanut_ansiosidonnainen,karenssia_jaljella=\
                self.move_to_unemp(wage,pension,old_wage,age,kansanelake,tyoelake,toe,toekesto,pinkslip,tyoura,
                    used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,alkanut_ansiosidonnainen,toe58,spouse=spouse)
        elif action == 2 or action == 5: # 
                employment_status,pension,wage,time_in_state,tyoura,pinkslip=\
                self.move_to_work(wage,pension,wage,age,time_in_state,tyoura,pinkslip)
        elif action == 3: # 
            #wage=self.get_wage(age,wage_reduction)        
            employment_status,pension,wage,time_in_state,tyoura,pinkslip=\
                self.move_to_parttime(wage,pension,wage,age,tyoura,time_in_state)
        elif action==11: # tk
            employment_status,pension,kansanelake,tyoelake,wage,time_in_state,ove_paid=\
                self.move_to_disab(wage,pension,old_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,spouse)
        else:
            print('Error 25')
            
        return employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
               pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella

    def stay_student(self,wage,employment_status,kansanelake,tyoelake,pension,time_in_state,toe,toekesto,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,alkanut_ansiosidonnainen,
                        toe58,ove_paid,children_under3,spouse_value,spouse):
        '''
        Pysy tilassa opiskelija (12)
        '''

        pinkslip=0
        karenssia_jaljella=0
        if sattuma[5]>=self.student_outrate[intage,g]:
            employment_status = 12 # unchanged
            time_in_state+=self.timestep
            pension=self.pension_accrual(age,0,pension,state=12)
            # opiskelu parantaa tuloja
        elif action == 0 or action == 1: # 
                employment_status,pension,wage,time_in_state,tyoura,pinkslip=\
                self.move_to_work(wage,pension,wage,age,0,tyoura,pinkslip)
        elif action == 2:
            employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
                used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,alkanut_ansiosidonnainen,karenssia_jaljella=\
                self.move_to_unemp(wage,pension,old_wage,age,kansanelake,tyoelake,toe,toekesto,pinkslip,tyoura,
                    used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,alkanut_ansiosidonnainen,toe58,spouse=spouse)
        elif action == 3 or action == 4 or action == 5:
            #wage=self.get_wage(age,wage_reduction)            
            employment_status,pension,wage,time_in_state,tyoura,pinkslip=\
                self.move_to_parttime(wage,pension,wage,age,tyoura,time_in_state)
        elif action == 11: # tk
            employment_status,pension,kansanelake,tyoelake,wage,time_in_state,ove_paid=\
                self.move_to_disab(wage,pension,old_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,spouse)
        else:
            print('error 29: ',action)
            
        return employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
               pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella

    def stay_oa_parttime(self,wage,employment_status,kansanelake,tyoelake,pension,time_in_state,toe,toekesto,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,alkanut_ansiosidonnainen,
                        toe58,ove_paid,children_under3,spouse_value,spouse):
        '''
        Pysy tilassa ve+(osa-aikatyö) (0)
        '''

        time_in_state+=self.timestep
        karenssia_jaljella=0
        
        # irtisanominen
        if sattuma[1]<self.pinkslip_intensity[g]:
            action=2 # ve:lle

        if age>=self.max_retirementage:
            employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                self.move_to_retirement(wage,pension,0,age,kansanelake,tyoelake,employment_status,unemp_after_ra,all_acc=True,scale_acc=False,spouse=spouse)
                
        elif action == 0 or action == 1: # jatkaa osa-aikatöissä, ei voi saada työttömyyspäivärahaa
            employment_status = 8 # unchanged
            parttimewage=0.5*wage
            pension=self.pension_accrual(age,parttimewage,pension,state=employment_status)
            tyoelake = self.elakeindeksi*tyoelake
            kansanelake = kansanelake * self.kelaindeksi
        elif action==2 or action==3: # jatkaa täysin töissä, ei voi saada työttömyyspäivärahaa
            employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                self.move_to_oa_fulltime(wage,pension,wage,age,kansanelake,tyoelake,employment_status,
                        0,all_acc=False,scale_acc=False)
        elif action == 4 or action == 5: # eläkkeelle, eläkeaikana karttunutta eläkettä ei vielä maksuun
                employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                self.move_to_retirement(wage,pension,wage,age,kansanelake,tyoelake,employment_status,
                    0,all_acc=False,scale_acc=False,spouse=spouse)
        elif action == 11:
            # no more working, move to "disab" with no change in paid_pension
            employment_status,kansanelake,tyoelake,pension,wage,time_in_state,kansanelake,tyoelake=\
                self.move_to_retdisab(wage,pension,0,age,time_in_state,kansanelake,tyoelake,unemp_after_ra)
        else:
            print('error 14, action {} age {}'.format(action,age))

        return employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
               pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella

    def stay_oa_fulltime(self,wage,employment_status,kansanelake,tyoelake,pension,time_in_state,toe,toekesto,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,alkanut_ansiosidonnainen,
                        toe58,ove_paid,children_under3,spouse_value,spouse):
        '''
        Pysy tilassa ve+työ (9)
        '''

        time_in_state+=self.timestep        
        karenssia_jaljella=0
        # irtisanominen
        if sattuma[1]<self.pinkslip_intensity[g]:
            action=2 # ve:lle

        if age>=self.max_retirementage:
            employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                self.move_to_retirement(wage,pension,old_wage,age,kansanelake,tyoelake,employment_status,
                    unemp_after_ra,all_acc=True,scale_acc=False,spouse=spouse)
        elif action == 0 or action == 1: # jatkaa töissä, ei voi saada työttömyyspäivärahaa
            employment_status = 9 # unchanged
            pension=self.pension_accrual(age,wage,pension,state=employment_status)
            tyoelake = self.elakeindeksi*tyoelake
            kansanelake = kansanelake * self.kelaindeksi
        elif action == 2: # jatkaa osa-aikatöissä, ei voi saada työttömyyspäivärahaa
            employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                self.move_to_oa_parttime(wage,pension,old_wage,age,kansanelake,tyoelake,employment_status,0,all_acc=False,scale_acc=False)
        elif action==3 or action == 4 or action == 5: # eläkkeelle, eläkeaikana karttunutta eläkettä ei vielä maksuun
                employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                self.move_to_retirement(wage,pension,wage,age,kansanelake,tyoelake,employment_status,0,
                    all_acc=False,scale_acc=False,spouse=spouse)
        elif action == 11:
            # no more working, move to "disab" with no change in paid_pension
            employment_status,kansanelake,tyoelake,pension,wage,time_in_state,kansanelake,tyoelake=\
                self.move_to_retdisab(wage,pension,old_wage,age,time_in_state,kansanelake,tyoelake,unemp_after_ra)
        else:
            print('error 14, action {} age {}'.format(action,age))
            
        return employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
               pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella

    def stay_parttime(self,wage,employment_status,kansanelake,tyoelake,pension,time_in_state,toe,toekesto,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,alkanut_ansiosidonnainen,
                        toe58,ove_paid,children_under3,spouse_value,spouse):
        '''
        Pysy tilassa osa-aikatyö (0)
        '''

        time_in_state+=self.timestep
        karenssia_jaljella=0
        
        # irtisanominen
        if sattuma[1]<self.pinkslip_intensity[g]:
            if age<self.min_retirementage:
                action=1 # unemp
                pinkslip=1
            else:
                action=2 # ve
                pinkslip=0
        else:
            pinkslip=0

        if ove_paid:
            tyoelake=tyoelake*self.elakeindeksi

        if action == 0 or ((action == 2 or action == 4) and age < self.min_retirementage) or (action == 5):
            employment_status = 10 # unchanged
            parttimewage=0.5*wage
            tyoura+=self.timestep
            
            if action == 5 and (not ove_paid) and (age>=self.min_ove_age):
                pension,tyoelake,ove_paid=self.move_to_ove(employment_status,pension,tyoelake,ove_paid,age,unemp_after_ra)
            
            pension=self.pension_accrual(age,parttimewage,pension,state=10)
        elif action == 1: # työttömäksi
            employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
                used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,alkanut_ansiosidonnainen,karenssia_jaljella=\
                self.move_to_unemp(wage,pension,old_wage,age,kansanelake,tyoelake,toe,toekesto,pinkslip,tyoura,
                    used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,alkanut_ansiosidonnainen,toe58,spouse=spouse)
        elif action==2:
            if age >= self.min_retirementage: # ve
                employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                    self.move_to_retirement(wage,pension,old_wage,age,kansanelake,tyoelake,employment_status,
                        unemp_after_ra,all_acc=True,spouse=spouse)
        elif action==3:
                employment_status,pension,wage,time_in_state,tyoura,pinkslip=\
                self.move_to_work(wage,pension,old_wage,age,0,tyoura,pinkslip)
        elif action==4: # move to oa_work
                employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                self.move_to_oa_parttime(wage,pension,wage,age,kansanelake,tyoelake,employment_status,
                        unemp_after_ra,all_acc=True,scale_acc=True)
        elif action==11: # tk
            employment_status,pension,kansanelake,tyoelake,wage,time_in_state,ove_paid=\
                self.move_to_disab(wage,pension,old_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,spouse)
        else:
            print('error 12')
            
        return employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
               pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella

    def stay_outsider(self,wage,employment_status,kansanelake,tyoelake,pension,time_in_state,toe,toekesto,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,alkanut_ansiosidonnainen,
                        toe58,ove_paid,children_under3,spouse_value,spouse):
        '''
        Pysy tilassa työvoiman ulkopuolella (11)
        '''
        karenssia_jaljella=0

        if age>=self.min_retirementage:
            employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                self.move_to_retirement(wage,pension,old_wage,age,kansanelake,tyoelake,employment_status,
                unemp_after_ra,all_acc=True,spouse=spouse)
        elif sattuma[5]>=self.outsider_outrate[intage,g]:
            time_in_state+=self.timestep
            employment_status = 11 # unchanged
            pension=self.pension_accrual(age,wage,pension,state=11)
        elif action == 0 or action == 1: # 
                employment_status,pension,wage,time_in_state,tyoura,pinkslip=\
                self.move_to_work(wage,pension,wage,age,time_in_state,tyoura,pinkslip)
        elif action == 2: # 
            pinkslip=0
            employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
                used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,alkanut_ansiosidonnainen,karenssia_jaljella=\
                self.move_to_unemp(wage,pension,old_wage,age,kansanelake,tyoelake,toe,toekesto,pinkslip,tyoura,
                    used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,alkanut_ansiosidonnainen,toe58,spouse=spouse)
        elif action == 3 or action == 4 or action == 5: # 
                employment_status,pension,wage,time_in_state,tyoura,pinkslip=\
                self.move_to_parttime(wage,pension,wage,age,tyoura,time_in_state)
        elif action == 11: # tk
            employment_status,pension,kansanelake,tyoelake,wage,time_in_state,ove_paid=\
                self.move_to_disab(wage,pension,old_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,spouse)
            pinkslip=0
        else:
            print('error 19: ',action)

        return employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
               pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella
               
    def get_benefits(self,empstate,wage,kansanelake,tyoelake,pension,time_in_state,pinkslip,old_wage,unempwage,
                        unempwage_basis,karenssia_jaljella,age,children_under3,children_under7,children_under18,ove_paid,
                        puoliso,puoliso_tila,puoliso_palkka,puoliso_kansanelake,puoliso_tyoelake,puoliso_old_wage,
                        puoliso_pinkslip,puoliso_karenssia_jaljella,puoliso_time_in_state,puoliso_unempwage,puoliso_unempwage_basis):
        '''
        This could be handlend better
        '''
        #karenssia_jaljella=0 # ok?
        tis=0
        if empstate==0:
            wage=0
            old_wage=unempwage_basis
        elif empstate==1:
            wage=wage
            old_wage=0
        elif empstate==2:
            wage=0
            old_wage=0
        elif empstate==3:
            wage=0
            old_wage=0
        elif empstate==4:
            wage=0
            old_wage=unempwage_basis
        elif empstate==5:
            wage=0
            old_wage=max(unempwage,0.5*(wage+old_wage)) # ei tarkkaa laskentaa
        elif empstate==6:
            wage=0
            old_wage=max(unempwage,0.5*(wage+old_wage)) # ei tarkkaa laskentaa
        elif empstate==7:
            wage=0
            old_wage=0
        elif empstate==8:
            parttimewage=0.5*wage
            wage=parttimewage
            old_wage=0
        elif empstate==9:
            wage=wage
            old_wage=0
        elif empstate==10:
            parttimewage=0.5*wage
            wage=parttimewage
            old_wage=0
        elif empstate==11:
            wage=0
            old_wage=old_wage
        elif empstate==12:
            wage=0
            old_wage=0
        elif empstate==13:
            wage=0
            old_wage=unempwage_basis
        else:
            print('unknown state',empstate)
        puoliso_tis=0
        if puoliso_tila==0:
            puoliso_palkka=0
            puoliso_old_wage=puoliso_unempwage_basis
        elif puoliso_tila==1:
            puoliso_palkka=puoliso_palkka
            puoliso_old_wage=0
        elif puoliso_tila==2:
            puoliso_palkka=0
            puoliso_old_wage=0
        elif puoliso_tila==3:
            puoliso_palkka=0
            puoliso_old_wage=0
        elif puoliso_tila==4:
            puoliso_palkka=0
            puoliso_old_wage=puoliso_unempwage_basis
        elif puoliso_tila==5:
            puoliso_palkka=0
            puoliso_old_wage=max(puoliso_unempwage,0.5*(puoliso_palkka+puoliso_old_wage)) # ei tarkkaa laskentaa
        elif puoliso_tila==6:
            puoliso_palkka=0
            puoliso_old_wage=max(puoliso_unempwage,0.5*(puoliso_palkka+puoliso_old_wage)) # ei tarkkaa laskentaa
        elif puoliso_tila==7:
            puoliso_palkka=0
            puoliso_old_wage=0
        elif puoliso_tila==8:
            spouse_parttimewage=0.5*puoliso_palkka
            puoliso_palkka=spouse_parttimewage
            puoliso_old_wage=0
        elif puoliso_tila==9:
            puoliso_palkka=puoliso_palkka
            puoliso_old_wage=0
        elif puoliso_tila==10:
            spouse_parttimewage=0.5*puoliso_palkka
            puoliso_palkka=spouse_parttimewage
            puoliso_old_wage=0
        elif puoliso_tila==11:
            puoliso_palkka=0
            puoliso_old_wage=puoliso_old_wage
        elif puoliso_tila==12:
            puoliso_palkka=0
            puoliso_old_wage=0
        elif puoliso_tila==13:
            puoliso_palkka=0
            puoliso_old_wage=puoliso_unempwage_basis
        else:
            print('unknown state',empstate)
            
        paid_pension=kansanelake+tyoelake
        puoliso_paid_pension=puoliso_kansanelake+puoliso_tyoelake
            
        netto,benq,netto_omat,netto_puoliso=self.comp_benefits(wage,old_wage,kansanelake,tyoelake,empstate,tis,children_under3,children_under7,children_under18,age,
                                puoliso,puoliso_tila,puoliso_palkka,puoliso_kansanelake,puoliso_tyoelake,puoliso_old_wage,puoliso_time_in_state,
                                irtisanottu=pinkslip,karenssia_jaljella=karenssia_jaljella,
                                puoliso_irtisanottu=puoliso_pinkslip,puoliso_karenssia_jaljella=puoliso_karenssia_jaljella)
            
        return netto,benq,netto_omat,netto_puoliso

    def pension_accrual(self,age,wage,pension,state=1,ove_paid=0):
        '''
        Eläkkeen karttumisrutiini
        '''
        if age>=52 and age<63:
            acc=self.acc_over_52
        else:
            acc=self.acc

        if state in set([0,4]):
            if age>=52 and age<63:
                acc=self.acc_unemp_over_52
            else:
                acc=self.acc_unemp
                
            if ove_paid>0:
                acc=0
            
            if age<self.min_retirementage:
                pension=pension*self.palkkakerroin+acc*wage
            else: # muuten ei karttumaa
                pension=pension*self.palkkakerroin
        elif state in set([1,10]):
            if age<self.max_retirementage:
                pension=pension*self.palkkakerroin+acc*wage
            else:
                pension=pension*self.palkkakerroin
        elif state in set([5,6]):
            if age>=52 and age<63:
                acc=self.acc_family_over_52
            else:
                acc=self.acc_family

            if age<self.max_retirementage:
                pension=pension*self.palkkakerroin+acc*max(wage,self.accbasis_kht)
            else:
                pension=pension*self.palkkakerroin
        elif state == 7:
            if age<self.max_retirementage:
                pension=pension*self.palkkakerroin+acc*self.accbasis_kht
            else:
                pension=pension*self.palkkakerroin
        elif state in set([8,9]):
            acc=self.acc # ei korotettua
            if age<self.max_retirementage:
                pension=pension*self.palkkakerroin+acc*wage
            else:
                pension=pension*self.palkkakerroin
        elif state == 13: # tm-tuki
            pension=pension*self.palkkakerroin # ei karttumaa!
        else: # 2,3,11,12,18 # ei karttumaa
            pension=pension*self.palkkakerroin # vastainen eläke, ei alkanut, ei karttumaa
            
        return pension

    def update_wage_reduction_baseline(self,state,wage_reduction,initial_reduction=False):
        '''
        Pidetään kirjaa siitä, kuinka paljon palkkaa alennetaan työttömyyden keston suhteen,
        ja miten siitä palaudutaan
        '''
        if state in set([1,10]): # töissä
            wage_reduction=max(0,wage_reduction-self.salary_const_up)
        if state in set([8,9]): # ve+töissä
            wage_reduction=max(0,wage_reduction-self.salary_const_up)
        elif state==12: # opiskelee
            wage_reduction=max(0,wage_reduction-self.salary_const_student)
        elif state in set([0,4,13,11]): # työtön tai työelämän ulkopuolella
            wage_reduction=min(1.0,wage_reduction+self.salary_const)
        elif state in set([5,6]): # äitiys-, isyys- tai vanhempainvapaa
            #wage_reduction+=self.salary_const
            wage_reduction=wage_reduction
        elif state in set([3]):
            wage_reduction=0.60 # vastaa määritelmää
        elif state in set([7,2]): # kotihoidontuki tai ve tai tk
            wage_reduction=min(1.0,wage_reduction+self.salary_const)
        elif state in set([14]): # ei muutosta
            wage_reduction=wage_reduction
        else: # ylivuoto, ei tiloja
            wage_reduction=wage_reduction
        
        return wage_reduction

    def update_wage_reduction_sigma(self,state,wage_reduction,initial_reduction=False):
        '''
        Pidetään kirjaa siitä, kuinka paljon palkkaa alennetaan työttömyyden keston suhteen,
        ja miten siitä palaudutaan
        
        Tämä malli ei mene koskaan nollaan.
        '''
        
        if initial_reduction:
            wage_reduction=max(0,1.0-(1-self.wage_initial_reduction)*(1-wage_reduction))
        
        if state in set([1,10]): # töissä
            wage_reduction=max(0,wage_reduction-self.salary_const_up)
        if state in set([8,9]): # ve+töissä
            wage_reduction=max(0,wage_reduction-self.salary_const_up)
        elif state==12: # opiskelee
            wage_reduction=max(0,wage_reduction-self.salary_const_student)
        elif state in set([0,4,13,11]): # työtön tai työelämän ulkopuolella
            wage_reduction=max(0,1.0-(1-self.salary_const)*(1-wage_reduction))
        elif state in set([5,6]): # isyys tai vanhempainvapaa, ei vaikutusta
            wage_reduction=wage_reduction
        elif state in set([3]):
            wage_reduction=0.60 # vastaa määritelmää
        elif state in set([7,2]): # kotihoidontuki tai ve
            #wage_reduction=max(0,1.0-((1-self.salary_const)**self.timestep)*(1-wage_reduction))
            wage_reduction=max(0,1.0-(1-self.salary_const)*(1-wage_reduction))
        elif state in set([14]): # ei muutosta
            wage_reduction=wage_reduction
        else: # ylivuoto, ei tiloja
            wage_reduction=wage_reduction
        
        return wage_reduction
        
    def get_family_wage(self,age,g):
        if g>2: # puoliso mies (yksinkertaistus)
            palkka=self.palkat_ika_miehet[self.map_age(age)]*self.g_r[self.map_age(age),g]
        else: # puoliso mies (yksinkertaistus)
            palkka=self.palkat_ika_naiset[self.map_age(age)]*self.g_r[self.map_age(age),g-3]
            
        return palkka
        
    def update_family(self,puoliso,age):
        '''
        Päivitä puolison/potentiaalisen puolison tila & palkka
        Päivitä avioliitto/avoliitto
        '''
        sattuma = np.random.uniform(size=2)
        
        # update marital status
        intage=int(np.floor(age))
        if puoliso>0:
            if self.divorce_rate[intage]>sattuma[0]:
                puoliso=0
            else:
                puoliso=1
        else:
            if self.marriage_rate[intage]>sattuma[0]:
                puoliso=1
            else:
                puoliso=0
    
        return puoliso
        
    def step(self, action, dynprog=False, debug=False):
        '''
        Open AI interfacen mukainen step-funktio, joka tekee askeleen eteenpäin
        toiminnon action mukaan 

        Keskeinen funktio simuloinnissa
        '''
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        emp_action=int(action[0])
        spouse_action=int(action[1])

        employment_status,g,pension,old_wage,age,time_in_state,paid_pension,pinkslip,toe,toekesto,\
            tyoura,used_unemp_benefit,wage_reduction,unemp_after_ra,\
            unempwage,unempwage_basis,prefnoise,children_under3,children_under7,children_under18,\
            unemp_left,alkanut_ansiosidonnainen,toe58,ove_paid,jasen,\
            puoliso,puoliso_tila,spouse_g,puoliso_old_wage,puoliso_pension,puoliso_wage_reduction,\
            puoliso_paid_pension,puoliso_next_wage,\
            puoliso_used_unemp_benefit,puoliso_unemp_benefit_left,\
            puoliso_unemp_after_ra,puoliso_unempwage,puoliso_unempwage_basis,\
            puoliso_alkanut_ansiosidonnainen,puoliso_toe58,\
            puoliso_toe,puoliso_toekesto,puoliso_tyoura,puoliso_time_in_state,puoliso_pinkslip,puoliso_ove_paid,\
            kansanelake,puoliso_kansanelake,tyoelake_maksussa,puoliso_tyoelake_maksussa,\
            next_wage\
                =self.state_decode(self.state)

        wage=next_wage                
        spouse_wage=puoliso_next_wage                
            
        intage=int(np.floor(age))
        t=int((age-self.min_age)/self.timestep)
        moved=False
        spouse_moved=False
        use_func=True
        
        if self.randomness:
            # kaikki satunnaisuus kerralla
            sattuma = np.random.uniform(size=7)
            sattuma2 = np.random.uniform(size=7)
            
            if self.include_spouses:
                puoliso=self.update_family(puoliso,age)
            else:
                puoliso=0
            
            # siirtymät
            move_prob=self.disability_intensity[intage,g]+self.birth_intensity[intage,g]+self.student_inrate[intage,g]+self.outsider_inrate[intage,g]

            if sattuma[0]<move_prob:
                s1=self.disability_intensity[intage,g]
                s2=s1+self.birth_intensity[intage,g]
                s3=s2+self.student_inrate[intage,g]
                #s4=s3+self.outsider_inrate[intage,g]
            
                # tk-alkavuus, siisti kuntoon!
                if sattuma[2]<s1/move_prob: # age<self.min_retirementage and 
                    emp_action=11 # disability
                elif sattuma[2]<s2/move_prob:
                    if self.infostat_can_have_children(age): # lasten väli vähintään vuosi, ei työkyvyttömyyseläkkeellä
                        if g>2: # naiset
                            if employment_status!=3:
                                employment_status,pension,wage,time_in_state,pinkslip=\
                                    self.move_to_motherleave(wage,pension,old_wage,age)
                                pinkslip=0
                                karenssia_jaljella=0
                                moved=True
                                wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)
                            if sattuma[4]<0.35 and puoliso_tila!=3: # orig 0.5
                                puoliso_tila,puoliso_pension,spouse_wage,puoliso_time_in_state,puoliso_pinkslip=\
                                    self.move_to_fatherleave(spouse_wage,puoliso_pension,puoliso_old_wage,age)
                                puoliso_karenssia_jaljella=0
                                spouse_moved=True
                                puoliso_wage_reduction=self.update_wage_reduction(puoliso_tila,puoliso_wage_reduction)
                        else: # miehet
                            # ikä valittu äidin iän mukaan. oikeastaan tämä ei mene ihan oikein miehille
                            if puoliso_tila!=3:
                                puoliso_tila,puoliso_pension,spouse_wage,puoliso_time_in_state,puoliso_pinkslip=\
                                    self.move_to_motherleave(spouse_wage,puoliso_pension,puoliso_old_wage,age)
                                puoliso_karenssia_jaljella=0
                                puoliso_wage_reduction=self.update_wage_reduction(puoliso_tila,puoliso_wage_reduction)
                                spouse_moved=True
                            if sattuma[4]<0.35 and employment_status!=3: # orig 0.5
                                employment_status,pension,wage,time_in_state,pinkslip=\
                                    self.move_to_fatherleave(wage,pension,old_wage,age)
                                karenssia_jaljella=0
                                wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)
                                moved=True
                elif sattuma[2]<s3/move_prob:
                    if employment_status not in set([2,3,5,6,7,8,9,11,12,18]): # and False:
                        employment_status,pension,wage,time_in_state,pinkslip=\
                            self.move_to_student(wage,pension,old_wage,age,time_in_state,tyoura,pinkslip)
                        karenssia_jaljella=0
                        wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)
                        moved=True
                #elif sattuma[2]<s4/move_prob: # and False:
                else:
                    if employment_status not in set([2,3,5,6,7,8,9,11,12,18]):
                        employment_status,pension,wage,time_in_state,pinkslip=\
                            self.move_to_outsider(wage,pension,old_wage,age,pinkslip)
                        karenssia_jaljella=0
                        wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)
                        moved=True
                        
            move_prob=self.disability_intensity[intage,spouse_g]+self.student_inrate[intage,spouse_g]+self.outsider_inrate[intage,spouse_g]

            if sattuma2[0]<move_prob and not spouse_moved:
                s1=self.disability_intensity[intage,spouse_g]
                s3=s1+self.student_inrate[intage,spouse_g]
                #s4=s3+self.outsider_inrate[intage,g]
            
                # tk-alkavuus, siisti kuntoon!
                if sattuma2[2]<s1/move_prob: # age<self.min_retirementage and 
                    spouse_action=11 # disability
                elif sattuma2[2]<s3/move_prob:
                    if puoliso_tila not in set([2,3,5,6,7,8,9,11,12,18]): # and False:
                        puoliso_tila,puoliso_pension,spouse_wage,puoliso_time_in_state,puoliso_pinkslip=\
                            self.move_to_student(spouse_wage,puoliso_pension,puoliso_old_wage,age,puoliso_time_in_state,puoliso_tyoura,
                                puoliso_pinkslip)
                        puoliso_karenssia_jaljella=0
                        puoliso_wage_reduction=self.update_wage_reduction(puoliso_tila,puoliso_wage_reduction)
                        puoliso_moved=True
                else:
                    if puoliso_tila not in set([2,3,5,6,7,8,9,11,12,18]):
                        puoliso_tila,puoliso_pension,spouse_wage,puoliso_time_in_state,puoliso_pinkslip=\
                            self.move_to_outsider(spouse_wage,puoliso_pension,puoliso_old_wage,age,puoliso_pinkslip)
                        puoliso_karenssia_jaljella=0
                        puoliso_wage_reduction=self.update_wage_reduction(puoliso_tila,puoliso_wage_reduction)
                        puoliso_moved=True
            # voi aiheuttaa epästabiilisuutta
            if sattuma[3]<self.mort_intensity[intage,g] and self.include_mort: 
                employment_status,pension,wage,time_in_state,netto=self.move_to_deceiced(pension,old_wage,age)
                
            if sattuma2[3]<self.mort_intensity[intage,spouse_g] and self.include_mort:  # puoliso, FIXME
                puoliso_tila,puoliso_pension,spouse_wage,puoliso_time_in_state,netto=self.move_to_deceiced(puoliso_pension,puoliso_old_wage,age)
        else:
            # tn ei ole koskaan alle rajan, jos tämä on 1
            sattuma = np.ones(7)
            sattuma2 = np.ones(7)
            
        if employment_status==14: # deceiced
            #time_in_state+=self.timestep
            if not self.include_mort:
                print('emp state 14')
            wage=old_wage
            nextwage=wage
            toe=0
            if self.mortstop:
                done=True
            else:
                done = age >= self.max_age
                done = bool(done)

            self.state = self.state_encode(employment_status,g,pension,wage,age+self.timestep,
                            time_in_state,tyoelake_maksussa,pinkslip,toe,tyoura,nextwage,
                            used_unemp_benefit,wage_reduction,unemp_after_ra,unempwage,unempwage_basis,
                            children_under3,children_under7,children_under18,
                            0,alkanut_ansiosidonnainen,toe58,ove_paid,0,
                            puoliso,puoliso_tila,spouse_wage,puoliso_pension,
                            puoliso_wage_reduction,puoliso_tyoelake_maksussa,puoliso_next_wage,
                            puoliso_used_unemp_benefit,puoliso_unemp_benefit_left,
                            puoliso_unemp_after_ra,puoliso_unempwage,puoliso_unempwage_basis,
                            puoliso_alkanut_ansiosidonnainen,puoliso_toe58,
                            puoliso_toe,puoliso_toekesto,puoliso_tyoura,puoliso_time_in_state,puoliso_pinkslip,puoliso_ove_paid,
                            kansanelake,puoliso_kansanelake,
                            prefnoise)
                            
            netto,benq=self.comp_benefits(0,0,0,14,0,children_under3,children_under7,children_under18,age,puoliso,puoliso_tila,puoliso_palkka,retq=True)
                            
            reward=0
            equivalent=0
            return np.array(self.state), reward, done, benq
        else:
            if age>=self.max_retirementage and employment_status not in set([2,3]):
                employment_status,kansanelake,tyoelake_maksussa,pension,wage,time_in_state,ove_paid\
                    =self.move_to_retirement(wage,pension,0,age,kansanelake,tyoelake_maksussa,employment_status,unemp_after_ra,all_acc=True,spouse=puoliso)
                wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)
                karenssia_jaljella=0
                moved=True
            
            if age>=self.max_retirementage and puoliso_tila not in set([2,3]):
                puoliso_tila,puoliso_kansanelake,puoliso_tyoelake_maksussa,puoliso_pension,spouse_wage,puoliso_time_in_state,puoliso_ove_paid\
                    =self.move_to_retirement(spouse_wage,puoliso_pension,0,age,puoliso_kansanelake,puoliso_tyoelake_maksussa,puoliso_tila,\
                        puoliso_unemp_after_ra,all_acc=True,spouse=puoliso)
                puoliso_wage_reduction=self.update_wage_reduction(puoliso_tila,puoliso_wage_reduction)
                puoliso_karenssia_jaljella=0
                puoliso_pinkslip=0
                spouse_moved=True
                        
            if not moved:
                # hoidetaan tilasiirtymät ja -pysymiset alirutiineilla, joita kutsutaan mäppäämällä tila funktioksi,
                # jota sitten kutsutaan
                spouse_value=False
                employment_status,kansanelake,tyoelake_maksussa,pension,wage,time_in_state,pinkslip,\
                unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella\
                    = self.map_stays[employment_status](wage,employment_status,kansanelake,tyoelake_maksussa,pension,time_in_state,toe,toekesto,
                                   tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,unempwage_basis,
                                   emp_action,age,sattuma,intage,g,alkanut_ansiosidonnainen,toe58,ove_paid,children_under3,puoliso,
                                   spouse_value)
                wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)
                                   
            if not spouse_moved:
                # hoidetaan tilasiirtymät ja -pysymiset alirutiineilla, joita kutsutaan mäppäämällä tila funktioksi,
                # jota sitten kutsutaan
                spouse_value=True
                puoliso_tila,puoliso_kansanelake,puoliso_tyoelake_maksussa,puoliso_pension,spouse_wage,puoliso_time_in_state,puoliso_pinkslip,\
                puoliso_unemp_after_ra,puoliso_tyoura,puoliso_used_unemp_benefit,puoliso_unempwage_basis,puoliso_alkanut_ansiosidonnainen,puoliso_ove_paid,\
                puoliso_karenssia_jaljella\
                    = self.map_stays[puoliso_tila](spouse_wage,puoliso_tila,puoliso_kansanelake,puoliso_tyoelake_maksussa,puoliso_pension,puoliso_time_in_state,puoliso_toe,puoliso_toekesto,
                                   puoliso_tyoura,puoliso_used_unemp_benefit,puoliso_pinkslip,puoliso_unemp_after_ra,puoliso_old_wage,puoliso_unempwage,puoliso_unempwage_basis,
                                   spouse_action,age,sattuma2,intage,spouse_g,puoliso_alkanut_ansiosidonnainen,puoliso_toe58,puoliso_ove_paid,children_under3,puoliso,
                                   spouse_value)
                puoliso_wage_reduction=self.update_wage_reduction(puoliso_tila,puoliso_wage_reduction)
                
            netto,benq,netto_omat,netto_puoliso=self.get_benefits(employment_status,wage,kansanelake,tyoelake_maksussa,pension,
                        time_in_state,pinkslip,old_wage,unempwage,unempwage_basis,karenssia_jaljella,age,
                        children_under3,children_under7,children_under18,ove_paid,
                        puoliso,puoliso_tila,spouse_wage,puoliso_kansanelake,puoliso_tyoelake_maksussa,puoliso_old_wage,
                        puoliso_pinkslip,puoliso_karenssia_jaljella,puoliso_time_in_state,puoliso_unempwage,puoliso_unempwage_basis)

        #self.check_q(benq,10)

        done = age >= self.max_age
        done = bool(done)
        
        # after this, preparing for the next step
        age=age+self.timestep
        
        toe58=self.check_toe58(age,toe,tyoura,toe58)
        puoliso_toe58=self.check_toe58(age,puoliso_toe,puoliso_tyoura,puoliso_toe58)
        
        work={1,10}
        retired={2,8,9}
        self.update_infostate(t,int(employment_status),wage,unempwage_basis,spouse=False)
        toe,toekesto,unempwage,children_under3,children_under7,children_under18=self.comp_infostats(age,spouse=False)
        if employment_status in work and self.tyossaoloehto(toe,tyoura,age):
            used_unemp_benefit=0
            alkanut_ansiosidonnainen=0
            #if alkanut_ansiosidonnainen>0:
            #    if not self.infostat_check_aareset(age):
            #        alkanut_ansiosidonnainen=0
        elif employment_status in retired:
            alkanut_ansiosidonnainen=0
        if alkanut_ansiosidonnainen<1:
            unempwage_basis=0
            
        self.update_infostate(t,int(puoliso_tila),spouse_wage,puoliso_unempwage_basis,spouse=True)
        puoliso_toe,puoliso_toekesto,puoliso_unempwage,_,_,_=self.comp_infostats(age,spouse=True)
        if puoliso_tila in work and self.tyossaoloehto(puoliso_toe,puoliso_tyoura,age):
            puoliso_used_unemp_benefit=0
            puoliso_alkanut_ansiosidonnainen=0
            #if alkanut_ansiosidonnainen>0:
            #    if not self.infostat_check_aareset(age):
            #        alkanut_ansiosidonnainen=0
        elif puoliso_tila in retired:
            puoliso_alkanut_ansiosidonnainen=0
        if puoliso_alkanut_ansiosidonnainen<1:
            puoliso_unempwage_basis=0
            
        if self.porrasta_toe and (employment_status in set([0,4]) or alkanut_ansiosidonnainen>0):
            old_toe=self.comp_oldtoe(spouse=False)
            pvr_jaljella=self.comp_unempdays_left(used_unemp_benefit,tyoura,age,toe,employment_status,alkanut_ansiosidonnainen,toe58,old_toe,printti=False)
        else:
            pvr_jaljella=self.comp_unempdays_left(used_unemp_benefit,tyoura,age,toe,employment_status,alkanut_ansiosidonnainen,toe58,toekesto)

        if self.porrasta_toe and (puoliso_tila in set([0,4]) or puoliso_alkanut_ansiosidonnainen>0):
            p_old_toe=self.comp_oldtoe(spouse=True)
            puoliso_pvr_jaljella=self.comp_unempdays_left(puoliso_used_unemp_benefit,puoliso_tyoura,age,puoliso_toe,puoliso_tila,puoliso_alkanut_ansiosidonnainen,puoliso_toe58,p_old_toe,printti=False)
        else:
            puoliso_pvr_jaljella=self.comp_unempdays_left(puoliso_used_unemp_benefit,puoliso_tyoura,age,puoliso_toe,puoliso_tila,puoliso_alkanut_ansiosidonnainen,puoliso_toe58,puoliso_toekesto)
            
        kassanjasenyys=self.get_kassanjasenyys()
        
        #self.render_infostate()

        if not done:
            if self.additive_logutil:
                reward_omat,omat_equivalent = self.log_utility(netto_omat,int(employment_status),age,g=g,pinkslip=pinkslip,spouse=0)
                reward_puoliso,spouse_equivalent = self.log_utility(netto_puoliso,int(puoliso_tila),age,g=spouse_g,pinkslip=puoliso_pinkslip,spouse=0)
                reward=reward_omat+reward_puoliso
                equivalent=omat_equivalent+spouse_equivalent
                
                if not np.isfinite(reward_omat):
                    print('omat',netto_omat,reward_omat)
                if not np.isfinite(reward_puoliso):
                    print('puoliso',netto_puoliso,reward_puoliso)
            else:
                omat_equivalent,spouse_equivalent=0,0
                reward,equivalent = self.log_utility(netto,int(employment_status),age,g=g,spouse_g=spouse_g,pinkslip=pinkslip,spouse_pinkslip=puoliso_pinkslip,
                                                    spouse=puoliso,spouse_empstate=puoliso_tila)
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            
            paid_pension += self.elinaikakerroin*pension # ei ihan oikein! lykkäyskorotus puuttuu, optimoijan pitäisi huomata, joten ei vaikutusta?
            pension=0
            puoliso_paid_pension += self.elinaikakerroin*puoliso_pension # ei ihan oikein! lykkäyskorotus puuttuu, optimoijan pitäisi huomata, joten ei vaikutusta?
            puoliso_pension=0
            
            netto,benq,netto_omat,netto_puoliso=self.get_benefits(employment_status,wage,kansanelake,tyoelake_maksussa,pension,time_in_state,pinkslip,old_wage,
                unempwage,unempwage_basis,karenssia_jaljella,age,children_under3,children_under7,children_under18,ove_paid,
                puoliso,puoliso_tila,spouse_wage,puoliso_kansanelake,puoliso_tyoelake_maksussa,puoliso_old_wage,puoliso_pinkslip,puoliso_karenssia_jaljella,
                puoliso_time_in_state,puoliso_unempwage,puoliso_unempwage_basis)

            #netto,benq,netto_omat,netto_puoliso=self.comp_benefits(0,old_wage,paid_pension,employment_status,time_in_state,children_under3,children_under7,children_under18,age,
            #    puoliso,puoliso_tila,spouse_wage,puoliso_paid_pension,puoliso_old_wage,puoliso_time_in_state)
                
            if employment_status in set([2,3,8,9]): # retired
                # pitäisi laskea tarkemmin, ei huomioi eläkkeen indeksointia!
                if self.include_npv_mort:
                    npv,npv0,npv_pension=self.comp_npv_simulation(g)
                    p_npv,p_npv0,p_npv_pension=self.comp_npv_simulation(spouse_g)
                    if self.additive_logutil:
                        error('fixme')
                        omat_equivalent,spouse_equivalent=0,0
                    else:
                        reward,equivalent,omat_equivalent,spouse_equivalent = self.log_utility(netto,employment_status,age,pinkslip=0,
                                                            spouse_pinkslip=puoliso_pinkslip,
                                                            spouse=puoliso,spouse_empstate=puoliso_tila,g=g,spouse_g=spouse_g)
                        reward*=0.5*(npv+p_npv) # approx
                else:
                    p_npv,p_npv0,p_npv_pension=self.npv[spouse_g],self.npv0[spouse_g],self.npv_pension[spouse_g]
                    npv,npv0,npv_pension=self.npv[g],self.npv0[g],self.npv_pension[g]
                    if self.additive_logutil:
                        reward_omat,omat_equivalent = self.log_utility(netto_omat,int(employment_status),age,g=g,pinkslip=pinkslip,spouse=0)
                        reward_puoliso,spouse_equivalent = self.log_utility(netto_puoliso,int(puoliso_tila),age,g=spouse_g,pinkslip=puoliso_pinkslip,spouse=0)
                        reward_omat*=self.npv[g]
                        reward_puoliso*=self.npv[spouse_g]
                        reward=reward_omat+reward_puoliso
                        equivalent=omat_equivalent+spouse_equivalent
                    else:
                        omat_equivalent,spouse_equivalent=0,0
                        reward,equivalent = self.log_utility(netto,employment_status,age,pinkslip=0,spouse=puoliso,spouse_pinkslip=puoliso_pinkslip,
                                                            spouse_empstate=puoliso_tila,g=g,spouse_g=spouse_g)
                        reward*=0.5*(self.npv[g]+self.npv[spouse_g])

                
                # npv0 is undiscounted
                benq=self.scale_q(npv,npv0,npv_pension,p_npv,p_npv0,p_npv_pension,benq,age)
            else:
                # giving up the pension
                reward = 0.0 #-self.npv[g]*self.log_utility(netto,employment_status,age)
                equivalent = 0.0
                omat_equivalent = 0.0
                spouse_equivalent = 0.0
                
            pinkslip=0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0
            equivalent = 0.0
            omat_equivalent = 0.0
            spouse_equivalent = 0.0

        # seuraava palkka tiedoksi valuaatioapproksimaattorille
        if employment_status in set([3,14]):
            next_wage=0
        else:
            next_wage=self.get_wage(age,wage_reduction)
            
        #self.check_q(benq,99)
        
        if puoliso_tila in set([3,14]):
            puoliso_next_wage=0
        else:
            puoliso_next_wage=self.get_spousewage(age,puoliso_wage_reduction)
        
        self.state = self.state_encode(employment_status,g,pension,wage,age,time_in_state,
                                tyoelake_maksussa,pinkslip,toe,toekesto,tyoura,next_wage,used_unemp_benefit,
                                wage_reduction,unemp_after_ra,unempwage,unempwage_basis,
                                children_under3,children_under7,children_under18,
                                pvr_jaljella,alkanut_ansiosidonnainen,toe58,ove_paid,kassanjasenyys,
                                puoliso,puoliso_tila,spouse_wage,puoliso_pension,
                                puoliso_wage_reduction,puoliso_tyoelake_maksussa,puoliso_next_wage,
                                puoliso_used_unemp_benefit,puoliso_pvr_jaljella,
                                puoliso_unemp_after_ra,puoliso_unempwage,puoliso_unempwage_basis,
                                puoliso_alkanut_ansiosidonnainen,puoliso_toe58,
                                puoliso_toe,puoliso_toekesto,puoliso_tyoura,puoliso_time_in_state,puoliso_pinkslip,puoliso_ove_paid,
                                kansanelake,puoliso_kansanelake,
                                prefnoise)

        if self.plotdebug:
            self.render(done=done,reward=reward,netto=netto,benq=benq,netto_omat=netto_omat,netto_puoliso=netto_puoliso)

        benq['omat_eq']=omat_equivalent
        benq['puoliso_eq']=spouse_equivalent
        benq['eq']=equivalent
        #print('eq',benq['eq'],benq['omat_eq'],benq['puoliso_eq'])
        return np.array(self.state), reward, done, benq
        
    def check_q(self,q,num=-1):
        for person in set(['omat_','puoliso_']):
            d1=q[person+'verot']
            d2=q[person+'valtionvero']+q[person+'kunnallisvero']+q[person+'ptel']+q[person+'tyotvakmaksu']+\
                q[person+'ylevero']+q[person+'sairausvakuutusmaksu']
            
            if np.abs(d2-d1)>1e-6:
                print(f'check_q {num}: {person} {d2-d1}')
        
    def scale_q(self,npv,npv0,npv_pension,p_npv,p_npv0,p_npv_pension,benq,age):
        '''
        Scaling the incomes etc by a discounted nominal present value
        '''
        omat='omat_'
        puoliso='puoliso_'

        v_pens={omat: npv_pension, puoliso: p_npv_pension}
        v0_pens={omat: npv0, puoliso: p_npv0}
        for alku in set([omat,puoliso]):
            p1=v_pens[alku]
            p2=v0_pens[alku]
            benq[alku+'verot']*=p1
            benq[alku+'etuustulo_brutto']*=p1
            benq[alku+'ylevero']*=p1
            benq[alku+'alv']*=p1
            benq[alku+'valtionvero']*=p1
            benq[alku+'kunnallisvero']*=p1
            benq[alku+'asumistuki']*=p1
            benq[alku+'tyotvakmaksu']*=p1 # tätä ei oikeastaan tarvita, mutta ei haittaa
            benq[alku+'sairausvakuutusmaksu']*=p1 # sairaanhoitomaksu maksetaan myös eläketulosta
            benq[alku+'elake_maksussa']*=p1
            benq[alku+'tyoelake']*=p1
            benq[alku+'kansanelake']*=p1 # p.o. kelaindeksillä
            benq[alku+'takuuelake']*=p1 # p.o. kelaindeksillä
            benq[alku+'kokoelake']*=p1
            benq[alku+'perustulo']*=p1
            benq[alku+'toimeentulotuki']*=p1
            benq[alku+'netto']*=p1
            benq[alku+'etuustulo_netto']*=p1
            benq[alku+'multiplier']=p2

        benq['verot']=benq[omat+'verot']+benq[puoliso+'verot']
        benq['etuustulo_brutto']=benq[omat+'etuustulo_brutto']+benq[puoliso+'etuustulo_brutto']
        benq['ylevero']=benq[omat+'ylevero']+benq[puoliso+'ylevero']
        benq['alv']=benq[omat+'alv']+benq[puoliso+'alv']
        benq['valtionvero']=benq[omat+'valtionvero']+benq[puoliso+'valtionvero']
        benq['kunnallisvero']=benq[omat+'kunnallisvero']+benq[puoliso+'kunnallisvero']
        benq['asumistuki']=benq[omat+'asumistuki']+benq[puoliso+'asumistuki']
        benq['tyotvakmaksu']=benq[omat+'tyotvakmaksu']
        benq['sairausvakuutusmaksu']=benq[omat+'sairausvakuutusmaksu']+benq[puoliso+'sairausvakuutusmaksu']
        benq['elake_maksussa']=benq[omat+'elake_maksussa']+benq[puoliso+'elake_maksussa']
        benq['tyoelake']=benq[omat+'tyoelake']+benq[puoliso+'tyoelake']
        benq['kansanelake']=benq[omat+'kansanelake']+benq[puoliso+'kansanelake']
        benq['takuuelake']=benq[omat+'takuuelake']+benq[puoliso+'takuuelake']
        benq['kokoelake']=benq[omat+'kokoelake']+benq[puoliso+'kokoelake']
        benq['perustulo']=benq[omat+'perustulo']+benq[puoliso+'perustulo']
        benq['toimeentulotuki']=benq[omat+'toimeentulotuki']+benq[puoliso+'toimeentulotuki']
        benq['netto']=benq[omat+'netto']+benq[puoliso+'netto']
        benq['etuustulo_netto']=benq[omat+'etuustulo_netto']+benq[puoliso+'etuustulo_netto']
        benq['multiplier']=(benq[omat+'multiplier']+benq[puoliso+'multiplier'])/2

        return benq             

#  Perussetti, tuottaa korkean elastisuuden

    def log_utility_default_params(self):
        # paljonko työstä poissaolo vaikuttaa palkkaan
    
        if self.include_mort:
            if self.perustulo:
                self.men_perustulo_extra=0.10
                self.women_perustulo_extra=0.10
            else:
                self.men_perustulo_extra=0.0
                self.women_perustulo_extra=0.0

            self.salary_const=0.0707*self.timestep
            self.salary_const_up=0.04*self.timestep # työssäolo palauttaa ansioita tämän verran vuodessa
            self.salary_const_student=0.05*self.timestep # opiskelu nostaa leikkausta tämän verran vuodessa
            self.wage_initial_reduction=0.5*self.salary_const
            self.men_kappa_fulltime=0.705 # 0.635 # 0.665
            self.men_mu_scale=0.130 # 0.14 # 0.30 # 0.16 # how much penalty is associated with work increase with age after mu_age
            self.men_mu_age=59.0 # P.O. 60??
            self.men_kappa_osaaika=0.365
            self.men_kappa_hoitovapaa=0.05
            self.men_kappa_ve=0.09 # ehkä 0.10?
            self.men_kappa_pinkslip_young=0.035
            self.men_kappa_pinkslip=0.05
            self.women_kappa_fulltime=0.655 # 0.605 # 0.58
            self.women_mu_scale=0.130 # 0.25 # 0.25 # 0.17 # how much penalty is associated with work increase with age after mu_age
            self.women_mu_age=59.0 # 61 # P.O. 60??
            self.women_kappa_osaaika=0.325
            self.women_kappa_hoitovapaa=0.35
            self.women_kappa_ve=0.09 # ehkä 0.10?
            self.women_kappa_pinkslip_young=0.035
            self.women_kappa_pinkslip=0.04
        else:
            self.salary_const=0.045*self.timestep # työttömyydestä palkka alenee tämän verran vuodessa
            self.salary_const_up=0.04*self.timestep # työssäolo palauttaa ansioita tämän verran vuodessa
            self.salary_const_student=0.05*self.timestep # opiskelu pienentää leikkausta tämän verran vuodessa
            self.wage_initial_reduction=0.005 # työttömäksi siirtymisestä tuleva alennus tuleviin palkkoihin
            
            self.men_kappa_fulltime=0.630 # 0.675 #0.682 # 0.670 # vapaa-ajan menetyksestä rangaistus miehille
            self.men_mu_scale=0.06 #18 # 0.14 # 0.30 # 0.16 # how much penalty is associated with work increase with age after mu_age
            self.men_mu_age=self.min_retirementage-4.0 # P.O. 60??
            self.men_kappa_osaaika_young=0.40 # vapaa-ajan menetyksestä rangaistus miehille osa-aikatyön teosta, suhteessa kokoaikaan
            self.men_kappa_osaaika_middle=0.63 # vapaa-ajan menetyksestä rangaistus miehille osa-aikatyön teosta, suhteessa kokoaikaan
            self.men_kappa_osaaika_old=0.40 # vapaa-ajan menetyksestä rangaistus miehille osa-aikatyön teosta, suhteessa kokoaikaan, alle 35v
            self.men_kappa_hoitovapaa=0.20 # hyöty hoitovapaalla olosta
            self.men_kappa_ve=0.00 # 0.03 # ehkä 0.10?
            self.men_kappa_pinkslip_young=0.00
            self.men_kappa_pinkslip_middle=0.20
            self.men_kappa_pinkslip_elderly=0.10
            
            self.women_kappa_fulltime=0.620 # 0.605 # 0.640 # 0.620 # 0.610 # vapaa-ajan menetyksestä rangaistus naisille
            self.women_mu_scale=0.06 # 0.25 # how much penalty is associated with work increase with age after mu_age
            self.women_mu_age=self.min_retirementage-3.5 # 61 #5 P.O. 60??
            self.women_kappa_osaaika_young=0.30
            self.women_kappa_osaaika_middle=0.55
            self.women_kappa_osaaika_old=0.35
            self.women_kappa_hoitovapaa=0.20 # 0.08
            self.women_kappa_ve=0.00 # 0.03 # ehkä 0.10?
            self.women_kappa_pinkslip_young=0.07
            self.women_kappa_pinkslip_middle=0.30
            self.women_kappa_pinkslip_elderly=0.30

    def set_parameters(self,**kwargs):
        if 'kwargs' in kwargs:
            kwarg=kwargs['kwargs']
        else:
            kwarg=kwargs

        for key, value in kwarg.items():
            if key=='step':
                if value is not None:
                    self.timestep=value
            elif key=='unemp_limit_reemp':
                if value is not None:
                    self.unemp_limit_reemp=value
            elif key=='mortstop':
                if value is not None:
                    self.mortstop=value
            elif key=='gamma':
                if value is not None:
                    gamma=value
            elif key=='use_sigma_reduction':
                if value is not None:
                    self.use_sigma_reduction=value
            elif key=='train':
                if value is not None:
                    self.train=value
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
            elif key=='mortality':
                if value is not None:
                    self.include_mort=value
            if key=='ansiopvraha_toe':
                if value is not None:
                    self.ansiopvraha_toe=value
            elif key=='ansiopvraha_kesto500':
                if value is not None:
                    self.ansiopvraha_kesto500=value
            elif key=='ansiopvraha_kesto400':
                if value is not None:
                    self.ansiopvraha_kesto400=value
            elif key=='ansiopvraha_kesto300':
                if value is not None:
                    self.ansiopvraha_kesto300=value  
            elif key=='perustulo':
                if value is not None:
                    self.perustulo=value  
                    self.universalcredit=False
            elif key=='universalcredit':
                if value is not None:
                    self.perustulo=False  
                    self.universalcredit=value
            elif key=='randomness':
                if value is not None:
                    self.randomness=value  
            elif key=='pinkslip':
                if value is not None:
                    self.include_pinkslip=value  
            elif key=='karenssi_kesto':
                if value is not None:
                    self.karenssi_kesto=value  
            elif key=='include_putki':
                if value is not None:
                    self.include_putki=value  
            elif key=='include_preferencenoise':
                if value is not None:
                    self.include_preferencenoise=value  
            elif key=='plotdebug':
                if value is not None:
                    self.plotdebug=value  
            elif key=='preferencenoise_level':
                if value is not None:
                    self.preferencenoise_std=value  
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
            elif key=='include_halftoe':
                if value is not None:
                    self.include_halftoe=value
            elif key=='porrasta_toe':
                if value is not None:
                    self.porrasta_toe=value 
            elif key=='include_ove':
                if value is not None:
                    self.include_ove=value  
               

    def set_utility_params(self,**kwargs):
        if 'kwargs' in kwargs:
            kwarg=kwargs['kwargs']
        else:
            kwarg={}

        for key, value in kwarg.items():
            if key=='men_kappa_fulltime':
                if value is not None:
                    self.men_kappa_fulltime=value
            elif key=='women_kappa_fulltime':
                if value is not None:
                    self.women_kappa_fulltime=value
            elif key=='men_mu_scale':
                if value is not None:
                    self.men_mu_scale=value
            elif key=='women_mu_scale':
                if value is not None:
                    self.women_mu_scale=value
            elif key=='men_mu_age':
                if value is not None:
                    self.men_mu_age=value
            elif key=='women_mu_age':
                if value is not None:
                    self.women_mu_age=value
            elif key=='men_kappa_osaaika':
                if value is not None:
                    self.men_kappa_osaaika=value
            elif key=='women_kappa_osaaika':
                if value is not None:
                    self.women_kappa_osaaika=value
            elif key=='men_kappa_hoitovapaa':
                if value is not None:
                    self.men_kappa_hoitovapaa=value
            elif key=='women_kappa_hoitovapaa':
                if value is not None:
                    self.women_kappa_hoitovapaa=value
            elif key=='men_kappa_ve':
                if value is not None:
                    self.men_kappa_ve=value
            elif key=='women_kappa_ve':
                if value is not None:
                    self.women_kappa_ve=value
            elif key=='kappa_pinkslip':
                if value is not None:
                    self.kappa_pinkslip=value
                    
    def log_get_kappa(self,age,g,employment_state,pinkslip):
        # kappa tells how much person values free-time
        if g<3: # miehet
            kappa_kokoaika=self.men_kappa_fulltime
            mu_scale=self.men_mu_scale
            mu_age=self.men_mu_age
            # lognormaali
            #if self.include_preferencenoise:
            #    kappa_kokoaika += prefnoise
        
            if age<28: # ikääntyneet preferoivat osa-aikatyötä
                kappa_osaaika=self.men_kappa_osaaika_young*kappa_kokoaika
            elif age<50: # ikääntyneet preferoivat osa-aikatyötä
                kappa_osaaika=self.men_kappa_osaaika_middle*kappa_kokoaika
            else:
                kappa_osaaika=self.men_kappa_osaaika_old*kappa_kokoaika
                
            kappa_hoitovapaa=self.men_kappa_hoitovapaa
            kappa_ve=self.men_kappa_ve
            if age>50:
                kappa_pinkslip=self.men_kappa_pinkslip_elderly
            elif age>28:
                kappa_pinkslip=self.men_kappa_pinkslip_middle
            else:
                kappa_pinkslip=self.men_kappa_pinkslip_young
        else: # naiset
            kappa_kokoaika=self.women_kappa_fulltime
            mu_scale=self.women_mu_scale
            mu_age=self.women_mu_age
            # lognormaali
            #if self.include_preferencenoise:
            #    kappa_kokoaika += prefnoise
        
            if age<28: # ikääntyneet preferoivat osa-aikatyötä
                kappa_osaaika=self.women_kappa_osaaika_young*kappa_kokoaika
            elif age<50: # ikääntyneet preferoivat osa-aikatyötä
                kappa_osaaika=self.women_kappa_osaaika_middle*kappa_kokoaika
            else:
                kappa_osaaika=self.women_kappa_osaaika_old*kappa_kokoaika
            kappa_hoitovapaa=self.women_kappa_hoitovapaa
            kappa_ve=self.women_kappa_ve
            if age>50:
                kappa_pinkslip=self.women_kappa_pinkslip_elderly
            elif age>28:
                kappa_pinkslip=self.women_kappa_pinkslip_middle
            else:
                kappa_pinkslip=self.women_kappa_pinkslip_young
                
        if pinkslip>0: # irtisanottu
            kappa_pinkslip = 0 # irtisanotuille ei vaikutuksia
        
        if age>mu_age:
            mage=min(self.min_retirementage+8,age)
            kappa_kokoaika += mu_scale*max(0,mage-mu_age)
            kappa_osaaika += mu_scale*max(0,mage-mu_age)
            #kappa_kokoaika *= (1+mu_scale*max(0,age-mu_age))
            #kappa_osaaika *= (1+mu_scale*max(0,age-mu_age))
            #kappa_kokoaika *= (1+mu_scale*max(0,min(10,age-mu_age)))
            #kappa_osaaika *= (1+mu_scale*max(0,min(10,age-mu_age)))

        if employment_state in set([1,9]):
            kappa= -kappa_kokoaika
        elif employment_state in set([8,10]):
            kappa= -kappa_osaaika
        elif employment_state in set([0,4]):
            kappa= -kappa_pinkslip
        elif employment_state in set([13]):
            if self.perustulo:
                kappa= 0
            else:
                kappa= -kappa_pinkslip
        elif employment_state == 2:
            kappa=kappa_ve
        elif employment_state == 7:
            kappa=kappa_hoitovapaa
        elif employment_state == 11:
            kappa=0 #kappa_outsider
        elif employment_state == 12:
            kappa=0 #kappa_opiskelija
        else: # states 3, 5, 6, 7, 14
            kappa=0  
            
        return kappa  
                    
    def log_utility(self,income,employment_state,age,g=0,spouse_g=0,pinkslip=0,spouse_pinkslip=0,prefnoise=0,spouse=0,spouse_empstate=-1,debug=False):
        '''
        Log-utiliteettifunktio muokattuna lähteestä Määttänen, 2013 & Hakola & Määttänen, 2005

        Tulot _income_ ovat vuositasolla, jotta askelpituuden muutos ei vaikuta vapaa-aika-vakioihin
        Tämä versio on parametrisoitu optimoijaa varten
        '''

        kappa=self.log_get_kappa(age,g,employment_state,pinkslip)

        if spouse>0:
            kappa2=self.log_get_kappa(age,spouse_g,spouse_empstate,spouse_pinkslip)
            kappa += kappa2
        
        # hyöty/score
        if self.include_preferencenoise:
            # normaali
            u=np.log(prefnoise*(income)/self.inflationfactor)+kappa
            equ=(income/self.inflationfactor)*np.exp(kappa)
        else:
            u=np.log(income/self.inflationfactor)+kappa
            equ=(income/self.inflationfactor)*np.exp(kappa)

        if u is np.inf and debug:
            print(f'inf: state {employment_state} spouse {spouse} sp_state {spouse_empstate} netto {income}')

        if income<1 and debug:
            print(f'inf: state {employment_state} spouse {spouse} sp_state {spouse_empstate} netto {income}')
            
        return u/10,equ # tulot ovat vuositasolla, mutta skaalataan hyöty

    def get_spousewage_step(self,age,reduction):
        '''
        palkka age-ikäiselle time_in_state-vähennyksellä työllistymispalkkaan, step-kohtaisesti, ei vuosikohtaisesti
        '''
        intage=self.map_age(age)
        if age<self.max_age and age>=self.min_age-1:
            return np.maximum(self.min_salary,self.spousesalary[intage])*max(0,(1-reduction))
        else:
            return 0

    def get_wage_step(self,age,reduction):
        '''
        palkka age-ikäiselle time_in_state-vähennyksellä työllistymispalkkaan, step-kohtaisesti, ei vuosikohtaisesti
        '''
        intage=self.map_age(age)
        if age<self.max_age and age>=self.min_age-1:
            return np.maximum(self.min_salary,self.salary[intage])*max(0,(1-reduction))
        else:
            return 0

    def map_age(self,age,start_zero=False):
        if start_zero:
            return int((age)*self.inv_timestep)
        else:
            return int((age-self.min_age)*self.inv_timestep)
    
    def wage_process_TK_v3(self,w,a0=3300*12,a1=3300*12,g=1):
        '''
        Palkkaprosessi muokattu lähteestä Määttänen, 2013 
        '''
        group_sigmas=np.array([0.05,0.07,0.10])*np.sqrt(self.timestep)
        sigma=group_sigmas[g]
        eps=np.random.normal(loc=0,scale=sigma,size=1)[0]
        c1=0.89**self.timestep
        if w>0:
             # pidetään keskiarvo/a1 samana kuin w/a0
            wt=a1*np.exp(c1*np.log(w/a0)+eps-0.5*sigma*sigma)
        else:
            wt=a1*np.exp(eps)

        # täysiaikainen vuositulo vähintään self.min_salary
        wt=np.maximum(self.min_salary,wt)

        return wt

#     def compute_salary_TK_v3(self,group=1,debug=False,initial_salary=None):
#         '''
#         Alussa ajettava funktio, joka tekee palkat yhtä episodia varten
#         '''
#         n_time = int(np.round((self.max_age-self.min_age)*self.inv_timestep))+2
#         self.salary=np.zeros(n_time)
#         self.spousesalary=np.zeros(n_time)
# 
#         if group>2: # naiset
#             r=self.g_r[0,group-3]
#             if initial_salary is not None:
#                 a0=initial_salary
#             else:
#                 a0=self.palkat_ika_naiset[0]*r
#             
#             a1=self.palkat_ika_naiset[0]*r/5
#             self.salary[0]=np.maximum(self.min_salary,np.random.normal(loc=a0,scale=a1,size=1)[0]) # e/y
#             self.spousesalary[0]=np.maximum(self.min_salary,np.random.normal(loc=a0,scale=a1,size=1)[0]) # e/y
# 
#             k=0
#             r0=self.g_r[0,group-3]
#             a0=self.palkat_ika_naiset[0]*r0
#             s0=self.salary[0]
#             s0s=self.spousesalary[0]
#             for age in np.arange(self.min_age,self.max_age,self.timestep):
#                 r1=self.g_r[k,group-3]
#                 a1=self.palkat_ika_naiset[k]*r1
#                 self.salary[self.map_age(age)]=self.wage_process_TK_v3(s0,a0,a1,g=group-3)
#                 self.spousesalary[self.map_age(age)]=self.wage_process_TK_v3(s0s,a0,a1,g=group-3)
#                 k=k+1
#                 s0=self.salary[self.map_age(age)]
#                 s0s=self.spousesalary[self.map_age(age)]
#                 a0=a1
#                 r0=r1
#         else: # miehet
#             r=self.g_r[0,group]
#             if initial_salary is not None:
#                 a0=initial_salary
#             else:
#                 a0=self.palkat_ika_miehet[0]*r
#                 
#             a1=self.palkat_ika_miehet[0]*r/5
#             self.salary[0]=np.maximum(self.min_salary,np.random.normal(loc=a0,scale=a1,size=1)[0]) # e/y
#             self.spousesalary[0]=np.maximum(self.min_salary,np.random.normal(loc=a0,scale=a1,size=1)[0]) # e/y
# 
#             k=0
#             r0=self.g_r[0,group]
#             a0=self.palkat_ika_miehet[0]*r0
#             s0=self.salary[0]
#             s0s=self.spousesalary[0]
#             for age in np.arange(self.min_age,self.max_age,self.timestep):
#                 r1=self.g_r[k,group]
#                 a1=self.palkat_ika_miehet[k]*r1
#                 self.salary[self.map_age(age)]=self.wage_process_TK_v3(s0,a0,a1,g=group)
#                 self.spousesalary[self.map_age(age)]=self.wage_process_TK_v3(s0,a0,a1,g=group)
#                 k=k+1
#                 s0=self.salary[self.map_age(age)]
#                 s0s=self.spousesalary[self.map_age(age)]
#                 a0=a1
#                 r0=r1
                
    def compute_salary_v4(self,group=1,debug=False,initial_salary=None,spouse=False):
        '''
        Alussa ajettava funktio, joka tekee palkat yhtä episodia varten
        '''
        n_time = int(np.round((self.max_age-self.min_age)*self.inv_timestep))+2
        salary=np.zeros(n_time)

        if group>2: # naiset
            r=self.g_r[0,group-3]
            if initial_salary is not None:
                a0=initial_salary
            else:
                a0=self.palkat_ika_naiset[0]*r
            
            a1=self.palkat_ika_naiset[0]*r/5
            salary[0]=np.maximum(self.min_salary,np.random.normal(loc=a0,scale=a1,size=1)[0]) # e/y

            k=0
            r0=self.g_r[0,group-3]
            a0=self.palkat_ika_naiset[0]*r0
            s0=salary[0]
            for age in np.arange(self.min_age,self.max_age,self.timestep):
                r1=self.g_r[k,group-3]
                a1=self.palkat_ika_naiset[k]*r1
                salary[self.map_age(age)]=self.wage_process_TK_v3(s0,a0,a1,g=group-3)
                k=k+1
                s0=salary[self.map_age(age)]
                a0=a1
                r0=r1
        else: # miehet
            r=self.g_r[0,group]
            if initial_salary is not None:
                a0=initial_salary
            else:
                a0=self.palkat_ika_miehet[0]*r
                
            a1=self.palkat_ika_miehet[0]*r/5
            salary[0]=np.maximum(self.min_salary,np.random.normal(loc=a0,scale=a1,size=1)[0]) # e/y

            k=0
            r0=self.g_r[0,group]
            a0=self.palkat_ika_miehet[0]*r0
            s0=salary[0]
            for age in np.arange(self.min_age,self.max_age,self.timestep):
                r1=self.g_r[k,group]
                a1=self.palkat_ika_miehet[k]*r1
                salary[self.map_age(age)]=self.wage_process_TK_v3(s0,a0,a1,g=group)
                k=k+1
                s0=salary[self.map_age(age)]
                a0=a1
                r0=r1
                
        if spouse:
            self.spousesalary=salary
        else:
            self.salary=salary
            
    def state_encode(self,emp,g,pension,old_wage,age,time_in_state,tyoelake_maksussa,pink,
                        toe,toekesto,tyohist,next_wage,used_unemp_benefit,wage_reduction,
                        unemp_after_ra,unempwage,unempwage_basis,
                        children_under3,children_under7,children_under18,
                        unemp_benefit_left,alkanut_ansiosidonnainen,toe58,ove_paid,kassanjasenyys,
                        puoliso,puoliso_tila,puoliso_old_wage,puoliso_pension,
                        puoliso_wage_reduction,puoliso_tyoelake_maksussa,puoliso_next_wage,
                        puoliso_used_unemp_benefit,puoliso_unemp_benefit_left,
                        puoliso_unemp_after_ra,puoliso_unempwage,puoliso_unempwage_basis,
                        puoliso_alkanut_ansiosidonnainen,puoliso_toe58,
                        puoliso_toe,puoliso_toekesto,puoliso_tyoura,puoliso_time_in_state,
                        puoliso_pinkslip,puoliso_ove_paid,kansanelake,puoliso_kansanelake,
                        prefnoise):     
        '''
        Tilan koodaus neuroverkkoa varten. Arvot skaalataan ja tilat one-hot-enkoodataan

        Käytetään, jos kuolleisuus ei mukana
        '''
        if self.include_preferencenoise:
            d=np.zeros(self.n_empl+self.n_groups+self.n_empl+48)
        else:
            d=np.zeros(self.n_empl+self.n_groups+self.n_empl+47)
                    
        states=self.n_empl
        # d2=np.zeros(n_empl,1)
        # d2[emp]=1
        d[0:states]=self.state_encoding[emp,:]
        if emp==14 and not self.include_mort:
            print('no state 14 in state_encode_nomort')
        elif emp>14:
            print('state_encode error '+str(emp))

        states2=states+self.n_groups
        d[states:states2]=self.group_encoding[g,:]

        if self.log_transform:
            d[states2]=np.log(pension/20_000+self.eps) # vastainen eläke
            d[states2+1]=np.log(old_wage/40_000+self.eps)
            d[states2+4]=np.log(tyoelake_maksussa/20_000+self.eps) # alkanut eläke
            d[states2+10]=np.log(next_wage/40_000+self.eps)
            d[states2+14]=np.log(unempwage/40_000+self.eps)
            d[states2+15]=np.log(unempwage_basis/40_000+self.eps)
        else:
            d[states2]=(pension-40_000)/40_000 # vastainen eläke
            d[states2+1]=(old_wage-40_000)/40_000
            d[states2+4]=(tyoelake_maksussa-40_000)/40_000 # alkanut eläke
            d[states2+10]=(next_wage-40_000)/40_000
            d[states2+14]=(unempwage-40_000)/40_000
            d[states2+15]=(unempwage_basis-40_000)/40_000

        d[states2+2]=(age-(self.max_age+self.min_age)/2)/20
        d[states2+3]=(time_in_state-10)/10
        if age>=self.min_retirementage:
            retaged=1
        else:
            retaged=0

        d[states2+5]=pink # irtisanottu vai ei 
        d[states2+6]=toe-14/12 # työssäoloehto
        d[states2+7]=(tyohist-10)/20 # tyohistoria: 300/400 pv
        d[states2+8]=(self.min_retirementage-age)/43
        d[states2+9]=unemp_benefit_left-1 #retaged
        d[states2+11]=used_unemp_benefit-1
        d[states2+12]=wage_reduction
        d[states2+13]=(unemp_after_ra-1)/2
        d[states2+16]=retaged
        d[states2+17]=alkanut_ansiosidonnainen
        d[states2+18]=(children_under3-5)/10
        d[states2+19]=(children_under7-5)/10
        d[states2+20]=(children_under18-5)/10
        d[states2+21]=toe58
        d[states2+22]=ove_paid
        if age>=self.min_ove_age:
            d[states2+23]=1
        
        d[states2+24]=kassanjasenyys
        d[states2+25]=toekesto-14/12
        d[states2+26]=puoliso
        states3=states2+27+self.n_spouseempl
        d[(states2+27):states3]=self.spousestate_encoding[puoliso_tila,:]        

        d[states3]=(puoliso_old_wage-40_000)/40_000
        d[states3+1]=(puoliso_pension-40_000)/40_000
        d[states3+2]=puoliso_wage_reduction
        d[states3+3]=(puoliso_tyoelake_maksussa-40_000)/40_000 # alkanut eläke
        d[states3+4]=(puoliso_next_wage-40_000)/40_000
        d[states3+5]=puoliso_used_unemp_benefit-1
        d[states3+6]=puoliso_unemp_benefit_left-1
        d[states3+7]=(puoliso_unemp_after_ra-1)/2
        d[states3+8]=(puoliso_unempwage-40_000)/40_000
        d[states3+9]=(puoliso_unempwage_basis-40_000)/40_000
        d[states3+10]=puoliso_alkanut_ansiosidonnainen
        d[states3+11]=puoliso_toe58
        d[states3+12]=puoliso_toe-14/12
        d[states3+13]=puoliso_toekesto-14/12
        d[states3+14]=(puoliso_tyoura-10)/20
        d[states3+15]=(puoliso_time_in_state-10)/10
        d[states3+16]=puoliso_pinkslip
        d[states3+17]=puoliso_ove_paid
        
        d[states3+18]=(kansanelake-40_000)/40_000
        d[states3+19]=(puoliso_kansanelake-40_000)/40_000
        
        if self.include_preferencenoise:
            d[states3+20]=prefnoise
        
        return d

    def get_spouse_g(self,g):
        '''
        Gives spouse group
        The assumption here is that spouse's gender is different (for simplicity) and
        that otherwise the group is the same (for simplicity)
        '''
        if g>2:
            spouse_g=g-3
        else:
            spouse_g=g+3
            
        return spouse_g

    def state_decode(self,vec):
        '''
        Tilan dekoodaus laskentaa varten

        Käytetään, jos aina
        '''

        emp=-1
        for k in range(self.n_empl):
            if vec[k]>0:
                emp=k
                break

        if emp<0:
            print('state error '+str(vec))

        g=-1
        pos=self.n_empl+self.n_groups
        for k in range(self.n_empl,pos):
            if vec[k]>0:
                g=k-self.n_empl
                break

        if g<0:
            print('state error '+str(vec))
        
        g=int(g)
        spouse_g=self.get_spouse_g(g)

        if self.log_transform:
            pension=(np.exp(vec[pos])-self.eps)*20_000
            wage=(np.exp(vec[pos+1])-self.eps)*40_000
            next_wage=(np.exp(vec[pos+10])-self.eps)*40_000
            tyoelake_maksussa=(np.exp(vec[pos+4])-self.eps)*20_000
            unempwage=(np.exp(vec[pos+14])-self.eps)*40_000
            unempwage_basis=(np.exp(vec[pos+15])-self.eps)*40_000
        else:
            pension=vec[pos]*40_000+40_000
            wage=vec[pos+1]*40_000+40_000 
            next_wage=vec[pos+10]*40_000+40_000 
            tyoelake_maksussa=vec[pos+4]*40_000+40_000
            unempwage=vec[pos+14]*40_000+40_000 
            unempwage_basis=vec[pos+15]*40_000+40_000 

        age=vec[pos+2]*20+(self.max_age+self.min_age)/2
        time_in_state=vec[pos+3]*10+10
        #if self.include300:
        pink=int(vec[pos+5]) # irtisanottu vai ei 
        toe=vec[pos+6]+14/12 # työssäoloehto, kesto
        tyohist=vec[pos+7]*20+10 # työhistoria
        used_unemp_benefit=vec[pos+11]+1 # käytetty työttömyyspäivärahapäivien määrä
        wage_reduction=vec[pos+12]
        unemp_after_ra=vec[pos+13]*2+1
        unemp_left=vec[pos+9]+1
        alkanut_ansiosidonnainen=int(vec[pos+17])
        children_under3=int(vec[pos+18]*10+5)
        children_under7=int(vec[pos+19]*10+5)
        children_under18=int(vec[pos+20]*10+5)
        toe58=int(vec[pos+21])
        ove_paid=int(vec[pos+22])
        kassanjasen=int(vec[pos+24])
        toekesto=vec[pos+25]+14/12
        puoliso=int(vec[pos+26])
        #puoliso_tila=int(vec[pos+27])
        pos2=pos+27+self.n_empl
        
        puoliso_tila=-1
        for k in range(self.n_spouseempl):
            if vec[pos+27+k]>0:
                puoliso_tila=k
                break
                
        puoliso_old_wage=vec[pos2+0]*40_000+40_000
        puoliso_pension=vec[pos2+1]*40_000+40_000
        puoliso_wage_reduction=vec[pos2+2]
        puoliso_tyoelake_maksussa=vec[pos2+3]*40_000+40_000
        puoliso_next_wage=vec[pos2+4]*40_000+40_000
        puoliso_used_unemp_benefit=vec[pos2+5]+1
        puoliso_unemp_benefit_left=vec[pos2+6]+1
        puoliso_unemp_after_ra=2*vec[pos2+7]+1
        puoliso_unempwage=vec[pos2+8]*40_000+40_000
        puoliso_unempwage_basis=vec[pos2+9]*40_000+40_000
        puoliso_alkanut_ansiosidonnainen=int(vec[pos2+10])
        puoliso_toe58=int(vec[pos2+11])
        puoliso_toe=vec[pos2+12]+14/12
        puoliso_toekesto=vec[pos2+13]+14/12
        puoliso_tyoura=vec[pos2+14]*20+10
        puoliso_time_in_state=vec[pos2+15]*10+10
        puoliso_pinkslip=int(vec[pos2+16])
        puoliso_ove_paid=int(vec[pos2+17])
        
        kansanelake=vec[pos2+18]*40_000+40_000
        puoliso_kansanelake=vec[pos2+19]*40_000+40_000
        paid_pension=tyoelake_maksussa+kansanelake
        puoliso_paid_pension=puoliso_tyoelake_maksussa+puoliso_kansanelake

        if self.include_preferencenoise:
            prefnoise=vec[pos2+20]
        else:
            prefnoise=0
        #else:
        #    children_under3=0
        #    children_under7=0
        #    children_under18=0
        #    if self.include_preferencenoise:
        #        prefnoise=vec[pos+18]
        #    else:
        #        prefnoise=0

        return int(emp),g,pension,wage,age,time_in_state,paid_pension,pink,toe,toekesto,\
               tyohist,used_unemp_benefit,wage_reduction,unemp_after_ra,\
               unempwage,unempwage_basis,prefnoise,children_under3,children_under7,children_under18,\
               unemp_left,alkanut_ansiosidonnainen,toe58,ove_paid,kassanjasen,puoliso,puoliso_tila,spouse_g,\
               puoliso_old_wage,puoliso_pension,puoliso_wage_reduction,puoliso_paid_pension,puoliso_next_wage,\
               puoliso_used_unemp_benefit,puoliso_unemp_benefit_left,\
               puoliso_unemp_after_ra,puoliso_unempwage,puoliso_unempwage_basis,\
               puoliso_alkanut_ansiosidonnainen,puoliso_toe58,\
               puoliso_toe,puoliso_toekesto,puoliso_tyoura,puoliso_time_in_state,puoliso_pinkslip,puoliso_ove_paid,\
               kansanelake,puoliso_kansanelake,tyoelake_maksussa,puoliso_tyoelake_maksussa,next_wage
                              
    def unit_test_code_decode(self):
        for k in range(10):
            emp=random.randint(0,3)
            g=np.random.randint(0,6)
            pension=np.random.uniform(0,80_000)
            old_wage=np.random.uniform(0,80_000)
            age=np.random.randint(0,60)
            time_in_state=np.random.uniform(0,30)
            pink=np.random.randint(2)
            toe=np.random.uniform(0,3)
            tyohist=np.random.uniform(0,20)
            next_wage=np.random.uniform(0,80_000)
            used_unemp_benefit=np.random.uniform(0,20)
            wage_reduction=np.random.uniform(0,1.0)
            unemp_after_ra=np.random.uniform(0,10.0)
            unempwage=np.random.uniform(0,80_000)
            unempwage_basis=np.random.uniform(0,80_000)
            prefnoise=np.random.uniform(-1,1)
            children_under3=np.random.randint(0,10)
            children_under7=np.random.randint(0,10)
            children_under18=np.random.randint(0,10)
            unemp_benefit_left=np.random.randint(0,10)
            alkanut_ansiosidonnainen=np.random.randint(0,2)
            toe58=np.random.randint(0,2)
            ove_paid=np.random.randint(0,2)
            kassanjasenyys=np.random.randint(0,2)
            toekesto=np.random.uniform(0,3)
            puoliso=np.random.randint(0,2)
            puoliso_tila=np.random.randint(0,3)
            puoliso_wage_reduction=np.random.uniform(0,50000)
            puoliso_pension=np.random.uniform(0,50000)
            puoliso_old_wage=np.random.uniform(0,50000)
            puoliso_next_wage=np.random.uniform(0,50000)
            puoliso_used_unemp_benefit=np.random.uniform(0,10)
            puoliso_unemp_benefit_left=np.random.uniform(0,10)
            puoliso_unemp_after_ra=np.random.uniform(0,10)
            puoliso_unempwage=np.random.uniform(0,50000)
            puoliso_unempwage_basis=np.random.uniform(0,50000)
            puoliso_alkanut_ansiosidonnainen=np.random.randint(0,3)
            puoliso_toe58=np.random.randint(0,2)
            puoliso_toe=np.random.uniform(0,20)
            puoliso_toekesto=np.random.uniform(0,20)
            puoliso_tyoura=np.random.uniform(0,40)
            puoliso_time_in_state=np.random.uniform(0,40)
            puoliso_pinkslip=np.random.randint(0,2)
            puoliso_ove_paid=np.random.randint(0,2)
            kansanelake=np.random.uniform(0,50000)
            puoliso_kansanelake=np.random.uniform(0,50000)
            tyoelake_maksussa=np.random.uniform(0,50000)
            puoliso_tyoelake_maksussa=np.random.uniform(0,50000)
            paid_pension=kansanelake+tyoelake_maksussa
            puoliso_paid_pension=puoliso_kansanelake+puoliso_tyoelake_maksussa
        
            vec=self.state_encode(emp,g,pension,old_wage,age,time_in_state,tyoelake_maksussa,pink,
                                toe,toekesto,tyohist,next_wage,used_unemp_benefit,wage_reduction,
                                unemp_after_ra,unempwage,unempwage_basis,children_under3,
                                children_under7,children_under18,unemp_benefit_left,alkanut_ansiosidonnainen,
                                toe58,ove_paid,kassanjasenyys,puoliso,puoliso_tila,puoliso_old_wage,puoliso_pension,
                                puoliso_wage_reduction,puoliso_tyoelake_maksussa,puoliso_next_wage,
                                puoliso_used_unemp_benefit,puoliso_unemp_benefit_left,
                                puoliso_unemp_after_ra,puoliso_unempwage,puoliso_unempwage_basis,
                                puoliso_alkanut_ansiosidonnainen,puoliso_toe58,
                                puoliso_toe,puoliso_toekesto,puoliso_tyoura,puoliso_time_in_state,
                                puoliso_pinkslip,puoliso_ove_paid,kansanelake,puoliso_kansanelake,
                                prefnoise)
                                
            emp2,g2,pension2,wage2,age2,time_in_state2,paid_pension2,pink2,toe2,toekesto2,\
                tyohist2,used_unemp_benefit2,wage_reduction2,unemp_after_ra2,\
                unempwage2,unempwage_basis2,prefnoise2,\
                children_under3_2,children_under7_2,children_under18_2,unemp_benefit_left2,\
                alkanut_ansiosidonnainen2,toe58_2,ove_paid_2,jasen_2,p2,p2_tila,p2_g,p2_old_wage,p2_pension,\
                p2_wage_reduction,p2_paid_pension,p2_next_wage,p2_used_unemp_benefit,p2_unemp_benefit_left,\
                p2_unemp_after_ra,p2_unempwage,p2_unempwage_basis,p2_alkanut_ansiosidonnainen,p2_toe58,p2_toe,\
                p2_toekesto,p2_tyoura,p2_time_in_state,p2_pinkslip,p2_ove_paid,\
                kansanelake2,puoliso_kansanelake2,tyoelake_maksussa2,puoliso_tyoelake_maksussa2,next_wage2\
                =self.state_decode(vec)
                
            self.check_state(emp,g,pension,old_wage,age,time_in_state,paid_pension,pink,
                                toe,tyohist,next_wage,used_unemp_benefit,wage_reduction,
                                unemp_after_ra,unempwage,unempwage_basis,
                                prefnoise,children_under3,children_under7,children_under18,
                                unemp_benefit_left,alkanut_ansiosidonnainen,toe58,ove_paid,
                                kassanjasenyys,puoliso,puoliso_tila,puoliso_old_wage,puoliso_pension,
                                puoliso_wage_reduction,puoliso_paid_pension,puoliso_next_wage,
                                puoliso_used_unemp_benefit,puoliso_unemp_benefit_left,
                                puoliso_unemp_after_ra,puoliso_unempwage,puoliso_unempwage_basis,
                                puoliso_alkanut_ansiosidonnainen,puoliso_toe58,
                                puoliso_toe,puoliso_toekesto,puoliso_tyoura,puoliso_time_in_state,puoliso_pinkslip,puoliso_ove_paid,
                                kansanelake,puoliso_kansanelake,tyoelake_maksussa,puoliso_tyoelake_maksussa,
                                emp2,g2,pension2,wage2,age2,time_in_state2,paid_pension2,pink2,toe2,
                                tyohist2,used_unemp_benefit2,wage_reduction2,unemp_after_ra2,
                                unempwage2,unempwage_basis2,prefnoise2,
                                children_under3_2,children_under7_2,children_under18_2,
                                unemp_benefit_left2,alkanut_ansiosidonnainen2,toe58_2,
                                ove_paid_2,jasen_2,p2,p2_tila,p2_old_wage,p2_pension,
                                p2_wage_reduction,p2_paid_pension,p2_next_wage,
                                p2_used_unemp_benefit,p2_unemp_benefit_left,
                                p2_unemp_after_ra,p2_unempwage,p2_unempwage_basis,p2_alkanut_ansiosidonnainen,p2_toe58,
                                p2_toe,p2_toekesto,p2_tyoura,p2_time_in_state,p2_pinkslip,p2_ove_paid,
                                kansanelake2,puoliso_kansanelake2,tyoelake_maksussa2,puoliso_tyoelake_maksussa2,
                                next_wage2)
        
    
    def check_state(self,emp,g,pension,old_wage,age,time_in_state,paid_pension,pink,
                                toe,tyohist,next_wage,used_unemp_benefit,wage_reduction,
                                unemp_after_ra,unempwage,unempwage_basis,
                                prefnoise,children_under3,children_under7,children_under18,
                                unemp_benefit_left,alkanut_ansiosidonnainen,toe58,ove_paid,
                                jasen,puoliso,puoliso_tila,puoliso_old_wage,puoliso_pension,
                                puoliso_wage_reduction,puoliso_paid_pension,puoliso_next_wage,
                                puoliso_used_unemp_benefit,puoliso_unemp_benefit_left,
                                puoliso_unemp_after_ra,puoliso_unempwage,puoliso_unempwage_basis,
                                puoliso_alkanut_ansiosidonnainen,puoliso_toe58,
                                puoliso_toe,puoliso_toekesto,puoliso_tyoura,puoliso_time_in_state,puoliso_pinkslip,puoliso_ove_paid,
                                kansanelake,puoliso_kansanelake,tyoelake_maksussa,puoliso_tyoelake_maksussa,
                                emp2,g2,pension2,old_wage2,age2,time_in_state2,paid_pension2,pink2,toe2,
                                tyohist2,used_unemp_benefit2,wage_reduction2,unemp_after_ra2,
                                unempwage2,unempwage_basis2,prefnoise2,
                                children_under3_2,children_under7_2,children_under18_2,
                                unemp_benefit_left2,alkanut_ansiosidonnainen2,toe58_2,
                                ove_paid_2,jasen2,p2,p2_tila,p2_old_wage,p2_pension,
                                p2_wage_reduction,p2_paid_pension,p2_next_wage,
                                p2_used_unemp_benefit,p2_unemp_benefit_left,
                                p2_unemp_after_ra,p2_unempwage,p2_unempwage_basis,p2_alkanut_ansiosidonnainen,p2_toe58,
                                p2_toe,p2_toekesto,p2_tyoura,p2_time_in_state,p2_pinkslip,p2_ove_paid,
                                kansanelake2,puoliso_kansanelake2,tyoelake_maksussa2,puoliso_tyoelake_maksussa2,
                                next_wage2):
        if not emp==emp2:  
            print('emp: {} vs {}'.format(emp,emp2))
        if not g==g2:  
            print('g: {} vs {}'.format(g,g2))
        if not math.isclose(pension,pension2):  
            print('pension: {} vs {}'.format(pension,pension2))
        if not math.isclose(old_wage,old_wage2):  
            print('old_wage: {} vs {}'.format(old_wage,old_wage2))
        if not age==age2:  
            print('age: {} vs {}'.format(age,age2))
        if not math.isclose(time_in_state,time_in_state2):  
            print('time_in_state: {} vs {}'.format(time_in_state,time_in_state2))
        if not math.isclose(paid_pension,paid_pension2):
            print('paid_pension: {} vs {}'.format(paid_pension,paid_pension2))
        if not pink==pink2:  
            print('pink: {} vs {}'.format(pink,pink2))
        if not math.isclose(tyohist,tyohist2):  
            print('tyohist: {} vs {}'.format(tyohist,tyohist2))
        if not math.isclose(next_wage,next_wage2):
            print('next_wage: {} vs {}'.format(next_wage,next_wage2))
        if not math.isclose(used_unemp_benefit,used_unemp_benefit2):  
            print('used_unemp_benefit: {} vs {}'.format(used_unemp_benefit,used_unemp_benefit2))
        if not math.isclose(wage_reduction,wage_reduction2):  
            print('wage_reduction: {} vs {}'.format(wage_reduction,wage_reduction2))
        if not unemp_after_ra==unemp_after_ra2:  
            print('unemp_after_ra: {} vs {}'.format(unemp_after_ra,unemp_after_ra2))
        if not math.isclose(unempwage,unempwage2):  
            print('unempwage: {} vs {}'.format(unempwage,unempwage2))
        if not math.isclose(unempwage_basis,unempwage_basis2):  
            print('unempwage_basis: {} vs {}'.format(unempwage_basis,unempwage_basis2))
        if self.include_preferencenoise:
            if not prefnoise==prefnoise2:  
                print('prefnoise: {} vs {}'.format(prefnoise,prefnoise2))
        if not children_under3==children_under3_2:  
            print('children_under3: {} vs {}'.format(children_under3,children_under3_2))
        if not children_under7==children_under7_2:  
            print('children_under7: {} vs {}'.format(children_under7,children_under7_2))
        if not children_under18==children_under18_2:  
            print('children_under18: {} vs {}'.format(children_under18,children_under18_2))
        if not math.isclose(unemp_benefit_left,unemp_benefit_left2):  
            print('unemp_benefit_left2: {} vs {}'.format(unemp_benefit_left,unemp_benefit_left2))
        if not alkanut_ansiosidonnainen==alkanut_ansiosidonnainen2:  
            print('alkanut_ansiosidonnainen: {} vs {}'.format(alkanut_ansiosidonnainen,alkanut_ansiosidonnainen2))
        if not toe58==toe58_2:  
            print('toe58: {} vs {}'.format(toe58,toe58_2))
        if not ove_paid==ove_paid_2:  
            print('ove_paid: {} vs {}'.format(ove_paid,ove_paid_2))
        if not jasen==jasen2:  
            print('jasen: {} vs {}'.format(jasen,jasen2))
        if not puoliso==p2:  
            print('puoliso: {} vs {}'.format(puoliso,p2))
        if not puoliso_tila==p2_tila:  
            print('puoliso_tila: {} vs {}'.format(puoliso_tila,p2_tila))
        if not math.isclose(puoliso_old_wage,p2_old_wage):  
            print('puoliso_old_wage: {} vs {}'.format(puoliso_old_wage,p2_old_wage))
        if not math.isclose(puoliso_wage_reduction,p2_wage_reduction):  
            print('puoliso_wage_reduction: {} vs {}'.format(puoliso_wage_reduction,p2_wage_reduction))
        if not math.isclose(puoliso_paid_pension,p2_paid_pension):  
            print('puoliso_paid_pension: {} vs {}'.format(puoliso_paid_pension,p2_paid_pension))
        if not math.isclose(puoliso_next_wage,p2_next_wage):  
            print('puoliso_next_wage: {} vs {}'.format(puoliso_next_wage,p2_next_wage))
        if not math.isclose(puoliso_used_unemp_benefit,p2_used_unemp_benefit):  
            print('puoliso_next_wage: {} vs {}'.format(puoliso_used_unemp_benefit,p2_used_unemp_benefit))
        if not math.isclose(puoliso_unemp_benefit_left,p2_unemp_benefit_left):  
            print('puoliso_unemp_benefit_left: {} vs {}'.format(puoliso_unemp_benefit_left,p2_unemp_benefit_left))
        if not math.isclose(puoliso_unemp_after_ra,p2_unemp_after_ra):  
            print('puoliso_unemp_after_ra: {} vs {}'.format(puoliso_unemp_after_ra,p2_unemp_after_ra))
        if not math.isclose(puoliso_unempwage,p2_unempwage):  
            print('puoliso_unempwage: {} vs {}'.format(puoliso_unempwage,p2_unempwage))
        if not math.isclose(puoliso_unempwage_basis,p2_unempwage_basis):  
            print('puoliso_unempwage_basis: {} vs {}'.format(puoliso_unempwage_basis,p2_unempwage_basis))
        if not math.isclose(puoliso_alkanut_ansiosidonnainen,p2_alkanut_ansiosidonnainen):  
            print('puoliso_alkanut_ansiosidonnainen: {} vs {}'.format(puoliso_alkanut_ansiosidonnainen,p2_alkanut_ansiosidonnainen))
        if not math.isclose(puoliso_toe58,p2_toe58):  
            print('puoliso_toe58: {} vs {}'.format(puoliso_toe58,p2_toe58))
        if not math.isclose(puoliso_toe,p2_toe):  
            print('puoliso_toe: {} vs {}'.format(puoliso_toe,p2_toe))
        if not math.isclose(puoliso_toekesto,p2_toekesto):  
            print('puoliso_toekesto: {} vs {}'.format(puoliso_toekesto,p2_toekesto))
        if not math.isclose(puoliso_tyoura,p2_tyoura):  
            print('puoliso_tyoura: {} vs {}'.format(puoliso_tyoura,p2_tyoura))
        if not math.isclose(puoliso_time_in_state,p2_time_in_state):  
            print('puoliso_time_in_state: {} vs {}'.format(puoliso_time_in_state,p2_time_in_state))
        if not math.isclose(puoliso_pinkslip,p2_pinkslip):  
            print('puoliso_pinkslip: {} vs {}'.format(puoliso_pinkslip,p2_pinkslip))
        if not math.isclose(puoliso_ove_paid,p2_ove_paid):  
            print('puoliso_ove_paid: {} vs {}'.format(puoliso_ove_paid,p2_ove_paid))
        if not math.isclose(kansanelake,kansanelake2):  
            print('kansanelake: {} vs {}'.format(kansanelake2,kansanelake2))
        if not math.isclose(puoliso_kansanelake,kansanelake):  
            print('puoliso_kansanelake: {} vs {}'.format(puoliso_kansanelake,puoliso_kansanelake2))
        if not math.isclose(tyoelake_maksussa,tyoelake_maksussa2):  
            print('tyoelake_maksussa: {} vs {}'.format(tyoelake_maksussa,tyoelake_maksussa2))
        if not math.isclose(puoliso_tyoelake_maksussa,puoliso_tyoelake_maksussa2):  
            print('puoliso_tyoelake_maksussa: {} vs {}'.format(puoliso_tyoelake_maksussa,puoliso_tyoelake_maksussa2))
    
    def reset(self,init=None):
        '''
        Open AI-interfacen mukainen reset-funktio, joka nollaa laskennan alkutilaan
        '''
        self.init_state()
        self.steps_beyond_done = None
        
        if self.plotdebug:
            self.render()

        return np.array(self.state)
        
#     def get_initstate(self):
#         if employment_state==0:
#             tyohist=1.0
#             toe=0.0
#             toekesto=1.0
#             wage_reduction=np.random.uniform(low=0.05,high=0.35)
#             used_unemp_benefit=0.0
#             unempwage_basis=old_wage
#             alkanut_ansiosidonnainen=1
#             unempwage=0
#         elif employment_state==13:
#             tyohist=0.0
#             toe=0.0
#             wage_reduction=np.random.uniform(low=0.10,high=0.50) # 20-70
#             used_unemp_benefit=2.0
#         elif employment_state==12:
#             tyohist=0.0
#             toe=0.0
#             wage_reduction=np.random.uniform(low=0.10,high=0.30)
#             used_unemp_benefit=0.0
#         elif employment_state==11:
#             tyohist=0.0
#             toe=0.0
#             wage_reduction=np.random.uniform(low=0.15,high=0.50) # 15-50
#         elif employment_state==3:
#             wage5y=next_wage
#             paid_pension=pension
#             # takuueläke voidaan huomioida jo tässä
#             paid_pension=self.ben.laske_kokonaiselake(age,paid_pension/12,include_kansanelake=self.include_kansanelake,include_takuuelake=False,disability=True)*12
#             pension=0
#         elif employment_state==2:
#             wage5y=next_wage
#             paid_pension=pension
#             # takuueläke voidaan huomioida jo tässä
#             paid_pension=self.ben.laske_kokonaiselake(age,paid_pension/12,include_kansanelake=self.include_kansanelake,include_takuuelake=False,disability=True)*12
#             pension=0    
    
    def get_initial_state(self,puoliso,is_spouse=False,g=-1):    
        '''
        Alusta tila
        '''
        age=int(self.min_age)
        pension=0
        time_in_state=0
        pink=0
        toe=0
        toekesto=0
        tyohist=0
        wage_reduction=0
        used_unemp_benefit=0        
        unemp_after_ra=0
        unempwage=0
        unempwage_basis=0
        children_under3=0
        children_under7=0
        children_under18=0
        alkanut_ansiosidonnainen=0
        toe58=0
        unemp_benefit_left=0
        ove_paid=0
        kassanjasenyys=0
        paid_pension=0
        kansanelake=0
        tyoelake_maksussa=0
        
        # set up salary for the entire career
        if is_spouse:
            group=g
        else:
            g=random.choices(np.array([0,1,2],dtype=int),weights=[0.3,0.5,0.2])[0]
            gender=random.choices(np.array([0,1],dtype=int),weights=[0.5,0.5])[0]
            group=int(g+gender*3)
                
        employment_state=random.choices(np.array([13,0,1,10,3,11,12],dtype=int),
                weights=self.initial_weights[group,:])[0]
                
        #if employment_state==3:
        #    self.dis_ratio+=1
        #self.nnn+=1
        
        #print(self.dis_ratio/self.nnn*100)
                
        #print(self.initial_weights[group,:],'yht',np.sum(self.initial_weights[group,:]))
                
    
        self.init_infostate(age=age,spouse=is_spouse)

        initial_salary=None
        reset_exp=False
#         if self.reset_exploration_go and self.train:
#             if self.reset_exploration_ratio>np.random.uniform():
#                 #print('exploration')
#                 initial_salary=np.random.uniform(low=1_000,high=100_000)
#                 pension=random.uniform(0,80_000)
#                 kassanjasenyys=np.random.randint(2)
#                 
#                 if random.random()<0.5:
#                     age=int(np.random.uniform(low=self.min_age,high=self.max_age-1))
#                 #else:
#                 #    age=int(np.random.uniform(low=62,high=self.max_age-1))
#                 if age<60:
#                     employment_state=random.choices(np.array([13,0,1,10,3,11,12],dtype=int),
#                         weights=[0.1,0.1,0.6,0.2,0.05,0.05,0.05])[0]
#                 elif age<self.min_retirementage:
#                     employment_state=random.choices(np.array([13,0,1,10,3,11,12,4],dtype=int),
#                         weights=[0.1,0.1,0.6,0.2,0.05,0.05,0.05,0.1])[0]
#                 else:
#                     employment_state=random.choices(np.array([1,2,8,9,3,10],dtype=int),
#                         weights=[0.2,0.5,0.2,0.1,0.1,0.2])[0]
#                         
#                 initial_salary=np.random.uniform(low=1_000,high=100_000)
#                 toe=random.choices(np.array([0,0.25,0.5,0.75,1.0,1.5,2.0,2.5],dtype=float),
#                     weights=[0.3,0.1,0.1,0.1,0.1,0.1,0.1,0.1])[0]
#                 tyohist=random.choices(np.array([0,0.25,0.5,0.75,1.0,1.5,2.0,2.5],dtype=float),
#                     weights=[0.3,0.1,0.1,0.1,0.1,0.1,0.1,0.1])[0]
#                 reset_exp=True
        
        self.compute_salary_v4(group=group,initial_salary=initial_salary,spouse=is_spouse)
            
        if not reset_exp:
            if employment_state==0:
                wage_reduction=np.random.uniform(low=0.05,high=0.35)
            elif employment_state==13:
                wage_reduction=np.random.uniform(low=0.10,high=0.50) # 20-70
            elif employment_state==12:
                wage_reduction=np.random.uniform(low=0.10,high=0.30)
            elif employment_state==11:
                wage_reduction=np.random.uniform(low=0.15,high=0.50) # 15-50
            elif employment_state==3:
                pension=0
            elif employment_state==2:
                pension=0
        else:        
            if employment_state==0:
                wage_reduction=np.random.uniform(low=0.0,high=0.45)
            elif employment_state==13:
                wage_reduction=np.random.uniform(low=0.0,high=0.60)
            elif employment_state==10:
                wage_reduction=np.random.uniform(low=0.0,high=0.60)
            elif employment_state==12:
                wage_reduction=np.random.uniform(low=0.0,high=0.50)
            elif employment_state==11:
                wage_reduction=np.random.uniform(low=0.0,high=0.35)
            elif employment_state==3:
                pension=0
            elif employment_state==2:
                pension=0
        
        if is_spouse:
            old_wage=self.get_spousewage(self.min_age,wage_reduction)
        else:
            old_wage=self.get_wage(self.min_age,wage_reduction)
        next_wage=old_wage
        
        if not reset_exp:
            if employment_state==0:
                tyohist=1.0
                toe=0.0
                toekesto=1.0
                used_unemp_benefit=0.0
                unempwage_basis=old_wage
                alkanut_ansiosidonnainen=1
                unempwage=0
            elif employment_state==13:
                tyohist=0.0
                toe=0.0
                used_unemp_benefit=2.0
            elif employment_state==12:
                tyohist=0.0
                toe=0.0
                used_unemp_benefit=0.0
            elif employment_state==11:
                tyohist=0.0
                toe=0.0
            elif employment_state==3:
                wage5y=next_wage
                tyoelake_maksussa=pension
                # takuueläke voidaan huomioida jo tässä
                #paid_pension=self.ben.laske_kokonaiselake_v2(age,paid_pension/12,kansanelake/12,include_kansanelake=self.include_kansanelake,include_takuuelake=False,disability=True)*12
                kansanelake = self.ben.laske_kansanelake(age,tyoelake_maksussa/12,1-puoliso)*12 # ben-modulissa palkat kk-tasolla
                pension=0
            elif employment_state==2:
                wage5y=next_wage
                tyoelake_maksussa=pension
                # takuueläke voidaan huomioida jo tässä
                #paid_pension=self.ben.laske_kokonaiselake_v2(age,paid_pension/12,kansanelake/12,include_kansanelake=self.include_kansanelake,include_takuuelake=False,disability=True)*12
                kansanelake = self.ben.laske_kansanelake(age,tyoelake_maksussa/12,1-puoliso)*12 # ben-modulissa palkat kk-tasolla
                pension=0
        else:        
            if employment_state==0:
                tyohist=np.random.uniform(low=0.0,high=age-18)
                toe=np.random.uniform(low=0.0,high=28/12)
                toekesto=toe
                used_unemp_benefit=np.random.uniform(low=0.0,high=2.0)
                unempwage_basis=old_wage
                alkanut_ansiosidonnainen=1
                unempwage=np.random.uniform(low=0.0,high=90_000.0)
            elif employment_state==13:
                tyohist=np.random.uniform(low=0.0,high=age-18)
                toe=0.0
                toekesto=toe
                used_unemp_benefit=2.0
            elif employment_state==10:
                tyohist=np.random.uniform(low=0.0,high=age-18)
                toe=0.0
                toekesto=toe
                used_unemp_benefit=2.0
            elif employment_state==12:
                tyohist=0.0
                toe=0.0
                used_unemp_benefit=0.0
            elif employment_state==11:
                tyohist=0.0
                toe=0.0
            elif employment_state==3:
                wage5y=next_wage
                tyoelake_maksussa=np.random.uniform(low=0.0,high=30_000)
                # takuueläke voidaan huomioida jo tässä
                #paid_pension=self.ben.laske_kokonaiselake_v2(age,paid_pension/12,kansanelake/12,include_kansanelake=self.include_kansanelake,include_takuuelake=False,disability=True)*12
                kansanelake = self.ben.laske_kansanelake(age,tyoelake_maksussa/12,1-puoliso)*12 # ben-modulissa palkat kk-tasolla
                pension=0
            elif employment_state==2:
                wage5y=next_wage
                tyoelake_maksussa=np.random.uniform(low=0.0,high=40_000)
                # takuueläke voidaan huomioida jo tässä
                #paid_pension=self.ben.laske_kokonaiselake_v2(age,paid_pension/12,kansanelake/12,include_kansanelake=self.include_kansanelake,include_takuuelake=False,disability=True)*12
                kansanelake = self.ben.laske_kansanelake(age,tyoelake_maksussa/12,1-puoliso)*12 # ben-modulissa palkat kk-tasolla
                pension=0
                
        if employment_state in set([1,10]):
            unempwage=old_wage
            
        unemp_benefit_left=self.comp_unempdays_left(used_unemp_benefit,tyohist,age,toe,employment_state,alkanut_ansiosidonnainen,toe58,toe)    

        kassanjasenyys=self.get_kassanjasenyys()
        if employment_state in set([0,4]):
            self.set_kassanjasenyys(1)
            
        return employment_state,group,pension,old_wage,age,time_in_state,paid_pension,pink,toe,toekesto,tyohist,next_wage,\
            used_unemp_benefit,wage_reduction,unemp_after_ra,unempwage,unempwage_basis,\
            children_under3,children_under7,children_under18,unemp_benefit_left,alkanut_ansiosidonnainen,toe58,\
            ove_paid,kassanjasenyys,kansanelake,tyoelake_maksussa

    
    def init_state(self):
    
        rn = np.random.uniform(size=1)
        if self.rates.get_initial_marriage_ratio()>rn:
            puoliso=1
        else:
            puoliso=0

        employment_state,group,pension,old_wage,age,time_in_state,paid_pension,pink,toe,toekesto,tyohist,next_wage,\
            used_unemp_benefit,wage_reduction,unemp_after_ra,unempwage,unempwage_basis,\
            children_under3,children_under7,children_under18,unemp_benefit_left,alkanut_ansiosidonnainen,toe58,\
            ove_paid,kassanjasenyys,kansanelake,tyoelake_maksussa\
            =self.get_initial_state(puoliso)
            
        spouse_g=self.get_spouse_g(group)

        puoliso_tila,_,puoliso_pension,puoliso_old_wage,_,puoliso_time_in_state,puoliso_paid_pension,puoliso_pink,puoliso_toe,\
            puoliso_toekesto,puoliso_tyohist,puoliso_next_wage,\
            puoliso_used_unemp_benefit,puoliso_wage_reduction,puoliso_unemp_after_ra,puoliso_unempwage,puoliso_unempwage_basis,\
            _,_,_,puoliso_unemp_benefit_left,puoliso_alkanut_ansiosidonnainen,puoliso_toe58,\
            puoliso_ove_paid,puoliso_kassanjasenyys,puoliso_kansanelake,puoliso_tyoelake_maksussa\
            =self.get_initial_state(puoliso,is_spouse=True,g=spouse_g)
            
        # tarvitseeko alkutilassa laskea muita tietoja uusiksi? ei kait

        if self.plotdebug:
            print('emp {} g {} old_wage {} next_wage {} age {} kassanjäsen {}'.format(employment_state,group,old_wage,next_wage,age,kassanjasenyys))
            print('emp {} g {} old_wage {} next_wage {} age {} kassanjäsen {}'.format(puoliso_tila,spouse_g,puoliso_old_wage,puoliso_next_wage,age,kassanjasenyys))

        if self.include_preferencenoise:
            # lognormaali
            #prefnoise=np.random.normal(loc=-0.5*self.preferencenoise_std*self.preferencenoise_std,scale=self.preferencenoise_std,size=1)[0]
            # normaali
            prefnoise=min(2.0,max(1e-6,np.random.normal(loc=1.0,scale=self.preferencenoise_std,size=1)[0]))
        else:
            prefnoise=0
            
        self.state = self.state_encode(employment_state,group,pension,old_wage,age,
                                       time_in_state,tyoelake_maksussa,pink,toe,toekesto,tyohist,next_wage,
                                       used_unemp_benefit,wage_reduction,unemp_after_ra,
                                       unempwage,unempwage_basis,
                                       children_under3,children_under7,children_under18,
                                       unemp_benefit_left,alkanut_ansiosidonnainen,toe58,
                                       ove_paid,kassanjasenyys,
                                       puoliso,puoliso_tila,puoliso_old_wage,puoliso_pension,
                                       puoliso_wage_reduction,puoliso_tyoelake_maksussa, puoliso_next_wage,
                                       puoliso_used_unemp_benefit,puoliso_unemp_benefit_left,
                                       puoliso_unemp_after_ra,
                                       puoliso_unempwage,puoliso_unempwage_basis,
                                       puoliso_alkanut_ansiosidonnainen,puoliso_toe58,puoliso_toe,
                                       puoliso_toekesto,puoliso_tyohist,puoliso_time_in_state,puoliso_pink,puoliso_ove_paid,
                                       kansanelake,puoliso_kansanelake,
                                       prefnoise)

    def render(self,mode='human',close=False,done=False,reward=None,netto=None,render_omat=True,render_puoliso=True,benq=None,netto_omat=None,netto_puoliso=None):
        '''
        Tulostus-rutiini
        '''
        emp,g,pension,wage,age,time_in_state,paid_pension,pink,toe,toekesto,tyohist,used_unemp_benefit,\
            wage_red,unemp_after_ra,unempwage,unempwage_basis,prefnoise,c3,c7,c18,\
            unemp_left,oikeus,toe58,ove_paid,jasen,\
            puoliso,puoliso_tila,spouse_g,puoliso_old_wage,puoliso_pension,\
            puoliso_wage_reduction,puoliso_paid_pension,puoliso_next_wage,\
            puoliso_used_unemp_benefit,puoliso_unemp_benefit_left,\
            puoliso_unemp_after_ra,puoliso_unempwage,puoliso_unempwage_basis,\
            puoliso_alkanut_ansiosidonnainen,puoliso_toe58,puoliso_toe,\
            puoliso_toekesto,puoliso_tyohist,puoliso_time_in_state,puoliso_pink,puoliso_ove_paid,\
            kansanelake,puoliso_kansanelake,tyoelake_maksussa,puoliso_tyoelake_maksussa,\
            next_wage\
                =self.state_decode(self.state)
            
        #if emp in set([0,4]):
        #    old_toe=self.comp_oldtoe()
        #    print(old_toe)
        #else:
        #    old_toe=0

        if jasen:
            kassassa='+'
        else:
            kassassa='-'
            
        out=f's {emp:d} g {g:d} w {wage:.0f} nw {next_wage:.0f} a {age:.2f} tis {time_in_state:.2f} pen {pension:.0f} paid_e {tyoelake_maksussa:.0f} paid_k {kansanelake:.0f}'+\
            f' irti {pink:d} toe {toe:.2f}{kassassa} tk{toekesto:.2f} ura {tyohist:.2f} ueb {used_unemp_benefit:.2f} red {wage_red:.2f} uew {unempwage:.0f}'+\
            f' (b {unempwage_basis:.0f}) c3 {c3:.0f} c7 {c7:.0f} c18 {c18:.0f} uleft {unemp_left:.2f} aa {oikeus:.0f} 58 {toe58:.0f} ove {ove_paid:.0f}'
            
        if reward is not None:
            out+=f' r {reward:.4f}'
        if netto_omat is not None:
            out+=f' n {netto_omat:.0f}'
            
        if render_omat:
            print(out)

        puoliso=f'ps {puoliso_tila:d} g {spouse_g:d} w {puoliso_old_wage:.0f} nw {puoliso_next_wage:.0f} a {age:.2f} p {puoliso:d} red {puoliso_wage_reduction:.2f}'+\
                f' pen {puoliso_pension:.0f} paid_e {puoliso_tyoelake_maksussa:.0f} paid_k {puoliso_kansanelake:.0f} ueb {puoliso_used_unemp_benefit:.2f}'+\
                f' uleft {puoliso_unemp_benefit_left:.2f} puar {puoliso_unemp_after_ra:.2f} uew {puoliso_unempwage:.0f} (b {puoliso_unempwage_basis:.0f})'+\
                f' aa {puoliso_alkanut_ansiosidonnainen:d} 58 {puoliso_toe58:d} toe {puoliso_toe:.2f}{kassassa} tk {puoliso_toekesto:.2f}'+\
                f' pink {puoliso_pink:d} ura {puoliso_tyohist:.2f}'
                
        #if benq is not None:
        #    print('omat',benq['omat_kokoelake'],'puoliso',benq['puoliso_kokoelake'])

        if reward is not None:
            puoliso+=f' r {reward:.4f}'
        if netto_puoliso is not None:
            puoliso+=f' n {netto_puoliso:.0f}'

        if render_puoliso:
            print(puoliso)
        
        if done:
            print('-------------------------------------------------------------------------------------------------------')

    def close(self):
        '''
        Ei käytössä
        '''
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def set_state_limits(self,debug=True):
        '''
        Rajat tiloille
        '''
        if self.log_transform:
            pension_min=np.log(0/20_000+self.eps) # vastainen eläke
            pension_max=np.log(200_000/20_000+self.eps) # vastainen eläke
            wage_max=np.log(500_000/40_000+self.eps)
            wage_min=np.log(0/40_000+self.eps)
            paid_pension_max=np.log(200_00/20_000+self.eps) # alkanut eläke
            paid_pension_min=np.log(0/20_000+self.eps) # alkanut eläke
        else:
            pension_max=(200_000-40_000)/40_000 # vastainen eläke
            pension_min=(0-40_000)/40_000 # vastainen eläke
            wage_max=(300_000-40_000)/40_000
            wage_min=(0-40_000)/40_000
            paid_pension_min=(0-40_000)/40_000 # alkanut eläke
            paid_pension_max=(200_000-40_000)/40_000 # alkanut eläke

        age_max=(self.max_age-(self.max_age+self.min_age)/2)/20
        age_min=(self.min_age-(self.max_age+self.min_age)/2)/20
        tis_max=(self.max_age-self.min_age-10)/10
        tis_min=-10/10
        pink_min=0 # irtisanottu vai ei 
        pink_max=1 # irtisanottu vai ei 
        toe_min=0-self.max_toe*0.5 # työssäoloehto
        toe_max=self.max_toe-self.max_toe*0.5 # työssäoloehto
        thist_min=-10/20 # tyohistoria: 300/400 pv
        thist_max=(self.max_age-self.min_age-10)/20 # tyohistoria: 300/400 pv
        out_max=100
        out_min=0

        group_min=0
        group_max=1
        state_min=0
        state_max=1
        ben_min=-1
        ben_max=2
        wr_min=0
        wr_max=1
        pref_min=-5
        pref_max=5
        unra_min=-1
        unra_max=1
        child_min=-1
        child_max=1
        tr_min=-1
        tr_max=1
        left_min=-1
        left_max=1

        low = [
            state_min,
            state_min,
            state_min,
            state_min,
            state_min,
            state_min,
            state_min,
            state_min,
            state_min,
            state_min,
            state_min,
            state_min,
            state_min,
            state_min,
            state_min,
            group_min,
            group_min,
            group_min,
            group_min,
            group_min,
            group_min,
            pension_min,
            wage_min,
            age_min,
            tis_min,
            paid_pension_min,
            pink_min,
            toe_min,
            thist_min,
            tr_min,
            left_min,
            wage_min,
            ben_min,
            wr_min,
            unra_min,
            wage_min,
            wage_min,
            state_min,
            state_min,
            child_min,
            child_min,
            child_min,
            state_min,
            pension_min, # ove määrä
            state_min,
            state_min,
            state_min,
            state_min,
            state_min,
            state_min,
            state_min,
            state_min,
            state_min,
            state_min,
            state_min,
            state_min,
            state_min,
            state_min,
            state_min,
            state_min,
            paid_pension_min,
            paid_pension_min,
            paid_pension_min,
            paid_pension_min,
            toe_min,
            toe_min,
            toe_min,
            unra_min,
            wage_min,
            wage_min,
            state_min,
            state_min,
            toe_min,
            toe_min,
            thist_min,
            tis_min,
            state_min,
            state_min,
            paid_pension_min,
            paid_pension_min
            ]            
            
            
        high = [
            state_max,
            state_max,
            state_max,
            state_max,
            state_max,
            state_max,
            state_max,
            state_max,
            state_max,
            state_max,
            state_max,
            state_max,
            state_max,
            state_max,
            state_max,
            group_max,
            group_max,
            group_max,
            group_max,
            group_max,
            group_max,
            pension_max,
            wage_max,
            age_max,
            tis_max,
            paid_pension_max,
            pink_max,
            toe_max,
            thist_max,
            tr_max,
            left_max,
            wage_max,
            ben_max,
            wr_max,
            unra_max,
            wage_max,
            wage_max,
            state_max,
            state_max,
            child_max,
            child_max,
            child_max,
            state_max,
            pension_max, # ove määrä
            state_max,
            state_max,
            state_max,
            state_max,
            state_max,
            state_max,
            state_max,
            state_max,
            state_max,
            state_max,
            state_max,
            state_max,
            state_max,
            state_max,
            state_max,
            state_max,
            paid_pension_max,
            paid_pension_max,
            paid_pension_max,
            paid_pension_max,
            toe_max,
            toe_max,
            toe_max,
            unra_max,
            wage_max,
            wage_max,
            state_max,
            state_max,
            toe_max,
            toe_max,
            thist_max,
            tis_max,
            state_max,
            state_max,
            paid_pension_max,
            paid_pension_max
            ]
            
        #if self.include_mort: # if mortality is included, add one more state
        #      low.insert(0,state_min)
        #      high.insert(0,state_max)
              
        # puoliso
        low.extend([state_min,state_min,wage_min])
        high.extend([state_max,state_max,wage_max])
        
        if self.include_preferencenoise:
            low.append(pref_min)
            high.append(pref_max)
                
        self.low=np.array(low)
        self.high=np.array(high)

    def explain(self):
        '''
        Tulosta laskennan parametrit
        '''
        print('Parameters of lifecycle:\ntimestep {}\ngamma {} ({} per anno)\nmin_age {}\nmax_age {}\nmin_retirementage {}\n'.format(self.timestep,self.gamma,self.gamma**(1.0/self.timestep),self.min_age,self.max_age,self.min_retirementage)+\
            'max_retirementage {}\nansiopvraha_kesto300 {}\nansiopvraha_kesto400 {}\nansiopvraha_kesto500 {}\nansiopvraha_toe {}\n'.format(self.max_retirementage,self.ansiopvraha_kesto300,self.ansiopvraha_kesto400,self.ansiopvraha_kesto500,self.ansiopvraha_toe)+\
            'perustulo {}\nkarenssi_kesto {}\nmortality {}\nrandomness {}\n'.format(self.perustulo,self.karenssi_kesto,self.include_mort,self.randomness)+\
            'include_putki {}\ninclude_pinkslip {}\n'.format(self.include_putki,self.include_pinkslip)+\
            f'perustulo {self.perustulo}\nsigma_reduction {self.use_sigma_reduction}\nplotdebug {self.plotdebug}\n'+\
            'additional_tyel_premium {}\nscale_tyel_accrual {}\ninclude_ove {}\n'.format(self.additional_tyel_premium,self.scale_tyel_accrual,self.include_ove)+\
            f'unemp_limit_reemp {self.unemp_limit_reemp}\n')

    def unempright_left(self,emp,tis,bu,ika,tyohistoria):
        '''
        Tilastointia varten lasketaan jäljellä olevat ansiosidonnaiset työttömyysturvapäivät
        '''
        if ika>=self.minage_500 and tyohistoria>=self.tyohistoria_vaatimus500:
            kesto=self.apvkesto500 #ansiopvraha_kesto500
        elif tyohistoria>=self.tyohistoria_vaatimus:
            kesto=self.apvkesto400 #ansiopvraha_kesto400
        else:
            kesto=self.apvkesto300 #ansiopvraha_kesto300
        
        #kesto=kesto/(12*21.5)
        #if irtisanottu<1 and time_in_state<self.karenssi_kesto: # karenssi, jos ei irtisanottu
        
        if emp==13:
            return tis
        else:
            return kesto-bu
            
    def init_inforate(self):
        #self.infostat_kassanjasenyys_rate()
        self.kassanjasenyys_joinrate,self.kassanjasenyys_rate=self.rates.infostat_kassanjasenyys_rate()
            
    def init_infostate(self,lapsia=0,lasten_iat=np.zeros(15),lapsia_paivakodissa=0,age=18,spouse=False):
        '''
        Alustaa infostate-dictorionaryn
        Siihen talletetaan tieto aiemmista tiloista, joiden avulla lasketaan statistiikkoja
        '''
        self.infostate={}
        states,latest,enimaika,palkka,voc_unempbasis=self.infostat_vocabulary(spouse=False)

        self.infostate[states]=np.zeros(self.n_time)-1
        self.infostate[palkka]=np.zeros(self.n_time)-1
        self.infostate[voc_unempbasis]=np.zeros(self.n_time)-1
        self.infostate[latest]=0
        self.infostate['children_n']=0
        self.infostate['children_date']=np.zeros(15)
        self.infostate[enimaika]=0
        
        states,latest,enimaika,palkka,voc_unempbasis=self.infostat_vocabulary(spouse=True)
        self.infostate[states]=np.zeros(self.n_time)-1
        self.infostate[palkka]=np.zeros(self.n_time)-1
        self.infostate[voc_unempbasis]=np.zeros(self.n_time)-1
        self.infostate[latest]=0
        self.infostate[enimaika]=0
        #self.infostate['kassanjasen']=0
        sattuma = np.random.uniform(size=1)
        t=int((age-self.min_age)/self.timestep)
        
        #print('age {} sattuma {} rate {}'.format(age,sattuma,self.kassanjasenyys_rate[t]))
        
        if sattuma<self.kassanjasenyys_rate[t]:
            self.infostate['kassanjasen']=1
        else:
            self.infostate['kassanjasen']=0
        
    def infostate_add_child(self,age):
        if self.infostate['children_n']<14:
            self.infostate['children_date'][self.infostate['children_n']]=age
            self.infostate['children_n']=self.infostate['children_n']+1
            
    def infostate_set_enimmaisaika(self,age,spouse=False):
        states,latest,enimaika,palkka,voc_unempbasis=self.infostat_vocabulary(spouse=spouse)
        t=int((age-self.min_age)/self.timestep)
        self.infostate[enimaika]=t
        
    def update_infostate(self,t,state,wage,unempbasis,spouse=False):
        states,latest,enimaika,palkka,voc_unempbasis=self.infostat_vocabulary(spouse=spouse)
    
        self.infostate[states][t]=state
        self.infostate[latest]=int(t)
        self.infostate[voc_unempbasis][t]=unempbasis
        if state==1:
            self.infostate[palkka][t]=wage
        elif state==10:
            self.infostate[palkka][t]=wage*0.5
        elif state in set([5,6]):
            self.infostate[palkka][t]=wage
        else:
            self.infostate[palkka][t]=0
        
    def render_infostate(self):
        print('states {}'.format(self.infostate['states']))
        
    def get_kassanjasenyys(self):
        return self.infostate['kassanjasen']

    def set_kassanjasenyys(self,value):
        self.infostate['kassanjasen']=value

    def infostat_kassanjasenyys_update(self,age):
        if self.infostate['kassanjasen']<1:
            sattuma = np.random.uniform(size=1)
            intage=self.map_age(age)
            if sattuma<self.kassanjasenyys_joinrate[intage]:
                self.infostate['kassanjasen']=1
        
    def comp_toe_wage_nykytila(self,spouse=False):
        states,latest,enimaika,palkka,voc_unempbasis=self.infostat_vocabulary(spouse=spouse)
        lstate=int(self.infostate[states][self.infostate[latest]])
        toes=0
        wage=0
        n_toe=int(np.floor(self.max_toe/self.timestep))
        emp_states={1,10}
        unemp_states={0,4}
        family_states={5,6,7}
        ret_states={2,3,8,9}
        
        if self.infostate['kassanjasen']>0:
            if lstate not in ret_states:
                if lstate in family_states:
                    # laskee, onko ollut riittävä toe ansiosidonnaiseen, ei onko päiviä jäljellä
                    t2=self.infostate[latest]
                    nt=0
                    while nt<n_toe and t2>=0:
                        emps=self.infostate[states][t2]
                        if emps in family_states:
                            pass
                        elif emps in emp_states:
                            w=self.infostate[palkka][t2]
                            if w>self.min_toewage:
                                toes+=self.timestep
                                wage+=w*self.timestep
                            elif self.include_halftoe and w>self.min_halftoewage and emps==10:
                                toes+=0.5*self.timestep
                                wage+=w*self.timestep
                            nt=nt+1
                        elif emps in unemp_states:
                            nt=nt+1
                        else:
                            nt=nt+1
                        t2=t2-1
                else:
                    # laskee, onko toe täyttynyt viimeisimmän ansiosidonnaisen alkamisen jälkeen
                    t2=self.infostate[latest]
                    nt=0
                    t0=self.infostate[enimaika]
                    while nt<n_toe and t2>=t0:
                        emps=self.infostate[states][t2]
                        if emps in family_states:
                            pass
                        elif emps in emp_states:
                            w=self.infostate[palkka][t2]
                            if w>self.min_toewage:
                                toes+=self.timestep
                                wage+=w*self.timestep
                            elif self.include_halftoe and w>self.min_halftoewage and emps==10:
                                toes+=0.5*self.timestep
                                wage+=w*self.timestep
                            nt=nt+1
                        elif emps in unemp_states:
                            nt=nt+1
                        else:
                            nt=nt+1
                        t2=t2-1
                if toes>=self.ansiopvraha_toe and toes>0:
                    wage=wage/toes
                else:
                    wage=0
        else:
            wage=0
            toes=0
            
        toekesto=toes
            
        return toes,toekesto,wage
        
    def comp_toe_wage_porrastus(self,spouse=False):
        states,latest,enimaika,palkka,voc_unempbasis=self.infostat_vocabulary(spouse=spouse)
        lstate=int(self.infostate[states][self.infostate[latest]])
        toes=0
        toekesto=0
        wage=0
        n_toe=int(np.floor(self.max_toe/self.timestep))
        emp_states={1,10}
        unemp_states={0,4}
        family_states={5,6,7}
        ret_states={2,3,8,9}
        
        
        if self.infostate['kassanjasen']>0 and lstate not in ret_states:
            # laskee, onko toe täyttynyt viimeisimmän ansiosidonnaisen alkamisen jälkeen
            t2=self.infostate[latest]
            nt=0
            t0=self.infostate[enimaika]
            while nt<n_toe and t2>=t0:
                emps=self.infostate[states][t2]
                if emps in family_states:
                    pass
                elif emps in emp_states:
                    w=self.infostate[palkka][t2]
                    if w>self.min_toewage:
                        toes+=self.timestep
                    elif self.include_halftoe and w>self.min_halftoewage and emps==10:
                        toes+=0.5*self.timestep
                    nt=nt+1
                #elif emps in unemp_states:
                #    nt=nt+1
                else:
                    nt=nt+1
                t2=t2-1
        
            # laskee, onko ollut riittävä toe ansiosidonnaiseen, ei onko päiviä jäljellä
            t2=self.infostate[latest]
            nt=0
            while nt<n_toe and t2>=0:
                emps=self.infostate[states][t2]
                if emps in family_states:
                    pass
                elif emps in emp_states:
                    w=self.infostate[palkka][t2]
                    if w>self.min_toewage:
                        toekesto+=self.timestep
                        wage+=w*self.timestep
                    elif self.include_halftoe and w>self.min_halftoewage and emps==10:
                        toekesto+=0.5*self.timestep
                        wage+=w*self.timestep
                    nt=nt+1
                elif emps in unemp_states:
                    nt=nt+1
                else:
                    nt=nt+1
                t2=t2-1
                
            if toekesto>=self.ansiopvraha_toe and toekesto>0:
                wage=wage/toekesto
            else:
                wage=0
        
            if lstate in family_states:
                toes=toekesto
        else:
            wage=0
            toes=0
            toekesto=0
            
        return toes,toekesto,wage        
        
    def comp_infostats(self,age,spouse=False):
        # laske työssäoloehto tarkasti
        # laske työttömyysturvaan vaikuttavat lasten määrät
        
        self.infostat_kassanjasenyys_update(age)
        
        toes,toekesto,wage=self.comp_toe_wage(spouse=spouse)
                
        #print('toes',toes)
        
        #start_t=max(self.infostate['latest']-n_toe,self.infostate['enimmaisaika_alkaa'])
        #print('-->',start_t,self.infostate['latest'],self.infostate['states'][start_t:self.infostate['latest']],
        #    self.infostate['wage'][start_t:self.infostate['latest']],toes)
        
        children_under18=0
        children_under7=0
        children_under3=0
        for k in range(self.infostate['children_n']):
            c_age=age-self.infostate['children_date'][k]
            if c_age<18:
                children_under18=children_under18+1
                if c_age<7:
                    children_under7=children_under7+1
                    if c_age<3:
                        children_under3=children_under3+1

        return toes,toekesto,wage,children_under3,children_under7,children_under18

    def infostat_comp_5y_ave_wage(self,spouse=False):
        emp_states={1,10}
        unemp_states={0,4,13}
        family_states={5,6}
        
        states,latest,enimaika,voc_wage,voc_unempbasis=self.infostat_vocabulary(spouse=spouse)
        #print(states,latest,enimaika,voc_wage,voc_unempbasis)
    
        lstate=int(self.infostate[latest])
        n=int(np.ceil(5/self.timestep))
        wage=0
        for x in range(lstate-n,lstate):
            if x<0:
                pass
            else:
                if self.infostate[states][x] in emp_states:
                    wage+=self.infostate[voc_wage][x]*self.timestep/5
                elif self.infostate[states][x] in family_states:
                    wage+=self.infostate[voc_wage][x]*self.timestep/5
                elif self.infostate[states][x] in unemp_states:
                    wage+=self.infostate[voc_unempbasis][x]*self.timestep/5
                elif self.infostate[states][x]==13:
                    wage+=self.disabbasis_tmtuki*self.timestep/5
                elif self.infostate[states][x]==12:
                    wage+=self.disabbasis_tmtuki*self.timestep/5

        return wage

    def infostat_can_have_children(self,age):
        children_under1=0
        for k in range(self.infostate['children_n']):
            c_age=age-self.infostate['children_date'][k]
            if c_age<1.01:
                children_under1=1
                break

        if children_under1>0:
            return False
        else:
            return True

    def infostat_vocabulary(self,spouse=False):
        if spouse:
            states='spouse_states'
            latest='spouse_latest'
            enimaika='spouse_enimmaisaika_alkaa'
            palkka='spouse_wage'
            unempbasis='spouse_unempbasis'
        else:
            states='states'
            latest='latest'
            enimaika='enimmaisaika_alkaa'
            palkka='wage'
            unempbasis='unempbasis'
            
        return states,latest,enimaika,palkka,unempbasis

    def infostat_check_aareset(self,age,spouse=False):
        states,latest,enimaika,palkka,voc_unempbasis=self.infostat_vocabulary(spouse=spouse)
    
        t=int((age-self.min_age)/self.timestep)
        ed_t=self.infostate[enimaika]
        if (t-ed_t)<1.0/self.timestep:
            return True
        else:
            return False
        
    def setup_state_encoding(self):
        self.state_encoding=np.zeros((self.n_empl,self.n_empl))
        for s in range(self.n_empl):
            self.state_encoding[s,s]=1
        self.group_encoding=np.zeros((self.n_groups,self.n_groups))
        for s in range(self.n_groups):
            self.group_encoding[s,s]=1
        self.spousestate_encoding=np.zeros((self.n_spouseempl,self.n_spouseempl))
        for s in range(self.n_spouseempl):
            self.spousestate_encoding[s,s]=1

    def print_q(self,a):
        '''
        pretty printer for dict
        '''
        for x in a.keys():
            if a[x]>0 or a[x]<0:
                print('{}:{:.2f} '.format(x,a[x]),end='')
                
        print('')
        
    def comp_oldtoe(self,printti=False,spouse=False):
        '''
        laske työttömyysjaksoa edeltävä työssäoloehto tarkasti
        '''
        toes=0
        n_toe=int(np.floor(self.max_toe/self.timestep))
        emp_states={1,10}
        unemp_states={0,4,13}
        family_states={5,6,7}
        ret_states={2,3,8,9}
        wage=0
        
        states,latest,enimaika,palkka,voc_unempbasis=self.infostat_vocabulary(spouse=spouse)
        
        lstate=int(self.infostate[states][self.infostate[latest]])
        
        #if lstate!=0:
        #    return 0
        
        nt=0
        t2=max(0,self.infostate[enimaika]-1)
        emp_states={1,10}
        unemp_states={0,4}
        family_states={5,6,7}
        while nt<n_toe:
            emps=self.infostate[states][t2]
            if printti:
                print('emps {} t2 {} toes {}'.format(emps,t2,toes))
            if emps in family_states:
                pass
            elif emps in emp_states:
                if self.infostate[palkka][t2]>self.min_toewage:
                    toes+=self.timestep
                elif self.include_halftoe and self.infostate[palkka][t2]>=self.min_halftoewage and emps==10:
                    toes+=0.5*self.timestep
                nt=nt+1
            elif emps in unemp_states:
                nt=nt+1
            else:
                nt=nt+1
            t2=t2-1

        return toes

    def check_toe58_v1(self,age,toe,tyoura,toe58):
        if age<self.minage_500:
            return 0
        elif self.tyossaoloehto(toe,tyoura,age) and tyoura>=self.tyohistoria_vaatimus500:
            return 1
        else:
            return 0

    def check_toe58(self,age,toe,tyoura,toe58,spouse=False):
        '''
        laske työttömyysjaksoa edeltävä työssäoloehto tarkasti
        '''
        states,latest,enimaika,palkka,voc_unempbasis=self.infostat_vocabulary(spouse=spouse)
        
        toes=0
        n_toe=int(np.floor(self.max_toe/self.timestep))
        emp_states={1,10}
        unemp_states={0,4}
        family_states={5,6,7}
        ret_states={2,3,8,9}
        lstate=int(self.infostate[states][self.infostate[latest]])
        
        if age<self.minage_500 or lstate in ret_states:
            return 0

        t=self.map_age(age)
        t58=self.map_age(58)
        
        #if lstate!=0:
        #    return 0
        
        nt=0
        if lstate in unemp_states:
            t2=max(0,self.infostate[enimaika]-1)
        else:
            t2=max(0,self.infostate[latest])
        
        while nt<n_toe and nt<t-t58:
            emps=self.infostate[states][t2]
            if emps in family_states:
                pass
            elif emps in emp_states:
                if self.infostate[palkka][t2]>self.min_toewage:
                    toes+=self.timestep
                elif self.include_halftoe and self.infostate[palkka][t2]>=self.min_halftoewage and emps==10:
                    toes+=0.5*self.timestep
                nt=nt+1
            elif emps in unemp_states:
                nt=nt+1
            else:
                nt=nt+1
            t2=t2-1

        if self.tyossaoloehto(toes,tyoura,age) and tyoura>=self.tyohistoria_vaatimus500:
            return 1
        else:
            return 0
