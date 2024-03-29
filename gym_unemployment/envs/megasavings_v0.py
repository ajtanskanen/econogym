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
    - has independent wage and all other state parameters (a first class operator)
    
    
    Savings:
    - full-scale savings included
    
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
#from . util import pretty_print
from . wages_v1 import Wages_v1
# class StayDict(dict):
#     '''
#     Apuluokka, jonka avulla tehdään virheenkorjausta 
#     '''
#     def __missing__(self, key):
#         return 'Unknown state '+key


class MegaSavingsEnv_v0(gym.Env):
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
        7    työssä-olo-Ehto               0          28/12
        8    työuran kesto                 0             50
        9    työstä pois (aika)            0            100 OK?
       10    Irtisanottu                   0              1
       11    Käytetty työttämyyskorvausaika
       12    Palkka-alennus
       13    Aika työttämänä ve-iässä
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
        13  Earnings-unrelated Unemployment (työmarkkinatuki)
        14  Sickness benefit
        15  Dead

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

    def __init__(self,**kwargs) -> None:
        '''
        Alkurutiinit
        '''
        super().__init__()

        self.setup_default_params()
        gamma=0.92
                
        # sets parameters based on kwargs
        self.set_parameters(**kwargs)
        
        if self.perustulo:
            self.ben = fin_benefits.BasicIncomeBenefits(**kwargs)
        elif self.universalcredit:
            self.ben = fin_benefits.UniversalCreditIncomeBenefits(**kwargs)
        else:
            #self.ben = fin_benefits.CyBenefits(**kwargs)
            self.ben = fin_benefits.Benefits(**kwargs)
        
        self.r_mean = (1+self.r_mean)**self.timestep-1
        self.r_std = self.r_std*np.sqrt(self.timestep)
        
        self.gamma=gamma**self.timestep # discounting
        self.palkkakerroin=(0.8*1+0.2*1.0/self.reaalinen_palkkojenkasvu)**self.timestep
        self.elakeindeksi=(0.2*1+0.8*1.0/self.reaalinen_palkkojenkasvu)**self.timestep
        #self.kelaindeksi=(1.0/self.reaalinen_palkkojenkasvu)**self.timestep
        #self.kelaindeksi=self.elakeindeksi # oletetaan, että KELA-indeksi ei jää jälkeen eläkeindeksistä (PTS: 50-50-indeksi)
        self.kelaindeksi=(0.5*1+0.5*1.0/self.reaalinen_palkkojenkasvu)**self.timestep # oletetaan, että KELA-indeksi = PTS: 50-50-indeksi
        self.n_age = self.max_age-self.min_age+1
        self.inv_timestep=int(np.round(1/self.timestep)) # pitäisi olla kokonaisluku
        self.n_time = int(np.round((self.n_age-1)*self.inv_timestep))+1

        self.returns=self.stock_returns()
             
        self.n_savings=21 # -20,...,-4,-3,-2,-1,0,1,2,3,4,...,20
        self.mid_sav_act=np.floor(self.n_savings/2)
            
        # karttumaprosentit
        if self.scale_tyel_accrual:
            acc_scaling=1+self.scale_additional_tyel_accrual
        else:
            acc_scaling=1
        
        self.acc=0.015*self.timestep*acc_scaling
        self.acc_sv=0.62 # sairauspäiväraja, ei skaalata
        self.acc_over_52=0.019*self.timestep*acc_scaling
        #self.acc_over_52=self.acc
        self.acc_family=1.15*self.acc
        self.acc_family_over_52=1.15*self.acc_over_52
        self.acc_unemp=0.75*self.acc
        self.acc_unemp_over_52=0.75*self.acc_over_52
        #self.min_family_accwage=12*757

        if self.include_mort:
            if self.include_ove:
                self.log_utility_mort_ove_params()
            else:
                self.log_utility_mort_noove_params()
        else:
            self.log_utility_nomort_noove_params()
        
        #self.log_utility_default_params()

        self.n_age=self.max_age-self.min_age+1

        if not self.train: # get stats right
            self.mortstop=False
            
        if not self.randomness:
            self.include_npv_mort=False
        
        if self.include_mort: # and not self.mortstop:
            if not self.silent:
                if self.include_mort and self.mortstop:
                    print('Mortality included, stopped')
                else:
                    print('Mortality included, not stopped')

            self.n_empl=16 # state of employment, 0,1,2,3,4
        else:
            if not self.silent:
                print('No mortality included')
            self.n_empl=16 # state of employment, 0,1,2,3,4
        self.set_state_limits()
            
        self.n_spouseempl=self.n_empl
            
        self.setup_state_encoding()

        self.set_year(self.year)
        
        if self.include_ove:
            self.n_actions=6
            self.n_spouse_actions=self.n_actions 
        else:
            self.n_actions=5
            self.n_spouse_actions=self.n_actions 

        if self.include_investment:
            self.action_space = spaces.MultiDiscrete([self.n_actions,self.n_spouse_actions,self.n_savings,self.n_savings])
        else:
            self.action_space = spaces.MultiDiscrete([self.n_actions,self.n_spouse_actions])
        
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        
        if self.use_sigma_reduction:
            self.update_wage_reduction=self.update_wage_reduction_sigma
        else:
            self.update_wage_reduction=self.update_wage_reduction_baseline

        #self.seed()
        self.viewer = None
        self.state = None
        
        inflation_raw=np.array([1.0,1.011,1.010,1.009,1.037,1.01,1.01]) # 2018 2019 2020 2021 2022 2023
        self.inflation=np.cumprod(inflation_raw)
        
        self.steps_beyond_done = None
        
        # normitetaan työttämyysjaksot askelvälin monikerroiksi
        scale=21.5*12
        self.apvkesto300=np.round(self.ansiopvraha_kesto300/scale/self.timestep)*self.timestep
        self.apvkesto400=np.round(self.ansiopvraha_kesto400/scale/self.timestep)*self.timestep
        self.apvkesto500=np.round(self.ansiopvraha_kesto500/scale/self.timestep)*self.timestep
                
        self.init_inforate()
        
        if not self.silent:
            self.explain()
        
        if self.plotdebug:
            self.unit_test_code_decode()
            
    def setup_default_params(self):
        # käytetäänkä exp/log-muunnosta tiloissa vai ei?
        self.log_transform=False
        self.eps=1e-20

        self.dis_ratio=0
        self.nnn=0
        
        self.bequest=0 # perintö utiliteetissa
    
        self.include_investment=True # sijoittaminen mukana
        self.r_mean=0.10 # 10 % tuotto
        self.r_std=0.10 # 10 % volatiliteetti
        self.random_returns=False
        self.CRRA_eta=1.5
        self.use_utility=0 # log; 1 CRRA
        self.include_wealth=False

        # male low income, male mid, male high, female low, female mid, female high income
        self.n_groups=6

        self.ansiopvraha_toe=0.5 # = 6kk
        self.karenssi_kesto=0.25 #0.25 # = 3kk
        self.isyysvapaa_kesto=0.25 # = 3kk
        self.aitiysvapaa_kesto=0.75 # = 9kk ml vanhempainvapaa
        self.mies_jatkaa_kotihoidontuelle=0.05 # 50 % koko äitiysvapaan käyttäneistä menee myös kotihoindotuelle
        self.nainen_jatkaa_kotihoidontuelle=0.5 # 50 % koko äitiysvapaan käyttäneistä menee myös kotihoindotuelle
        self.aitiysvapaa_pois=0.06 # per vuosi
        self.min_tyottputki_ika=61 # vuotta. Ikä, jonka täytyttyö pääsee putkeen
        self.tyohistoria_tyottputki=5 # vuotta. vähimmäistyöura putkeenpääsylle
        self.kht_kesto=2.0 # kotihoidontuen kesto 2 v
        self.tyohistoria_vaatimus=3.0 # 3 vuotta
        self.tyohistoria_vaatimus500=10.0 # p.o. 5 vuotta 20v aikana; 10v tarkoittaa, että 18-38 välin ollut töissä + 5v/20v 
        self.ansiopvraha_kesto400=400 # päivää
        self.ansiopvraha_kesto300=300 # päivää
        self.ansiopvraha_kesto500=500 # päivää
        self.minage_500=58 # minimi-ikä 500 päivälle
        self.min_salary=1000 # julkaistujen laskelmien jälkeen
        
        self.map_stays={0: self.stay_unemployed,  1: self.stay_employed,         2: self.stay_retired,       3: self.stay_disabled,
                        4: self.stay_pipeline,    5: self.stay_motherleave,      6: self.stay_fatherleave,   7: self.stay_kht,
                        8: self.stay_oa_parttime, 9: self.stay_oa_fulltime,     10: self.stay_parttime,     11: self.stay_outsider,
                       12: self.stay_student,    13: self.stay_tyomarkkinatuki, 14: self.stay_svpaivaraha}


        self.timestep=0.25
        self.max_age=70
        self.min_age=18
        self.min_retirementage=63.5 #65
        self.max_retirementage=68 # 70
        self.max_unemploymentbenefitage=65

        self.syntymavuosi=1980
        #self.elinaikakerroin=0.925 # etk:n arvio 1962 syntyneille
        self.elinaikakerroin=0.96344 # vuoden 2017 kuolleisuutta käytettäessä myös elinaikakerroin on sieltä
        #self.elinaikakerroin=self.laske_elinaikakerroin(self.syntymavuosi)
        
        self.reaalinen_palkkojenkasvu=1.016
        
        # exploration does not really work here due to the use of history
        self.reset_exploration_go=False
        self.reset_exploration_ratio=0.4
        
        self.train=False

        self.include_spouses=True # Puolisot mukana?
        self.include_mort=True # onko kuolleisuus mukana laskelmissa
        self.include_npv_mort=True # onko kuolleisuus mukana laskelmissa
        self.include_preferencenoise=False # onko työllisyyspreferenssissä hajonta mukana 
        self.perustulo=False # onko Kelan perustulo laskelmissa
        self.universalcredit=False # Yleistuki
        self.randomness=True # onko stokastiikka mukana
        self.mortstop=False # pysäytä kuolleisuuden jälkeen
        self.include_putki=True # työttämyysputki mukana
        self.include_pinkslip=True # irtisanomiset mukana
        self.use_sigma_reduction=True # kumpi palkkareduktio
        self.include_kansanelake=True
        self.include_takuuelake=True
        self.preferencenoise_std=0.1
        self.silent=False
        
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
        self.mortplot=False
        
        self.unemp_limit_reemp=True # työttämästä työlliseksi tn, jos hakee täitä
        
        # etuuksien laskentavuosi
        self.year=2018
        
        # OVE-parametrit
        self.ove_ratio=0.5
        self.min_ove_age=61

        self.plotdebug=False # tulostetaanko rivi riviltä tiloja
        self.plottkdebug=False    
        
        # lainaus
        self.max_debt=0 #-20_000
            
    def set_annual_params(self,year : int) -> None:
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
        self.min_disab_tulevaaika=17_000/10 # tämä jaettu kymmenellä, koska vertailukohta on vuosittainen keskiarvoansios
        
        # luvut 2022 tasossa
        kerroin=self.inflation[year-2018]/self.inflation[2024-2018]
        self.opiskelija_asumismenot_toimeentulo=150*kerroin
        self.opiskelija_asumismenot_asumistuki=150*kerroin
        self.elakelainen_asumismenot_toimeentulo=200*kerroin
        self.elakelainen_asumismenot_asumistuki=200*kerroin
        self.muu_asumismenot_toimeentulo=290*kerroin
        self.muu_asumismenot_asumistuki=290*kerroin
        self.muu_asumismenot_lisa=120*kerroin

    def get_retirementage(self) -> float:
        return self.min_retirementage

    def set_retirementage(self,year : int) -> None:
        if year==2018:
            self.min_retirementage=63.5
            self.max_retirementage=68
        elif year==2019:
            self.min_retirementage=63.75
            self.max_retirementage=68
        elif year==2020:
            self.min_retirementage=64.0
            self.max_retirementage=68
        elif year==2021:
            self.min_retirementage=64.25
            self.max_retirementage=68
        elif year==2022:
            self.min_retirementage=64.5
            self.max_retirementage=68
        elif year==2023:
            self.min_retirementage=64.75
            self.max_retirementage=68
        elif year==2024:
            self.min_retirementage=65
            self.max_retirementage=69
        else:
            error('retirement_age')
            
    def set_year(self,year : int) -> None:
        self.year=year
        self.set_annual_params(year)
        self.set_retirementage(year)
        self.ben.set_year(year)
        self.rates=Rates(year=self.year,silent=self.silent,max_age=self.max_age,
            n_groups=self.n_groups,timestep=self.timestep,inv_timestep=self.inv_timestep,n_empl=self.n_empl)

        #self.palkat_ika_miehet,self.palkat_ika_naiset,self.g_r=self.rates.setup_salaries_v4(self.min_retirementage)
        self.palkat_ika_miehet,self.palkat_ika_naiset,self.g_r=self.rates.setup_salaries_v4(self.min_retirementage)
        self.wages_spouse=Wages_v1(year=self.year,silent=self.silent,max_age=self.max_age,
            n_groups=self.n_groups,timestep=self.timestep,inv_timestep=self.inv_timestep,
            min_retirementage=self.min_retirementage,min_salary=self.min_salary)
        self.wages_main=Wages_v1(year=self.year,silent=self.silent,max_age=self.max_age,
            n_groups=self.n_groups,timestep=self.timestep,inv_timestep=self.inv_timestep,
            min_retirementage=self.min_retirementage,min_salary=self.min_salary)

        #self.get_wage=self.get_wage_step        
        #self.get_spousewage=self.get_spousewage_step

        self.get_wage=self.wages_main.get_wage        
        self.get_spousewage=self.wages_spouse.get_wage
        
        # reemployment probability
        prob_3m=self.rates.get_reemp_prob() # 0.5
        prob_1y=1-(1-prob_3m)**(1./0.25)
        self.unemp_reemp_prob=1-(1-prob_1y)**self.timestep # kolmessa kuukaudessa
        #print('unemp_reemp_prob',self.unemp_reemp_prob)

        prob_3m=0.2
        prob_1y=1-(1-prob_3m)**(1./0.25)
        self.oa_reemp_prob=1-(1-prob_1y)**self.timestep # kolmessa kuukaudessa
        
        # moving from parttime work to fulltime work
        prob_3m=0.5
        prob_1y=1-(1-prob_3m)**(1./0.25)
        self.parttime_fullemp_prob=1-(1-prob_1y)**self.timestep # kolmessa kuukaudessa
        
        self.disability_intensity,self.svpaivaraha_disabilityrate=self.rates.get_eff_disab_rate_v4()
        self.pinkslip_intensity=self.rates.get_pinkslip_rate()*self.timestep
        self.birth_intensity=self.rates.get_birth_rate_v2(symmetric=False)
        self.mort_intensity=self.rates.get_mort_rate()
        self.student_inrate,self.student_outrate=self.rates.get_student_rate_v2() # myös armeijassa olevat tässä
        self.outsider_inrate,self.outsider_outrate=self.rates.get_outsider_rate()
        self.divorce_rate=self.rates.get_divorce_rate()
        self.marriage_rate=self.rates.get_marriage_rate()

        self.npv,self.npv0,self.npv_pension,self.npv_gpension=self.comp_npv()
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
        return 104
        
    def test_comp_npv(self):
        npv,npv0,cpsum_pension=self.comp_npv()
        
        n=10000
        snpv=np.zeros((6,n))
        snpv0=np.zeros((6,n))
        scpsum_pension=np.zeros((6,n))
        
        for g in range(6):
            for k in range(n):
                snpv[g,k],snpv0[g,k],scpsum_pension[g,k],scpsum_gpension[g,k]=self.comp_npv_simulation(g)
                
        print(npv)
        for g in range(6):
            print('{}: {}'.format(g,np.mean(snpv[g,:])))

    def comp_npv(self):
        '''
        lasketaan montako timestep:iä (diskontattuna) max_age:n jälkeen henkilä on vanhuuseläkkeellä 
        hyvin yksinkertainen toteutus. Tulos on odotettu lukumäärä timestep:jä
        
        npv <- diskontattu
        npv0 <- ei ole diskontattu
        '''
        npv=np.zeros(self.n_groups)
        npv0=np.zeros(self.n_groups)
        npv_pension=np.zeros(self.n_groups)
        npv_gpension=np.zeros(self.n_groups)

        for g in range(self.n_groups):
            cpsum=1.0
            cpsum0=1.0
            cpsum_gpension=1.0
            cpsum_pension=1.0
            for x in np.arange(100,self.max_age,-self.timestep):
                intx=int(np.floor(x))
                m=self.mort_intensity[intx,g]
                cpsum=m*1+(1-m)*(1+self.gamma*cpsum) # gamma discounted
                cpsum0=m*1+(1-m)*(1+cpsum0) # no discount
                cpsum_gpension=m*1+(1-m)*(1+cpsum_gpension*self.gamma*self.elakeindeksi)  # gamma + pension indexing discount
                cpsum_pension=m*1+(1-m)*(1+cpsum_pension*self.elakeindeksi)  # pension indexing discount
            npv[g]=cpsum
            npv0[g]=cpsum0
            npv_pension[g]=cpsum_pension
            npv_gpension[g]=cpsum_gpension
            #print(g,npv0)
            
        if self.plotdebug:
            print('npv:',npv)
            
        return npv,npv0,npv_pension,npv_gpension

    def comp_npv_simulation(self,g : int):

        '''
        simuloidaan npv jokaiselle erikseen montako timestep:iä (diskontattuna) max_age:n jälkeen henkilä on vanhuuseläkkeellä 
        hyvin yksinkertainen toteutus. Tulos on odotettu lukumäärä timestep:jä
        
        npv <- diskontattu
        npv0 <- ei ole diskontattu
        '''
        npv=0.0
        npv0=0.0
        npv_pension=0.0
        npv_gpension=0.0

        cpsum=1.0
        cpsum0=1.0
        cpsum_pension=1.0
        cpsum_gpension=1.0
        
        alive=True
        num=int(np.ceil(100-self.max_age+2)/self.timestep)
        sattuma = np.random.uniform(size=num)
        x=self.max_age+self.timestep
        k=0
        while alive and x<100+self.timestep:
            intx=int(np.floor(x))
            print(intx,cpsum)
            if sattuma[k]>self.mort_intensity[intx,g]:
                cpsum=1.0+self.gamma*cpsum
                cpsum0=1.0+cpsum0
                cpsum_pension=1.0+cpsum_pension*self.elakeindeksi
                cpsum_gpension=1.0+cpsum_pension*self.gamma*self.elakeindeksi
            else:
                alive=False
            k=k+1
            x=x+self.timestep
                
        npv=cpsum
        npv0=cpsum0
        npv_pension=cpsum_pension
        npv_gpension=cpsum_gpension
            
        if self.plotdebug:
            print('comp_npv_simulation npv:',npv)

        return npv,npv0,npv_pension,npv_gpension

    def setup_children(self,p : dict,puoliso : int,employment_state : int,puoliso_tila : int,
                    children_under3 : int,children_under7 : int,children_under18 : int,lapsikorotus_lapsia : int) -> None:
        # tässä ei alku+, koska lapset yhteisiä
        if puoliso>0:
            p['lapsia']=children_under18
            p['lapsia_paivahoidossa']=children_under7
            p['lapsia_alle_kouluikaisia']=children_under7
            p['lapsia_alle_3v']=children_under3
            p['lapsia_kotihoidontuella']=0
            p['lapsikorotus_lapsia']=lapsikorotus_lapsia
        
            if employment_state==5 or puoliso_tila==5: # äitiysvapaa
                p['lapsia_paivahoidossa']=0
            if employment_state==6 or puoliso_tila==6: # äitiysvapaa
                p['lapsia_paivahoidossa']=0
            if employment_state==7 or puoliso_tila==7: # hoitovapaa
                p['lapsia_paivahoidossa']=0
                p['lapsia_kotihoidontuella']=children_under7
            p['saa_elatustukea']=0
        else:
            p['lapsia']=children_under18
            p['lapsia_paivahoidossa']=children_under7
            p['lapsia_alle_kouluikaisia']=children_under7
            p['lapsia_alle_3v']=children_under3
            p['lapsia_kotihoidontuella']=0
            p['lapsikorotus_lapsia']=lapsikorotus_lapsia
        
            if employment_state==5 or puoliso_tila==5: # äitiysvapaa
                p['lapsia_paivahoidossa']=0
            if employment_state==6 or puoliso_tila==6: # äitiysvapaa
                p['lapsia_paivahoidossa']=0
            if employment_state==7 or puoliso_tila==7: # hoitovapaa
                p['lapsia_paivahoidossa']=0
                p['lapsia_kotihoidontuella']=children_under7

            if children_under18>0:
                p['saa_elatustukea']=1 # vain yksinhuoltaja
#                 print('saa elatustukea',children_under18,'lasta')
#             else:
#                 print('ei saa elatustukea',children_under18,'lasta')

    def setup_benefits(self,wage : float,old_wage : float,kansanelake : float,tyoelake : float,employment_state : int,
                    time_in_state : float,ika : float,used_unemp_benefit : float,children_under3 : int,children_under7 : int,children_under18 : int,puoliso=0,
                    irtisanottu=0,karenssia_jaljella=0,alku='omat_',p2=None,puolisoalku='puoliso_') -> dict:
        if p2 is not None:
            p=p2.copy()
        else:
            p={}
            
        if self.perustulo:
            p[alku+'perustulo']=1
        else:
            p[alku+'perustulo']=0
            
        p[alku+'saa_elatustukea']=0
        p[alku+'opiskelija']=0
        p[alku+'elakkeella']=0
        p[alku+'toimeentulotuki_vahennys']=0
        p[alku+'ika']=ika
        p[alku+'tyoton']=0
        p[alku+'peruspaivarahalla']=0
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
        
        if employment_state==15:
            p[alku+'alive']=0
        else:
            p[alku+'alive']=1
        
        if employment_state==1:
            p[alku+'tyoton']=0 # voisi olla työtön siinä mielessä, että oikeutettu soviteltuun päivärahaan
            p[alku+'t']=wage/12
            p[alku+'vakiintunutpalkka']=wage/12
            p[alku+'saa_ansiopaivarahaa']=0
            p[alku+'tyoelake']=tyoelake/12 # ove
        elif employment_state==0: # työtön, ansiopäivärahalla
            if ika<self.max_unemploymentbenefitage:
                #self.render()
                p[alku+'tyoton']=1
                p[alku+'t']=0
                p[alku+'vakiintunutpalkka']=old_wage/12
                p[alku+'saa_ansiopaivarahaa']=1
                p[alku+'tyottomyyden_kesto']=12*21.5*used_unemp_benefit
                    
                if irtisanottu<1 and karenssia_jaljella>0:
                    p[alku+'saa_ansiopaivarahaa']=0
                    p[alku+'tyoton']=0
                    
                p[alku+'tyoelake']=tyoelake/12 # ove
            else:
                p[alku+'tyoton']=0 # ei oikeutta työttämyysturvaan
                p[alku+'t']=0
                p[alku+'vakiintunutpalkka']=0
                p[alku+'saa_ansiopaivarahaa']=0
                p[alku+'tyoelake']=tyoelake/12 # ove
        elif employment_state==13: # työmarkkinatuki
            if ika<self.max_unemploymentbenefitage:
                p[alku+'tyoton']=1
                p[alku+'peruspaivarahalla']=1
                p[alku+'t']=0
                p[alku+'vakiintunutpalkka']=0
                p[alku+'tyottomyyden_kesto']=12*21.5*time_in_state
                p[alku+'saa_ansiopaivarahaa']=0
                p[alku+'tyoelake']=tyoelake/12 # ove
            else:
                p[alku+'tyoton']=0 # ei oikeutta työttämyysturvaan
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
        elif employment_state==4: # työttämyysputki
            if ika<self.max_unemploymentbenefitage:
                p[alku+'tyoton']=1
                p[alku+'t']=0
                p[alku+'vakiintunutpalkka']=old_wage/12
                p[alku+'saa_ansiopaivarahaa']=1
                p[alku+'tyottomyyden_kesto']=12*21.5*time_in_state
                p[alku+'tyoelake']=tyoelake/12 # ove
            else:
                p[alku+'tyoton']=0 # ei oikeutta työttämyysturvaan
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
            p[alku+'kotihoidontuki_kesto']=time_in_state
            p[alku+'t']=0
            p[alku+'vakiintunutpalkka']=old_wage/12
            p[alku+'tyoelake']=tyoelake/12
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
        elif employment_state==14: # sv-päiväraha
            p[alku+'t']=0
            p[alku+'tyoelake']=tyoelake/12 # ove
            p[alku+'vakiintunutpalkka']=old_wage/12
            p[alku+'sairauspaivarahalla']=1
        elif employment_state==15: # kuollut
            p[alku+'t']=0
            p[alku+'tyoelake']=0 # ove
        else:
            print('Unknown employment_state ',employment_state)
        
        p[alku+'elake_maksussa']=p[alku+'tyoelake']+p[alku+'kansanelake']
            
        p['ansiopvrahan_suojaosa']=1
        p['ansiopvraha_lapsikorotus']=1

        if puoliso>0 and employment_state!=15:
            p['aikuisia']=2
        else:
            p['aikuisia']=1
            p[puolisoalku+'opiskelija']=0
            p[puolisoalku+'peruspaivarahalla']=0
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
            p[puolisoalku+'kansanelake']=0
            p[puolisoalku+'elakkeella']=0
            p[puolisoalku+'sairauspaivarahalla']=0
            p[puolisoalku+'disabled']=0
            p[puolisoalku+'t']=0
            
        return p    

    def setup_asumismenot(self,employment_state : int,puoliso : int,puoliso_tila : int,children_under18 : int,p : dict) -> None:
        if employment_state==12: # opiskelija
            if puoliso>0:
                p['asumismenot_toimeentulo']=self.opiskelija_asumismenot_toimeentulo*0.5+children_under18*self.muu_asumismenot_lisa
                p['asumismenot_asumistuki']=self.opiskelija_asumismenot_asumistuki*0.5+children_under18*self.muu_asumismenot_lisa
                if puoliso_tila==12:
                    p['asumismenot_toimeentulo']+=self.opiskelija_asumismenot_toimeentulo*0.5
                    p['asumismenot_asumistuki']+=self.opiskelija_asumismenot_asumistuki*0.5
                elif puoliso_tila in set([2,3,8,9]):
                    p['asumismenot_toimeentulo']+=self.elakelainen_asumismenot_toimeentulo*0.5
                    p['asumismenot_asumistuki']+=self.elakelainen_asumismenot_asumistuki*0.5
                else:
                    p['asumismenot_toimeentulo']+=self.muu_asumismenot_toimeentulo*0.5
                    p['asumismenot_asumistuki']+=self.muu_asumismenot_asumistuki*0.5
            else:
                p['asumismenot_toimeentulo']=self.opiskelija_asumismenot_toimeentulo+children_under18*self.muu_asumismenot_lisa
                p['asumismenot_asumistuki']=self.opiskelija_asumismenot_asumistuki+children_under18*self.muu_asumismenot_lisa
            
        elif employment_state in set([2,3,8,9]): # eläkeläinen
            if puoliso>0:
                p['asumismenot_toimeentulo']=self.elakelainen_asumismenot_toimeentulo*0.5+children_under18*self.muu_asumismenot_lisa
                p['asumismenot_asumistuki']=self.elakelainen_asumismenot_asumistuki*0.5+children_under18*self.muu_asumismenot_lisa
                if puoliso_tila==12:
                    p['asumismenot_toimeentulo']+=self.opiskelija_asumismenot_toimeentulo*0.5
                    p['asumismenot_asumistuki']+=self.opiskelija_asumismenot_asumistuki*0.5
                elif puoliso_tila in set([2,3,8,9]):
                    p['asumismenot_toimeentulo']+=self.elakelainen_asumismenot_toimeentulo*0.5
                    p['asumismenot_asumistuki']+=self.elakelainen_asumismenot_asumistuki*0.5
                else:
                    p['asumismenot_toimeentulo']+=self.muu_asumismenot_toimeentulo*0.5
                    p['asumismenot_asumistuki']+=self.muu_asumismenot_asumistuki*0.5
            else:
                p['asumismenot_toimeentulo']=self.elakelainen_asumismenot_toimeentulo*0.5+children_under18*self.muu_asumismenot_lisa
                p['asumismenot_asumistuki']=self.elakelainen_asumismenot_asumistuki*0.5+children_under18*self.muu_asumismenot_lisa            
        elif employment_state==15:
            p['asumismenot_toimeentulo']=0
            p['asumismenot_asumistuki']=0
        else: # muu
            if puoliso>0:
                p['asumismenot_toimeentulo']=self.muu_asumismenot_toimeentulo+children_under18*self.muu_asumismenot_lisa
                p['asumismenot_asumistuki']=self.muu_asumismenot_asumistuki+children_under18*self.muu_asumismenot_lisa
                if puoliso_tila==12:
                    p['asumismenot_toimeentulo']+=self.opiskelija_asumismenot_toimeentulo
                    p['asumismenot_asumistuki']+=self.opiskelija_asumismenot_asumistuki
                elif puoliso_tila in set([2,3,8,9]):
                    p['asumismenot_toimeentulo']+=self.elakelainen_asumismenot_toimeentulo
                    p['asumismenot_asumistuki']+=self.elakelainen_asumismenot_asumistuki
                else:
                    p['asumismenot_toimeentulo']+=self.muu_asumismenot_toimeentulo
                    p['asumismenot_asumistuki']+=self.muu_asumismenot_asumistuki
            else:
                p['asumismenot_toimeentulo']=self.muu_asumismenot_toimeentulo+children_under18*self.muu_asumismenot_lisa
                p['asumismenot_asumistuki']=self.muu_asumismenot_asumistuki+children_under18*self.muu_asumismenot_lisa

    def comp_benefits(self,wage : float,old_wage : float,kansanelake : float,tyoelake : float,employment_state : int,
                    time_in_state : float,children_under3 : int,children_under7 : int,children_under18 : int,ika : float,
                    puoliso : int,puoliso_tila : int,puoliso_palkka : float,puoliso_kansanelake : float,puoliso_tyoelake : float,
                    puoliso_old_wage : float,puoliso_time_in_state : float,
                    used_unemp_benefit : float, puoliso_used_unemp_benefit : float,
                    g : int,p_g : int,
                    irtisanottu=0,puoliso_irtisanottu=0,karenssia_jaljella=0,puoliso_karenssia_jaljella=0,
                    retq=True,ove=False,debug=False):
        '''
        Kutsuu fin_benefits-modulia, jonka avulla lasketaan etuudet ja huomioidaan verotus
        Laske etuuksien arvo, kun 
            wage on palkka
            old_wage on vanha palkka
            pension on maksettavan eläkkeen määrä
            employment_state on töissä olo (0)/työttämyys (1)/eläkkeellä olo (2) jne.
            time_in_state on kesto tilassa
            ika on henkilän ikä
        '''
        
        if puoliso>0 and not (employment_state==15 or puoliso_tila==15): # pariskunta
            p=self.setup_benefits(wage,old_wage,kansanelake,tyoelake,employment_state,time_in_state,ika,used_unemp_benefit,
                children_under3,children_under7,children_under18,puoliso=0,
                irtisanottu=irtisanottu,karenssia_jaljella=karenssia_jaljella,alku='')

            p=self.setup_benefits(puoliso_palkka,puoliso_old_wage,puoliso_kansanelake,puoliso_tyoelake,puoliso_tila,puoliso_time_in_state,ika,puoliso_used_unemp_benefit,
                      children_under3,children_under7,children_under18,puoliso=1,
                      irtisanottu=puoliso_irtisanottu,karenssia_jaljella=puoliso_karenssia_jaljella,
                      alku='puoliso_',p2=p)
                      
            self.setup_asumismenot(employment_state,puoliso,puoliso_tila,children_under18,p)
            self.setup_children(p,puoliso,employment_state,puoliso_tila,children_under3,children_under7,children_under18,children_under18)

            p['aikuisia']=2
            
            netto,benefitq=self.ben.laske_tulot_v2(p,include_takuuelake=self.include_takuuelake,omatalku='',puolisoalku='puoliso_')
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
            
            #if puoliso_tila in set([5,6,7]) or employment_state in set([5,6,7]):
            #    print(employment_state,puoliso_tila,benefitq['pvhoito'])
        else: # ei pariskunta
            if employment_state in [5,6,7]:
                if puoliso_tila not in [5,6,7] or (puoliso_tila in [5,6,7] and g>p_g):
                    p=self.setup_benefits(wage,old_wage,kansanelake,tyoelake,employment_state,time_in_state,ika,used_unemp_benefit,
                        children_under3,children_under7,children_under18,puoliso=0,
                        irtisanottu=irtisanottu,karenssia_jaljella=karenssia_jaljella,alku='')
                    self.setup_children(p,puoliso,employment_state,puoliso_tila,children_under3,children_under7,children_under18,children_under18)
                    self.setup_asumismenot(employment_state,0,-1,children_under18,p)
                else:
                    # lapset 0 tässä, yksinkertaistus
                    p=self.setup_benefits(wage,old_wage,kansanelake,tyoelake,employment_state,time_in_state,ika,used_unemp_benefit,
                        0,0,0,puoliso=0,irtisanottu=irtisanottu,karenssia_jaljella=karenssia_jaljella,alku='')
                    self.setup_children(p,puoliso,employment_state,puoliso_tila,0,0,0,children_under18)
                    self.setup_asumismenot(employment_state,0,-1,0,p)
            elif (employment_state!=15 and g>p_g) and puoliso_tila not in [5,6,7]:
                # lapset itsellä, ei puolisolla. tässä epäsymmetria
                p=self.setup_benefits(wage,old_wage,kansanelake,tyoelake,employment_state,time_in_state,ika,used_unemp_benefit,
                    children_under3,children_under7,children_under18,puoliso=0,
                    irtisanottu=irtisanottu,karenssia_jaljella=karenssia_jaljella,alku='')
                self.setup_children(p,puoliso,employment_state,puoliso_tila,children_under3,children_under7,children_under18,children_under18)
                self.setup_asumismenot(employment_state,0,-1,children_under18,p)
            else:
                # lapset 0 tässä, yksinkertaistus
                p=self.setup_benefits(wage,old_wage,kansanelake,tyoelake,employment_state,time_in_state,ika,used_unemp_benefit,
                    0,0,0,puoliso=0,irtisanottu=irtisanottu,karenssia_jaljella=karenssia_jaljella,alku='')
                self.setup_children(p,puoliso,employment_state,puoliso_tila,0,0,0,children_under18)
                self.setup_asumismenot(employment_state,0,-1,0,p)
    
            netto1,benefitq1=self.ben.laske_tulot_v2(p,include_takuuelake=self.include_takuuelake)
            
            if puoliso_tila in [5,6,7]:
                if employment_state not in [5,6,7] or (employment_state in [5,6,7] and p_g>g):
                    p2=self.setup_benefits(puoliso_palkka,puoliso_old_wage,puoliso_kansanelake,puoliso_tyoelake,puoliso_tila,puoliso_time_in_state,ika,puoliso_used_unemp_benefit,
                              children_under3,children_under7,children_under18,puoliso=0,irtisanottu=puoliso_irtisanottu,
                              karenssia_jaljella=puoliso_karenssia_jaljella,alku='')
                    self.setup_children(p2,puoliso,puoliso_tila,employment_state,children_under3,children_under7,children_under18,children_under18)
                    self.setup_asumismenot(puoliso_tila,0,-1,children_under18,p2)
                else:
                    p2=self.setup_benefits(puoliso_palkka,puoliso_old_wage,puoliso_kansanelake,puoliso_tyoelake,puoliso_tila,puoliso_time_in_state,ika,puoliso_used_unemp_benefit,
                              0,0,0,puoliso=0,irtisanottu=puoliso_irtisanottu,karenssia_jaljella=puoliso_karenssia_jaljella,
                              alku='',p2=p)
                    self.setup_children(p2,puoliso,puoliso_tila,-1,0,0,0,children_under18)
                    self.setup_asumismenot(puoliso_tila,0,-1,0,p2)
            elif (puoliso_tila!=15 and g<p_g) and employment_state not in [5,6,7]:
                # lapsilisat maksetaan puolisolle
                p2=self.setup_benefits(puoliso_palkka,puoliso_old_wage,puoliso_kansanelake,puoliso_tyoelake,puoliso_tila,puoliso_time_in_state,ika,puoliso_used_unemp_benefit,
                          children_under3,children_under7,children_under18,puoliso=0,irtisanottu=puoliso_irtisanottu,
                          karenssia_jaljella=puoliso_karenssia_jaljella,alku='')
                self.setup_children(p2,puoliso,puoliso_tila,employment_state,children_under3,children_under7,children_under18,children_under18)
                self.setup_asumismenot(puoliso_tila,0,-1,children_under18,p2)
            else:
                p2=self.setup_benefits(puoliso_palkka,puoliso_old_wage,puoliso_kansanelake,puoliso_tyoelake,puoliso_tila,puoliso_time_in_state,ika,puoliso_used_unemp_benefit,
                          0,0,0,puoliso=0,irtisanottu=puoliso_irtisanottu,karenssia_jaljella=puoliso_karenssia_jaljella,
                          alku='',p2=p)
                self.setup_children(p2,puoliso,puoliso_tila,-1,0,0,0,children_under18)
                self.setup_asumismenot(puoliso_tila,0,-1,0,p2)
                          
            netto2,benefitq2=self.ben.laske_tulot_v2(p2,include_takuuelake=self.include_takuuelake,omat='puoliso_',puoliso='omat_',omatalku='',puolisoalku='puoliso_') # switch order
            netto=netto1+netto2
            
            if netto1<1 and debug:
                print(f'netto<1, omat tila {employment_state}',wage,old_wage,kansanelake,tyoelake,time_in_state,ika,children_under3,children_under7,children_under18)
                print(benefitq1)
            if netto2<1 and puoliso>0 and debug:
                print(f'netto<1, spouse {puoliso_tila}',puoliso_palkka,puoliso_old_wage,puoliso_kansanelake,puoliso_tyoelake,puoliso_time_in_state,ika)
                print(benefitq2)

            if (benefitq1['omat_netto']<1e-6 or benefitq2['puoliso_netto']<1e-6) and debug:
                print(f'omat netto {employment_state}: ',benefitq1['omat_netto'])
                print(f'puoliso netto {puoliso_tila}: ',benefitq2['puoliso_netto'])

            netto_omat=benefitq1['omat_netto']*12
            netto_puoliso=benefitq2['puoliso_netto']*12
            
            #if puoliso_tila in set([5,6,7]):
            #    print(puoliso_tila,benefitq2['pvhoito'])
            #if  employment_state in set([5,6,7]):
            #    print(employment_state,benefitq1['pvhoito'])

            benefitq=self.ben.add_q(benefitq1,benefitq2)
            benefitq['netto']=netto
            netto=netto*12
            
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
        Lähtäarvot 18-vuotiaana
        '''

        initial_weight=np.zeros((6,7))
        if self.year==2018:
            # tilat [13,0,1,10,3,11,12] siis [tmtuki,ansiosidonnainen,kokoaikatyö,osaaikatyö,työvoimanulkopuolella,opiskelija]
            # lasketaan painotetut kertoimet eri tulotasoille
            # työttämät miehet, osuus väestästä
            m_tyoton=0.019
            m1,m2,m3=self.rates.get_wees(0,1.27,1.0,m_tyoton)
            # työttämät naiset, osuus väestästä
            w_tyoton=0.01609
            w1,w2,w3=self.rates.get_wees(1,1.2,1.0,w_tyoton)

            # työlliset, miehet
            om=0.2565  # miehet töissä + opiskelija
            om1,om2,om3=self.rates.get_wees(0,1.25,1.0,om)
            
            ow=0.3443 
            ow1,ow2,ow3=self.rates.get_wees(1,1.2,1.0,ow)
            # työvoiman ulkopuolella, miehet
            m_tyovoimanulkop=0.029
            um1,um2,um3=self.rates.get_wees(0,1.3,1.0,m_tyovoimanulkop)
            
            # työvoiman ulkopuolella, naiset
            w_tyovoimanulkop=0.025
            uw1,uw2,uw3=self.rates.get_wees(1,1.2,1.0,w_tyovoimanulkop)
            # työkyvyttömät, miehet
            md=0.008
            md1,md2,md3=self.rates.get_wees(0,1.3,1.0,md)
            # työkyvyttömät, naiset
            wd=0.00730
            wd1,wd2,wd3=self.rates.get_wees(1,1.2,1.0,wd)
            
            # opiskelijat+työlliset+muut työvoiman ulkop+työttämät, miehet
            ym=0.686+om+m_tyovoimanulkop+m_tyoton # miehet töissä + opiskelija
            # opiskelijat+työlliset+muut työvoiman ulkop+työttämät, naiset
            yw=0.6076+ow+w_tyovoimanulkop+w_tyoton
            
            # 13,0,1,10,3,11,12
            initial_weight[0,:]=[m1*4/5,m1*1/5,0.68*om1,0.32*om1,md1,um1,1-om1-um1-m1-md1]
            initial_weight[1,:]=[m2*4/5,m2*1/5,0.68*om2,0.32*om2,md2,um2,1-om2-um2-m2-md2]
            initial_weight[2,:]=[m3*4/5,m3*1/5,0.68*om3,0.32*om3,md3,um3,1-om3-um3-m3-md3]
            initial_weight[3,:]=[w1*4/5,w1*1/5,0.44*ow1,0.56*ow1,wd1,uw1,1-ow1-uw1-w1-wd1]
            initial_weight[4,:]=[w2*4/5,w2*1/5,0.44*ow2,0.56*ow2,wd2,uw2,1-ow2-uw2-w2-wd2]
            initial_weight[5,:]=[w3*4/5,w3*1/5,0.44*ow3,0.56*ow3,wd3,uw3,1-ow3-uw3-w3-wd3]
        elif self.year==2019: # 
            # tilat [13,0,1,10,3,11,12] siis [tmtuki,ansiosidonnainen,kokoaikatyö,osaaikatyö,työvoimanulkopuolella,opiskelija]
            # lasketaan painotetut kertoimet eri tulotasoille
            # työttämät miehet, osuus väestästä
            m_tyoton=0.02133
            m1,m2,m3=self.rates.get_wees(0,1.27,1.0,m_tyoton)
            # työttämät naiset, osuus väestästä
            w_tyoton=0.01653
            w1,w2,w3=self.rates.get_wees(1,1.2,1.0,w_tyoton)

            # työlliset, miehet
            om=0.2606  # miehet töissä + opiskelija
            om1,om2,om3=self.rates.get_wees(0,1.25,1.0,om)
            
            ow=0.3633
            ow1,ow2,ow3=self.rates.get_wees(1,1.2,1.0,ow)
            # työvoiman ulkopuolella, miehet
            m_tyovoimanulkop=0.0301
            um1,um2,um3=self.rates.get_wees(0,1.3,1.0,m_tyovoimanulkop)
            
            # työvoiman ulkopuolella, naiset
            w_tyovoimanulkop=0.02661
            uw1,uw2,uw3=self.rates.get_wees(1,1.2,1.0,w_tyovoimanulkop)
            
            # työkyvyttömät, miehet
            md=0.01031
            md1,md2,md3=self.rates.get_wees(0,1.3,1.0,md)
            # työkyvyttömät, naiset
            wd=0.007156
            wd1,wd2,wd3=self.rates.get_wees(1,1.2,1.0,wd)
            
            # 13,0,1,10,3,11,12
            initial_weight[0,:]=[m1*4/5,m1*1/5,0.68*om1,0.32*om1,md1,um1,1-om1-um1-m1-md1]
            initial_weight[1,:]=[m2*4/5,m2*1/5,0.68*om2,0.32*om2,md2,um2,1-om2-um2-m2-md2]
            initial_weight[2,:]=[m3*4/5,m3*1/5,0.68*om3,0.32*om3,md3,um3,1-om3-um3-m3-md3]
            initial_weight[3,:]=[w1*4/5,w1*1/5,0.44*ow1,0.56*ow1,wd1,uw1,1-ow1-uw1-w1-wd1]
            initial_weight[4,:]=[w2*4/5,w2*1/5,0.44*ow2,0.56*ow2,wd2,uw2,1-ow2-uw2-w2-wd2]
            initial_weight[5,:]=[w3*4/5,w3*1/5,0.44*ow3,0.56*ow3,wd3,uw3,1-ow3-uw3-w3-wd3]
        elif self.year==2020: # 
            # tilat [13,0,1,10,3,11,12] siis [tmtuki,ansiosidonnainen,kokoaikatyö,osaaikatyö,työvoimanulkopuolella,opiskelija]
            # lasketaan painotetut kertoimet eri tulotasoille
            # työttämät miehet, osuus väestästä
            m_tyoton=0.02399
            m1,m2,m3=self.rates.get_wees(0,1.27,1.0,m_tyoton)
            # työttämät naiset, osuus väestästä
            w_tyoton=0.02051
            w1,w2,w3=self.rates.get_wees(1,1.2,1.0,w_tyoton)

            # työlliset, miehet
            om=0.2053  # miehet töissä + opiskelija
            om1,om2,om3=self.rates.get_wees(0,1.25,1.0,om)
            
            ow=0.2732
            ow1,ow2,ow3=self.rates.get_wees(1,1.2,1.0,ow)
            # työvoiman ulkopuolella, miehet
            m_tyovoimanulkop=0.03278
            um1,um2,um3=self.rates.get_wees(0,1.3,1.0,m_tyovoimanulkop)
            
            # työvoiman ulkopuolella, naiset
            w_tyovoimanulkop=0.02433
            uw1,uw2,uw3=self.rates.get_wees(1,1.2,1.0,w_tyovoimanulkop)
            # työkyvyttömät, miehet
            md=0.00982
            md1,md2,md3=self.rates.get_wees(0,1.3,1.0,md)
            # työkyvyttömät, naiset
            wd=0.00685
            wd1,wd2,wd3=self.rates.get_wees(1,1.2,1.0,wd)
            
            # 13,0,1,10,3,11,12
            initial_weight[0,:]=[m1*4/5,m1*1/5,0.68*om1,0.32*om1,md1,um1,1-om1-um1-m1-md1]
            initial_weight[1,:]=[m2*4/5,m2*1/5,0.68*om2,0.32*om2,md2,um2,1-om2-um2-m2-md2]
            initial_weight[2,:]=[m3*4/5,m3*1/5,0.68*om3,0.32*om3,md3,um3,1-om3-um3-m3-md3]
            initial_weight[3,:]=[w1*4/5,w1*1/5,0.44*ow1,0.56*ow1,wd1,uw1,1-ow1-uw1-w1-wd1]
            initial_weight[4,:]=[w2*4/5,w2*1/5,0.44*ow2,0.56*ow2,wd2,uw2,1-ow2-uw2-w2-wd2]
            initial_weight[5,:]=[w3*4/5,w3*1/5,0.44*ow3,0.56*ow3,wd3,uw3,1-ow3-uw3-w3-wd3]
        elif self.year==2021: # PÄIVITÄ
            # tilat [13,0,1,10,3,11,12] siis [tmtuki,ansiosidonnainen,kokoaikatyö,osaaikatyö,työvoimanulkopuolella,opiskelija]
            # lasketaan painotetut kertoimet eri tulotasoille
            # työttämät miehet, osuus väestästä
            m_tyoton=0.021
            m1,m2,m3=self.rates.get_wees(0,1.27,1.0,m_tyoton)
            # työttämät naiset, osuus väestästä
            w_tyoton=0.017
            w1,w2,w3=self.rates.get_wees(1,1.2,1.0,w_tyoton)

            # työlliset, miehet
            om=0.2565  # miehet töissä + opiskelija
            om1,om2,om3=self.rates.get_wees(0,1.25,1.0,om)
            
            ow=0.360 
            ow1,ow2,ow3=self.rates.get_wees(1,1.2,1.0,ow)
            # työvoiman ulkopuolella, miehet
            m_tyovoimanulkop=0.030
            um1,um2,um3=self.rates.get_wees(0,1.3,1.0,m_tyovoimanulkop)
            
            # työvoiman ulkopuolella, naiset
            w_tyovoimanulkop=0.027
            uw1,uw2,uw3=self.rates.get_wees(1,1.2,1.0,w_tyovoimanulkop)
            # työkyvyttömät, miehet
            md=0.009
            md1,md2,md3=self.rates.get_wees(0,1.3,1.0,md)
            # työkyvyttömät, naiset
            wd=0.006
            wd1,wd2,wd3=self.rates.get_wees(1,1.2,1.0,wd)
            
            # 13,0,1,10,3,11,12
            initial_weight[0,:]=[m1*4/5,m1*1/5,0.68*om1,0.32*om1,md1,um1,1-om1-um1-m1-md1]
            initial_weight[1,:]=[m2*4/5,m2*1/5,0.68*om2,0.32*om2,md2,um2,1-om2-um2-m2-md2]
            initial_weight[2,:]=[m3*4/5,m3*1/5,0.68*om3,0.32*om3,md3,um3,1-om3-um3-m3-md3]
            initial_weight[3,:]=[w1*4/5,w1*1/5,0.44*ow1,0.56*ow1,wd1,uw1,1-ow1-uw1-w1-wd1]
            initial_weight[4,:]=[w2*4/5,w2*1/5,0.44*ow2,0.56*ow2,wd2,uw2,1-ow2-uw2-w2-wd2]
            initial_weight[5,:]=[w3*4/5,w3*1/5,0.44*ow3,0.56*ow3,wd3,uw3,1-ow3-uw3-w3-wd3]
        elif self.year==2022: # PÄIVITÄ
            # tilat [13,0,1,10,3,11,12] siis [tmtuki,ansiosidonnainen,kokoaikatyö,osaaikatyö,työvoimanulkopuolella,opiskelija]
            # lasketaan painotetut kertoimet eri tulotasoille
            # työttämät miehet, osuus väestästä
            m_tyoton=0.021
            m1,m2,m3=self.rates.get_wees(0,1.27,1.0,m_tyoton)
            # työttämät naiset, osuus väestästä
            w_tyoton=0.017
            w1,w2,w3=self.rates.get_wees(1,1.2,1.0,w_tyoton)

            # työlliset, miehet
            om=0.2565  # miehet töissä + opiskelija
            om1,om2,om3=self.rates.get_wees(0,1.25,1.0,om)
            
            ow=0.360 
            ow1,ow2,ow3=self.rates.get_wees(1,1.2,1.0,ow)
            # työvoiman ulkopuolella, miehet
            m_tyovoimanulkop=0.030
            um1,um2,um3=self.rates.get_wees(0,1.3,1.0,m_tyovoimanulkop)
            
            # työvoiman ulkopuolella, naiset
            w_tyovoimanulkop=0.027
            uw1,uw2,uw3=self.rates.get_wees(1,1.2,1.0,w_tyovoimanulkop)
            # työkyvyttömät, miehet
            md=0.009
            md1,md2,md3=self.rates.get_wees(0,1.3,1.0,md)
            # työkyvyttömät, naiset
            wd=0.006
            wd1,wd2,wd3=self.rates.get_wees(1,1.2,1.0,wd)
            
            # 13,0,1,10,3,11,12
            initial_weight[0,:]=[m1*4/5,m1*1/5,0.68*om1,0.32*om1,md1,um1,1-om1-um1-m1-md1]
            initial_weight[1,:]=[m2*4/5,m2*1/5,0.68*om2,0.32*om2,md2,um2,1-om2-um2-m2-md2]
            initial_weight[2,:]=[m3*4/5,m3*1/5,0.68*om3,0.32*om3,md3,um3,1-om3-um3-m3-md3]
            initial_weight[3,:]=[w1*4/5,w1*1/5,0.44*ow1,0.56*ow1,wd1,uw1,1-ow1-uw1-w1-wd1]
            initial_weight[4,:]=[w2*4/5,w2*1/5,0.44*ow2,0.56*ow2,wd2,uw2,1-ow2-uw2-w2-wd2]
            initial_weight[5,:]=[w3*4/5,w3*1/5,0.44*ow3,0.56*ow3,wd3,uw3,1-ow3-uw3-w3-wd3]
        elif self.year==2023: # PÄIVITÄ
            # tilat [13,0,1,10,3,11,12] siis [tmtuki,ansiosidonnainen,kokoaikatyö,osaaikatyö,työvoimanulkopuolella,opiskelija]
            # lasketaan painotetut kertoimet eri tulotasoille
            # työttämät miehet, osuus väestästä
            m_tyoton=0.021
            m1,m2,m3=self.rates.get_wees(0,1.27,1.0,m_tyoton)
            # työttämät naiset, osuus väestästä
            w_tyoton=0.017
            w1,w2,w3=self.rates.get_wees(1,1.2,1.0,w_tyoton)

            # työlliset, miehet
            om=0.2565  # miehet töissä + opiskelija
            om1,om2,om3=self.rates.get_wees(0,1.25,1.0,om)
            
            ow=0.360 
            ow1,ow2,ow3=self.rates.get_wees(1,1.2,1.0,ow)
            # työvoiman ulkopuolella, miehet
            m_tyovoimanulkop=0.030
            um1,um2,um3=self.rates.get_wees(0,1.3,1.0,m_tyovoimanulkop)
            
            # työvoiman ulkopuolella, naiset
            w_tyovoimanulkop=0.027
            uw1,uw2,uw3=self.rates.get_wees(1,1.2,1.0,w_tyovoimanulkop)
            # työkyvyttömät, miehet
            md=0.009
            md1,md2,md3=self.rates.get_wees(0,1.3,1.0,md)
            # työkyvyttömät, naiset
            wd=0.006
            wd1,wd2,wd3=self.rates.get_wees(1,1.2,1.0,wd)
            
            # 13,0,1,10,3,11,12
            initial_weight[0,:]=[m1*4/5,m1*1/5,0.68*om1,0.32*om1,md1,um1,1-om1-um1-m1-md1]
            initial_weight[1,:]=[m2*4/5,m2*1/5,0.68*om2,0.32*om2,md2,um2,1-om2-um2-m2-md2]
            initial_weight[2,:]=[m3*4/5,m3*1/5,0.68*om3,0.32*om3,md3,um3,1-om3-um3-m3-md3]
            initial_weight[3,:]=[w1*4/5,w1*1/5,0.44*ow1,0.56*ow1,wd1,uw1,1-ow1-uw1-w1-wd1]
            initial_weight[4,:]=[w2*4/5,w2*1/5,0.44*ow2,0.56*ow2,wd2,uw2,1-ow2-uw2-w2-wd2]
            initial_weight[5,:]=[w3*4/5,w3*1/5,0.44*ow3,0.56*ow3,wd3,uw3,1-ow3-uw3-w3-wd3]            
        else:
            print('Unsupported year',self.year)
            error(999)
            
        for k in range(6):
            scale=np.sum(initial_weight[k,:])
            initial_weight[k,:] /= scale
            
        return initial_weight

    def scale_pension(self,pension : float,age : float,scale=True,unemp_after_ra=0,elinaikakerroin=True):
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
        
    def move_to_parttime(self,wage : float,pension : float,tyoelake : float,old_wage : float,age : float,tyoura : float,time_in_state : float,
                        has_spouse=False,is_spouse=False):
        '''
        Siirtymä osa-aikaiseen työskentelyyn
        '''
        employment_status = 10 # switch to part-time work
        parttimewage=0.5*wage
        tyoura += self.timestep
        time_in_state=self.timestep
        old_wage=0
        pension=self.pension_accrual(age,parttimewage,pension,state=employment_status)
        pinkslip=0
        tyoelake = tyoelake * self.elakeindeksi

        return employment_status,pension,tyoelake,wage,time_in_state,tyoura,pinkslip

    def move_to_work(self,wage : float,pension : float,tyoelake : float,old_wage : float,age : float,time_in_state : float,tyoura : float,pinkslip : int,
                    has_spouse=False,is_spouse=False):
        '''
        Siirtymä täysiaikaiseen työskentelyyn
        '''
        employment_status = 1 # täihin
        pinkslip=0
        old_wage=0
        tyoura+=self.timestep
        pension=self.pension_accrual(age,wage,pension,state=employment_status)
        time_in_state=self.timestep
        tyoelake=tyoelake*self.elakeindeksi

        return employment_status,pension,tyoelake,wage,time_in_state,tyoura,pinkslip

    def move_to_oa_fulltime(self,wage : float,pension : float,old_wage : float,age : float,kansanelake : float,tyoelake : float,employment_status : int,
            unemp_after_ra : float,scale_acc=True,has_spouse=0,is_spouse=False):
        '''
        Siirtymä vanhuuseläkkeelle
        '''
        if age>=self.max_retirementage: # move to retirement state 2
            if employment_status in set([2,8,9]): # ve, ve+työ, ve+osatyö
                # ei lykkäyskorotusta
                tyoelake = self.elakeindeksi*tyoelake+self.scale_pension(pension,age,scale=False,unemp_after_ra=unemp_after_ra)
                kansanelake = kansanelake * self.kelaindeksi
                pension=0
                employment_status = 2
            else: # ei vielä eläkkeellä
                # lykkäyskorotus
                tyoelake = tyoelake*self.elakeindeksi + pension
                if self.include_kansanelake:
                    kansanelake = self.ben.laske_kansanelake(age,tyoelake*self.elinaikakerroin/12,1-has_spouse)*12 # ben-modulissa palkat kk-tasolla
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
            if employment_status in set([2,8,9]): # ve, ve+työ, ve+osatyö
                tyoelake = self.elakeindeksi*tyoelake
                kansanelake = kansanelake * self.kelaindeksi
                employment_status = 9
            elif employment_status==4:
                # lykkäyskorotus
                tyoelake = tyoelake*self.elakeindeksi + pension
                if self.include_kansanelake:
                    # vähentämätön kansaneläke
                    kansanelake = self.ben.laske_kansanelake(age,tyoelake*self.elinaikakerroin/12,1-has_spouse,disability=True)*12 # ben-modulissa palkat kk-tasolla
                else:
                    kansanelake = 0
                # lykkäys ei vähennä kansaneläkettä
                tyoelake += (self.scale_pension(pension,age,scale=scale_acc,unemp_after_ra=unemp_after_ra) - pension)
                pension=0
                employment_status = 9
            else:
                # lykkäyskorotus
                tyoelake = tyoelake*self.elakeindeksi + pension
                if self.include_kansanelake:
                    kansanelake = self.ben.laske_kansanelake(age,tyoelake*self.elinaikakerroin/12,1-has_spouse)*12 # ben-modulissa palkat kk-tasolla
                else:
                    kansanelake = 0
                # lykkäys ei vähennä kansaneläkettä
                tyoelake += (self.scale_pension(pension,age,scale=scale_acc,unemp_after_ra=unemp_after_ra) - pension)
                pension=0
                employment_status = 9

            time_in_state=self.timestep
            alkanut_ansiosidonnainen=0
            pension=self.pension_accrual(age,wage,pension,state=employment_status)
            ove_paid=0
        else: # työvoiman ulkopuolella
            time_in_state=0
            employment_status = 2 
            pension = pension * self.palkkakerroin
            wage=old_wage
            time_in_state+=self.timestep
            ove_paid=0

        return employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid

    def move_to_student(self,wage : float,pension : float,tyoelake : float,old_wage : float,age : float,time_in_state : float,tyoura : float,pinkslip : int):
        '''
        Siirtymä opiskelijaksi
        Tässä ei muuttujien päivityksiä, koska se tehdään jo muualla!
        '''
        employment_status = 12
        time_in_state=self.timestep
        pinkslip=0
        tyoelake=tyoelake*self.elakeindeksi
        pension=pension*self.palkkakerroin

        return employment_status,pension,tyoelake,wage,time_in_state,pinkslip
        
    def move_to_oa_parttime(self,wage : float,pension : float,old_wage : float,age : float,kansanelake : float,tyoelake : float,employment_status : float,
            unemp_after_ra : float,scale_acc=True,has_spouse=0,is_spouse=False):
        '''
        Siirtymä vanhuuseläkkeelle
        '''
        if age>=self.max_retirementage: # move to retirement state 2
            if employment_status in set([2,8,9]): # ve, ve+työ, ve+osatyö
                # ei lykkäyskorotusta
                tyoelake = self.elakeindeksi*tyoelake+self.scale_pension(pension,age,scale=False,unemp_after_ra=unemp_after_ra)
                kansanelake = kansanelake * self.kelaindeksi
                pension=0
                employment_status = 2
            else: # ei vielä eläkkeellä
                # lykkäyskorotus
                tyoelake = tyoelake*self.elakeindeksi + pension
                if self.include_kansanelake:
                    kansanelake = self.ben.laske_kansanelake(age,tyoelake*self.elinaikakerroin/12,1-has_spouse)*12 # ben-modulissa palkat kk-tasolla
                else:
                    kansanelake = 0
                    
                # lykkäys ei vähennä kansaneläkettä
                tyoelake += (self.scale_pension(pension,age,scale=scale_acc,unemp_after_ra=unemp_after_ra) - pension)
                pension=0
                employment_status = 2
                
                #print('kansaneläke',kansanelake,'tyoelake',tyoelake)
                
            time_in_state=self.timestep
            alkanut_ansiosidonnainen=0
            ove_paid=0
        elif age>=self.min_retirementage:
            if employment_status in set([2,8,9]): # ve, ve+työ, ve+osatyö
                tyoelake = self.elakeindeksi * tyoelake
                kansanelake = kansanelake * self.kelaindeksi
                employment_status = 8
            elif employment_status==4:
                # lykkäyskorotus
                tyoelake = tyoelake*self.elakeindeksi + pension
                if self.include_kansanelake:
                    kansanelake = self.ben.laske_kansanelake(age,tyoelake*self.elinaikakerroin/12,1-has_spouse,disability=True)*12 # ben-modulissa palkat kk-tasolla
                else:
                    kansanelake = 0
                # lykkäys ei vähennä kansaneläkettä
                tyoelake += (self.scale_pension(pension,age,scale=scale_acc,unemp_after_ra=unemp_after_ra) - pension)
                pension=0
                employment_status = 8
            else:
                # lykkäyskorotus
                tyoelake = tyoelake*self.elakeindeksi + pension
                if self.include_kansanelake:
                    kansanelake = self.ben.laske_kansanelake(age,tyoelake*self.elinaikakerroin/12,1-has_spouse)*12 # ben-modulissa palkat kk-tasolla
                else:
                    kansanelake = 0
                # lykkäys ei vähennä kansaneläkettä
                tyoelake += (self.scale_pension(pension,age,scale=scale_acc,unemp_after_ra=unemp_after_ra) - pension)
                pension=0
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

    def move_to_ove(self,employment_status : int,pension : float,tyoelake : float,ove_paid : float,age : float,unemp_after_ra : float):
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
            tyoelake = self.scale_pension(self.ove_ratio*pension,age,scale=True,unemp_after_ra=unemp_after_ra)/self.elakeindeksi # ei eläkeindeksi tässä
            kansanelake = 0
            pension=(1-self.ove_ratio)*pension # *self.palkkakerroin, tässä ei indeksoida, koska pension_accrual hoitaa tämän
            ove_paid=1

        return pension,tyoelake,ove_paid

    def move_to_retirement(self,wage : float,pension : float,old_wage : float,age : float,kansanelake : float,tyoelake : float,employment_status : float,unemp_after_ra : float,
        all_acc=True,scale_acc=True,has_spouse=0,is_spouse=False):
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
                    kansanelake = self.ben.laske_kansanelake(age,tyoelake*self.elinaikakerroin/12,1-has_spouse)*12 # ben-modulissa palkat kk-tasolla
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
                    pension = pension * self.palkkakerroin
                    tyoelake = self.elakeindeksi * tyoelake
                    kansanelake = kansanelake * self.kelaindeksi
                elif employment_status==4: # putki
                    # lykkäyskorotus
                    tyoelake = tyoelake * self.elakeindeksi + pension
                    if self.include_kansanelake: # ei varhennnusvähennystä putken tapauksessa
                        kansanelake = self.ben.laske_kansanelake(age,tyoelake*self.elinaikakerroin/12,1-has_spouse,disability=True)*12 # ben-modulissa palkat kk-tasolla
                    else:
                        kansanelake = 0
                    # lykkäys ei vähennä kansaneläkettä
                    tyoelake += (self.scale_pension(pension,age,scale=scale_acc,unemp_after_ra=unemp_after_ra) - pension)
                    pension=0
                    employment_status = 2 
                else:
                    # lykkäyskorotus
                    tyoelake = tyoelake * self.elakeindeksi + pension
                    if self.include_kansanelake:
                        kansanelake = self.ben.laske_kansanelake(age,tyoelake*self.elinaikakerroin/12,1-has_spouse)*12 # ben-modulissa palkat kk-tasolla
                    else:
                        kansanelake = 0
                    # lykkäys ei vähennä kansaneläkettä
                    tyoelake += (self.scale_pension(pension,age,scale=scale_acc,unemp_after_ra=unemp_after_ra) - pension)
                    pension=0
                    employment_status = 2 
                    
            elif employment_status in set([8,9]): # ve, ve+työ, ve+osatyö
                tyoelake = self.elakeindeksi * tyoelake
                kansanelake = kansanelake * self.kelaindeksi
                pension = pension * self.palkkakerroin
                employment_status = 2 
            else:
                print('error 289')

            time_in_state=self.timestep
            alkanut_ansiosidonnainen=0
            ove_paid=0
        else: # työvoiman ulkopuolella
            ove_paid=0
            time_in_state = 0
            employment_status = 2 
            wage = old_wage
            time_in_state += self.timestep
            print('retired before retirement age!!!!')

        return employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid

    def move_to_retdisab(self,wage : float,pension : float,old_wage : float,age : float,time_in_state : float,kansanelake : float,tyoelake : float,unemp_after_ra : float):   
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
        
    def tyossaoloehto(self,toe : float,tyoura : float,age : float):
        '''
        täyttyykä työssäoloehto
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
    
    def comp_unempdays_left_nykytila(self,kesto : float,tyoura : float,age : float,toe : float,emp : int,alkanut_ansiosidonnainen : int,toe58 : int,old_toe : float,printti=False):
        '''
        Nykytilan mukainen jäljellä olevien työttämyyspäivärahapäivien laskenta
        '''
        if emp in set([2,3,8,9,13]):
            return 0
            
        if self.get_kassanjasenyys()<1:
            return 0
    
        toe_tayttyy=self.tyossaoloehto(toe,tyoura,age)

        if self.include_putki and (
            emp==4 
            or (emp==0 and age>=self.min_tyottputki_ika and tyoura>=self.tyohistoria_tyottputki)
            or (emp in set ([1,10]) and age>=self.min_tyottputki_ika and tyoura>=self.tyohistoria_tyottputki and toe_tayttyy)
            ):
            return max(0,self.max_unemploymentbenefitage-age)

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
         
        return max(0,min(toekesto,self.max_unemploymentbenefitage-age))

    def paivarahapaivia_jaljella_nykytila(self,kesto : float,tyoura : float,age : float,toe58 : int,toe: int):
        '''
        Onko työttämyyspäivärahapäiviä jäljellä?
        '''
        if age>=self.max_unemploymentbenefitage:
            return False
            
        if self.get_kassanjasenyys()<1:
            return False

        if ((tyoura>=self.tyohistoria_vaatimus500 and kesto>=self.apvkesto500 and age>=self.minage_500 and toe58>0) \
            or (tyoura>=self.tyohistoria_vaatimus and kesto>=self.apvkesto400 and (age<self.minage_500 or tyoura<self.tyohistoria_vaatimus500 or toe58<1)) \
            or (tyoura<self.tyohistoria_vaatimus and kesto>=self.apvkesto300)):    
            return False
        else:
            return True
            
    def comp_unempdays_left_porrastus(self,kesto : float,tyoura : float,age : float,toe : float,emp : int,alkanut_ansiosidonnainen : int,toe58 : int,old_toe,printti=False):
        if emp in set([2,3,8,9,13]):
            return 0
    
        if emp==4:
            return min(0,self.max_unemploymentbenefitage-age)
            
        if self.get_kassanjasenyys()<1:
            return 0
        
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
            
        return max(0,min(ret,self.max_unemploymentbenefitage-age))

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
        if age>=self.max_unemploymentbenefitage:
            return False

        if (tyoura>=self.tyohistoria_vaatimus500 and kesto>=self.apvkesto500 and age>=self.minage_500 and toe58>0 and not self.porrastus500) \
            or ((not self.toe_porrastus_kesto(kesto,toe,tyoura)) and (age<self.minage_500 or tyoura<self.tyohistoria_vaatimus500 or toe58<1)):
            return False
        else:
            return True
            
    def update_unempwage_basis(self,unempwage_basis,unempwage,use80percent):
        '''
        Tähän 80% sääntä (jos edellisestä uudelleenmäärittelystä alle vuosi, 80% suojaosa)
        '''
        if use80percent:
            return max(unempwage_basis*0.8,unempwage)
        else:
            return unempwage

    def move_to_unemp(self,wage,pension,old_wage,age,kansanelake,tyoelake,toe,toekesto,irtisanottu,tyoura,
                    used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,alkanut_ansiosidonnainen,toe58,ove_paid,
                    has_spouse,is_spouse):
        '''
        Siirtymä työttämyysturvalle
        '''
        if age>=self.min_retirementage: # ei uusia työttämiä enää alimman ve-iän jälkeen, vanhat jatkavat
            pinkslip=0
            employment_status=0
            unempwage_basis=0
            alkanut_ansiosidonnainen = 0
            used_unemp_benefit = 0
            karenssia_jaljella=0
            
            employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                self.move_to_retirement(wage,pension,old_wage,age,kansanelake,tyoelake,employment_status,
                    unemp_after_ra,all_acc=True,scale_acc=True,has_spouse=has_spouse,is_spouse=is_spouse)
                
            return employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
                   used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,\
                   alkanut_ansiosidonnainen,karenssia_jaljella,ove_paid
        else:
            #if toe>=self.ansiopvraha_toe: # täyttyykä työssäoloehto
            tehto=self.tyossaoloehto(toe,tyoura,age)
            if tehto or alkanut_ansiosidonnainen>0:
                if tehto:
                    kesto=0
                    used_unemp_benefit=0
                    self.infostate_set_enimmaisaika(age,is_spouse=is_spouse) # resetoidaan enimmäisaika
                    if age>=58: # suojasääntö 58v täyttäneille
                        unempwage_basis=max(unempwage,self.update_unempwage_basis(unempwage_basis,unempwage,False))
                    else:
                        if self.infostate_check_aareset(age,is_spouse=is_spouse):
                            unempwage_basis=self.update_unempwage_basis(unempwage_basis,unempwage,True)
                        else:
                            unempwage_basis=self.update_unempwage_basis(unempwage_basis,unempwage,False)
                    
                    jaljella=self.paivarahapaivia_jaljella(kesto,tyoura,age,toe58,toekesto)
                else:
                    kesto=used_unemp_benefit
                    if self.porrasta_toe:
                        jaljella=self.paivarahapaivia_jaljella(kesto,tyoura,age,toe58,self.comp_oldtoe(spouse=is_spouse))
                    else:
                        jaljella=self.paivarahapaivia_jaljella(kesto,tyoura,age,toe58,toekesto) # toe ei vaikuta
                    
                if jaljella:
                    employment_status  = 0 # siirto ansiosidonnaiselle
                    #if alkanut_ansiosidonnainen<1:
                    if irtisanottu>0 or alkanut_ansiosidonnainen>0: # or alkanut_ansiosidonnainen>0: # muuten ei oikeutta ansiopäivärahaan karenssi vuoksi
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
               unempwage_basis,alkanut_ansiosidonnainen,karenssia_jaljella,ove_paid

    def update_karenssi(self,karenssia_jaljella):
        karenssia_jaljella=max(0,karenssia_jaljella-self.timestep)
        return karenssia_jaljella
        

    def move_to_outsider(self,wage,pension,tyoelake,old_wage,age,irtisanottu):
        '''
        Siirtymä työvoiman ulkopuolelle
        '''
        employment_status = 11 # switch
        time_in_state=0
        pension=pension*self.palkkakerroin
        tyoelake=tyoelake*self.elakeindeksi

        time_in_state+=self.timestep
        pinkslip=0

        return employment_status,pension,tyoelake,wage,time_in_state,pinkslip
        
    def move_to_svraha(self,wage,pension,old_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,has_spouse,children_under18,is_spouse):
        '''
        Siirtymä työkyvyttämyyseläkkeelle aktiivista, ei eläkkeeltä
        '''
        employment_status = 14 # sv-päiväraha
        pension=pension*self.palkkakerroin
        tyoelake=tyoelake*self.elakeindeksi
        kansanelake = 0
        old_wage = wage

        time_in_state=self.timestep
        pinkslip=0

        return employment_status,pension,kansanelake,tyoelake,wage,time_in_state,ove_paid
        
    def move_to_disab(self,wage,pension,old_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,has_spouse,children_under18,is_spouse):
        '''
        Siirtymä työkyvyttämyyseläkkeelle aktiivista, ei eläkkeeltä
        '''
        if age<self.min_retirementage: # ensisijaisuusaika sv-päivärahaa
            return self.move_to_svraha(wage,pension,old_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,has_spouse,children_under18,is_spouse)
        else:
            return self.move_to_disab_state(wage,pension,old_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,has_spouse,children_under18,is_spouse)
        
    def move_to_disab_state(self,wage,pension,old_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,has_spouse,children_under18,is_spouse):
        '''
        Siirtymä työkyvyttämyyseläkkeelle aktiivista, ei eläkkeeltä
        '''
        employment_status = 3 # tk
        if age<self.min_retirementage:
            wage5y,realwage=self.infostate_comp_5y_ave_wage(is_spouse=is_spouse) 
            
            if realwage>self.min_disab_tulevaaika: # oikeus tulevaan aikaan, tosin tässä 5v ajalta laskettuna
                tyoelake=(tyoelake+self.elinaikakerroin*(pension+self.acc/self.timestep*wage5y*max(0,self.min_retirementage-age)))*self.elakeindeksi
            else:
                tyoelake=(tyoelake+self.elinaikakerroin*pension)*self.elakeindeksi
            
            if self.include_kansanelake:
                kansanelake = self.ben.laske_kansanelake(age,tyoelake/12,1-has_spouse,disability=True,lapsia=children_under18)*12 # ben-modulissa palkat kk-tasolla
            else:
                kansanelake = 0
            
            pension=0
            alkanut_ansiosidonnainen=0
            time_in_state=self.timestep
            wage=0
            ove_paid=0
            #print('move_to_disab at ',age,'ke',kansanelake,'te',tyoelake,'wage5y',wage5y)
            #wage_reduction=0.60 # vastaa määritelmää
        else:
            # siirtymä vanhuuseläkkeelle, lykkäyskorotus, ei tulevaa aikaa
            tyoelake = tyoelake*self.elakeindeksi + pension
            if self.include_kansanelake:
                kansanelake = self.ben.laske_kansanelake(age,tyoelake*self.elinaikakerroin/12,1-has_spouse,lapsia=children_under18)*12 # ben-modulissa palkat kk-tasolla
            else:
                kansanelake = 0
            # lykkäys ei vähennä kansaneläkettä
            tyoelake += (self.scale_pension(pension,age,scale=True,unemp_after_ra=unemp_after_ra) - pension)
            pension=0

            time_in_state=self.timestep
            alkanut_ansiosidonnainen=0
            ove_paid=0
            #wage_reduction=0.60 # vastaa määritelmää

        return employment_status,pension,kansanelake,tyoelake,wage,time_in_state,ove_paid
        
    def comp_familypension(self,puoliso,emp_state,puoliso_tila,tyoelake,pension,age,puoliso_tyoelake,puoliso_pension,children_under18,has_spouse,is_spouse):
        '''
        Siirtymä työkyvyttämyyseläkkeelle aktiivista, ei eläkkeeltä
        '''
        if puoliso_tila==15:
            return 0
        
        if has_spouse<1:
            return puoliso_tyoelake
            
        if emp_state in set([2,3,8,9]):
            add_pension=0.5*tyoelake
        elif age<self.min_retirementage:
            wage5y,_=self.infostate_comp_5y_ave_wage(is_spouse=not is_spouse) 
            dis_pension=(tyoelake+self.elinaikakerroin*(pension+self.acc/self.timestep*wage5y*max(0,self.min_retirementage-age)))*self.elakeindeksi
            add_pension=0.5*dis_pension
        else:
            tyoelake = tyoelake*self.elakeindeksi + self.scale_pension(pension,age,scale=True,unemp_after_ra=0)
            add_pension=0.5*tyoelake

        if puoliso_tila in set([2,3,8,9]):
            omaelake=puoliso_tyoelake
        elif age<self.min_retirementage:
            wage5y,_=self.infostate_comp_5y_ave_wage(is_spouse=is_spouse) 
            omaelake=(tyoelake+self.elinaikakerroin*(pension+self.acc/self.timestep*wage5y*max(0,self.min_retirementage-age)))*self.elakeindeksi
        else:
            omaelake = tyoelake*self.elakeindeksi + self.scale_pension(pension,age,scale=True,unemp_after_ra=0)
            
        vahennysperuste=732.50*12
        vahennys=0.5*max(0,omaelake-vahennysperuste)
        leskenelake=max(0,add_pension-vahennys)
        
        if self.plotdebug:
            print(f'puolison tyoelake {omaelake} ja leskenelake {leskenelake}')
        
        puoliso_tyoelake += leskenelake
        
        return puoliso_tyoelake

    def move_to_deceiced(self,pension,old_wage,age):
        '''
        Siirtymä tilaan kuollut
        '''
        employment_status = 15 # deceiced
        wage=old_wage
        pension=0
        tyoelake_maksussa=0
        #netto=0
        time_in_state=0
        alkanut_ansiosidonnainen=0
        puoliso=0
        
        if self.mortplot:
            self.plotdebug=True

        return employment_status,pension,wage,time_in_state,puoliso,tyoelake_maksussa

    def move_to_kht(self,wage : float,pension : float,tyoelake : float,old_wage : float,age : float):
        '''
        Siirtymä kotihoidontuelle
        '''
        employment_status = 7 # kotihoidontuelle
        pension=self.pension_accrual(age,old_wage,pension,state=7)
        tyoelake=tyoelake*self.elakeindeksi
        
        time_in_state=self.timestep

        return employment_status,pension,tyoelake,wage,time_in_state

    def move_to_fatherleave(self,wage : float,pension : float,tyoelake : float,old_wage : float,age : float):
        '''
        Siirtymä isyysvapaalle
        '''
        
        #self.infostate_add_child(age) # only for the mother
        employment_status = 6 # isyysvapaa
        time_in_state=0
        pension=self.pension_accrual(age,old_wage,pension,state=6)
        tyoelake=tyoelake*self.elakeindeksi
        time_in_state+=self.timestep        
        pinkslip=0
        
        return employment_status,pension,tyoelake,wage,time_in_state,pinkslip

    def move_to_motherleave(self,wage : float,pension : float,tyoelake : float,old_wage : float,age : float):
        '''
        Siirtymä äitiysvapaalle
        '''
        self.infostate_add_child(age)
        employment_status = 5 # äitiysvapaa
        time_in_state=0
        pension=self.pension_accrual(age,old_wage,pension,state=5)
        tyoelake=tyoelake*self.elakeindeksi
        time_in_state+=self.timestep
        pinkslip=0

        return employment_status,pension,tyoelake,wage,time_in_state,pinkslip

    def stay_unemployed(self,wage,employment_status,kansanelake,tyoelake,pension,time_in_state,toe,toekesto,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,alkanut_ansiosidonnainen,
                        toe58,ove_paid,children_under3,children_under18,has_spouse,is_spouse):
        '''
        Pysy tilassa työtön (0)
        '''
        time_in_state+=self.timestep
        karenssia_jaljella=0
        
        # if the aim is to be employed, there is a definite age-dependent probability that a person is reemployed
        if (action == 1 or action == 3 or (action==2 and children_under3<1 and age < self.min_retirementage) or
            action == 4) and self.unemp_limit_reemp:
            if sattuma[7]>self.unemp_reemp_prob[intage] and self.randomness:
                action = 0
            
        if age>=self.max_unemploymentbenefitage:
            employment_status,kansanelake,tyoelake,pension,wage,time_in_state,wove_paid\
                =self.move_to_retirement(wage,pension,0,age,kansanelake,tyoelake,employment_status,
                    unemp_after_ra,all_acc=True,scale_acc=True,has_spouse=has_spouse,is_spouse=is_spouse)
        elif action == 0 or action == 5:
            employment_status = 0 # unchanged
                
            if self.porrasta_toe:
                oldtoe=self.comp_oldtoe(spouse=spouse_value)
            else:
                oldtoe=0
                
            if action == 5 and (not ove_paid) and (age>=self.min_ove_age):
                pension,tyoelake,ove_paid=self.move_to_ove(employment_status,pension,tyoelake,ove_paid,age,unemp_after_ra)

            kesto=used_unemp_benefit
            tyoelake=tyoelake*self.elakeindeksi
                
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
            employment_status,pension,tyoelake,wage,time_in_state,tyoura,pinkslip=\
                self.move_to_work(wage,pension,tyoelake,old_wage,age,time_in_state,tyoura,pinkslip)
        elif action == 3: # osatyö 50%
            employment_status,pension,tyoelake,wage,time_in_state,tyoura,pinkslip=\
                self.move_to_parttime(wage,pension,tyoelake,old_wage,age,tyoura,time_in_state)
        elif action == 2:
            if age >= self.min_retirementage: # ve
                employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                    self.move_to_retirement(wage,pension,old_wage,age,kansanelake,tyoelake,employment_status,
                        unemp_after_ra,scale_acc=True,has_spouse=has_spouse,is_spouse=is_spouse)
            elif children_under3>0:
                employment_status,pension,tyoelake,wage,time_in_state=self.move_to_kht(wage,pension,tyoelake,old_wage,age)
            else:
                employment_status,pension,tyoelake,wage,time_in_state,tyoura,pinkslip=\
                    self.move_to_parttime(wage,pension,tyoelake,old_wage,age,tyoura,time_in_state)
        elif action == 4: # osatyö 50% + ve
            if age >= self.min_retirementage: # ve
                employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                    self.move_to_oa_parttime(wage,pension,old_wage,age,kansanelake,tyoelake,employment_status,
                            unemp_after_ra,scale_acc=True,has_spouse=has_spouse,is_spouse=is_spouse)
            else:
                employment_status,pension,tyoelake,wage,time_in_state,tyoura,pinkslip=\
                    self.move_to_parttime(wage,pension,tyoelake,old_wage,age,tyoura,time_in_state)
        elif action == 11: # tk
            employment_status,pension,kansanelake,tyoelake,wage,time_in_state,ove_paid=\
                self.move_to_disab(wage,pension,old_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,has_spouse,children_under18,is_spouse)
            pinkslip=0
        else:
            print('error 17')  
            
        return employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
            pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
            alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella

    def stay_tyomarkkinatuki(self,wage,employment_status,kansanelake,tyoelake,pension,time_in_state,toe,toekesto,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,alkanut_ansiosidonnainen,
                        toe58,ove_paid,children_under3,children_under18,has_spouse,is_spouse):
        '''
        Pysy tilassa työmarkkinatuki (13)
        '''
        time_in_state+=self.timestep
        karenssia_jaljella=0
                
        if (action == 1 or action == 3 or (action==2 and children_under3<1 and age < self.min_retirementage) or action == 4) and self.unemp_limit_reemp:
            if sattuma[7]>self.unemp_reemp_prob[intage] and self.randomness:
                action = 0

        if age>=self.max_unemploymentbenefitage:
            employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                self.move_to_retirement(wage,pension,0,age,kansanelake,tyoelake,employment_status,
                    unemp_after_ra,all_acc=True,scale_acc=True,has_spouse=has_spouse,is_spouse=is_spouse)
        elif action == 0 or action == 5:
            employment_status = 13 # unchanged
                
            if action == 5 and (not ove_paid) and (age>=self.min_ove_age):
                pension,tyoelake,ove_paid=self.move_to_ove(employment_status,pension,tyoelake,ove_paid,age,unemp_after_ra)

            tyoelake=tyoelake*self.elakeindeksi
            pension=self.pension_accrual(age,wage,pension,state=13)

            if age >= self.min_retirementage:
                unemp_after_ra+=self.timestep
        
        elif action == 1: # 
            employment_status,pension,tyoelake,wage,time_in_state,tyoura,pinkslip=\
                self.move_to_work(wage,pension,tyoelake,old_wage,age,time_in_state,tyoura,pinkslip)
        elif action == 3: # osatyö 50%
            employment_status,pension,tyoelake,wage,time_in_state,tyoura,pinkslip=\
                self.move_to_parttime(wage,pension,tyoelake,old_wage,age,tyoura,time_in_state)
        elif action == 2:
            if age >= self.min_retirementage: # ve
                employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                    self.move_to_retirement(wage,pension,old_wage,age,kansanelake,tyoelake,employment_status,
                        unemp_after_ra,scale_acc=True,has_spouse=has_spouse,is_spouse=is_spouse)
            elif children_under3>0:
                employment_status,pension,tyoelake,wage,time_in_state=self.move_to_kht(wage,pension,tyoelake,old_wage,age)
            else:
                employment_status,pension,tyoelake,wage,time_in_state,tyoura,pinkslip=\
                    self.move_to_parttime(wage,pension,tyoelake,old_wage,age,tyoura,time_in_state)
        elif action == 4: # osatyö 50% + ve
            if age >= self.min_retirementage: # ve
                employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                    self.move_to_oa_parttime(wage,pension,old_wage,age,kansanelake,tyoelake,employment_status,
                            unemp_after_ra,scale_acc=False,has_spouse=has_spouse,is_spouse=is_spouse)
            else:
                employment_status,pension,tyoelake,wage,time_in_state,tyoura,pinkslip=\
                    self.move_to_parttime(wage,pension,tyoelake,old_wage,age,tyoura,time_in_state)
        elif action==11: # tk
            employment_status,pension,kansanelake,tyoelake,wage,time_in_state,ove_paid=\
                self.move_to_disab(wage,pension,old_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,has_spouse,children_under18,is_spouse)
        else:
            print('error 17')        
                
        return employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
               pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella
                
    def stay_pipeline(self,wage,employment_status,kansanelake,tyoelake,pension,time_in_state,toe,toekesto,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,alkanut_ansiosidonnainen,
                        toe58,ove_paid,children_under3,children_under18,has_spouse,is_spouse):
        '''
        Pysy tilassa työttämyysputki (4)
        '''
        time_in_state+=self.timestep
        karenssia_jaljella=0

        if (action == 1 or action == 3 or (action==2 and age < self.min_retirementage) or action == 4) and self.unemp_limit_reemp:
            if sattuma[7]>self.unemp_reemp_prob[intage] and self.randomness:
                action = 0
        
        if age>=self.max_unemploymentbenefitage:
            employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                self.move_to_retirement(wage,pension,0,age,kansanelake,tyoelake,employment_status,
                    unemp_after_ra,all_acc=True,scale_acc=True,has_spouse=has_spouse,is_spouse=is_spouse)
        elif action == 0 or action == 5:
            employment_status  = 4 # unchanged
            pension=self.pension_accrual(age,unempwage_basis,pension,state=4,ove_paid=ove_paid)
                
            if action == 5 and (not ove_paid) and (age>=self.min_ove_age):
                pension,tyoelake,ove_paid=self.move_to_ove(employment_status,pension,tyoelake,ove_paid,age,unemp_after_ra)
                
            tyoelake=tyoelake*self.elakeindeksi
            used_unemp_benefit+=self.timestep
            if age >= self.min_retirementage:
                unemp_after_ra+=self.timestep
                
        elif action == 1:
            employment_status,pension,tyoelake,wage,time_in_state,tyoura,pinkslip=\
                self.move_to_work(wage,pension,tyoelake,old_wage,age,time_in_state,tyoura,pinkslip)
        elif action == 3:
            employment_status,pension,tyoelake,wage,time_in_state,tyoura,pinkslip=\
                self.move_to_parttime(wage,pension,tyoelake,old_wage,age,tyoura,time_in_state)
        elif action==2:
            if age >= self.min_retirementage: # ve
                employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                    self.move_to_retirement(wage,pension,old_wage,age,kansanelake,tyoelake,employment_status,
                        unemp_after_ra,all_acc=True,scale_acc=True,has_spouse=has_spouse,is_spouse=is_spouse)
                pinkslip=0
            else:
                employment_status,pension,tyoelake,wage,time_in_state,tyoura,pinkslip=\
                    self.move_to_parttime(wage,pension,tyoelake,old_wage,age,time_in_state,tyoura,pinkslip)
        elif action == 4: # osatyö 50% + ve
            if age >= self.min_retirementage: # ve
                employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                    self.move_to_oa_parttime(wage,pension,old_wage,age,kansanelake,tyoelake,employment_status,
                        unemp_after_ra,scale_acc=True,has_spouse=has_spouse,is_spouse=is_spouse)
            else:
                employment_status,pension,tyoelake,wage,time_in_state,tyoura,pinkslip=\
                    self.move_to_parttime(wage,pension,tyoelake,old_wage,age,tyoura,time_in_state)
        elif action==11: # tk
            employment_status,pension,kansanelake,tyoelake,wage,time_in_state,ove_paid=\
                self.move_to_disab(wage,pension,old_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,has_spouse,children_under18,is_spouse)
        else:
            print('error 1: ',action)
            
        return employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
               pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella
        
    def stay_employed(self,wage,employment_status,kansanelake,tyoelake,pension,time_in_state,toe,toekesto,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,alkanut_ansiosidonnainen,
                        toe58,ove_paid,children_under3,children_under18,has_spouse,is_spouse):
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
            
        if action == 3 or action == 4:
            if sattuma[7]>self.parttime_fullemp_prob and self.randomness:
                action = 0

        if action == 0 or action == 5:
            employment_status = 1 # unchanged
            
            if action == 5 and (not ove_paid) and (age>self.min_ove_age):
                pension,tyoelake,ove_paid=self.move_to_ove(employment_status,pension,tyoelake,ove_paid,age,unemp_after_ra)
                            
            tyoelake=tyoelake*self.elakeindeksi
            tyoura+=self.timestep
            pension=self.pension_accrual(age,wage,pension,state=1)
        elif action == 1: # työttömäksi
            employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
                used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,alkanut_ansiosidonnainen,karenssia_jaljella,ove_paid=\
                self.move_to_unemp(wage,pension,old_wage,age,kansanelake,tyoelake,toe,toekesto,pinkslip,tyoura,
                    used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,alkanut_ansiosidonnainen,toe58,ove_paid,has_spouse,is_spouse)
        elif action == 3: # osatyö 50%
            employment_status,pension,tyoelake,wage,time_in_state,tyoura,pinkslip=\
                self.move_to_parttime(wage,pension,tyoelake,old_wage,age,tyoura,0)
        elif action==2:
            if age >= self.min_retirementage:
                employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                    self.move_to_retirement(wage,pension,old_wage,age,kansanelake,tyoelake,
                        employment_status,unemp_after_ra,has_spouse=has_spouse,is_spouse=is_spouse,all_acc=True,scale_acc=True) 
            else:
                employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
                    used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,alkanut_ansiosidonnainen,karenssia_jaljella,ove_paid=\
                    self.move_to_unemp(wage,pension,old_wage,age,kansanelake,tyoelake,toe,toekesto,pinkslip,tyoura,
                        used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,alkanut_ansiosidonnainen,ove_paid,toe58,has_spouse,is_spouse)
        elif action == 4: # osatyö 50% + ve
            if age >= self.min_retirementage:
                employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                    self.move_to_oa_parttime(wage,pension,old_wage,age,kansanelake,tyoelake,employment_status,
                        unemp_after_ra,scale_acc=True,has_spouse=has_spouse,is_spouse=is_spouse)
            else:
                employment_status,pension,tyoelake,wage,time_in_state,tyoura,pinkslip=\
                    self.move_to_parttime(wage,pension,tyoelake,old_wage,age,tyoura,unemp_after_ra,has_spouse=has_spouse,is_spouse=is_spouse)
        elif action == 11: # tk
            employment_status,pension,kansanelake,tyoelake,wage,time_in_state,ove_paid=\
                self.move_to_disab(wage,pension,old_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,has_spouse,children_under18,is_spouse)
        else:
            print('error 12')    
            
        return employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
              pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
              alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella
           
    def stay_disabled(self,wage,employment_status,kansanelake,tyoelake,pension,time_in_state,toe,toekesto,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,alkanut_ansiosidonnainen,
                        toe58,ove_paid,children_under3,children_under18,has_spouse,is_spouse):
            
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
        
        if math.isclose(time_in_state,5.0) and age<55.0:
            # kertakorotus
            if age<31:
                korotus=1.25
            else:
                korotus=1.25-0.01*max(0,age-31.0)
            tyoelake=tyoelake*korotus
        
        if age>=self.max_retirementage:
            tyoelake = tyoelake+self.scale_pension(pension,age,scale=False)/self.elakeindeksi
            pension=0           
        else:
            pension=pension*self.palkkakerroin
            
        wage=0
        kansanelake = kansanelake*self.kelaindeksi

        return employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
               pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella

    def stay_retired(self,wage,employment_status,kansanelake,tyoelake,pension,time_in_state,toe,toekesto,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,alkanut_ansiosidonnainen,
                        toe58,ove_paid,children_under3,children_under18,has_spouse,is_spouse):
        '''
        Pysy tilassa vanhuuseläke (2)
        '''
        karenssia_jaljella=0
        if age >= self.min_retirementage: # ve
            time_in_state += self.timestep
            
            if (action in set([1,2,3,4,5])) and self.unemp_limit_reemp:
                #if sattuma[7]>self.unemp_reemp_prob[intage] and self.randomness:
                if sattuma[7]>self.oa_reemp_prob and self.randomness:
                    action = 0

            if age>=self.max_retirementage:
                employment_status = 2 # unchanged
                tyoelake = tyoelake*self.elakeindeksi+self.scale_pension(pension,age,scale=False,unemp_after_ra=unemp_after_ra)
                pension=0           
                kansanelake = kansanelake * self.kelaindeksi
            elif action == 0:
                employment_status = 2 # unchanged
                tyoelake = self.elakeindeksi*tyoelake
                kansanelake = kansanelake * self.kelaindeksi
                pension=pension*self.palkkakerroin 
            elif action == 2 or action == 1 or action == 4:
                employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                    self.move_to_oa_parttime(wage,pension,wage,age,kansanelake,tyoelake,employment_status,0,
                        scale_acc=False,has_spouse=has_spouse,is_spouse=is_spouse)
            elif action == 3 or action == 5:
                employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                    self.move_to_oa_fulltime(wage,pension,wage,age,kansanelake,tyoelake,employment_status,0,
                        scale_acc=False,has_spouse=has_spouse,is_spouse=is_spouse)
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
                employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
                    used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,alkanut_ansiosidonnainen,karenssia_jaljella,ove_paid=\
                    self.move_to_unemp(wage,pension,old_wage,age,kansanelake,tyoelake,toe,toekesto,0,tyoura,
                        used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,alkanut_ansiosidonnainen,toe58,ove_paid,has_spouse,is_spouse)
            elif action == 2: # täihin
                employment_status,pension,tyoelake,wage,time_in_state,tyoura,pinkslip=\
                    self.move_to_work(wage,pension,tyoelake,wage,age,time_in_state,tyoura,pinkslip)
            elif action == 3: # osatyö 50%
                employment_status,pension,tyoelake,wage,time_in_state,tyoura,pinkslip=\
                    self.move_to_parttime(wage,pension,tyoelake,wage,age,tyoura,time_in_state)
            elif action == 11: # tk
                employment_status,pension,kansanelake,tyoelake,wage,time_in_state,ove_paid=\
                    self.move_to_disab(wage,pension,old_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,has_spouse,children_under18,is_spouse)
            else:
                print('error 12')
                
        return employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
               pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella

    def stay_motherleave(self,wage,employment_status,kansanelake,tyoelake,pension,time_in_state,toe,toekesto,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,alkanut_ansiosidonnainen,
                        toe58,ove_paid,children_under3,children_under18,has_spouse,is_spouse):
        '''
        Pysy tilassa äitiysvapaa (5)
        '''
        #exit_prb=np.random.uniform(0,80_000)
        karenssia_jaljella=0
        if time_in_state>=self.aitiysvapaa_kesto or sattuma[5]<self.aitiysvapaa_pois:
            if time_in_state>=self.aitiysvapaa_kesto and sattuma[7]<self.nainen_jatkaa_kotihoidontuelle:
                action = 3
                
            pinkslip=0
            if action == 0:
                employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
                    used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,alkanut_ansiosidonnainen,karenssia_jaljella,ove_paid=\
                    self.move_to_unemp(wage,pension,old_wage,age,kansanelake,tyoelake,toe,toekesto,pinkslip,tyoura,
                        used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,alkanut_ansiosidonnainen,toe58,ove_paid,has_spouse,is_spouse)
            elif action == 1: # 
                employment_status,pension,tyoelake,wage,time_in_state,tyoura,pinkslip=\
                    self.move_to_work(wage,pension,tyoelake,old_wage,age,time_in_state,tyoura,pinkslip)
            elif action == 3 or action == 2: # 
                employment_status,pension,tyoelake,wage,time_in_state=self.move_to_kht(wage,pension,tyoelake,old_wage,age)
            elif action == 4 or action == 5:
                employment_status,pension,tyoelake,wage,time_in_state,tyoura,pinkslip=\
                    self.move_to_parttime(wage,pension,tyoelake,old_wage,age,tyoura,time_in_state)
            elif action==11: # tk
                employment_status,pension,kansanelake,tyoelake,wage,time_in_state,ove_paid=\
                    self.move_to_disab(wage,pension,old_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,has_spouse,children_under18,is_spouse)
            else:
                print('Error 21')
        else:
            pension=self.pension_accrual(age,old_wage,pension,state=5)
            tyoelake=tyoelake*self.elakeindeksi
            time_in_state+=self.timestep
                
        return employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
               pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella

    def stay_fatherleave(self,wage,employment_status,kansanelake,tyoelake,pension,time_in_state,toe,toekesto,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,alkanut_ansiosidonnainen,
                        toe58,ove_paid,children_under3,children_under18,has_spouse,is_spouse):
        '''
        Pysy tilassa isyysvapaa (6)
        '''
        karenssia_jaljella=0
        if time_in_state>=self.isyysvapaa_kesto:
            if sattuma[7]<self.mies_jatkaa_kotihoidontuelle:
                action = 3
        
            pinkslip=0
            if action == 0:
                employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
                    used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,alkanut_ansiosidonnainen,karenssia_jaljella,ove_paid=\
                    self.move_to_unemp(wage,pension,old_wage,age,kansanelake,tyoelake,toe,toekesto,pinkslip,tyoura,
                        used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,alkanut_ansiosidonnainen,toe58,ove_paid,has_spouse,is_spouse)
            elif action == 1: # 
                # ei vaikutusta palkkaan
                employment_status,pension,tyoelake,wage,time_in_state,tyoura,pinkslip=\
                    self.move_to_work(wage,pension,tyoelake,old_wage,age,0,tyoura,pinkslip)
            elif action == 3 or action == 2: # 
                employment_status,pension,tyoelake,wage,time_in_state=self.move_to_kht(wage,pension,tyoelake,old_wage,age)
            elif action == 4 or action == 5:
                employment_status,pension,tyoelake,wage,time_in_state,tyoura,pinkslip=\
                    self.move_to_parttime(wage,pension,tyoelake,old_wage,age,tyoura,0)
            elif action==11: # tk
                employment_status,pension,kansanelake,tyoelake,wage,time_in_state,ove_paid=\
                    self.move_to_disab(wage,pension,old_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,has_spouse,children_under18,is_spouse)
            else:
                print('Error 23')
        else:
            pension=self.pension_accrual(age,old_wage,pension,state=6)
            tyoelake=tyoelake*self.elakeindeksi
            time_in_state+=self.timestep

        return employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
               pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella

    def stay_kht(self,wage,employment_status,kansanelake,tyoelake,pension,time_in_state,toe,toekesto,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,alkanut_ansiosidonnainen,
                        toe58,ove_paid,children_under3,children_under18,has_spouse,is_spouse):
        '''
        Pysy tilassa kotihoidontuki (0)
        '''
        karenssia_jaljella=0

        if (action == 0) and (time_in_state>self.kht_kesto or children_under3<1): # jos etuus loppuu, siirtymä satunnaisesti
            if self.randomness:
                s=sattuma[5] #np.random.uniform()
            else:
                s=0
                
            if s<1/3:
                action=1
            elif s<2/3:
                action=2
            else:
                action=3
                
        if (action == 2 or action == 5 or action == 3) and self.unemp_limit_reemp:
            if sattuma[7]>self.unemp_reemp_prob[intage] and self.randomness:
                action = 1

        if age >= self.min_retirementage: # ve
            employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                self.move_to_retirement(wage,pension,old_wage,age,kansanelake,tyoelake,employment_status,
                    unemp_after_ra,all_acc=True,scale_acc=True,has_spouse=has_spouse,is_spouse=is_spouse)
        elif (action == 0) and ((time_in_state<=self.kht_kesto and children_under3>0) or self.perustulo): # jos perustulo, ei aikarajoitetta
            employment_status  = 7 # stay
            time_in_state+=self.timestep
            pension=self.pension_accrual(age,wage,pension,state=7)
            tyoelake=tyoelake*self.elakeindeksi
        elif action == 1 or action == 4: # 
            pinkslip=0
            employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
                used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,alkanut_ansiosidonnainen,karenssia_jaljella,ove_paid=\
                self.move_to_unemp(wage,pension,old_wage,age,kansanelake,tyoelake,toe,toekesto,pinkslip,tyoura,
                    used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,alkanut_ansiosidonnainen,toe58,ove_paid,has_spouse,is_spouse)
        elif action == 2 or action == 5: # 
            employment_status,pension,tyoelake,wage,time_in_state,tyoura,pinkslip=\
                self.move_to_work(wage,pension,tyoelake,wage,age,time_in_state,tyoura,pinkslip)
        elif action == 3: # 
            employment_status,pension,tyoelake,wage,time_in_state,tyoura,pinkslip=\
                self.move_to_parttime(wage,pension,tyoelake,wage,age,tyoura,time_in_state)
        elif action==11: # tk
            employment_status,pension,kansanelake,tyoelake,wage,time_in_state,ove_paid=\
                self.move_to_disab(wage,pension,old_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,has_spouse,children_under18,is_spouse)
        else:
            print('Error 25')
            
        return employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
               pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella

    def stay_student(self,wage,employment_status,kansanelake,tyoelake,pension,time_in_state,toe,toekesto,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,alkanut_ansiosidonnainen,
                        toe58,ove_paid,children_under3,children_under18,has_spouse,is_spouse):
        '''
        Pysy tilassa opiskelija (12)
        '''

        pinkslip=0
        karenssia_jaljella=0
        
        if age>=self.min_retirementage:
            employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                self.move_to_retirement(wage,pension,old_wage,age,kansanelake,tyoelake,employment_status,
                    unemp_after_ra,all_acc=True,scale_acc=True,has_spouse=has_spouse,is_spouse=is_spouse)
            unempwage_basis,alkanut_ansiosidonnainen=0,0

        else:
            if (action == 0 or action == 1 or action == 3 or action == 4 or action == 5) and self.unemp_limit_reemp: # and intage<40:
                if sattuma[7]>self.unemp_reemp_prob[intage] and self.randomness:
                    action = 2
        
            if sattuma[5]>=self.student_outrate[intage,g]:
                employment_status = 12 # unchanged
                time_in_state+=self.timestep
                pension=self.pension_accrual(age,0,pension,state=12)
                tyoelake=tyoelake*self.elakeindeksi
            elif action == 0: # 
                employment_status,pension,tyoelake,wage,time_in_state,tyoura,pinkslip=\
                    self.move_to_work(wage,pension,tyoelake,wage,age,0,tyoura,pinkslip)
            elif action == 1: # 
                if children_under3>0:
                    employment_status,pension,tyoelake,wage,time_in_state=self.move_to_kht(wage,pension,tyoelake,old_wage,age)
                else:
                    employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
                        used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,alkanut_ansiosidonnainen,karenssia_jaljella,ove_paid=\
                        self.move_to_unemp(wage,pension,old_wage,age,kansanelake,tyoelake,toe,toekesto,pinkslip,tyoura,
                            used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,alkanut_ansiosidonnainen,toe58,ove_paid,has_spouse,is_spouse)
            elif action == 2:
                employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
                    used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,alkanut_ansiosidonnainen,karenssia_jaljella,ove_paid=\
                    self.move_to_unemp(wage,pension,old_wage,age,kansanelake,tyoelake,toe,toekesto,pinkslip,tyoura,
                        used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,alkanut_ansiosidonnainen,toe58,ove_paid,has_spouse,is_spouse)
            elif action == 3 or action == 4 or action == 5:
                employment_status,pension,tyoelake,wage,time_in_state,tyoura,pinkslip=\
                    self.move_to_parttime(wage,pension,tyoelake,wage,age,tyoura,time_in_state)
            elif action == 11: # tk
                employment_status,pension,kansanelake,tyoelake,wage,time_in_state,ove_paid=\
                    self.move_to_disab(wage,pension,old_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,has_spouse,children_under18,is_spouse)
            else:
                print('error 29: ',action)
            
        return employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
               pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella

    def stay_oa_parttime(self,wage,employment_status,kansanelake,tyoelake,pension,time_in_state,toe,toekesto,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,alkanut_ansiosidonnainen,
                        toe58,ove_paid,children_under3,children_under18,has_spouse,is_spouse):
        '''
        Pysy tilassa ve+(osa-aikatyö) (0)
        '''

        karenssia_jaljella=0
        
        # irtisanominen
        if sattuma[1]<self.pinkslip_intensity[g]:
            action=4 # ve:lle
            pinkslip=1

        if age>=self.max_retirementage:
            employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                self.move_to_retirement(wage,pension,0,age,kansanelake,tyoelake,employment_status,
                    unemp_after_ra,all_acc=True,scale_acc=False,has_spouse=has_spouse,is_spouse=is_spouse)
        elif action == 0: # jatkaa osa-aikatöissä, ei voi saada työttämyyspäivärahaa
            employment_status = 8 # unchanged
            time_in_state+=self.timestep
            parttimewage = 0.5*wage
            pension = self.pension_accrual(age,parttimewage,pension,state=employment_status)
            tyoelake = self.elakeindeksi * tyoelake
            kansanelake = kansanelake * self.kelaindeksi
        elif action==2 or action==3: # jatkaa täysin töissä, ei voi saada työttämyyspäivärahaa
            employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                self.move_to_oa_fulltime(wage,pension,wage,age,kansanelake,tyoelake,employment_status,
                        0,scale_acc=False,has_spouse=has_spouse,is_spouse=is_spouse)
        elif action == 4 or action == 5 or action == 1: # eläkkeelle, eläkeaikana karttunutta eläkettä ei vielä maksuun
            employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                self.move_to_retirement(wage,pension,wage,age,kansanelake,tyoelake,employment_status,
                    0,all_acc=False,scale_acc=False,has_spouse=has_spouse,is_spouse=is_spouse)
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
                        toe58,ove_paid,children_under3,children_under18,has_spouse,is_spouse):
        '''
        Pysy tilassa ve+työ (9)
        '''

        karenssia_jaljella=0
        
        # irtisanominen
        if sattuma[1]<self.pinkslip_intensity[g]:
            action=4 # ve:lle
            pinkslip=1

        if age>=self.max_retirementage:
            employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                self.move_to_retirement(wage,pension,old_wage,age,kansanelake,tyoelake,employment_status,
                    unemp_after_ra,all_acc=True,scale_acc=False,has_spouse=has_spouse,is_spouse=is_spouse)
        elif action == 0: # jatkaa töissä, ei voi saada työttämyyspäivärahaa
            employment_status = 9 # unchanged
            time_in_state+=self.timestep        
            pension=self.pension_accrual(age,wage,pension,state=employment_status)
            tyoelake = tyoelake * self.elakeindeksi
            kansanelake = kansanelake * self.kelaindeksi
        elif action == 2 or action == 5 or action == 1: # jatkaa osa-aikatöissä, ei voi saada työttämyyspäivärahaa
            employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                self.move_to_oa_parttime(wage,pension,old_wage,age,kansanelake,tyoelake,employment_status,0,
                    scale_acc=False,has_spouse=has_spouse,is_spouse=is_spouse)
        elif action==3 or action == 4: # eläkkeelle, eläkeaikana karttunutta eläkettä ei vielä maksuun
            employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                self.move_to_retirement(wage,pension,wage,age,kansanelake,tyoelake,employment_status,0,
                    all_acc=False,scale_acc=False,has_spouse=has_spouse,is_spouse=is_spouse)
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
                        toe58,ove_paid,children_under3,children_under18,has_spouse,is_spouse):
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
                pinkslip=1
        else:
            pinkslip=0

        if (action == 3) or (action == 4):
            if sattuma[7]>self.parttime_fullemp_prob and self.randomness:
                action = 0

        if action == 0 or action == 5:
            employment_status = 10 # unchanged
            parttimewage=0.5*wage
            tyoura+=self.timestep
            
            if action == 5 and (not ove_paid) and (age>=self.min_ove_age):
                pension,tyoelake,ove_paid=self.move_to_ove(employment_status,pension,tyoelake,ove_paid,age,unemp_after_ra)
            
            tyoelake=tyoelake*self.elakeindeksi
            pension=self.pension_accrual(age,parttimewage,pension,state=10)
        elif action == 1: # työttömäksi
            employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
                used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,alkanut_ansiosidonnainen,karenssia_jaljella,ove_paid=\
                self.move_to_unemp(wage,pension,old_wage,age,kansanelake,tyoelake,toe,toekesto,pinkslip,tyoura,
                    used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,alkanut_ansiosidonnainen,toe58,ove_paid,has_spouse,is_spouse)
        elif action==3:
            employment_status,pension,tyoelake,wage,time_in_state,tyoura,pinkslip=\
                self.move_to_work(wage,pension,tyoelake,old_wage,age,0,tyoura,pinkslip)
        elif action==2:
            if age >= self.min_retirementage:
                employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                    self.move_to_retirement(wage,pension,old_wage,age,kansanelake,tyoelake,employment_status,
                        unemp_after_ra,all_acc=True,scale_acc=True,has_spouse=has_spouse,is_spouse=is_spouse)
            elif children_under3>0:
                employment_status,pension,tyoelake,wage,time_in_state=self.move_to_kht(wage,pension,tyoelake,old_wage,age)
            else:
                employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
                    used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,alkanut_ansiosidonnainen,karenssia_jaljella,ove_paid=\
                    self.move_to_unemp(wage,pension,old_wage,age,kansanelake,tyoelake,toe,toekesto,pinkslip,tyoura,
                        used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,alkanut_ansiosidonnainen,toe58,ove_paid,has_spouse,is_spouse)
        elif action==4: # move to oa_work
            if age >= self.min_retirementage:
                employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                    self.move_to_oa_parttime(wage,pension,wage,age,kansanelake,tyoelake,employment_status,
                        unemp_after_ra,scale_acc=True,has_spouse=has_spouse,is_spouse=is_spouse)
            else:
                employment_status,pension,tyoelake,wage,time_in_state,tyoura,pinkslip=\
                    self.move_to_work(wage,pension,tyoelake,old_wage,age,0,tyoura,pinkslip)
        elif action==11: # tk
            employment_status,pension,kansanelake,tyoelake,wage,time_in_state,ove_paid=\
                self.move_to_disab(wage,pension,old_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,has_spouse,children_under18,is_spouse)
        else:
            print('error 12')
            
        return employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
               pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella

    def stay_outsider(self,wage,employment_status,kansanelake,tyoelake,pension,time_in_state,toe,toekesto,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,alkanut_ansiosidonnainen,
                        toe58,ove_paid,children_under3,children_under18,has_spouse,is_spouse):
        '''
        Pysy tilassa työvoiman ulkopuolella (11)
        '''
        karenssia_jaljella=0

        if age>=self.min_retirementage:
            employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                self.move_to_retirement(wage,pension,old_wage,age,kansanelake,tyoelake,employment_status,
                    unemp_after_ra,all_acc=True,scale_acc=True,has_spouse=has_spouse,is_spouse=is_spouse)
        elif sattuma[5]>=self.outsider_outrate[intage,g]:
            time_in_state+=self.timestep
            employment_status = 11 # unchanged
            pension = pension * self.palkkakerroin
            tyoelake = tyoelake * self.elakeindeksi
        else:
            if (action in set ([0,1,4,5])) and self.unemp_limit_reemp:
                if sattuma[7]>self.unemp_reemp_prob[intage] and self.randomness:
                    action = 2
        
            if action == 0 or action == 1: # 
                employment_status,pension,tyoelake,wage,time_in_state,tyoura,pinkslip=\
                    self.move_to_work(wage,pension,tyoelake,wage,age,time_in_state,tyoura,pinkslip)
            elif action == 3:
                if children_under3>0:
                    employment_status,pension,tyoelake,wage,time_in_state=self.move_to_kht(wage,pension,tyoelake,old_wage,age)
                else:
                    pinkslip=0
                    employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
                        used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,alkanut_ansiosidonnainen,karenssia_jaljella,ove_paid=\
                        self.move_to_unemp(wage,pension,old_wage,age,kansanelake,tyoelake,toe,toekesto,pinkslip,tyoura,
                            used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,alkanut_ansiosidonnainen,toe58,ove_paid,has_spouse,is_spouse)
            elif action == 2: # 
                pinkslip=0
                employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
                    used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,alkanut_ansiosidonnainen,karenssia_jaljella,ove_paid=\
                    self.move_to_unemp(wage,pension,old_wage,age,kansanelake,tyoelake,toe,toekesto,pinkslip,tyoura,
                        used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,alkanut_ansiosidonnainen,toe58,ove_paid,has_spouse,is_spouse)
            elif action == 4 or action == 5: # 
                employment_status,pension,tyoelake,wage,time_in_state,tyoura,pinkslip=\
                    self.move_to_parttime(wage,pension,tyoelake,wage,age,tyoura,time_in_state)
            elif action == 11: # tk
                employment_status,pension,kansanelake,tyoelake,wage,time_in_state,ove_paid=\
                    self.move_to_disab(wage,pension,old_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,has_spouse,children_under18,is_spouse)
                pinkslip=0
            else:
                print('error 19: ',action)

        return employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
               pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella
               
    def stay_svpaivaraha(self,wage,employment_status,kansanelake,tyoelake,pension,time_in_state,toe,toekesto,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,alkanut_ansiosidonnainen,
                        toe58,ove_paid,children_under3,children_under18,has_spouse,is_spouse):
        '''
        Pysy sairauspäivärahalla (14)
        '''
        karenssia_jaljella=0

        if age>=self.max_unemploymentbenefitage:
                employment_status,pension,kansanelake,tyoelake,wage,time_in_state,ove_paid=\
                    self.move_to_disab_state(wage,pension,old_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,has_spouse,children_under18,is_spouse)
                pinkslip=0
        elif time_in_state<1.0:
            if age>=self.min_retirementage and action==1:
                employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                    self.move_to_retirement(wage,pension,old_wage,age,kansanelake,tyoelake,employment_status,
                        unemp_after_ra,all_acc=True,scale_acc=True,has_spouse=has_spouse,is_spouse=is_spouse)
            elif action==11:
                employment_status,pension,kansanelake,tyoelake,wage,time_in_state,ove_paid=\
                    self.move_to_disab_state(wage,pension,old_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,has_spouse,children_under18,is_spouse)
                pinkslip=0
            else:
                time_in_state+=self.timestep
                employment_status = 14 # unchanged
                tyoelake = tyoelake * self.elakeindeksi
                pension=self.pension_accrual(age,old_wage,pension,state=14)
        else:
            if sattuma[5]<self.svpaivaraha_disabilityrate[intage,g] or action==11:
                employment_status,pension,kansanelake,tyoelake,wage,time_in_state,ove_paid=\
                    self.move_to_disab_state(wage,pension,old_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,has_spouse,children_under18,is_spouse)
                pinkslip=0
            else:
                if (action in set ([0,1,4,5])) and self.unemp_limit_reemp:
                    if sattuma[7]>self.unemp_reemp_prob[intage] and self.randomness:
                        action = 2
        
                if action == 0 or action == 1: # 
                    employment_status,pension,tyoelake,wage,time_in_state,tyoura,pinkslip=\
                        self.move_to_work(wage,pension,tyoelake,wage,age,time_in_state,tyoura,pinkslip)
                elif action == 3:
                    if children_under3>0:
                        employment_status,pension,tyoelake,wage,time_in_state=self.move_to_kht(wage,pension,tyoelake,old_wage,age)
                    else:
                        pinkslip=0
                        employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
                            used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,alkanut_ansiosidonnainen,karenssia_jaljella,ove_paid=\
                            self.move_to_unemp(wage,pension,old_wage,age,kansanelake,tyoelake,toe,toekesto,pinkslip,tyoura,
                                used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,alkanut_ansiosidonnainen,toe58,ove_paid,has_spouse,is_spouse)
                elif action == 2: # 
                    pinkslip=0
                    employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
                        used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,alkanut_ansiosidonnainen,karenssia_jaljella,ove_paid=\
                        self.move_to_unemp(wage,pension,old_wage,age,kansanelake,tyoelake,toe,toekesto,pinkslip,tyoura,
                            used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,alkanut_ansiosidonnainen,toe58,ove_paid,has_spouse,is_spouse)
                elif action == 4:
                    if age>=self.min_retirementage:
                        employment_status,kansanelake,tyoelake,pension,wage,time_in_state,ove_paid=\
                            self.move_to_retirement(wage,pension,old_wage,age,kansanelake,tyoelake,employment_status,
                                unemp_after_ra,all_acc=True,scale_acc=True,has_spouse=has_spouse,is_spouse=is_spouse)
                    else:
                        pinkslip=0
                        employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
                            used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,alkanut_ansiosidonnainen,karenssia_jaljella,ove_paid=\
                            self.move_to_unemp(wage,pension,old_wage,age,kansanelake,tyoelake,toe,toekesto,pinkslip,tyoura,
                                used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,alkanut_ansiosidonnainen,toe58,ove_paid,has_spouse,is_spouse)
                elif action == 5: # 
                    employment_status,pension,tyoelake,wage,time_in_state,tyoura,pinkslip=\
                        self.move_to_parttime(wage,pension,tyoelake,wage,age,tyoura,time_in_state)
                else:
                    print('error 19: ',action)

        return employment_status,kansanelake,tyoelake,pension,wage,time_in_state,\
               pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella
               
    def get_benefits(self,empstate,wage,kansanelake,tyoelake,pension,time_in_state,pinkslip,old_wage,unempwage,
                        unempwage_basis,karenssia_jaljella,age,children_under3,children_under7,children_under18,ove_paid,used_unemp_benefit,
                        puoliso,puoliso_tila,puoliso_palkka,puoliso_kansanelake,puoliso_tyoelake,puoliso_old_wage,
                        puoliso_pinkslip,puoliso_karenssia_jaljella,puoliso_time_in_state,puoliso_unempwage,puoliso_unempwage_basis,puoliso_used_unemp_benefit,
                        g,p_g):
        '''
        This could be handled better
        '''
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
        elif empstate==14:
            wage=0
            old_wage=old_wage
        elif empstate==15:
            wage=0
            old_wage=0
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
        elif puoliso_tila==14:
            puoliso_palkka=0
            puoliso_old_wage=puoliso_old_wage
        elif puoliso_tila==15:
            puoliso_palkka=0
            puoliso_old_wage=0
        else:
            print('unknown state',puoliso_tila)
            
        paid_pension=kansanelake+tyoelake
        puoliso_paid_pension=puoliso_kansanelake+puoliso_tyoelake
            
        netto,benq,netto_omat,netto_puoliso=self.comp_benefits(wage,old_wage,kansanelake,tyoelake,empstate,tis,
                                children_under3,children_under7,children_under18,age,
                                puoliso,puoliso_tila,puoliso_palkka,puoliso_kansanelake,puoliso_tyoelake,puoliso_old_wage,puoliso_time_in_state,
                                used_unemp_benefit,puoliso_used_unemp_benefit,
                                g,p_g,
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
        elif state == 14:
            if age<self.max_retirementage: 
                pension=pension*self.palkkakerroin+acc*wage*self.acc_sv
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
        else: # 2,3,11,12,14,15 # ei karttumaa
            pension=pension*self.palkkakerroin # vastainen eläke, ei alkanut, ei karttumaa
            
        return pension

    def update_wage_reduction_baseline(self,state,wage_reduction,pinkslip,time_in_state,initial_reduction=False):
        '''
        Pidetään kirjaa siitä, kuinka paljon palkkaa alennetaan työttämyyden keston suhteen,
        ja miten siitä palaudutaan
        '''
        if state in set([1,10]): # töissä
            wage_reduction=max(0,wage_reduction-self.salary_const_up)
        if state in set([8,9]): # ve+töissä, reduction ei parane enää, vaan jää eläkkeellejääntihetken tasoon
            wage_reduction=max(0,wage_reduction) #-self.salary_const_up)
        elif state==12: # opiskelee
            wage_reduction=max(0,wage_reduction-self.salary_const_student)
        elif state in set([0,4,13,11]): # työtön tai työelämän ulkopuolella
            wage_reduction=min(1.0,wage_reduction+self.salary_const)
        elif state in set([5,6]): # äitiys-, isyys- tai vanhempainvapaa
            #wage_reduction+=self.salary_const
            wage_reduction=min(1.0,wage_reduction+self.salary_const)
        elif state in set([3]):
            wage_reduction=0.60 # vastaa määritelmää
        elif state in set([7]): # kotihoidontuki tai ve tai tk
            wage_reduction=min(1.0,wage_reduction+self.salary_const)
        elif state in set([2]): # kotihoidontuki tai ve tai tk
            wage_reduction=min(1.0,wage_reduction+self.salary_const_retirement)
        elif state in set([14]): # sairaspäiväraha
            wage_reduction=min(1.0,wage_reduction+self.salary_const_svpaiva)
        elif state in set([15]): # ei muutosta
            wage_reduction=wage_reduction
        else: # ylivuoto, ei tiloja
            wage_reduction=wage_reduction
        
        return wage_reduction

    def update_wage_reduction_sigma(self,state,wage_reduction,pinkslip,time_in_state,initial_reduction=False):
        '''
        Pidetään kirjaa siitä, kuinka paljon palkkaa alennetaan työttämyyden keston suhteen,
        ja miten siitä palaudutaan
        
        Tämä malli ei mene koskaan nollaan.
        
        Irtisanottuja työttömiä ei reduktio koske ensimmäisenä vuonna
        '''
        
        min_reduction=-0.05
        
        if initial_reduction:
            wage_reduction=max(min_reduction,1.0-(1.0-self.wage_initial_reduction)*(1.0-wage_reduction))
        
        if state in set([1]): # töissä
            wage_reduction=max(min_reduction,wage_reduction-self.salary_const_up)
        elif state in set([10]): # osatöissä
            wage_reduction=max(min_reduction,wage_reduction-self.salary_const_up_osaaika)
        elif state in set([8,9]): # ve+töissä, reduction ei parane enää, vaan jää eläkkeellejääntihetken tasoon
            wage_reduction=max(min_reduction,wage_reduction) #-self.salary_const_up)
        elif state==12: # opiskelee
            wage_reduction=max(min_reduction,wage_reduction-self.salary_const_student)
        elif state in set([11]): # työelämän ulkopuolella
            wage_reduction=max(min_reduction,1.0-(1.0-self.salary_const)*(1.0-wage_reduction))
        elif state in set([0,4,13]): # työtön
            if pinkslip<1 or (pinkslip>0 and time_in_state>0.49): # time_in_state ei ole ihan oikein tässä
                wage_reduction=max(min_reduction,1.0-(1.0-self.salary_const)*(1.0-wage_reduction))
        elif state in set([5,6]): # isyys-, äitiys- tai vanhempainvapaa
            wage_reduction=max(min_reduction,1.0-(1.0-self.salary_const)*(1.0-wage_reduction))
        elif state in set([3]):
            wage_reduction=0.60 # vastaa määritelmää
        elif state in set([7]): # kotihoidontuki 
            wage_reduction=max(min_reduction,1.0-(1.0-self.salary_const)*(1.0-wage_reduction))
        elif state in set([2]): # ve
            wage_reduction=max(min_reduction,1.0-(1.0-self.salary_const_retirement)*(1.0-wage_reduction))
        elif state in set([14]): # sairaspäiväraha
            wage_reduction=max(min_reduction,1.0-(1.0-self.salary_const_svpaiva)*(1.0-wage_reduction))
        elif state in set([15]): # ei muutosta
            wage_reduction=wage_reduction
        else: # ylivuoto, ei tiloja
            wage_reduction=wage_reduction
        
        return wage_reduction
        
#     def get_family_wage(self,age,g):
#         if g>2: # puoliso mies (yksinkertaistus)
#             #palkka=self.palkat_ika_miehet[self.map_age(age)]*self.g_r[self.map_age(age),g]
#             palkka=self.palkat_ika_miehet[self.map_age(age)]*self.g_r[self.map_age(age),g-3]
#         else: # puoliso nainen (yksinkertaistus)
#             #palkka=self.palkat_ika_naiset[self.map_age(age)]*self.g_r[self.map_age(age),g]
#             palkka=self.palkat_ika_naiset[self.map_age(age)]*self.g_r[self.map_age(age),g+3]
#             
#         return palkka
#         
    def update_family(self,puoliso,age,employment_status,puoliso_tila,sattuma):
        '''
        Päivitä puolison/potentiaalisen puolison tila & palkka
        Päivitä avioliitto/avoliitto
        '''
        #if self.randomness:
        #    sattuma = np.random.uniform(size=2)
        #else:
        #    sattuma = np.array([0,0])
        
        # update marital status
        intage=int(np.floor(age))
        if puoliso>0:
            if self.divorce_rate[intage]>sattuma[6]:
                puoliso=0
            else:
                puoliso=1
        else:
            if self.marriage_rate[intage]>sattuma[6]:
                puoliso=1
            else:
                puoliso=0
                
        if employment_status==15 or puoliso_tila==15:
            puoliso=0
    
        return puoliso
        
    def move_to_mort(self,age,children_under3,children_under7,children_under18,g,puoliso,prefnoise):
        #time_in_state+=self.timestep
        if not self.include_mort:
            print('mort not included but emp state 15')
        
        employment_status,puoliso_tila=15,15
        
        wage=0
        nextwage=0
        toe=0
        if self.mortstop:
            done=True
        else:
            age=age+self.timestep
            done = age >= self.max_age
            done = bool(done)
    
        #print(f'********* employment_status {employment_status} puoliso_tila {puoliso_tila} age {age}')
        
        pension,wage,nextwage,time_in_state=0,0,0,0
        tyoelake_maksussa=0
        pinkslip,toe,toekesto=0,0,0
        tyoura=0
        used_unemp_benefit=0
        wage_reduction=0
        unemp_after_ra=0
        unempwage=0
        unempwage_basis=0
        children_under3=0
        children_under7=0
        children_under18=0
        alkanut_ansiosidonnainen=0
        toe58=0
        ove_paid=0
        
        spouse_wage,puoliso_pension=0,0
        puoliso_wage_reduction=0
        puoliso_tyoelake_maksussa=0
        puoliso_next_wage=0
        puoliso_used_unemp_benefit=0
        puoliso_unemp_benefit_left=0
        puoliso_unemp_after_ra=0
        puoliso_unempwage=0
        puoliso_unempwage_basis=0
        puoliso_alkanut_ansiosidonnainen=0
        puoliso_toe58=0
        puoliso_toe=0
        puoliso_toekesto=0
        puoliso_tyoura=0
        puoliso_time_in_state=0
        puoliso_pinkslip=0
        puoliso_ove_paid=0
        kansanelake=0
        puoliso_kansanelake=0
        savings,puoliso_savings=0,0
                                
        self.state = self.state_encode(employment_status,g,pension,wage,age,
                        time_in_state,tyoelake_maksussa,pinkslip,toe,toekesto,tyoura,nextwage,
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
                        savings,puoliso_savings,
                        prefnoise)
                    
        if self.plotdebug:
            self.render()
                    
        netto,benq,netto_omat,netto_puoliso=self.get_benefits(15,0,0,0,0,
                    0,0,0,0,0,0,age,0,
                    children_under3,children_under7,children_under18,0,
                    0,15,0,0,0,0,0,
                    0,0,0,0,0,0,0)
                
        benq['omat_eq']=0
        benq['puoliso_eq']=0
        benq['eq']=0
        benq['mod_sav_action']=0    
        benq['mod_p_sav_action']=0    
        #print('done',done)
        #print(f'********* employment_status {employment_status} puoliso_tila {puoliso_tila} age {age}')
                    
        reward=0
        equivalent=0
        return np.array(self.state), reward, done, benq
            
    def step(self, action, dynprog=False, debug=False):
        '''
        Open AI interfacen mukainen step-funktio, joka tekee askeleen eteenpäin
        toiminnon action mukaan 

        Keskeinen funktio simuloinnissa
        '''
        #a=int(act/5)
        #action=[a,act%5]
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        emp_action=int(action[0])
        spouse_action=int(action[1])
        sav_action=self.map_save_action(int(action[2]))
        spouse_sav_action=self.map_save_action(int(action[3]))

        if self.plotdebug and False:
            print(f'act {emp_action} s_act {spouse_action}')
        if self.plotdebug and False:
            print(f'sact {sav_action} s_sact {spouse_sav_action}')

        employment_status,g,pension,old_wage,age,time_in_state,paid_pension,pinkslip,toe,\
            toekesto,tyoura,used_unemp_benefit,wage_reduction,unemp_after_ra,unempwage,\
            unempwage_basis,prefnoise,children_under3,children_under7,children_under18,\
            unemp_left,alkanut_ansiosidonnainen,toe58,ove_paid,jasen,\
            puoliso,puoliso_tila,spouse_g,puoliso_old_wage,puoliso_pension,puoliso_wage_reduction,\
            puoliso_paid_pension,puoliso_next_wage,puoliso_used_unemp_benefit,\
            puoliso_unemp_benefit_left,puoliso_unemp_after_ra,puoliso_unempwage,\
            puoliso_unempwage_basis,puoliso_alkanut_ansiosidonnainen,puoliso_toe58,\
            puoliso_toe,puoliso_toekesto,puoliso_tyoura,puoliso_time_in_state,puoliso_pinkslip,\
            puoliso_ove_paid,kansanelake,puoliso_kansanelake,tyoelake_maksussa,\
            puoliso_tyoelake_maksussa,next_wage,savings,spouse_savings\
                =self.state_decode(self.state)
                
        wage=next_wage                
        spouse_wage=puoliso_next_wage                
            
        intage=int(np.floor(age))
        t=int((age-self.min_age)/self.timestep)
        moved=False
        spouse_moved=False
        use_func=True
        
        prevstate=employment_status
        prevstatepuol=puoliso_tila
        if self.randomness:
            # kaikki satunnaisuus kerralla
            sattuma = np.random.uniform(size=8)
            sattuma2 = np.random.uniform(size=8)
            
            if self.include_spouses:
                puoliso=self.update_family(puoliso,age,employment_status,puoliso_tila,sattuma)
            else:
                puoliso=0
            
            # siirtymät, tässä disability_intensity iässä age+1, koska ensin svpäivärahakausi
            move_prob=self.disability_intensity[intage+1,g,employment_status]+self.birth_intensity[intage,g]\
                +self.student_inrate[intage,g]+self.outsider_inrate[intage,g]

            if sattuma[0]<move_prob:
                s2=self.birth_intensity[intage,g]
                s1=s2+self.disability_intensity[intage,g,employment_status]
                s3=s1+self.student_inrate[intage,g]
                #s4=s3+self.outsider_inrate[intage,g]
            
                if sattuma[2]<s2/move_prob: # vanhempainvapaa
                    if self.infostate_can_have_children(age) and not (employment_status==15 or puoliso_tila==15): 
                        # lasten väli vähintään vuosi, ei työkyvyttämyyseläkkeellä
                        if g>2: # naiset
                            if employment_status not in set([3]):
                                employment_status,pension,tyoelake_maksussa,wage,time_in_state,pinkslip=\
                                    self.move_to_motherleave(wage,pension,tyoelake_maksussa,old_wage,age)
                                pinkslip=0
                                karenssia_jaljella=0
                                moved=True
                                wage_reduction=self.update_wage_reduction(employment_status,wage_reduction,pinkslip,time_in_state)
                                if sattuma[4]<0.39 and puoliso_tila not in set([3]): # orig 0.5
                                    puoliso_tila,puoliso_pension,puoliso_tyoelake_maksussa,spouse_wage,puoliso_time_in_state,puoliso_pinkslip=\
                                        self.move_to_fatherleave(spouse_wage,puoliso_pension,puoliso_tyoelake_maksussa,puoliso_old_wage,age)
                                    puoliso_karenssia_jaljella=0
                                    puoliso_pinkslip=0
                                    spouse_moved=True
                                    puoliso_wage_reduction=self.update_wage_reduction(puoliso_tila,puoliso_wage_reduction,puoliso_pinkslip,puoliso_time_in_state)
                        else: # miehet
                            # ikä valittu äidin iän mukaan. oikeastaan tämä ei mene ihan oikein miehille
                            if puoliso_tila not in set([3]):
                                puoliso_tila,puoliso_pension,tyoelake_maksussa,spouse_wage,puoliso_time_in_state,puoliso_pinkslip=\
                                    self.move_to_motherleave(spouse_wage,puoliso_pension,puoliso_tyoelake_maksussa,puoliso_old_wage,age)
                                puoliso_karenssia_jaljella=0
                                puoliso_pinkslip=0
                                puoliso_wage_reduction=self.update_wage_reduction(puoliso_tila,puoliso_wage_reduction,puoliso_pinkslip,puoliso_time_in_state)
                                spouse_moved=True
                                if sattuma[4]<0.39 and employment_status not in set([3]): # orig 0.5
                                    employment_status,pension,tyoelake_maksussa,wage,time_in_state,pinkslip=\
                                        self.move_to_fatherleave(wage,pension,tyoelake_maksussa,old_wage,age)
                                    karenssia_jaljella=0
                                    pinkslip=0
                                    wage_reduction=self.update_wage_reduction(employment_status,wage_reduction,pinkslip,time_in_state)
                                    moved=True
                elif sattuma[2]<s1/move_prob: # age<self.min_retirementage and 
                    if not employment_status in set([15]):
                        emp_action=11 # disability
                elif sattuma[2]<s3/move_prob:
                    if employment_status in set([0,1,4,10,13]) and age<self.min_retirementage: # not in set([2,3,5,6,7,8,9,11,12,15]):
                        employment_status,pension,tyoelake_maksussa,wage,time_in_state,pinkslip=\
                            self.move_to_student(wage,pension,tyoelake_maksussa,old_wage,age,time_in_state,tyoura,pinkslip)
                        karenssia_jaljella=0
                        pinkslip=0
                        wage_reduction=self.update_wage_reduction(employment_status,wage_reduction,pinkslip,time_in_state)
                        moved=True
                else:
                    if employment_status in set([0,1,4,10,12,13]) and age<self.min_retirementage: # not in set([2,3,5,6,7,8,9,11,15]):
                        employment_status,pension,tyoelake_maksussa,wage,time_in_state,pinkslip=\
                            self.move_to_outsider(wage,pension,tyoelake_maksussa,old_wage,age,pinkslip)
                        karenssia_jaljella=0
                        pinkslip=0
                        wage_reduction=self.update_wage_reduction(employment_status,wage_reduction,pinkslip,time_in_state)
                        moved=True
                        
            # siirtymät, tässä disability_intensity iässä age+1, koska ensin svpäivärahakausi
            move_prob=self.disability_intensity[intage+1,spouse_g,puoliso_tila]+self.student_inrate[intage,spouse_g]+self.outsider_inrate[intage,spouse_g]

            if sattuma2[0]<move_prob and not spouse_moved and puoliso_tila!=15:
                s1=self.disability_intensity[intage,spouse_g,puoliso_tila]
                s3=s1+self.student_inrate[intage,spouse_g]
                #s4=s3+self.outsider_inrate[intage,g]
            
                # tk-alkavuus, siisti kuntoon!
                if sattuma2[2]<s1/move_prob: # age<self.min_retirementage and 
                    spouse_action=11 # disability
                elif sattuma2[2]<s3/move_prob:
                    if puoliso_tila in set([0,1,4,10,13]) and age<self.min_retirementage: # not in et([2,3,5,6,7,8,9,11,12,15]):
                        puoliso_tila,puoliso_pension,puoliso_tyoelake_maksussa,spouse_wage,puoliso_time_in_state,puoliso_pinkslip=\
                            self.move_to_student(spouse_wage,puoliso_pension,puoliso_tyoelake_maksussa,puoliso_old_wage,
                                age,puoliso_time_in_state,puoliso_tyoura,puoliso_pinkslip)
                        puoliso_karenssia_jaljella=0
                        puoliso_pinkslip=0
                        puoliso_wage_reduction=self.update_wage_reduction(puoliso_tila,puoliso_wage_reduction,puoliso_pinkslip,puoliso_time_in_state)
                        puoliso_moved=True
                else:
                    if puoliso_tila in set([0,1,4,10,12,13]) and age<self.min_retirementage: # not in set([2,3,5,6,7,8,9,11,12,15]):
                        puoliso_tila,puoliso_pension,puoliso_tyoelake_maksussa,spouse_wage,puoliso_time_in_state,puoliso_pinkslip=\
                            self.move_to_outsider(spouse_wage,puoliso_pension,puoliso_tyoelake_maksussa,puoliso_old_wage,age,puoliso_pinkslip)
                        puoliso_karenssia_jaljella=0
                        puoliso_pinkslip=0
                        puoliso_wage_reduction=self.update_wage_reduction(puoliso_tila,puoliso_wage_reduction,puoliso_pinkslip,puoliso_time_in_state)
                        puoliso_moved=True
                        
            # voi aiheuttaa epästabiilisuutta
            if sattuma[3]<self.mort_intensity[intage,g] and self.include_mort and employment_status!=15: 
                if puoliso>0: # avo- tai avioliitossa
                    # huomioi vain maksussa olevan puolison eläkkeen. Ei taida olla oikein.
                    puoliso_tyoelake_maksussa=self.comp_familypension(puoliso,employment_status,puoliso_tila,
                        tyoelake_maksussa,pension,age,puoliso_tyoelake_maksussa,
                        puoliso_pension,children_under18,puoliso,False)
                    spouse_savings += savings # perintöverot??
                    savings = 0
                        
                employment_status,pension,wage,time_in_state,puoliso,tyoelake_maksussa=\
                    self.move_to_deceiced(pension,old_wage,age)
                # self.bequest*
                
            if sattuma2[3]<self.mort_intensity[intage,spouse_g] and self.include_mort and puoliso_tila!=15:
                if puoliso>0: # avo- tai avioliitossa
                    # huomioi vain maksussa olevan puolison eläkkeen. Ei taida olla oikein.
                    tyoelake_maksussa=self.comp_familypension(puoliso,puoliso_tila,employment_status,
                        puoliso_tyoelake_maksussa,puoliso_pension,age,tyoelake_maksussa,
                        pension,children_under18,puoliso,True)
                    savings += spouse_savings # perintöverot??
                    spouse_savings = 0

                puoliso_tila,puoliso_pension,spouse_wage,puoliso_time_in_state,puoliso,puoliso_tyoelake_maksussa=\
                    self.move_to_deceiced(puoliso_pension,puoliso_old_wage,age)
                # self.bequest*
                
        else:
            # tn ei ole koskaan alle rajan, jos tämä on 1
            sattuma = np.ones(7)
            sattuma2 = np.ones(7)
            
        if employment_status==15 and puoliso_tila==15: # both deceiced
            return self.move_to_mort(age,children_under3,children_under7,children_under18,g,puoliso,prefnoise)
        else:
            karenssia_jaljella,puoliso_karenssia_jaljella=0,0
            if age>=self.max_retirementage and employment_status not in set([2,3,15]):
                employment_status,kansanelake,tyoelake_maksussa,pension,wage,time_in_state,ove_paid\
                    =self.move_to_retirement(wage,pension,0,age,kansanelake,tyoelake_maksussa,
                        employment_status,unemp_after_ra,all_acc=True,scale_acc=True,has_spouse=puoliso,is_spouse=False)
                pinkslip=0
                wage_reduction=self.update_wage_reduction(employment_status,wage_reduction,pinkslip,time_in_state)
                karenssia_jaljella=0
                moved=True
            
            if age>=self.max_retirementage and puoliso_tila not in set([2,3,15]):
                puoliso_tila,puoliso_kansanelake,puoliso_tyoelake_maksussa,puoliso_pension,spouse_wage,puoliso_time_in_state,puoliso_ove_paid\
                    =self.move_to_retirement(spouse_wage,puoliso_pension,0,age,puoliso_kansanelake,puoliso_tyoelake_maksussa,puoliso_tila,\
                        puoliso_unemp_after_ra,all_acc=True,scale_acc=True,has_spouse=puoliso,is_spouse=True)
                puoliso_pinkslip=0
                puoliso_wage_reduction=self.update_wage_reduction(puoliso_tila,puoliso_wage_reduction,puoliso_pinkslip,puoliso_time_in_state)
                puoliso_karenssia_jaljella=0
                puoliso_pinkslip=0
                spouse_moved=True
                        
            if (not moved) and employment_status != 15:
                # hoidetaan tilasiirtymät ja -pysymiset alirutiineilla, joita kutsutaan mäppäämällä tila funktioksi,
                # jota sitten kutsutaan
                is_spouse=False
                employment_status,kansanelake,tyoelake_maksussa,pension,wage,time_in_state,pinkslip,\
                unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella\
                    = self.map_stays[employment_status](wage,employment_status,kansanelake,tyoelake_maksussa,pension,time_in_state,toe,toekesto,
                                   tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,unempwage_basis,
                                   emp_action,age,sattuma,intage,g,alkanut_ansiosidonnainen,toe58,ove_paid,children_under3,children_under18,puoliso,
                                   is_spouse)
                wage_reduction=self.update_wage_reduction(employment_status,wage_reduction,pinkslip,time_in_state)

            if (not spouse_moved) and puoliso_tila != 15:
                # hoidetaan tilasiirtymät ja -pysymiset alirutiineilla, joita kutsutaan mäppäämällä tila funktioksi,
                # jota sitten kutsutaan
                is_spouse=True
                puoliso_tila,puoliso_kansanelake,puoliso_tyoelake_maksussa,puoliso_pension,spouse_wage,puoliso_time_in_state,puoliso_pinkslip,\
                puoliso_unemp_after_ra,puoliso_tyoura,puoliso_used_unemp_benefit,puoliso_unempwage_basis,puoliso_alkanut_ansiosidonnainen,puoliso_ove_paid,\
                puoliso_karenssia_jaljella\
                    = self.map_stays[puoliso_tila](spouse_wage,puoliso_tila,puoliso_kansanelake,puoliso_tyoelake_maksussa,puoliso_pension,puoliso_time_in_state,puoliso_toe,puoliso_toekesto,
                                   puoliso_tyoura,puoliso_used_unemp_benefit,puoliso_pinkslip,puoliso_unemp_after_ra,puoliso_old_wage,puoliso_unempwage,puoliso_unempwage_basis,
                                   spouse_action,age,sattuma2,intage,spouse_g,puoliso_alkanut_ansiosidonnainen,puoliso_toe58,puoliso_ove_paid,children_under3,children_under18,puoliso,
                                   is_spouse)
                puoliso_wage_reduction=self.update_wage_reduction(puoliso_tila,puoliso_wage_reduction,puoliso_pinkslip,puoliso_time_in_state)
            
            netto,benq,netto_omat,netto_puoliso=self.get_benefits(employment_status,wage,kansanelake,tyoelake_maksussa,pension,
                        time_in_state,pinkslip,old_wage,unempwage,unempwage_basis,karenssia_jaljella,age,
                        children_under3,children_under7,children_under18,ove_paid,used_unemp_benefit,
                        puoliso,puoliso_tila,spouse_wage,puoliso_kansanelake,puoliso_tyoelake_maksussa,puoliso_old_wage,
                        puoliso_pinkslip,puoliso_karenssia_jaljella,puoliso_time_in_state,puoliso_unempwage,puoliso_unempwage_basis,
                        puoliso_used_unemp_benefit,
                        g,spouse_g)

        #check_q(benq,10)
        
        # after this, preparing for the next step
        age=age+self.timestep
        
        if self.include_investment:
            if employment_status==15:
                mod_sav_action=0
            else:
                netto_omat,savings,mod_sav_action = self.update_savings(netto_omat,savings,sav_action,employment_status,age) 

            if puoliso_tila==15:
                mod_spouse_sav_action=0
            else:
                netto_puoliso,spouse_savings,mod_spouse_sav_action = self.update_savings(netto_puoliso,spouse_savings,spouse_sav_action,puoliso_tila,age) 
        else:
            savings,mod_sav_action=0,0
            spouse_savings,mod_spouse_sav_action=0,0
                        
        netto = netto_omat+netto_puoliso

        done = age >= self.max_age
        done = bool(done)
        
        toe58=self.check_toe58(age,toe,tyoura,toe58)
        puoliso_toe58=self.check_toe58(age,puoliso_toe,puoliso_tyoura,puoliso_toe58)
        
        work={1,10}
        retired={2,8,9}
        self.update_infostate(t,int(employment_status),wage,unempwage_basis,is_spouse=False)
        toe,toekesto,unempwage,children_under3,children_under7,children_under18=self.comp_infostats(age,is_spouse=False)
        if employment_status in work and self.tyossaoloehto(toe,tyoura,age):
            used_unemp_benefit=0
            alkanut_ansiosidonnainen=0
            #if alkanut_ansiosidonnainen>0:
            #    if not self.infostate_check_aareset(age):
            #        alkanut_ansiosidonnainen=0
        elif employment_status in retired:
            alkanut_ansiosidonnainen=0
        if alkanut_ansiosidonnainen<1 or age>self.max_unemploymentbenefitage:
            unempwage_basis=0
            
        self.update_infostate(t,int(puoliso_tila),spouse_wage,puoliso_unempwage_basis,is_spouse=True)
        puoliso_toe,puoliso_toekesto,puoliso_unempwage,_,_,_=self.comp_infostats(age,is_spouse=True)
        if puoliso_tila in work and self.tyossaoloehto(puoliso_toe,puoliso_tyoura,age):
            puoliso_used_unemp_benefit=0
            puoliso_alkanut_ansiosidonnainen=0
            #if alkanut_ansiosidonnainen>0:
            #    if not self.infostate_check_aareset(age):
            #        alkanut_ansiosidonnainen=0
        elif puoliso_tila in retired:
            puoliso_alkanut_ansiosidonnainen=0
        if puoliso_alkanut_ansiosidonnainen<1 or age>self.max_unemploymentbenefitage:
            puoliso_unempwage_basis=0
            
        if employment_status not in set([2,3,8,9,15]) and age<self.max_unemploymentbenefitage:
            if self.porrasta_toe and (employment_status in set([0,4]) or alkanut_ansiosidonnainen>0):
                old_toe=self.comp_oldtoe(spouse=False)
                pvr_jaljella=self.comp_unempdays_left(used_unemp_benefit,tyoura,age,toe,employment_status,alkanut_ansiosidonnainen,toe58,old_toe,printti=False)
            else:
                pvr_jaljella=self.comp_unempdays_left(used_unemp_benefit,tyoura,age,toe,employment_status,alkanut_ansiosidonnainen,toe58,toekesto)
        else:
            pvr_jaljella=0

        if puoliso_tila not in set([2,3,8,9,15]) and age<self.max_unemploymentbenefitage:
            if self.porrasta_toe and (puoliso_tila in set([0,4]) or puoliso_alkanut_ansiosidonnainen>0):
                p_old_toe=self.comp_oldtoe(spouse=True)
                puoliso_pvr_jaljella=self.comp_unempdays_left(puoliso_used_unemp_benefit,puoliso_tyoura,age,puoliso_toe,puoliso_tila,puoliso_alkanut_ansiosidonnainen,puoliso_toe58,p_old_toe,printti=False)
            else:
                puoliso_pvr_jaljella=self.comp_unempdays_left(puoliso_used_unemp_benefit,puoliso_tyoura,age,puoliso_toe,puoliso_tila,puoliso_alkanut_ansiosidonnainen,puoliso_toe58,puoliso_toekesto)
        else:
            puoliso_pvr_jaljella=0
            
        kassanjasenyys=self.get_kassanjasenyys()
        
        #self.render_infostate()

        if not done:
            reward_omat,omat_equivalent = self.log_utility(netto_omat,int(employment_status),age,g=g,pinkslip=pinkslip)
            #print('c',netto_puoliso)
            reward_puoliso,spouse_equivalent = self.log_utility(netto_puoliso,int(puoliso_tila),age,g=spouse_g,pinkslip=puoliso_pinkslip)
            reward=reward_omat+reward_puoliso
            #print(reward_puoliso,spouse_equivalent,netto_puoliso)
            equivalent=omat_equivalent+spouse_equivalent
            
            if not np.isfinite(reward_omat):
                print('omat',netto_omat,reward_omat)
            if not np.isfinite(reward_puoliso):
                print('puoliso',netto_puoliso,reward_puoliso)

            benq['omat_eq']=omat_equivalent
            benq['puoliso_eq']=spouse_equivalent
            benq['eq']=equivalent
            benq['mod_sav_action']=mod_sav_action
            benq['mod_p_sav_action']=mod_spouse_sav_action
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            
            paid_pension += self.elinaikakerroin*pension # ei ihan oikein! lykkäyskorotus puuttuu, optimoijan pitäisi huomata, joten ei vaikutusta?
            pension=0.0
            puoliso_paid_pension += self.elinaikakerroin*puoliso_pension # ei ihan oikein! lykkäyskorotus puuttuu, optimoijan pitäisi huomata, joten ei vaikutusta?
            puoliso_pension=0.0
            
            netto,benq,netto_omat,netto_puoliso=self.get_benefits(employment_status,wage,kansanelake,tyoelake_maksussa,pension,time_in_state,pinkslip,old_wage,
                unempwage,unempwage_basis,karenssia_jaljella,age,children_under3,children_under7,children_under18,ove_paid,used_unemp_benefit,
                puoliso,puoliso_tila,spouse_wage,puoliso_kansanelake,puoliso_tyoelake_maksussa,puoliso_old_wage,puoliso_pinkslip,puoliso_karenssia_jaljella,
                puoliso_time_in_state,puoliso_unempwage,puoliso_unempwage_basis,puoliso_used_unemp_benefit,g,spouse_g)

            if employment_status in set([2,3,8,9]): # retired
                if self.include_npv_mort:
                    npv,npv0,npv_pension,npv_gpension=self.comp_npv_simulation(g)
                else:
                    npv,npv0,npv_pension,npv_gpension=self.npv[g],self.npv0[g],self.npv_pension[g],self.npv_gpension[g]
                
                if self.include_investment:    
                    savings_component=savings/npv_pension # ostetaan sijoituksilla annuiteetti, FIXME
                    netto_omat=netto_omat+savings_component
                    savings=0

                reward_omat,omat_equivalent = self.log_utility(netto_omat,int(employment_status),age,g=g,pinkslip=pinkslip)
                reward_omat *= npv_gpension # discounting with pension index & gamma
            else:
                # giving up the pension
                reward_omat,omat_equivalent=0.0,0.0
                npv,npv0,npv_pension,npv_gpension=0.0,0.0,0.0,0.0
                
            if puoliso_tila in set([2,3,8,9]): # retired
                if self.include_npv_mort:
                    p_npv,p_npv0,p_npv_pension,p_npv_gpension=self.comp_npv_simulation(spouse_g)
                else:
                    p_npv,p_npv0,p_npv_pension,p_npv_gpension=self.npv[spouse_g],self.npv0[spouse_g],self.npv_pension[spouse_g],self.npv_gpension[spouse_g]
                    
                if self.include_investment:    
                    spouse_savings_component=spouse_savings/p_npv_pension # ostetaan sijoituksilla annuiteetti
                    netto_puoliso=netto_puoliso+spouse_savings_component
                    spouse_savings=0

                reward_puoliso,spouse_equivalent = self.log_utility(netto_puoliso,int(puoliso_tila),age,g=spouse_g,pinkslip=puoliso_pinkslip)
                reward_puoliso *= p_npv_gpension
            else:
                # giving up the pension
                reward_puoliso,spouse_equivalent=0.0,0.0
                p_npv,p_npv0,p_npv_pension,p_npv_gpension=0.0,0.0,0.0,0.0
            
            netto=netto_omat+netto_puoliso
            benq['omat_eq']=omat_equivalent
            benq['puoliso_eq']=spouse_equivalent
            benq['mod_sav_action']=mod_sav_action
            benq['mod_p_sav_action']=mod_spouse_sav_action
            
            # updates benq directly
            self.scale_q(npv,npv0,npv_pension,npv_gpension,p_npv,p_npv0,p_npv_pension,p_npv_gpension,benq,age)
            
            # total reward is
            reward=reward_omat+reward_puoliso
            equivalent=omat_equivalent+spouse_equivalent
            benq['eq']=equivalent
            pinkslip=0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0
            equivalent = 0.0
            omat_equivalent = 0.0
            spouse_equivalent = 0.0
            
            benq['omat_eq']=omat_equivalent
            benq['puoliso_eq']=spouse_equivalent
            benq['eq']=equivalent
            benq['mod_sav_action']=mod_sav_action
            benq['mod_p_sav_action']=mod_spouse_sav_action

        # seuraava palkka tiedoksi valuaatioapproksimaattorille
        if employment_status in set([3,15]):
            next_wage=0
        else:
            next_wage=self.get_wage(age,wage_reduction)
            
        #check_q(benq,99)
        
        if puoliso_tila in set([3,15]):
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
                                puoliso_toe,puoliso_toekesto,puoliso_tyoura,puoliso_time_in_state,
                                puoliso_pinkslip,puoliso_ove_paid,kansanelake,puoliso_kansanelake,
                                savings,spouse_savings,
                                prefnoise)

        if self.plotdebug:
            self.render(done=done,reward=reward,netto=netto,benq=benq,netto_omat=netto_omat,netto_puoliso=netto_puoliso,
                savact=mod_sav_action,p_savact=mod_spouse_sav_action)

        return np.array(self.state), reward, done, benq
        
    def scale_q(self,npv,npv0,npv_pension,npv_gpension,p_npv,p_npv0,p_npv_pension,p_npv_gpension,benq,age):
        '''
        Scaling the incomes etc by a discounted nominal present value
        '''
        omat='omat_'
        puoliso='puoliso_'
        
#         print('omat npv:',npv0)
#         print(type(npv0))
#         print('puoliso npv:',npv_pension)
#         print(type(npv_pension))

        v_pens={omat: npv_pension, puoliso: p_npv_pension}
        v0_pens={omat: npv0, puoliso: p_npv0}
        for alku in set([omat,puoliso]):
            p1=v_pens[alku]
            p2=v0_pens[alku]
            #print(p1,p2)
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
            benq[alku+'eq']*=p1
            benq[alku+'multiplier']=p2

        #for alku in set([omat,puoliso]):
        #    print(alku,benq[alku+'multiplier'])

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
        benq['eq']=benq[omat+'eq']+benq[puoliso+'eq']
        benq['multiplier']=(benq[omat+'multiplier']+benq[puoliso+'multiplier'])/2

#  Perussetti, tuottaa korkean elastisuuden

    def get_mortstate(self):
        return 15
        
    def log_utility_mort_noove_params(self):
        #
        # OVE - NO
        # MORT - YES
        #
    
        # paljonko työstä poissaolo vaikuttaa palkkaan
        self.salary_const=0.04*self.timestep # 0.038 työttämyydestä palkka alenee tämän verran vuodessa
        self.salary_const_retirement=0.10*self.timestep # vanhuuseläkkeellä muutos nopeampaa
        self.salary_const_svpaiva=0.20*self.timestep # pitkällä svpäivärahalla muutos nopeaa
        self.salary_const_up=0.030*self.timestep # 0.04 työssäolo palauttaa ansioita tämän verran vuodessa
        self.salary_const_up_osaaika=0.030*self.timestep # 0.04 osa-aikainen työssäolo palauttaa ansioita tämän verran vuodessa
        self.salary_const_student=0.01*self.timestep # 0.05 opiskelu pienentää leikkausta tämän verran vuodessa
        self.wage_initial_reduction=0.015 # työttömäksi siirtymisestä tuleva alennus tuleviin palkkoihin, NOT USED!
        
        self.max_mu_age=self.min_retirementage+6.0 # 
        
        self.men_kappa_fulltime=0.736 # vapaa-ajan menetyksestä rangaistus miehille
        self.men_mu_scale_kokoaika=0.0518 #0.075 # 0.075 #18 # 0.14 # 0.30 # 0.16 # how much penalty is associated with work increase with age after mu_age
        self.men_mu_scale_osaaika=0.0350 #0.075 # 0.075 #18 # 0.14 # 0.30 # 0.16 # how much penalty is associated with work increase with age after mu_age
        self.men_mu_age=self.min_retirementage-8.75 #5.5 # P.O. 60??
        self.men_kappa_osaaika_young=0.515 # vapaa-ajan menetyksestä rangaistus miehille osa-aikatyön teosta, suhteessa kokoaikaan
        self.men_kappa_osaaika_middle=0.655 # vapaa-ajan menetyksestä rangaistus miehille osa-aikatyön teosta, suhteessa kokoaikaan
        #self.men_kappa_osaaika_old=0.62 # vapaa-ajan menetyksestä rangaistus miehille osa-aikatyön teosta, suhteessa kokoaikaan
        self.men_kappa_osaaika_old=0.62 # self.men_kappa_osaaika_middle
        self.men_kappa_osaaika_pension=0.65
        self.men_kappa_hoitovapaa=0.037 # hyäty hoitovapaalla olosta
        self.men_kappa_ve=0.25
        self.men_kappa_pinkslip_young=0.30
        self.men_kappa_pinkslip_middle=0.20
        self.men_kappa_pinkslip_elderly=0.15
        
        self.women_kappa_fulltime=0.581 # vapaa-ajan menetyksestä rangaistus naisille
        self.women_mu_scale_kokoaika=0.0518 #0.075 # 0.075 # 0how much penalty is associated with work increase with age after mu_age
        self.women_mu_scale_osaaika=0.0350 #0.075 # 0.075 # 0how much penalty is associated with work increase with age after mu_age
        self.women_mu_age=self.min_retirementage-4.5 #4.0 # 61 #5 P.O. 60??
        self.women_kappa_osaaika_young=0.376
        self.women_kappa_osaaika_middle=0.400
        #self.women_kappa_osaaika_old=0.48
        self.women_kappa_osaaika_old=0.400 # self.women_kappa_osaaika_middle
        self.women_kappa_osaaika_pension=0.55
        self.women_kappa_hoitovapaa=0.270 # 0.27
        self.women_kappa_ve=0.25
        self.women_kappa_pinkslip_young=0.35
        self.women_kappa_pinkslip_middle=0.20
        self.women_kappa_pinkslip_elderly=0.20
        self.kappa_svpaivaraha=0.5        
        
    def log_utility_mort_ove_params(self):
        #
        # OVE - YES
        # MORT - YES
        #
    
        self.salary_const=0.04*self.timestep # 0.038 työttämyydestä palkka alenee tämän verran vuodessa
        self.salary_const_retirement=0.10*self.timestep # vanhuuseläkkeellä muutos nopeampaa
        self.salary_const_svpaiva=0.20*self.timestep # pitkällä svpäivärahalla muutos nopeaa
        self.salary_const_up=0.030*self.timestep # 0.04 työssäolo palauttaa ansioita tämän verran vuodessa
        self.salary_const_up_osaaika=0.030*self.timestep # 0.04 osa-aikainen työssäolo palauttaa ansioita tämän verran vuodessa
        self.salary_const_student=0.01*self.timestep # 0.05 opiskelu pienentää leikkausta tämän verran vuodessa
        self.wage_initial_reduction=0.015 # työttömäksi siirtymisestä tuleva alennus tuleviin palkkoihin, NOT USED!
        
        self.max_mu_age=self.min_retirementage+7.0 # 
        
        self.men_kappa_fulltime=0.820 # vapaa-ajan menetyksestä rangaistus miehille
        self.men_mu_scale_kokoaika=0.02 # 0.0560 #0.075 # 0.075 #18 # 0.14 # 0.30 # 0.16 # how much penalty is associated with work increase with age after mu_age
        self.men_mu_scale_osaaika=0.01 # 0.0330 #0.075 # 0.075 #18 # 0.14 # 0.30 # 0.16 # how much penalty is associated with work increase with age after mu_age
        self.men_mu_age=self.min_retirementage-7.25 #5.5 # P.O. 60??
        self.men_kappa_osaaika_young=0.535 # vapaa-ajan menetyksestä rangaistus miehille osa-aikatyön teosta, suhteessa kokoaikaan
        self.men_kappa_osaaika_middle=0.660 # vapaa-ajan menetyksestä rangaistus miehille osa-aikatyön teosta, suhteessa kokoaikaan
        #self.men_kappa_osaaika_old=0.62 # vapaa-ajan menetyksestä rangaistus miehille osa-aikatyön teosta, suhteessa kokoaikaan
        self.men_kappa_osaaika_old=0.625 # self.men_kappa_osaaika_middle
        self.men_kappa_osaaika_pension=0.65
        self.men_kappa_hoitovapaa=0.04 # hyäty hoitovapaalla olosta
        self.men_kappa_ve=0.29
        self.men_kappa_pinkslip_young=0.25
        self.men_kappa_pinkslip_middle=0.15
        self.men_kappa_pinkslip_elderly=0.15
        
        self.women_kappa_fulltime=0.635 # vapaa-ajan menetyksestä rangaistus naisille
        self.women_mu_scale_kokoaika=0.02 # 0.0560 #0.075 # 0.075 # 0how much penalty is associated with work increase with age after mu_age
        self.women_mu_scale_osaaika=0.01 # 0.0330 #0.075 # 0.075 # 0how much penalty is associated with work increase with age after mu_age
        self.women_mu_age=self.min_retirementage-3.5 #4.0 # 61 #5 P.O. 60??
        self.women_kappa_osaaika_young=0.360
        self.women_kappa_osaaika_middle=0.400
        #self.women_kappa_osaaika_old=0.48
        self.women_kappa_osaaika_old=0.405 # self.women_kappa_osaaika_middle
        self.women_kappa_osaaika_pension=0.55
        self.women_kappa_hoitovapaa=0.300 # 0.27
        self.women_kappa_ve=0.29
        self.women_kappa_pinkslip_young=0.30
        self.women_kappa_pinkslip_middle=0.20
        self.women_kappa_pinkslip_elderly=0.20   
        self.kappa_svpaivaraha=0.5
             
    def log_utility_nomort_noove_params(self):
        #
        # OVE - NO/YES
        # MORT - YES
        #
        
        self.salary_const=0.045*self.timestep # työttämyydestä palkka alenee tämän verran vuodessa
        self.salary_const_retirement=0.10*self.timestep # vanhuuseläkkeellä muutos nopeampaa
        self.salary_const_svpaiva=0.20*self.timestep # pitkällä svpäivärahalla muutos nopeaa
        self.salary_const_up=0.04*self.timestep # työssäolo palauttaa ansioita tämän verran vuodessa
        self.salary_const_up_osaaika=0.030*self.timestep # 0.04 osa-aikainen työssäolo palauttaa ansioita tämän verran vuodessa
        self.salary_const_student=0.05*self.timestep # opiskelu pienentää leikkausta tämän verran vuodessa
        self.wage_initial_reduction=0.015 # työttömäksi siirtymisestä tuleva alennus tuleviin palkkoihin, NOT USED!
        
        self.max_mu_age=self.min_retirementage+6.0 # 

        self.men_kappa_fulltime=0.640 # 0.675 #0.682 # 0.670 # vapaa-ajan menetyksestä rangaistus miehille
        self.men_mu_scale_kokoaika=0.0518 #0.075 # 0.075 #18 # 0.14 # 0.30 # 0.16 # how much penalty is associated with work increase with age after mu_age
        self.men_mu_scale_osaaika=0.0350 #0.075 # 0.075 #18 # 0.14 # 0.30 # 0.16 # how much penalty is associated with work increase with age after mu_age
        self.men_mu_age=self.min_retirementage-4.0 # P.O. 60??
        self.men_kappa_osaaika_young=0.55 # vapaa-ajan menetyksestä rangaistus miehille osa-aikatyön teosta, suhteessa kokoaikaan
        self.men_kappa_osaaika_middle=0.62 # vapaa-ajan menetyksestä rangaistus miehille osa-aikatyön teosta, suhteessa kokoaikaan
        self.men_kappa_osaaika_old=0.40 # vapaa-ajan menetyksestä rangaistus miehille osa-aikatyön teosta, suhteessa kokoaikaan, alle 35v
        self.men_kappa_osaaika_pension=0.30
        self.men_kappa_hoitovapaa=0.30 # hyäty hoitovapaalla olosta
        self.men_kappa_ve=0.00 # 0.03 # ehkä 0.10?
        self.men_kappa_pinkslip_young=0.01
        self.men_kappa_pinkslip_middle=0.20
        self.men_kappa_pinkslip_elderly=0.05
        
        self.women_kappa_fulltime=0.640 # 0.605 # 0.640 # 0.620 # 0.610 # vapaa-ajan menetyksestä rangaistus naisille
        self.women_mu_scale_kokoaika=0.0518 #0.075 # 0.075 # 0how much penalty is associated with work increase with age after mu_age
        self.women_mu_scale_osaaika=0.0350 #0.075 # 0.075 # 0how much penalty is associated with work increase with age after mu_age
        self.women_mu_age=self.min_retirementage-3.5 # 61 #5 P.O. 60??
        self.women_kappa_osaaika_young=0.45
        self.women_kappa_osaaika_middle=0.50
        self.women_kappa_osaaika_old=0.30
        self.women_kappa_osaaika_pension=0.30
        self.women_kappa_hoitovapaa=0.70 # 0.08
        self.women_kappa_ve=0.00 # 0.03 # ehkä 0.10?
        self.women_kappa_pinkslip_young=0.10
        self.women_kappa_pinkslip_middle=0.27
        self.women_kappa_pinkslip_elderly=0.25
        self.kappa_svpaivaraha=0.5

    def log_get_kappa(self,age : float,g : int,employment_state : int,pinkslip : int):
        # kappa tells how much person values free-time
        if g<3: # miehet
            kappa_kokoaika=self.men_kappa_fulltime
            mu_scale_kokoaika=self.men_mu_scale_kokoaika
            mu_scale_osaaika=self.men_mu_scale_osaaika
            mu_age=self.men_mu_age
            # lognormaali
            #if self.include_preferencenoise:
            #    kappa_kokoaika += prefnoise
        
            if age<28: # ikääntyneet preferoivat osa-aikatyötä
                kappa_osaaika=self.men_kappa_osaaika_young
            elif age<58: # ikääntyneet preferoivat osa-aikatyötä
                kappa_osaaika=self.men_kappa_osaaika_middle
            else:
                kappa_osaaika=self.men_kappa_osaaika_old

            kappa_osaaika_pension=self.men_kappa_osaaika_pension
                
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
            mu_scale_kokoaika=self.women_mu_scale_kokoaika
            mu_scale_osaaika=self.women_mu_scale_osaaika
            mu_age=self.women_mu_age
            # lognormaali
            #if self.include_preferencenoise:
            #    kappa_kokoaika += prefnoise
        
            if age<28: # ikääntyneet preferoivat osa-aikatyötä
                kappa_osaaika=self.women_kappa_osaaika_young
            elif age<58: # ikääntyneet preferoivat osa-aikatyötä
                kappa_osaaika=self.women_kappa_osaaika_middle
            else:
                kappa_osaaika=self.women_kappa_osaaika_old
                
            kappa_osaaika_pension=self.women_kappa_osaaika_pension
                
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
        
        kappa_osaaika = kappa_osaaika*kappa_kokoaika
        if age>mu_age:
            mage=max(0,min(self.max_mu_age,age)-mu_age)
            kappa_osaaika += mu_scale_osaaika*mage
            kappa_kokoaika += mu_scale_kokoaika*mage
            
            #print(age,kappa_osaaika,mu_scale_osaaika*mage,kappa_kokoaika,mu_scale_kokoaika*mage)
        
        if employment_state in set([1,9]):
            kappa= -kappa_kokoaika
        elif employment_state in set([10]):
            kappa= -kappa_osaaika
        elif employment_state in set([8]):
            kappa= -kappa_osaaika #_pension*kappa_kokoaika
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
            kappa=0
        elif employment_state == 12:
            kappa=0
        else: # states 3, 5, 6, 7, 14, 15
            kappa=0  
            
        return kappa  
                    
    def log_utility(self,income : float,employment_state : int,age,g=0,pinkslip=0,prefnoise=0,spouse=0,debug=False):
        '''
        Log-utiliteettifunktio muokattuna lähteestä Määttänen, 2013 & Hakola & Määttänen, 2005

        Tulot _income_ ovat vuositasolla, jotta askelpituuden muutos ei vaikuta vapaa-aika-vakioihin
        Tämä versio on parametrisoitu optimoijaa varten
        '''
        
        if employment_state==15:
            return 0,0

        kappa=self.log_get_kappa(age,g,employment_state,pinkslip)

        # hyäty/score
        if self.include_preferencenoise:
            # normaali
            u=np.log(prefnoise*income/self.inflationfactor)+kappa
            equ=(income/self.inflationfactor)*np.exp(kappa)
        else:
            u=np.log(income/self.inflationfactor)+kappa
            equ=(income/self.inflationfactor)*np.exp(kappa)

        if u is np.inf and debug:
            print(f'inf: state {employment_state} spouse {spouse} sp_state {spouse_empstate} netto {income}')

        if income<1 and debug:
            print(f'inf: state {employment_state} spouse {spouse} sp_state {spouse_empstate} netto {income}')
            
        return u/20,equ # tulot ovat vuositasolla, mutta skaalataan hyäty

    def CRRA(self,net):
        '''
        Isoelastic utility function
        A common setting is self.CRRA_eta=1.5
        When CRRA_eta==1.0, it reduces to log utility
        '''
        return ((net)**(1.0-self.CRRA_eta)-1.0)/(1-self.CRRA_eta)

    def CRRA_utility(self,income : float,employment_state : int,age,g=0,pinkslip=0,prefnoise=0,spouse=0,debug=False):
        '''
        CRRA-utiliteettifunktio 

        Tulot _income_ ovat vuositasolla, jotta askelpituuden muutos ei vaikuta vapaa-aika-vakioihin
        Tämä versio on parametrisoitu optimoijaa varten
        '''
        
        if employment_state==15:
            return 0,0

        kappa=self.log_get_kappa(age,g,employment_state,pinkslip)

        # hyäty/score
        if self.include_preferencenoise:
            # normaali
            u=self.CRRA(prefnoise*income/self.inflationfactor)+kappa
            equ=(income/self.inflationfactor)*np.exp(kappa)
        else:
            u=self.CRRA(income/self.inflationfactor)+kappa
            equ=(income/self.inflationfactor)*np.exp(kappa)

        if u is np.inf and debug:
            print(f'inf: state {employment_state} spouse {spouse} sp_state {spouse_empstate} netto {income}')

        if income<1 and debug:
            print(f'inf: state {employment_state} spouse {spouse} sp_state {spouse_empstate} netto {income}')
            
        return u/20,equ # tulot ovat vuositasolla, mutta skaalataan hyäty



#     def get_spousewage_step(self,age,reduction):
#         '''
#         palkka age-ikäiselle time_in_state-vähennyksellä työllistymispalkkaan, step-kohtaisesti, ei vuosikohtaisesti
#         '''
#         intage=self.map_age(age)
#         if age<self.max_age and age>=self.min_age-1:
#             return np.maximum(self.min_salary,self.spousesalary[intage])*max(0,(1-reduction))
#         else:
#             return 0
# 
#     def get_wage_step(self,age,reduction):
#         '''
#         palkka age-ikäiselle time_in_state-vähennyksellä työllistymispalkkaan, step-kohtaisesti, ei vuosikohtaisesti
#         '''
#         intage=self.map_age(age)
#         if age<self.max_age and age>=self.min_age-1:
#             return np.maximum(self.min_salary,self.salary[intage])*max(0,(1-reduction))
#         else:
#             return 0

    def set_parameters(self,**kwargs):
        if 'kwargs' in kwargs:
            kwarg=kwargs['kwargs']
        else:
            kwarg=kwargs

        for key, value in kwarg.items():
            if key=='step':
                if value is not None:
                    self.timestep=value
            elif key=='mortplot':
                if value is not None:
                    self.mortplot=value
            elif key=='random_returns':
                if value is not None:
                    self.random_returns=value
            elif key=='use_utility':
                if value is not None:
                    self.use_utility=value
            elif key=='CRRA_eta':
                if value is not None:
                    self.CRRA_eta=value
            elif key=='investment':
                if value is not None:
                    self.include_investment=value
            elif key=='wealth':
                if value is not None:
                    self.include_wealth=value
            elif key=='r_mean':
                if value is not None:
                    self.r_mean=value
            elif key=='r_std':
                if value is not None:
                    self.r_std=value
            elif key=='silent':
                if value is not None:
                    self.silent=value
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
                    
    def map_age(self,age : float,start_zero=False):
        if start_zero:
            return int((age)*self.inv_timestep)
        else:
            return int((age-self.min_age)*self.inv_timestep)
          
    def state_encode(self,emp : int,g : int,pension : float,old_wage : float,age : float,time_in_state : float,tyoelake_maksussa : float,pink : int,
                        toe : float,toekesto : float,tyohist : float,next_wage : float,used_unemp_benefit : float,wage_reduction : float,
                        unemp_after_ra : float,unempwage : float,unempwage_basis : float,
                        children_under3 : int,children_under7 : int,children_under18 : int,
                        unemp_benefit_left : float,alkanut_ansiosidonnainen : int,toe58 : int,ove_paid : float,kassanjasenyys : int,
                        puoliso : int,puoliso_tila : int,puoliso_old_wage : float,puoliso_pension : float,
                        puoliso_wage_reduction : float,puoliso_tyoelake_maksussa : float,puoliso_next_wage : float,
                        puoliso_used_unemp_benefit : float,puoliso_unemp_benefit_left : float,
                        puoliso_unemp_after_ra : float,puoliso_unempwage : float,puoliso_unempwage_basis : float,
                        puoliso_alkanut_ansiosidonnainen : float,puoliso_toe58 : float,
                        puoliso_toe : float,puoliso_toekesto : float,puoliso_tyoura : float,puoliso_time_in_state : float,
                        puoliso_pinkslip : int,puoliso_ove_paid : float,kansanelake : float,puoliso_kansanelake : float,
                        savings : float,puoliso_savings : float,
                        prefnoise : float):     
        '''
        Tilan koodaus neuroverkkoa varten. Arvot skaalataan ja tilat one-hot-enkoodataan

        Käytetään, jos kuolleisuus ei mukana
        '''
        if self.include_investment:    
            if self.include_preferencenoise:
                d=np.zeros(self.n_empl+self.n_groups+self.n_spouseempl+50)
            else:
                d=np.zeros(self.n_empl+self.n_groups+self.n_spouseempl+49)
        else:
            if self.include_preferencenoise:
                d=np.zeros(self.n_empl+self.n_groups+self.n_spouseempl+48)
            else:
                d=np.zeros(self.n_empl+self.n_groups+self.n_spouseempl+47)
                    
        states=self.n_empl
        # d2=np.zeros(n_empl,1)
        # d2[emp]=1
        d[0:states]=self.state_encoding[emp,:]
        if emp==15 and not self.include_mort:
            print('no state 15 in state_encode_nomort')
        elif emp>15:
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
        
        if self.include_investment:    
            d[states3+20]=(savings-100_000)/100_000
            d[states3+21]=(puoliso_savings-100_000)/100_000
            if self.include_preferencenoise:
                d[states3+22]=prefnoise
        else:
            if self.include_preferencenoise:
                d[states3+20]=prefnoise
            
        return d

    def get_spouse_g(self,g : int):
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
        used_unemp_benefit=vec[pos+11]+1 # käytetty työttämyyspäivärahapäivien määrä
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
        
        if self.include_investment:    
            savings=vec[pos2+20]*100_000+100_000
            puoliso_savings=vec[pos2+21]*100_000+100_000
        else:
            savings=0
            puoliso_savings=0

        if self.include_preferencenoise:
            if self.include_investment:    
                prefnoise=vec[pos2+22]
            else:
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
               kansanelake,puoliso_kansanelake,tyoelake_maksussa,puoliso_tyoelake_maksussa,next_wage,\
               savings,puoliso_savings
                              
    def random_init_state(self,minage=18,maxage=70):
        emp=random.randint(0,15)
        g=np.random.randint(0,6)
        pension=np.random.uniform(0,80_000)
        old_wage=np.random.uniform(0,80_000)
        age=np.random.randint(minage/self.timestep,maxage/self.timestep-1)*self.timestep
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
        puoliso_tila=np.random.randint(0,15)
        puoliso_wage_reduction=np.random.uniform(0,1.0)
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
        kansanelake=np.random.uniform(0,10000)
        puoliso_kansanelake=np.random.uniform(0,10000)
        tyoelake_maksussa=np.random.uniform(0,50000)
        puoliso_tyoelake_maksussa=np.random.uniform(0,50000)
        if include_investment:
            savings=np.random.uniform(0,50000)
            puoliso_savings=np.random.uniform(0,50000)
        else:
            savings=0
            puoliso_savings=0
        
        paid_pension=kansanelake+tyoelake_maksussa
        puoliso_paid_pension=puoliso_kansanelake+puoliso_tyoelake_maksussa
        
        if age<63.5:
            if puoliso_tila in set([2,8,9]):
                puoliso_tila=0
            if emp in set([2,8,9]):
                emp=0
            kansanelake=0
            puoliso_kansanelake=0
            tyoelake_maksussa=0
            puoliso_tyoelake_maksussa=0
        
        vec=self.state_encode(emp,g,pension,old_wage,age,time_in_state,tyoelake_maksussa,pink,
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
                savings,puoliso_savings,
                prefnoise)
                            
        return vec
        
    def unit_test_code_decode(self):
        for k in range(10):
            emp=random.randint(0,15)
            g=np.random.randint(0,6)
            pension=np.random.uniform(0,80_000)
            old_wage=np.random.uniform(0,80_000)
            age=np.random.randint(0,70)
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
            puoliso_tila=np.random.randint(0,15)
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
            if self.include_investment:
                puoliso_savings=np.random.uniform(0,50000)
                savings=np.random.uniform(0,50000)
            else:
                puoliso_savings=0
                savings=0
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
                                savings,puoliso_savings,
                                prefnoise)
                                
            emp2,g2,pension2,wage2,age2,time_in_state2,paid_pension2,pink2,toe2,toekesto2,\
                tyohist2,used_unemp_benefit2,wage_reduction2,unemp_after_ra2,\
                unempwage2,unempwage_basis2,prefnoise2,\
                children_under3_2,children_under7_2,children_under18_2,unemp_benefit_left2,\
                alkanut_ansiosidonnainen2,toe58_2,ove_paid_2,jasen_2,p2,p2_tila,p2_g,p2_old_wage,p2_pension,\
                p2_wage_reduction,p2_paid_pension,p2_next_wage,p2_used_unemp_benefit,p2_unemp_benefit_left,\
                p2_unemp_after_ra,p2_unempwage,p2_unempwage_basis,p2_alkanut_ansiosidonnainen,p2_toe58,p2_toe,\
                p2_toekesto,p2_tyoura,p2_time_in_state,p2_pinkslip,p2_ove_paid,\
                kansanelake2,puoliso_kansanelake2,tyoelake_maksussa2,puoliso_tyoelake_maksussa2,next_wage2,\
                savings2,puoliso_savings2\
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
                                savings,puoliso_savings,
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
                                savings2,puoliso_savings2,
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
                    savings,puoliso_savings,
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
                    savings2,puoliso_savings2,
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
        if not math.isclose(puoliso_kansanelake,puoliso_kansanelake2):  
            print('puoliso_kansanelake: {} vs {}'.format(puoliso_kansanelake,puoliso_kansanelake2))
        if not math.isclose(tyoelake_maksussa,tyoelake_maksussa2):  
            print('tyoelake_maksussa: {} vs {}'.format(tyoelake_maksussa,tyoelake_maksussa2))
        if not math.isclose(puoliso_tyoelake_maksussa,puoliso_tyoelake_maksussa2):  
            print('puoliso_tyoelake_maksussa: {} vs {}'.format(puoliso_tyoelake_maksussa,puoliso_tyoelake_maksussa2))
        if not math.isclose(savings,savings2):  
            print('savings: {} vs {}'.format(savings,savings2))
        if not math.isclose(puoliso_savings,puoliso_savings2):  
            print('puoliso_savings2: {} vs {}'.format(puoliso_savings,puoliso_savings2))
    
    def check_state_vec(self,vec1,vec2):
        emp2,g2,pension2,old_wage2,age2,time_in_state2,paid_pension2,pink2,toe2,toekesto2,\
            tyohist2,used_unemp_benefit2,wage_reduction2,unemp_after_ra2,\
            unempwage2,unempwage_basis2,prefnoise2,\
            children_under3_2,children_under7_2,children_under18_2,unemp_benefit_left2,\
            alkanut_ansiosidonnainen2,toe58_2,ove_paid_2,jasen_2,p2,p2_tila,p2_g,p2_old_wage,p2_pension,\
            p2_wage_reduction,p2_paid_pension,p2_next_wage,p2_used_unemp_benefit,p2_unemp_benefit_left,\
            p2_unemp_after_ra,p2_unempwage,p2_unempwage_basis,p2_alkanut_ansiosidonnainen,p2_toe58,p2_toe,\
            p2_toekesto,p2_tyoura,p2_time_in_state,p2_pinkslip,p2_ove_paid,\
            kansanelake2,p2_kansanelake,tyoelake_maksussa2,p2_tyoelake_maksussa,next_wage2,savings2,puoliso_savings2\
            =self.state_decode(vec2)

        emp,g,pension,old_wage,age,time_in_state,paid_pension,pink,toe,toekesto,\
            tyohist,used_unemp_benefit,wage_reduction,unemp_after_ra,\
            unempwage,unempwage_basis,prefnoise,\
            children_under3,children_under7,children_under18,unemp_benefit_left,\
            alkanut_ansiosidonnainen,toe58,ove_paid,jasen,puoliso,p_tila,p_g,p_old_wage,p_pension,\
            p_wage_reduction,p_paid_pension,p_next_wage,p_used_unemp_benefit,p_unemp_benefit_left,\
            p_unemp_after_ra,p_unempwage,p_unempwage_basis,p_alkanut_ansiosidonnainen,p_toe58,p_toe,\
            p_toekesto,p_tyoura,p_time_in_state,p_pinkslip,p_ove_paid,\
            kansanelake,p_kansanelake,tyoelake_maksussa,p_tyoelake_maksussa,next_wage,savings,puoliso_savings\
            =self.state_decode(vec1)
    
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
        if not jasen==jasen_2:  
            print('jasen: {} vs {}'.format(jasen,jasen2))
        if not puoliso==p2:  
            print('puoliso: {} vs {}'.format(puoliso,p2))
        if not p_tila==p2_tila:  
            print('p_tila: {} vs {}'.format(p_tila,p2_tila))
        if not math.isclose(p_old_wage,p2_old_wage):  
            print('p_old_wage: {} vs {}'.format(p_old_wage,p2_old_wage))
        if not math.isclose(p_wage_reduction,p2_wage_reduction):  
            print('p_wage_reduction: {} vs {}'.format(p_wage_reduction,p2_wage_reduction))
        if not math.isclose(p_paid_pension,p2_paid_pension):  
            print('p_paid_pension: {} vs {}'.format(p_paid_pension,p2_paid_pension))
        if not math.isclose(p_next_wage,p2_next_wage):  
            print('p_next_wage: {} vs {}'.format(p_next_wage,p2_next_wage))
        if not math.isclose(p_used_unemp_benefit,p2_used_unemp_benefit):  
            print('p_next_wage: {} vs {}'.format(p_used_unemp_benefit,p2_used_unemp_benefit))
        if not math.isclose(p_unemp_benefit_left,p2_unemp_benefit_left):  
            print('p_unemp_benefit_left: {} vs {}'.format(p_unemp_benefit_left,p2_unemp_benefit_left))
        if not math.isclose(p_unemp_after_ra,p2_unemp_after_ra):  
            print('p_unemp_after_ra: {} vs {}'.format(p_unemp_after_ra,p2_unemp_after_ra))
        if not math.isclose(p_unempwage,p2_unempwage):  
            print('p_unempwage: {} vs {}'.format(p_unempwage,p2_unempwage))
        if not math.isclose(p_unempwage_basis,p2_unempwage_basis):  
            print('p_unempwage_basis: {} vs {}'.format(p_unempwage_basis,p2_unempwage_basis))
        if not math.isclose(p_alkanut_ansiosidonnainen,p2_alkanut_ansiosidonnainen):  
            print('p_alkanut_ansiosidonnainen: {} vs {}'.format(p_alkanut_ansiosidonnainen,p2_alkanut_ansiosidonnainen))
        if not math.isclose(p_toe58,p2_toe58):  
            print('p_toe58: {} vs {}'.format(p_toe58,p2_toe58))
        if not math.isclose(p_toe,p2_toe):  
            print('p_toe: {} vs {}'.format(p_toe,p2_toe))
        if not math.isclose(p_toekesto,p2_toekesto):  
            print('p_toekesto: {} vs {}'.format(p_toekesto,p2_toekesto))
        if not math.isclose(p_tyoura,p2_tyoura):  
            print('p_tyoura: {} vs {}'.format(p_tyoura,p2_tyoura))
        if not math.isclose(p_time_in_state,p2_time_in_state):  
            print('p_time_in_state: {} vs {}'.format(p_time_in_state,p2_time_in_state))
        if not math.isclose(p_pinkslip,p2_pinkslip):  
            print('p_pinkslip: {} vs {}'.format(p_pinkslip,p2_pinkslip))
        if not math.isclose(p_ove_paid,p2_ove_paid):  
            print('p_ove_paid: {} vs {}'.format(p_ove_paid,p2_ove_paid))
        if not math.isclose(kansanelake,kansanelake2):  
            print('kansanelake: {} vs {}'.format(kansanelake2,kansanelake2))
        if not math.isclose(p_kansanelake,p2_kansanelake):  
            print('p_kansanelake: {} vs {}'.format(p_kansanelake,p2_kansanelake))
        if not math.isclose(tyoelake_maksussa,tyoelake_maksussa2):  
            print('tyoelake_maksussa: {} vs {}'.format(tyoelake_maksussa,tyoelake_maksussa2))
        if not math.isclose(p_tyoelake_maksussa,p2_tyoelake_maksussa):  
            print('p_tyoelake_maksussa: {} vs {}'.format(p_tyoelake_maksussa,p2_tyoelake_maksussa))    
        if not math.isclose(savings,savings2):  
            print('savings: {} vs {}'.format(savings,savings2))
        if not math.isclose(puoliso_savings,puoliso_savings2):  
            print('puoliso_savings2: {} vs {}'.format(puoliso_savings,puoliso_savings2))

    def reset(self,init=None):
        '''
        Open AI-interfacen mukainen reset-funktio, joka nollaa laskennan alkutilaan
        '''
        self.init_state()
        self.steps_beyond_done = None

        self.returns=self.stock_returns()

        if self.mortplot:
            self.plotdebug=False
        
        if self.plotdebug:
            self.render()

        return np.array(self.state)
    
    def get_initial_state(self,puoliso : int,is_spouse=False,g=-1):    
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
#                 cc_salary=np.random.uniform(low=1_000,high=100_000)
#                 toe=random.choices(np.array([0,0.25,0.5,0.75,1.0,1.5,2.0,2.5],dtype=float),
#                     weights=[0.3,0.1,0.1,0.1,0.1,0.1,0.1,0.1])[0]
#                 tyohist=random.choices(np.array([0,0.25,0.5,0.75,1.0,1.5,2.0,2.5],dtype=float),
#                     weights=[0.3,0.1,0.1,0.1,0.1,0.1,0.1,0.1])[0]
#                 reset_exp=True
        
        if is_spouse:
            #self.spousesalary=self.compute_salary_v4(group=group,initial_salary=initial_salary,spouse=is_spouse)
            self.wages_spouse.compute_salary(group=group,initial_salary=initial_salary)
        else:
            #self.salary=self.compute_salary_v4(group=group,initial_salary=initial_salary,spouse=is_spouse)
            self.wages_main.compute_salary(group=group,initial_salary=initial_salary)
            
        if not reset_exp:
            if employment_state==0:
                wage_reduction=np.random.uniform(low=0.05,high=0.45)
            elif employment_state==13:
                wage_reduction=np.random.uniform(low=0.15,high=0.60) # 20-70
            elif employment_state==1:
                wage_reduction=np.random.uniform(low=0.0,high=0.10) # 20-70
            elif employment_state==10:
                wage_reduction=np.random.uniform(low=0.0,high=0.10) # 20-70
            elif employment_state==12:
                wage_reduction=np.random.uniform(low=0.10,high=0.40)
            elif employment_state==11:
                wage_reduction=np.random.uniform(low=0.20,high=0.60) # 15-50
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
                kansanelake = self.ben.laske_kansanelake(age,tyoelake_maksussa/12,1-puoliso,disability=True)*12 # ben-modulissa palkat kk-tasolla
                pension=0
            elif employment_state==2:
                wage5y=next_wage
                tyoelake_maksussa=pension
                # takuueläke voidaan huomioida jo tässä
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
                kansanelake = self.ben.laske_kansanelake(age,tyoelake_maksussa/12,1-puoliso,disability=True)*12 # ben-modulissa palkat kk-tasolla
                pension=0
            elif employment_state==2:
                wage5y=next_wage
                tyoelake_maksussa=np.random.uniform(low=0.0,high=40_000)
                # takuueläke voidaan huomioida jo tässä
                kansanelake = self.ben.laske_kansanelake(age,tyoelake_maksussa/12,1-puoliso)*12 # ben-modulissa palkat kk-tasolla
                pension=0
           
        if employment_state in set([1,10]):
            unempwage=old_wage
            
        unemp_benefit_left=self.comp_unempdays_left(used_unemp_benefit,tyohist,age,toe,employment_state,alkanut_ansiosidonnainen,toe58,toe)    

        if not is_spouse:
            if employment_state in set([0,4]):
                self.set_kassanjasenyys(1)
                kassanjasenyys=1
            else:
                kassanjasenyys=self.get_kassanjasenyys()
        else:
            kassanjasenyys=self.get_kassanjasenyys()
                    
        return employment_state,group,pension,old_wage,age,time_in_state,paid_pension,pink,toe,toekesto,tyohist,next_wage,\
            used_unemp_benefit,wage_reduction,unemp_after_ra,unempwage,unempwage_basis,\
            children_under3,children_under7,children_under18,unemp_benefit_left,alkanut_ansiosidonnainen,toe58,\
            ove_paid,kassanjasenyys,kansanelake,tyoelake_maksussa

    
    def init_state(self):
    
        if self.randomness:
            rn = np.random.uniform(size=1)
        else:
            rn = 0
            
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
            puoliso_ove_paid,_,puoliso_kansanelake,puoliso_tyoelake_maksussa\
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
            
        if self.include_wealth:
            savings=self.rates.initial_wealth(g)
            puoliso_savings=self.rates.initial_wealth(g)
        else:
            savings,puoliso_savings=0,0
            
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
                                       savings,puoliso_savings,
                                       prefnoise)

    def render(self,mode='human',close=False,done=False,reward=None,netto=None,render_omat=True,render_puoliso=False,benq=None,
            netto_omat=None,netto_puoliso=None,savact=None,p_savact=None,act=None,p_act=None):
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
            next_wage,savings,p_savings\
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

        if puoliso:
            onpuoliso='+'
        else:
            onpuoliso=''
            
        if emp==15:
            m='*'
        else:
            m=''
            
        kappa=self.log_get_kappa(age,g,emp,pink)

        out=f'{m}s{onpuoliso} {emp:2d} g {g:d} a {age:.2f} w {wage:.0f} nw {next_wage:.0f} red {wage_red:.2f} tis {time_in_state:.2f}'+\
            f' pen {pension:.0f} p_e {tyoelake_maksussa:.0f} p_k {kansanelake:.0f} ueb {used_unemp_benefit:.2f}'+\
            f' toe {toe:.2f}{kassassa} tk{toekesto:.2f} ura {tyohist:.2f} uew {unempwage:.0f}'+\
            f' (b {unempwage_basis:.0f}) ul {unemp_left:.2f} aa {oikeus:.0f} 58 {toe58:.0f} ove {ove_paid:.0f}'+\
            f' pink {pink:d} c{c3:.0f}/{c7:.0f}/{c18:.0f} k {kappa:.2f}'+\
            f' sav {savings:.0f}'
            
        if savact is not None:
            out+=f' sact {savact:.2f}'
            
        if reward is not None:
            out+=f' r {reward:.4f}'
        if netto_omat is not None:
            out+=f' n {netto_omat:.0f}'

        if render_omat:
            print(out)
            
        if puoliso_tila==15:
            m='*'
        else:
            m=''
            
        kappa=self.log_get_kappa(age,spouse_g,puoliso_tila,puoliso_pink)

        puoliso=f'{m}ps{onpuoliso} {puoliso_tila:d} g {spouse_g:d} a {age:.2f} w {puoliso_old_wage:.0f} nw {puoliso_next_wage:.0f} red {puoliso_wage_reduction:.2f} tis {puoliso_time_in_state:.2f}'+\
                f' pen {puoliso_pension:.0f} paid_e {puoliso_tyoelake_maksussa:.0f} paid_k {puoliso_kansanelake:.0f} ueb {puoliso_used_unemp_benefit:.2f}'+\
                f' toe {puoliso_toe:.2f}{kassassa} tk {puoliso_toekesto:.2f} ura {puoliso_tyohist:.2f} uew {puoliso_unempwage:.0f} (b {puoliso_unempwage_basis:.0f})'+\
                f' uleft {puoliso_unemp_benefit_left:.2f} 58 {puoliso_toe58:d} aa {puoliso_alkanut_ansiosidonnainen:d}'+\
                f' pink {puoliso_pink:d} k {kappa:.2f}'+\
                f' sav {p_savings:.0f}'

        if savact is not None:
            puoliso+=f' sact {p_savact:.2f}'

                
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
        Ei käytässä
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
        toe_min=0-28/12*0.5 # työssäoloehto
        toe_max=28/12-28/12*0.5 # työssäoloehto
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
        savings_min=-1
        savings_max=30
        
        low=[]
        high=[]
        low_mid = [
            pension_min, # vastainen eläke
            wage_min, # old_wage
            age_min, # age
            tis_min, # time_in_state
            paid_pension_min, # maksussa oleva eläke
            pink_min, # pink
            toe_min, # työssäoloehto
            thist_min, # tyhistoria
            tr_min, # aikaa vanhuuseläkeikään
            left_min, # unemp_left
            wage_min, # next_wage
            ben_min, # used_unemp_ben
            wr_min, # wage_reduction
            unra_min, # unemp after ret.age
            wage_min, # unempwage
            wage_min, # unempwage_basis
            state_min, # retaged
            state_min, # alkanut_ansiosidonnainen
            child_min, # children under 3
            child_min, # children under 7
            child_min, # children under 18
            state_min, # toe58
            pension_min,  # ove määrä
            state_min, # ove aged
            state_min, # kassan jäsen 0/1
            toe_min, # uuden toen laskenta
            state_min # on puolisoa tai ei
            ]
        
        low_end=[wage_min, # puoliso old wage
            paid_pension_min, # puoliso pension
            wr_min, # puoliso wr
            paid_pension_min, # työeläke maksussa
            wage_min, # puoliso next wage
            ben_min,
            ben_min,
            unra_min,
            wage_min,
            wage_min,
            state_min, # puoliso alkanut ansiosidonnainen
            state_min, # puoliso toe58
            toe_min,
            toe_min,
            thist_min,
            tis_min,
            state_min,
            state_min,
            paid_pension_min,
            paid_pension_min
            ]            
            
        savings_min = [
            savings_min, # sijoitukset
            savings_min # sijoitukset
            ]            
            
        high_mid = [
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
            state_max, # retaged
            state_max, # alkanut_ansiosidonnainen
            child_max,
            child_max,
            child_max,
            state_max, # toe58
            pension_max,  # ove määrä
            state_max, # ove aged
            state_max, # kassan jäsen 0/1
            toe_max, # uuden toen laskenta
            state_max # on puolisoa tai ei
            ]

        high_end=[wage_max, # puoliso old wage
            paid_pension_max, # puoliso pension
            wr_max, # puoliso wr
            paid_pension_max, # työeläke maksussa
            wage_max, # puoliso next wage
            ben_max,
            ben_max,
            unra_max,
            wage_max,
            wage_max,
            state_max, # puoliso alkanut ansiosidonnainen
            state_max, # puoliso toe58
            toe_max,
            toe_max,
            thist_max,
            tis_max,
            state_max,
            state_max,
            paid_pension_max,
            paid_pension_max
            ]
        savings_max = [
            savings_max, # sijoitukset
            savings_max # sijoitukset
            ]            
                    
        for k in range(self.n_empl):
            low.append(state_min)
            high.append(state_max)
        for k in range(self.n_groups):
            low.append(group_min)
            high.append(group_max)

        low.extend(low_mid)
        high.extend(high_mid)
            
        for k in range(self.n_empl):
            low.append(state_min)
            high.append(state_max)

        low.extend(low_end)
        high.extend(high_end)

        if self.include_investment:
            low.extend(savings_min)
            high.extend(savings_max)
            
        #if self.include_mort: # if mortality is included, add one more state
        #      low.insert(0,state_min)
        #      high.insert(0,state_max)
              
        if self.include_preferencenoise:
            low.append(pref_min)
            high.append(pref_max)
                
        self.low=np.array(low)
        self.high=np.array(high)

    def explain(self):
        '''
        Tulosta laskennan parametrit
        '''
        r_t=np.mean(self.returns)-1.0
        r_s=np.std(self.returns)
        r_ann=(1+self.r_mean)**(1.0/self.timestep)-1
        gamma_ann=self.gamma**(1.0/self.timestep)
        print(f'Parameters of lifecycle:\ntimestep {self.timestep}\ngamma {self.gamma:.6f} ({gamma_ann:.4f} per anno)\nmin_age {self.min_age}\nmax_age {self.max_age}\nmin_retirementage {self.min_retirementage}\n'+\
            f'max_retirementage {self.max_retirementage}\nansiopvraha_kesto300 {self.ansiopvraha_kesto300}\nansiopvraha_kesto400 {self.ansiopvraha_kesto400}\nansiopvraha_kesto500 {self.ansiopvraha_kesto500}\nansiopvraha_toe {self.ansiopvraha_toe}\n'+\
            f'perustulo {self.perustulo}\nkarenssi_kesto {self.karenssi_kesto}\nmortality {self.include_mort}\nrandomness {self.randomness}\n'+\
            f'include_putki {self.include_putki}\ninclude_pinkslip {self.include_pinkslip}\n'+\
            f'perustulo {self.perustulo}\nsigma_reduction {self.use_sigma_reduction}\nplotdebug {self.plotdebug}\n'+\
            f'additional_tyel_premium {self.additional_tyel_premium}\nscale_tyel_accrual {self.scale_tyel_accrual}\ninclude_ove {self.include_ove}\n'+\
            f'unemp_limit_reemp {self.unemp_limit_reemp}\n'+\
            f'investment {self.include_investment}\n'+\
            f'random_returns {self.random_returns}\n'+\
            f'r_mean {self.r_mean:.4f} (ann. {r_ann:.4f})\nr_std {self.r_std:.4f}\n'+\
            f'r_t {r_t:.4f}\nr_s {r_s:.4f}\n'+\
            f'wealth {self.include_wealth}')

    def unempright_left(self,emp : int,tis : float,bu : float,ika : float,tyohistoria : float):
        '''
        Tilastointia varten lasketaan jäljellä olevat ansiosidonnaiset työttämyysturvapäivät
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
        self.kassanjasenyys_joinrate,self.kassanjasenyys_rate=self.rates.get_kassanjasenyys_rate()
            
    def init_infostate(self,lapsia=0,lasten_iat=np.zeros(15),lapsia_paivakodissa=0,age=18,spouse=False):
        '''
        Alustaa infostate-dictorionaryn
        Siihen talletetaan tieto aiemmista tiloista, joiden avulla lasketaan statistiikkoja
        '''
        self.infostate={}
        states,latest,enimaika,palkka,voc_unempbasis,member=self.infostate_vocabulary(is_spouse=False)

        self.infostate[states]=np.zeros(self.n_time)-1
        self.infostate[palkka]=np.zeros(self.n_time)-1
        self.infostate[voc_unempbasis]=np.zeros(self.n_time)-1
        self.infostate[member]=np.zeros(self.n_time,dtype=np.int8)
        self.infostate[latest]=0
        self.infostate['children_n']=0
        self.infostate['children_date']=np.zeros(15)
        self.infostate[enimaika]=0
        
        states,latest,enimaika,palkka,voc_unempbasis,member=self.infostate_vocabulary(is_spouse=True)
        self.infostate[states]=np.zeros(self.n_time)-1
        self.infostate[palkka]=np.zeros(self.n_time)-1
        self.infostate[voc_unempbasis]=np.zeros(self.n_time)-1
        self.infostate[member]=np.zeros(self.n_time,dtype=np.int8)
        self.infostate[latest]=0
        self.infostate[enimaika]=0
        sattuma = np.random.uniform(size=1)
        t=int((age-self.min_age)/self.timestep)
        
        if sattuma[0]<self.kassanjasenyys_rate[t]:
            self.set_kassanjasenyys(1) #self.infostate['kassanjasen']=1
        else:
            self.set_kassanjasenyys(0) # self.infostate['kassanjasen']=0
        
    def infostate_add_child(self,age : float):
        if self.infostate['children_n']<14:
            self.infostate['children_date'][self.infostate['children_n']]=age
            self.infostate['children_n']=self.infostate['children_n']+1
            
    def infostate_set_enimmaisaika(self,age : float,is_spouse=False):
        states,latest,enimaika,palkka,voc_unempbasis,member=self.infostate_vocabulary(is_spouse=is_spouse)
        t=int((age-self.min_age)/self.timestep)
        self.infostate[enimaika]=t
        
    def update_infostate(self,t : int,state : int,wage : float,unempbasis : float,is_spouse=False):
        states,latest,enimaika,palkka,voc_unempbasis,member=self.infostate_vocabulary(is_spouse=is_spouse)
    
        self.infostate[states][t]=state
        self.infostate[latest]=int(t)
        self.infostate[voc_unempbasis][t]=unempbasis
        self.infostate[member][t]=self.infostate['kassanjasen']
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

    def set_kassanjasenyys(self,value : int):
        self.infostate['kassanjasen']=value
        #print('jasen ',value)

    def infostate_kassanjasenyys_update(self,age : float):
        if self.infostate['kassanjasen']<1:
            sattuma = np.random.uniform(size=1)
            #print(sattuma[0],self.kassanjasenyys_joinrate[self.map_age(age)])
            #print(self.kassanjasenyys_joinrate[self.map_age(age)])
            if sattuma[0]<self.kassanjasenyys_joinrate[self.map_age(age)] and self.randomness:
                #print('join')
                self.set_kassanjasenyys(1)
        
    def comp_toe_wage_nykytila(self,is_spouse=False):
        states,latest,enimaika,palkka,voc_unempbasis,member=self.infostate_vocabulary(is_spouse=is_spouse)
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
                        if self.infostate[member][t2]<1:
                            nt=nt+1
                        elif emps in family_states:
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
                        if self.infostate[member][t2]<1:
                            nt=nt+1
                        elif emps in family_states:
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
        
    def comp_toe_wage_porrastus(self,is_spouse=False):
        states,latest,enimaika,palkka,voc_unempbasis,member=self.infostate_vocabulary(is_spouse=is_spouse)
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
                if self.infostate[member][t2]<1:
                    nt=nt+1
                elif emps in family_states:
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
                if self.infostate[member][t2]<1:
                    nt=nt+1
                elif emps in family_states:
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
        
    def comp_infostats(self,age : float,is_spouse=False):
        # laske työssäoloehto tarkasti
        # laske työttämyysturvaan vaikuttavat lasten määrät
        
        if not is_spouse:
            self.infostate_kassanjasenyys_update(age)
        
        toes,toekesto,wage=self.comp_toe_wage(is_spouse=is_spouse)
                
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

    def infostate_comp_5y_ave_wage(self,is_spouse=False):
        emp_states={1,10}
        unemp_states={0,4}
        family_states={5,6}
        
        states,latest,enimaika,voc_wage,voc_unempbasis,member=self.infostate_vocabulary(is_spouse=is_spouse)
        #print(states,latest,enimaika,voc_wage,voc_unempbasis)
    
        lstate=int(self.infostate[latest])
        n=int(np.ceil(5/self.timestep))
        wage=0
        truewage=0
        for x in range(lstate-n,lstate):
            if x<0:
                pass
            else:
                empstate=self.infostate[states][x]
                if empstate in emp_states:
                    value=self.infostate[voc_wage][x]
                    w=value
                elif empstate in family_states:
                    value=self.infostate[voc_wage][x]
                    w=0
                elif empstate in unemp_states:
                    value=self.infostate[voc_unempbasis][x]
                    w=0
                elif empstate in set([7,12,13]):
                    value=self.disabbasis_tmtuki
                    w=0
                else:
                    value=0
                    w=0
                    
                if self.plottkdebug:
                    print(f'{empstate}: {value:.2f}')
                    
                wage += value*self.timestep/5
                truewage += w*self.timestep

        return wage,truewage

    def infostate_can_have_children(self,age : float):
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

    def infostate_vocabulary(self,is_spouse=False):
        if is_spouse:
            states='spouse_states'
            latest='spouse_latest'
            enimaika='spouse_enimmaisaika_alkaa'
            palkka='spouse_wage'
            unempbasis='spouse_unempbasis'
            jasen='spouse_unempmember'
        else:
            states='states'
            latest='latest'
            enimaika='enimmaisaika_alkaa'
            palkka='wage'
            unempbasis='unempbasis'
            jasen='unempmember'
            
        return states,latest,enimaika,palkka,unempbasis,jasen

    def infostate_check_aareset(self,age,is_spouse=False):
        states,latest,enimaika,palkka,voc_unempbasis,member=self.infostate_vocabulary(is_spouse=is_spouse)
    
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

    def comp_oldtoe(self,printti=False,is_spouse=False):
        '''
        laske työttämyysjaksoa edeltävä työssäoloehto tarkasti
        '''
        toes=0
        n_toe=int(np.floor(self.max_toe/self.timestep))
        emp_states={1,10}
        unemp_states={0,4,13}
        family_states={5,6,7}
        ret_states={2,3,8,9}
        wage=0
        
        states,latest,enimaika,palkka,voc_unempbasis,member=self.infostate_vocabulary(is_spouse=is_spouse)
        
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

    #def check_toe58_v1(self,age,toe,tyoura,toe58):
    #    if age<self.minage_500:
    #        return 0
    #    elif self.tyossaoloehto(toe,tyoura,age) and tyoura>=self.tyohistoria_vaatimus500:
    #        return 1
    #    else:
    #        return 0

    def check_toe58(self,age : float,toe : float,tyoura : float,toe58 : int,is_spouse=False):
        '''
        laske työttämyysjaksoa edeltävä työssäoloehto tarkasti
        '''
        states,latest,enimaika,palkka,voc_unempbasis,member=self.infostate_vocabulary(is_spouse=is_spouse)
        
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

        
    def swap_spouses(self,vec):
        employment_status,g,pension,wage,age,time_in_state,paid_pension,pinkslip,toe,\
            toekesto,tyoura,used_unemp_benefit,wage_reduction,unemp_after_ra,unempwage,\
            unempwage_basis,prefnoise,children_under3,children_under7,children_under18,\
            unemp_left,alkanut_ansiosidonnainen,toe58,ove_paid,jasen,\
            puoliso,puoliso_tila,spouse_g,puoliso_wage,puoliso_pension,puoliso_wage_reduction,\
            puoliso_paid_pension,puoliso_next_wage,puoliso_used_unemp_benefit,\
            puoliso_unemp_benefit_left,puoliso_unemp_after_ra,puoliso_unempwage,\
            puoliso_unempwage_basis,puoliso_alkanut_ansiosidonnainen,puoliso_toe58,\
            puoliso_toe,puoliso_toekesto,puoliso_tyoura,puoliso_time_in_state,puoliso_pinkslip,\
            puoliso_ove_paid,kansanelake,puoliso_kansanelake,tyoelake_maksussa,\
            puoliso_tyoelake_maksussa,next_wage,\
            savings,puoliso_saving\
                =self.state_decode(vec)

        return self.state_encode(puoliso_tila,spouse_g,puoliso_pension,puoliso_wage,age,puoliso_time_in_state,
            puoliso_tyoelake_maksussa,puoliso_pinkslip,puoliso_toe,puoliso_toekesto,puoliso_tyoura,puoliso_next_wage,puoliso_used_unemp_benefit,
            puoliso_wage_reduction,puoliso_unemp_after_ra,puoliso_unempwage,puoliso_unempwage_basis,
            children_under3,children_under7,children_under18,
            puoliso_unemp_benefit_left,puoliso_alkanut_ansiosidonnainen,puoliso_toe58,puoliso_ove_paid,jasen,
            puoliso,
            employment_status,wage,pension,
            wage_reduction,tyoelake_maksussa,next_wage,
            used_unemp_benefit,unemp_left,
            unemp_after_ra,unempwage,unempwage_basis,
            alkanut_ansiosidonnainen,toe58,
            toe,toekesto,tyoura,time_in_state,
            pinkslip,ove_paid,puoliso_kansanelake,kansanelake,
            puoliso_saving,savings,
            prefnoise)
                                
    def test_swap(self,minage=18,maxage=70):
        self.reset_exploration_go=True
        self.reset_exploration_ratio=0.4
        self.randomness=False
        self.silent=True
        
        for k in range(100):
            self.reset()
            self.steps_beyond_done=None
            vec0=self.random_init_state(minage=minage,maxage=maxage)
            self.state=vec0
            action0=random.randint(0,4)
            action1=random.randint(0,4)
            a=np.array([action0,action1])
            _,r0,_,q0=self.step(a)
            
            vec1=self.swap_spouses(vec0)
            a=np.array([action1,action0])

            self.state=vec1
            self.steps_beyond_done=None
            _,r1,_,q1=self.step(a)
            
            if not math.isclose(r0,r1):
                self.render()
                print('!!!! ERROR:')
            print(r0,r1)
            vec2=self.swap_spouses(vec1)
            self.check_state_vec(vec0,vec2)
            self.pretty_print(q0,q1)
            
            print('\n----------')

    def get_minimal(self):
        return False

    def stock_returns(self):
        if self.random_returns:
            return 1.0+np.random.randn(self.n_time)*self.r_std+self.r_mean
        else:
            return 1.0+np.ones((self.n_time,1))*self.r_mean

    def map_save_action(self,sav_act : int):
        if sav_act>0:
            s=(sav_act-self.mid_sav_act)/100*2
        else:
            s=(sav_act-self.mid_sav_act)/self.mid_sav_act
        
        return s
        
    def update_savings(self,netto : float,savings : float,sav_action : int,empstate : int,age : float):
        '''
        netto income is at annual level, not actual money earned during timestep
        => savings are at similarly scale level. to get real money, multiply by self.timestep
        '''
        t=self.map_age(age)
        interest = savings*(self.returns[t,0]-1.0)
        savings += interest

        osavings=savings
        onetto=netto        
        r=(self.returns[t,0]-1.0)

        if savings<0:
            payback = -0.20*savings # lyhennys vähintään  20 % vuodessa
        else:
            payback = 0

        if sav_action>0:
            save=min(0.9*netto,payback+sav_action*netto)
            netto -= save
            savings += save
            mod_sav_act=sav_action
            #if savings>100_000:
            #    print(f'!!! save {save} savings {savings}')
        elif sav_action<0:
            save=min(0.9*netto,sav_action*netto+payback)
    
            if savings+save<self.max_debt and save<0:
                save=max(save,self.max_debt-savings)
        
            if empstate in set([2,3,8,9]):
                if savings+save>0:
                    netto += -save
                    savings += save
                    mod_sav_act = sav_action
                else:
                    save=min(0.9*netto,payback)
                    netto -= save
                    savings += save
                    mod_sav_act = 0
            else:
                if savings+save>0:
                    netto += -save
                    savings += save
                    mod_sav_act = sav_action
                elif savings+save>self.max_debt and age<self.min_retirementage:
                    netto += -save
                    savings += save
                    mod_sav_act = sav_action
                else:
                    save=min(0.9*netto,payback)
                    netto -= save
                    savings += save
                    mod_sav_act = 0
        else:
            save=min(0.9*netto,payback)
            netto-=save
            savings+=save
            mod_sav_act=0
            
        #print(f'{age}: sact {sav_action} ({mod_sav_act}) save {save} netto {onetto} -> {netto} savings {osavings} -> {savings} interest {interest} ({r:.3f})')
                    
        return netto,savings,mod_sav_act