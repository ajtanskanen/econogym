"""

    unemployment_v9
    

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

    v6 modifies v5 so that the length of life is known at birth
    v7 adds time until disability, time until marriage/divorce, time until having a child as a state variable
    - this should improve fit
    v8 revised fit
    v9 
    - alimony slightly improved
    - child support fixed
    - refitted reemp probs + career impact

    fully random events still include:
    - next salary
    - transition from sick leave to disability
    - transition to studying and outside the labor force

    genders are here fixed: main is a man and spouse is a woman (to be improved later)
    - this enables a smaller state space
    - more flexible groups available

    FIXME:
    - lapsia syntyy samallalailla kaikille, pitäisi painottaa jo lapsia saaneita perheitä?

"""

import math
import random
import gym
from gym import spaces, logger, utils, error
from gym.utils import seeding
import numpy as np
import fin_benefits
from . rates import Rates
from scipy.interpolate import interp1d
from . util import compare_q_print,crosscheck_print,test_var
from . wages_v1 import Wages_v1
from . state_v8 import Statevector_v8
#from . infostate_v5 import Infostate

# class StayDict(dict):
#     '''
#     Apuluokka, jonka avulla tehdään virheenkorjausta
#     '''
#     def __missing__(self, key):
#         return 'Unknown state '+key


class UnemploymentEnv_v9(gym.Env):
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
       5x    Preferenssikohina, jos halutaan

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
        4    Ove etc

    Reward:
        Reward is the sum of wage and benefit for every step taken, including the termination step

    Starting State:
        Starting state in unemployed at age 18

    Step:
        Each step corresponds to three months in time (0.25 years)

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

        #self.reward_range = [0,100]

        #print('init start v5')

        self.setup_default_params()
        gamma=0.92

        # No render mode set
        self.render_mode = None

        # No render mode set
        self.reward_range = (0,10)

        # sets parameters based on kwargs
        self.set_parameters(**kwargs)

        self.gamma=gamma**self.timestep # discounting
        self.palkkakerroin = (0.8*1.0 + 0.2*1.0/self.reaalinen_palkkojenkasvu)**self.timestep
        self.elakeindeksi = (0.2*1.0 + 0.8*1.0/self.reaalinen_palkkojenkasvu)**self.timestep
        #self.kelaindeksi=(1.0/self.reaalinen_palkkojenkasvu)**self.timestep
        #self.kelaindeksi = self.elakeindeksi # oletetaan, että KELA-indeksi ei jää jälkeen eläkeindeksistä (PTS: 50-50-indeksi)
        self.kelaindeksi=(0.5*1.0 + 0.5*1.0/self.reaalinen_palkkojenkasvu)**self.timestep # oletetaan, että KELA-indeksi = PTS: 50-50-indeksi
        self.n_age = self.max_age-self.min_age+1
        self.inv_timestep=int(np.round(1/self.timestep)) # pitäisi olla kokonaisluku
        self.n_time = int(np.round((self.n_age-1)*self.inv_timestep))+1

        # karttumaprosentit
        if self.scale_tyel_accrual:
            acc_scaling=1+self.scale_additional_tyel_accrual
        else:
            acc_scaling=1

        # scaling for reward, corresponds to temperature in softmax
        self.temperature = 0.10

        self.acc=0.015*self.timestep*acc_scaling
        self.acc_sv=0.62 # sairauspäiväraja, ei skaalata
        self.acc_over_52=0.019*self.timestep*acc_scaling
        #self.acc_over_52 = self.acc
        self.acc_family=1.15*self.acc
        self.acc_family_over_52=1.15*self.acc_over_52
        self.acc_unemp=0.75*self.acc
        self.acc_unemp_over_52=0.75*self.acc_over_52
        #self.min_family_accwage=12*757

        deterministic=True
        if self.include_mort:
            self.log_utility_mort_ove_det_params()
        else:
            self.log_utility_mort_ove_det_params()

        self.n_age = self.max_age-self.min_age+1

        if not self.train: # get stats right
            self.mortstop=False

        if self.train: # get stats right
            self.include_emtr=False

        if not self.randomness:
            self.include_npv_mort=False

        if not self.silent:
            if self.mortstop:
                print('Mortality included, stopped')
            else:
                print('Mortality included, not stopped')

        self.n_empl=16 # state of employment, 0,1,2,3,4
        self.n_empl = self.n_empl

        self.set_year(self.year)

        if self.include_ove:
            self.n_actions=5
            self.n_spouse_actions = self.n_actions
        else:
            self.n_actions=5
            self.n_spouse_actions = self.n_actions

        self.include_savings=False
        if self.include_savings:
            self.n_savings=41 # -5,-4,-3,-2,-1,0,1,2,3,4,5
            self.mid_sav_act=np.floor(self.n_savings/2)

            if self.include_parttimeactions:
                self.n_parttime_action=3
                self.action_space = spaces.MultiDiscrete([self.n_actions,self.n_spouse_actions,self.n_parttime_action,self.n_parttime_action,self.n_savings,self.n_savings])
                self.parttime_actions = self.setup_parttime_actions()
            else:
                self.n_parttime_action=3
                self.action_space = spaces.MultiDiscrete([self.n_actions,self.n_spouse_actions,self.n_savings,self.n_savings])
                self.parttime_actions = self.setup_parttime_actions(debug=True)
        else:
            if self.include_parttimeactions:
                self.n_parttime_action=3
                self.action_space = spaces.MultiDiscrete([self.n_actions,self.n_spouse_actions,self.n_parttime_action,self.n_parttime_action])
                self.parttime_actions = self.setup_parttime_actions()
            else:
                self.n_parttime_action=3
                self.action_space = spaces.MultiDiscrete([self.n_actions,self.n_spouse_actions])
                self.parttime_actions = self.setup_parttime_actions(debug=True)

        #self.setup_state_encoding()
        self.states=Statevector_v8(self.n_empl,self.n_groups_encoding,self.n_parttime_action,self.include_mort,self.min_age,self.max_age,self.include_preferencenoise,self.min_retirementage,self.min_ove_age,self.get_paid_wage,self.timestep)
        self.low,self.high = self.states.set_state_limits()

        #self.action_space = spaces.MultiDiscrete([self.n_actions,self.n_spouse_actions])
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        if self.use_sigma_reduction:
            self.update_wage_reduction = self.update_wage_reduction_sigma
        else:
            self.update_wage_reduction = self.update_wage_reduction_baseline

        #self.seed()
        self.viewer = None
        self.state = None

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
            self.states.unit_test_code_decode()
            self.test_swap(n=10)

        #print('init done v5')

    def map_save_action(self,sav_act: int) -> float:
        if sav_act>0:
            s=(sav_act-self.mid_sav_act)/100*2
        else:
            s=(sav_act-self.mid_sav_act)/self.mid_sav_act

        return s

    def update_savings(self,age: float,netto: float,savings: float,sav_action: int,empstate: int):
        error('not fully implemented')
        t = self.map_age(age)
        interest=savings*(self.returns[t]-1.0)
        savings=savings+interest

        if savings<0:
            payback = -0.20*savings # lyhennys vähintään  20 % vuodessa
        else:
            payback = 0

        if sav_action>0:
            save = min(0.9*netto,payback+sav_action*netto)
            netto -= save
            savings += save
            mod_sav_act = sav_action
        elif sav_action<0:
            save=sav_action*netto
            if savings+save>self.max_debt and empstate!=2 and age<self.min_retirementage:
                save += payback
                save = min(save,0.9*netto)
                netto +=-save
                savings+=save
                mod_sav_act=sav_action
            elif savings>0 and empstate==2:
                save = sav_action*savings
                netto += -save
                savings+=save
                mod_sav_act=sav_action
            else:
                save=min(0.9*netto,payback)
                netto-=save
                savings+=save
                mod_sav_act=0
        else:
            save=min(0.9*netto,payback)
            netto-=save
            savings+=save
            mod_sav_act=0

        #print(f'netto {netto} savings {savings} mod_sav_act {mod_sav_act}')

        return netto,savings,mod_sav_act

    def setup_parttime_actions(self,debug: bool = False):
        #actions=np.zeros((self.n_empl,3))
        actions=np.zeros((17,3))
        if debug:
            work_actions=np.array([40/40,40/40,40/40])
            parttimework_actions=np.array([20/40,20/40,20/40])
            actions[1,:] = work_actions
            actions[8,:] = parttimework_actions
            actions[9,:] = work_actions
            actions[10,:] = parttimework_actions
        else:
            out_of_work_actions=np.array([0/40,0/40,0/40])
            actions[0,:] = out_of_work_actions
            actions[1,:] = np.array([32/40,40/40,48/40])
            actions[2,:] = out_of_work_actions
            actions[3,:] = out_of_work_actions
            actions[4,:] = out_of_work_actions
            actions[5,:] = out_of_work_actions
            actions[6,:] = out_of_work_actions
            actions[7,:] = out_of_work_actions
            actions[8,:] = np.array([8/40,16/40,24/40])
            actions[9,:] = np.array([32/40,40/40,48/40])
            actions[10,:] = np.array([8/40,16/40,24/40])
            actions[11,:] = out_of_work_actions
            actions[12,:] = np.array([0/40,8/40,16/40])
            actions[13,:] = out_of_work_actions
            actions[14,:] = out_of_work_actions
            actions[15,:] = out_of_work_actions
            actions[16,:] = np.array([0/40,8/40,16/40])

        return actions


    def setup_default_params(self):
        # käytetäänkä exp/log-muunnosta tiloissa vai ei?
        self.eps=1e-20

        self.dis_ratio=0
        self.nnn=0

        # male low income, male mid, male high, female low, female mid, female high income
        self.n_groups=6
        self.n_groups_encoding=3 # 3+3 = 6

        #self.osaaikatyonteko=0.65 # = 6kk
        self.ansiopvraha_toe=0.5 # = 6kk
        self.karenssi_kesto=0.25 #0.25 # = 3kk
        # perheistä 87 % prosenttia on käyttänyt hoitovapaata
        self.mies_jatkaa_kotihoidontuelle=0.05 # 50 % koko isyysvapaan käyttäneistä menee myös kotihoindotuelle
        self.nainen_jatkaa_kotihoidontuelle=0.8 # 50 % koko äitiysvapaan käyttäneistä menee myös kotihoindotuelle
        self.aitiysvapaa_pois=0.02 # per 3 kk
        self.tyohistoria_tyottputki=10.0 # vuotta. vähimmäistyöura putkeenpääsylle (oikeasti 5 v ed. 20 v aikana; tässä työura 35 v, josta 10 v)
        self.kht_kesto=2.0 # kotihoidontuen kesto 2 v, ei tsekata tarkkaan. (vain se että on alle 3v lapsi)
        self.tyohistoria_vaatimus=3.0 # 3 vuotta
        self.tyohistoria_vaatimus500=10.0 # p.o. 5 vuotta 20v aikana; 10v tarkoittaa, että 18-38 välin ollut töissä + 5v/20v
        self.ansiopvraha_kesto400=400 # päivää
        self.ansiopvraha_kesto300=300 # päivää
        self.ansiopvraha_kesto500=500 # päivää
        self.minage_500=58 # minimi-ikä 500 päivälle
        self.min_salary=1450*12 # 1000 # julkaistujen laskelmien jälkeen

        self.update_disab_wage_reduction=False # True # päivitä wage reduction, vaikka henkilö on työkyvyttömyyseläkkeellä

        self.map_stays={0: self.stay_unemployed,  1: self.stay_employed,         2: self.stay_retired,       3: self.stay_disabled,
                        4: self.stay_pipeline,    5: self.stay_motherleave,      6: self.stay_fatherleave,   7: self.stay_kht,
                        8: self.stay_oa_parttime, 9: self.stay_oa_fulltime,     10: self.stay_parttime,     11: self.stay_outsider,
                       12: self.stay_student,    13: self.stay_tyomarkkinatuki, 14: self.stay_svpaivaraha}


        self.timestep=0.25
        self.max_age=75
        self.min_age=18
        self.min_retirementage=63.5 #65
        self.max_retirementage=68 # 70
        self.max_unemploymentbenefitage=65
        self.max_svbenefitage=65

        self.syntymavuosi=1980
        #self.elinaikakerroin=0.925 # etk:n arvio 1962 syntyneille
        self.elinaikakerroin=0.96344 # vuoden 2017 kuolleisuutta käytettäessä myös elinaikakerroin on sieltä
        #self.elinaikakerroin = self.laske_elinaikakerroin(self.syntymavuosi)

        self.reaalinen_palkkojenkasvu=1.016

        # exploration does not really work here due to the use of history
        self.reset_exploration_go=False
        self.reset_exploration_ratio=0.4

        self.train=False

        self.include_parttimeactions=True
        self.include_spouses=True # Puolisot mukana?
        self.include_mort=True # onko kuolleisuus mukana laskelmissa
        #self.include_npv_mort=True # onko kuolleisuus eläkeaikana mukana laskelmissa
        self.include_npv_mort=False # onko kuolleisuus eläkeaikana mukana laskelmissa
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
        self.include_emtr=False

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
        self.include_ove=True
        self.mortplot=False
        self.suojasaanto_toe58=True # suojasääntö: toe-palkka ei alene 58v jälkeen

        self.unemp_limit_reemp=True # työttämästä työlliseksi tn, jos hakee täitä

        # etuuksien laskentavuosi
        self.year=2018

        # OVE-parametrit
        self.ove_ratio=0.5
        self.min_ove_age=61

        self.custom_ben=None # parametrized benefits module

        self.plotdebug=False # tulostetaanko rivi riviltä tiloja
        self.plottkdebug=False

    def set_annual_params(self,year: int) -> None:
        # arvot MAALISKUUN lukuja kuluttajahintaindeksin vuosimuutoksessa, https://stat.fi/tilasto/khi
        # nämä matchataan ansiotasoindeksin muutokseen
        #inflation_raw = np.array([1.0, # 2018
        #                          1.010248, # 1.011206, # 1.011399, # 2019
        #                          1.002917, # 1.006027, # 1.00381, # 2020
        #                          1.021936, # 1.013334622, # 1.020716, # 2021
        #                          1.071092, # 1.057976542, # 1.071655, # 2022
        #                          1.062916, # 1.079315007, # 1.043444, # 2023
        #                          1.015725, # 1.021795407, # 1.014, # 2024
        #                          1.005592, # 1.005312194, # 1.006896, # 2025
        #                          1.015, # 2026
        #                          1.020]) # 2027

        inflation_raw = self.rates.get_inflation()
        self.inflation = np.cumprod(inflation_raw)
        self.inflationfactor = self.inflation[year-2018]

        # arvot ansiotasoindeksi (y+1)Q1 vs yQ1 tasossa
        #salaryinflation_raw=np.array([1.0, # 2018
        #                              1.021210603, # 1.021243, # 2019
        #                              1.019460017, # 1.01987, # 2020
        #                              1.024215574, # 1.021613, # 2021
        #                              1.024079442, # 1.022646, # 2022
        #                              1.042374951, # 1.030886, # 2023
        #                              1.030721699, # 1.039005, # 2024
        #                              1.032, # 2025
        #                              1.038, # 2026
        #                              1.030]) # 2027

        salaryinflation_raw = self.rates.get_wageinflation()
        self.salaryinflation=np.cumprod(salaryinflation_raw)
        self.salaryinflationfactor = self.salaryinflation[year-2018]
        if year<2023:
            self.isyysvapaa_kesto=0.25 # = 3kk
            self.aitiysvapaa_kesto=0.75 # = 9kk ml vanhempainvapaa
        else:
            self.isyysvapaa_kesto=0.25 # = 3kk
            self.aitiysvapaa_kesto=0.75 # = 9kk ml vanhempainvapaa

        if self.porrasta_toe:
            self.max_toe=35/12
            if self.year<2024:
                self.min_toewage=844*12*self.salaryinflationfactor # vuoden 2019 luku tilanteessa, jossa ei tessiä
                self.min_halftoewage=422*12*self.salaryinflationfactor # vuoden 2019 luku tilanteessa, jossa ei tessiä
            else:
                self.min_toewage=930*12*self.salaryinflationfactor # vuoden 2024 luku 
                self.min_halftoewage=0.5*930*12*self.salaryinflationfactor # vuoden 2024 luku 
        else:
            self.max_toe=28/12
            self.min_toewage=1211*12*self.salaryinflationfactor # vuoden 2019 luku tilanteessa, jossa ei tessiä
            self.min_halftoewage=800*12*self.salaryinflationfactor # vuoden 2019 luku tilanteessa, jossa ei tessiä

        self.setup_unempdays_left(porrastus = self.porrasta_toe)

        self.accbasis_kht=719.0*12*self.salaryinflationfactor # palkkakertoin??
        self.accbasis_tmtuki=0 # 1413.75*12
        self.disabbasis_tmtuki=1413.75*12*self.salaryinflationfactor # palkkakertoin??
        self.min_disab_tulevaaika=17_000/10 # tämä jaettu kymmenellä, koska vertailukohta on vuosittainen keskiarvoansios

        kerroin = self.inflation[year-2018]/self.inflation[2021-2018]
        self.opiskelija_asumismenot_toimeentulo=150*kerroin
        self.opiskelija_asumismenot_asumistuki=150*kerroin
        self.elakelainen_asumismenot_toimeentulo=200*kerroin
        self.elakelainen_asumismenot_asumistuki=200*kerroin
        self.muu_asumismenot_toimeentulo=290*kerroin
        self.muu_asumismenot_asumistuki=290*kerroin
        self.muu_asumismenot_lisa=120*kerroin

        #putkiika=np.array([61,61,61,61,62,63,64,100,100,100,100])
        putkiika=np.array([61,61,61,61,61,62,62,62,63,63,64,99])

        self.initial_benefits_p() # setup initial p for benefits module

        self.min_tyottputki_ika=putkiika[year-2018] # vuotta. Ikä, jonka täytyttyä pääsee putkeen

    def set_retirementage(self,year: int) -> None:
        if year==2018:
            self.min_retirementage_putki=62.0
            self.min_retirementage=63.5
            self.max_retirementage=68
        elif year==2019:
            self.min_retirementage_putki=62.0
            self.min_retirementage=63.75
            self.max_retirementage=68
        elif year==2020:
            self.min_retirementage_putki=64.0
            self.min_retirementage=64.0
            self.max_retirementage=69
        elif year==2021:
            self.min_retirementage_putki=64.0
            self.min_retirementage=64.25
            self.max_retirementage=69
        elif year==2022:
            self.min_retirementage_putki=64.0
            self.min_retirementage=64.5
            self.max_retirementage=69
        elif year==2023:
            self.min_retirementage_putki=64.0
            self.min_retirementage=64.75
            self.max_retirementage=69
        elif year==2024:
            self.min_retirementage_putki=65.0
            self.min_retirementage=65
            self.max_retirementage=70
        elif year==2025:
            self.min_retirementage_putki=65.0
            self.min_retirementage=65
            self.max_retirementage=70
        elif year==2026:
            self.min_retirementage_putki=65.0
            self.min_retirementage=65
            self.max_retirementage=70
        elif year==2027:
            self.min_retirementage_putki=65.0
            self.min_retirementage=65
            self.max_retirementage=70
        else:
            error('retirement_age')

    def set_year(self,year: int) -> None:
        self.year=year
        self.rates=Rates(year = self.year,silent = self.silent,max_age = self.max_age,max_retage=self.max_retirementage,
            n_groups = self.n_groups,timestep = self.timestep,inv_timestep = self.inv_timestep,n_empl = self.n_empl)

        self.set_annual_params(year)
        self.set_retirementage(year)
        self.ben.set_year(year)
        self.marg=fin_benefits.Marginals(self.ben,year = self.year)

        self.group_weights = self.rates.get_group_weights()
        #print('self.group_weights',self.group_weights)

        #kassanjasenyys_joinrate,kassanjasenyys_rate = self.rates.get_kassanjasenyys_rate()
        #self.infostats=Infostate(self.n_time,self.timestep,self.min_age,kassanjasenyys_joinrate=kassanjasenyys_joinrate,
        #    kassanjasenyys_rate=kassanjasenyys_rate,include_halftoe = self.include_halftoe,min_toewage = self.min_toewage)

        #self.palkat_ika_miehet,self.palkat_ika_naiset,self.g_r = self.rates.setup_salaries_v4(self.min_retirementage)
        self.wages_spouse = Wages_v1(year = self.year,silent = self.silent,max_age = self.max_age,
            n_groups = self.n_groups,timestep = self.timestep,inv_timestep = self.inv_timestep,
            min_retirementage = self.min_retirementage,min_salary = self.min_salary, weights = self.group_weights)
        self.wages_main = Wages_v1(year = self.year,silent = self.silent,max_age = self.max_age,
            n_groups = self.n_groups,timestep = self.timestep,inv_timestep = self.inv_timestep,
            min_retirementage = self.min_retirementage,min_salary = self.min_salary, weights = self.group_weights)

        #self.palkat_ika_miehet,self.palkat_ika_naiset,_,_ = self.wages_main.get_salaries()

        self.get_wage = self.wages_main.get_wage
        self.get_spousewage = self.wages_spouse.get_wage

        # reemployment probability
        prob_ft_3m,prob_pt_3m,prob_3m_oa,parttime_fullemp_prob_3m,fulltime_pt_prob_3m = self.rates.get_reemp_prob_v9()
        #prob_1y=1-(1-prob_ft_3m)**(1./0.25)
        self.unemp_reemp_ft_prob=prob_ft_3m #1-(1-prob_1y)**self.timestep # kolmessa kuukaudessa
        self.student_reemp_ft_prob=prob_ft_3m
        self.student_reemp_ft_prob[:30,:]=self.student_reemp_ft_prob[:30,:]
        #prob_1y=1-(1-prob_pt_3m)**(1./0.25)
        self.unemp_reemp_pt_prob=prob_pt_3m #1-(1-prob_1y)**self.timestep # kolmessa kuukaudessa
        self.student_reemp_pt_prob=prob_pt_3m #1-(1-prob_1y)**self.timestep # kolmessa kuukaudessa
        self.student_reemp_pt_prob[:30,:]=self.student_reemp_pt_prob[:30,:]
        #prob_1y=1-(1-prob_3m_oa)**(1./0.25)
        #self.oa_reemp_prob=prob_3m_oa #1-(1-prob_1y)**self.timestep # kolmessa kuukaudessa

        # moving from parttime work to fulltime work
        #prob_1y=1-(1-parttime_fullemp_prob_3m)**(1./0.25)
        self.parttime_fullemp_prob=parttime_fullemp_prob_3m #1-(1-prob_1y)**self.timestep # kolmessa kuukaudessa
        #prob_1y=1-(1-fulltime_pt_prob_3m)**(1./0.25)
        self.fulltime_pt_prob=fulltime_pt_prob_3m #1-(1-prob_1y)**self.timestep # kolmessa kuukaudessa

        #if self.plotdebug:
        #    print('unemp_reemp_ft_prob',self.unemp_reemp_ft_prob)
        #    print('unemp_reemp_pt_prob',self.unemp_reemp_pt_prob)
        #    print('oa_reemp_prob',self.oa_reemp_prob)
        #    print('parttime_fullemp_prob',self.parttime_fullemp_prob)

        if self.year==2018:
            self.elinaikakerroin=0.96102
        elif self.year==2019:
            self.elinaikakerroin=0.95722
        elif self.year==2020:
            self.elinaikakerroin=0.95404
        elif self.year==2021:
            self.elinaikakerroin=0.94984
        elif self.year==2022:
            self.elinaikakerroin=0.94659
        elif self.year==2023:
            self.elinaikakerroin=0.94419
        elif self.year==2024:
            self.elinaikakerroin=0.94692
        elif self.year==2025:
            self.elinaikakerroin=0.94392
        elif self.year==2026:
            self.elinaikakerroin=0.94092
        elif self.year==2027:
            self.elinaikakerroin=0.93792
        else:
            print('unknown year',self.year)

        self.disability_intensity,self.svpaivaraha_disabilityrate,self.svpaivaraha_short3m = self.rates.get_eff_disab_rate_v9()
        #self.pinkslip_intensity = self.rates.get_pinkslip_rate()*self.timestep
        self.pinkslip_intensity = self.rates.get_pinkslip_rate_v8()*self.timestep
        self.birth_intensity = self.rates.get_birth_rate_v9(symmetric=False)
        self.mort_intensity = self.rates.get_mort_rate_v8(self.year) #get_mort_rate()
        self.student_inrate,self.student_outrate = self.rates.get_student_rate_v9() # myös armeijassa olevat tässä
        self.outsider_inrate,self.outsider_outrate = self.rates.get_outsider_rate_v9(self.max_retirementage)
        self.divorce_rate = self.rates.get_divorce_rate()
        self.marriage_rate, self.marriage_matrix = self.rates.get_marriage_rate()

        self.npv_pension,self.npv_gpension = self.comp_npv()
        self.initial_weights = self.rates.get_initial_weights_v9()

    def comp_npv(self):
        '''
        lasketaan montako timestep:iä (diskontattuna) max_age:n jälkeen henkilä on vanhuuseläkkeellä
        hyvin yksinkertainen toteutus. Tulos on odotettu lukumäärä timestep:jä

        npv <- diskontattu
        npv0 <- ei ole diskontattu
        '''
        maxk=int(100/self.timestep)
        npv_pension=np.zeros(maxk+1)
        npv_gpension=np.zeros(maxk+1)

        cpsum_pension  =0.0
        cpsum_gpension = 0.0
        for k in range(0,maxk):
            cpsum_pension=1+self.elakeindeksi*cpsum_pension
            cpsum_gpension=1+self.elakeindeksi*cpsum_gpension
            npv_pension[k+1] = cpsum_pension
            npv_gpension[k+1] = cpsum_gpension

        if self.plotdebug:
            print('npv_pension:',npv_pension)

        return npv_pension,npv_gpension

    def comp_time_until_v2(self,age: float,intensity,max_age: float=50,initial: bool=False,min_time: float=0.0) -> float:

        '''
        simuloidaan jokaiselle erikseen kauanko aikaa eventiin on. hyvin yksinkertainen toteutus. Tulos on vuosina.
        min_age:n jälkeen henkilöllä on.

        tässä oletetaan, että eventti voi sattua vain kerran. lasketaan aika siihen.

        npv <- diskontattu
        npv0 <- ei ole diskontattu
        '''
        docont = True
        sattuma = random.uniform(0,1)
        x = age #+ self.timestep
        b = intensity[int(np.floor(x))]
        if sattuma > b:
            while docont and x < max_age:
                if sattuma <= b:
                    docont = False
                    break

                x += self.timestep
                b += (1-b)*intensity[int(np.floor(x))]

            if docont:
                x = 100

        if initial:
            ret = x-age
        else:
            ret = max(min_time,x-age)

        return ret

    def comp_life_left(self,g: int,age: float,state: int=1) -> float:
        '''
        simuloidaan npv jokaiselle erikseen kauanko elinaikaa min_age:n jälkeen henkilöllä on.
        hyvin yksinkertainen toteutus. Tulos on vuosina.
        '''

        ret = self.comp_time_until_v2(age,self.mort_intensity[:,g,state],max_age=120,initial=False,min_time=0.0)
        if self.plotdebug:
            print(f'For age {age} group {g} comp_life_left x: {ret}')
        return ret

    def comp_until_disab(self,g: int,age: float,state: int=1) -> float:

        '''
        simuloidaan npv jokaiselle erikseen kauanko elinaikaa min_age:n jälkeen henkilöllä on.
        hyvin yksinkertainen toteutus. Tulos on vuosina.

        npv <- diskontattu
        npv0 <- ei ole diskontattu
        '''
        if state != 3:
            time = self.comp_time_until_v2(age,self.disability_intensity[:,g,state],max_age=self.max_age,initial=False,min_time=0.0)
        else:
            time = 100

        if self.plotdebug and True:
            print(f'For age {age} group {g} comp_until_disab x: {time}')

        return time

    def comp_until_birth(self,g: int,age: float,initial: bool=False) -> float:

        '''
        simuloidaan npv jokaiselle erikseen kauanko elinaikaa min_age:n jälkeen henkilöllä on.
        hyvin yksinkertainen toteutus. Tulos on vuosina.

        npv <- diskontattu
        npv0 <- ei ole diskontattu
        '''
        ret = self.comp_time_until_v2(age,self.birth_intensity[:,g],max_age=50,initial=initial,min_time=1.0)
        if self.plotdebug and True:
            print(f'For age {age} and group {g} comp_until_birth x: {ret}')

        return ret

    def test_birth(self,reps: int=1000,scale=1.0):
        npv0=np.zeros(self.n_groups)

        orig = self.birth_intensity.copy()
        self.birth_intensity = orig * scale

        startage = self.min_age
        min_age = 18
        max_age = 50
        syntyneita_basic = np.zeros(self.n_groups)
        for g in range(self.n_groups):
            num = int(np.ceil(max_age-min_age+2)/self.timestep)
            sattuma = np.random.uniform(0,1,size=(reps,num))
            for r in range(reps):
                i = 0
                for x in np.arange(min_age,max_age,self.timestep):
                    i = i+1
                    if sattuma[r,i] <= self.birth_intensity[int(np.floor(x)),g]:
                        syntyneita_basic[g] += 1

        syntyneita_v2 = np.zeros(self.n_groups)
        for g in range(self.n_groups):
            for r in range(reps):
                time_to = self.comp_until_birth(g,startage)
                age = startage
                while age + time_to < max_age:
                    syntyneita_v2[g] += 1
                    age = age + time_to
                    time_to = self.comp_until_birth(g,age)

        self.birth_intensity = orig.copy()

        print('syntyneita_basic',syntyneita_basic,syntyneita_basic/reps)
        #print('syntyneita_v0',syntyneita_v0,syntyneita_v0/reps)
        print('syntyneita_v2',syntyneita_v2,syntyneita_v2/reps)
        print('ratio syntyneita_v2/syntyneita_basic',syntyneita_v2/syntyneita_basic)


    def comp_time_to_marriage(self, puoliso: int,age: float,g: int, p_g: int) -> float:
        '''
        Päivitä puolison/potentiaalisen puolison tila
        Päivitä avioliitto/avoliitto
        '''
        if puoliso>0:
            x_d = self.comp_time_until_v2(age,self.divorce_rate[:],max_age=100,initial=False,min_time=0.0)
            if self.plotdebug and True:
                print(f'For age {age} and group {g} time_to_divorce x: {x_d}')
            x_m = 100
        else:
            x_m = self.comp_time_until_v2(age,self.marriage_matrix[:,g,p_g-3],max_age=100,initial=False,min_time=0.0)
            if self.plotdebug and True:
                print(f'For age {age} and group {g} time_to_marriage d_m: {x_m}')
            x_d = 100

        return x_m,x_d

    def comp_time_to_study(self,state: int,age: float,group: int) -> float:
        '''
        Päivitä puolison/potentiaalisen puolison tila
        Päivitä avioliitto/avoliitto
        '''

        if state == 12: # opiskelija
            x_m = self.comp_time_until_v2(age,self.student_outrate[:,group],max_age=self.max_age,initial=False,min_time=0.0)
        else:
            x_m = self.comp_time_until_v2(age,self.student_inrate[:,group],max_age=self.max_age,initial=False,min_time=0.0)

        if self.plotdebug and True:
            print(f'For age {age} time_to_student x_m: {x_m}')

        return x_m

    def comp_time_to_outsider(self,state: int,age: float,group: int) -> float:
        '''
        Päivitä aika outsideriksi/pois outsideristä
        '''

        if state == 11: # outsider
            x_m = self.comp_time_until_v2(age,self.outsider_outrate[:,group],max_age=self.max_age,initial=False,min_time=0.0)
        else:
            x_m = self.comp_time_until_v2(age,self.outsider_inrate[:,group],max_age=self.max_age,initial=False,min_time=0.0)

        if self.plotdebug and True:
            print(f'For age {age} time_to_outsider x_m: {x_m}')

        return x_m

    def setup_children(self,p : dict,puoliso: int,employment_state: int,spouse_empstate: int,
                    children_under3: int,children_under7: int,children_under18: int,lapsikorotus_lapsia: int) -> None:
        # tässä ei alku+, koska lapset yhteisiä
        if puoliso>0:
            p['lapsia'] = children_under18
            p['lapsia_paivahoidossa'] = children_under7
            p['lapsia_alle_kouluikaisia'] = children_under7
            p['lapsia_alle_3v'] = children_under3
            p['lapsia_kotihoidontuella'] = 0
            p['lapsikorotus_lapsia'] = lapsikorotus_lapsia

            if employment_state==5 or spouse_empstate==5: # äitiysvapaa
                p['lapsia_paivahoidossa'] = 0
            elif employment_state==6 or spouse_empstate==6: # isyysvapaa
                p['lapsia_paivahoidossa'] = 0
            elif employment_state in set([0,4,13]) or spouse_empstate in set([0,4,13]):
                p['lapsia_paivahoidossa'] = 0
            elif employment_state==7 or spouse_empstate==7: # hoitovapaa
                p['lapsia_paivahoidossa'] = 0
                p['lapsia_kotihoidontuella'] = children_under7

            if employment_state==10 or spouse_empstate==10:
                p['osaaikainen_paivahoito'] = 1 # 1 # lisää tähän tsekki että osa-aikatila on 0 tai 1 mutta ei 2
            else:
                p['osaaikainen_paivahoito'] = 0

            p['saa_elatustukea'] = 0
        else:
            p['lapsia'] = children_under18
            p['lapsia_paivahoidossa'] = children_under7
            p['lapsia_alle_kouluikaisia'] = children_under7
            p['lapsia_alle_3v'] = children_under3
            p['lapsia_kotihoidontuella'] = 0
            p['lapsikorotus_lapsia'] = lapsikorotus_lapsia

            if employment_state==5: # äitiysvapaa
                p['lapsia_paivahoidossa'] = 0
            elif employment_state==6: # isyysvapaa
                p['lapsia_paivahoidossa'] = 0
            elif employment_state in set([0,4,13]):
                p['lapsia_paivahoidossa'] = 0
            elif employment_state==7: # hoitovapaa
                p['lapsia_paivahoidossa'] = 0
                p['lapsia_kotihoidontuella'] = children_under7

            if employment_state==10:
                p['osaaikainen_paivahoito'] = 1
            else:
                p['osaaikainen_paivahoito'] = 0

            if children_under18>0:
                p['saa_elatustukea'] = 1 # vain yksinhuoltaja
            else:
                p['saa_elatustukea'] = 0

    def initial_benefits_p(self):
        p={}

        p['veromalli'] = 0
        p['kuntaryhma'] = 3
        p['lapsia_kotihoidontuella'] = 0

        if self.year>=2024:
            p['tyottomyysturva_suojaosa_taso']=0
            p['ansiopvrahan_suojaosa']=0
            p['ansiopvraha_lapsikorotus']=0
        elif self.year==2020:
            p['ansiopvrahan_suojaosa'] = 1
            p['ansiopvraha_lapsikorotus'] = 1
            p['tyottomyysturva_suojaosa_taso'] = 500
        else:
            p['ansiopvrahan_suojaosa'] = 1
            p['ansiopvraha_lapsikorotus'] = 1
            p['tyottomyysturva_suojaosa_taso'] = 300

        self.initial_p = p

    def setup_benefits(self,wage: float,benefitbasis: float,kansanelake: float,tyoelake: float,employment_state: int,pt_state: int,
                    time_in_state: float,ika: float,used_unemp_benefit: float,children_under3: int,children_under7: int,children_under18: int,puoliso: int=0,
                    irtisanottu: int= 0,karenssia_jaljella: float= 0,p2:dict = None,alku='omat_',puolisoalku='puoliso_') -> dict:
        if p2 is not None:
            p = p2.copy()
        else:
            p = self.initial_p.copy()

        p[alku+'perustulo'] = 0 # ei perustuloa tässä mallissa, FIXME!
        p[alku+'saa_elatustukea'] = 0
        p[alku+'opiskelija'] = 0
        p[alku+'elakkeella'] = 0
        p[alku+'toimeentulotuki_vahennys'] = 0
        p[alku+'ika'] = ika
        p[alku+'tyoton'] = 0
        p[alku+'peruspaivarahalla'] = 0
        p[alku+'saa_ansiopaivarahaa'] = 0
        p[alku+'vakiintunutpalkka'] = 0

        p[alku+'tyottomyyden_kesto'] = 0
        p[alku+'isyysvapaalla'] = 0
        p[alku+'aitiysvapaalla'] = 0
        p[alku+'kotihoidontuella'] = 0
        p[alku+'tyoelake'] = 0
        p[alku+'kansanelake'] = 0
        p[alku+'elake_maksussa'] = 0
        p[alku+'elakkeella'] = 0
        p[alku+'sairauspaivarahalla'] = 0
        p[alku+'disabled'] = 0
        p[alku+'tyoaika'] = 0

        if employment_state==15:
            p[alku+'alive'] = 0
        else:
            p[alku+'alive'] = 1

        if employment_state==1:
            p[alku+'tyoton'] = 0 # voisi olla työtön siinä mielessä, että oikeutettu soviteltuun päivärahaan
            p[alku+'t'] = wage/12
            p[alku+'vakiintunutpalkka'] = wage/12
            p[alku+'saa_ansiopaivarahaa'] = 0
            p[alku+'tyoelake'] = tyoelake/12 # ove
            if pt_state==0:
                p[alku+'tyoaika'] = 32
            elif pt_state==1:
                p[alku+'tyoaika'] = 40
            else:
                p[alku+'tyoaika'] = 48
        elif employment_state==0: # työtön, ansiopäivärahalla
            if ika<self.max_unemploymentbenefitage:
                #self.render()
                p[alku+'tyoton'] = 1
                p[alku+'t'] = 0
                p[alku+'vakiintunutpalkka'] = benefitbasis/12
                p[alku+'saa_ansiopaivarahaa'] = 1
                p[alku+'tyottomyyden_kesto'] = 12*21.5*(used_unemp_benefit-self.timestep)

                if irtisanottu<1 and karenssia_jaljella>0:
                    p[alku+'saa_ansiopaivarahaa'] = 0
                    p[alku+'tyoton'] = 0

                p[alku+'tyoelake'] = tyoelake/12 # ove
            else:
                p[alku+'tyoton'] = 0 # ei oikeutta työttömyysturvaan
                p[alku+'t'] = 0
                p[alku+'vakiintunutpalkka'] = 0
                p[alku+'saa_ansiopaivarahaa'] = 0
                p[alku+'tyoelake'] = tyoelake/12 # ove
        elif employment_state==13: # työmarkkinatuki
            if ika<self.max_unemploymentbenefitage:
                p[alku+'tyoton'] = 1
                p[alku+'peruspaivarahalla'] = 1
                p[alku+'t'] = 0
                p[alku+'vakiintunutpalkka'] = 0
                p[alku+'tyottomyyden_kesto'] = 12*21.5*(used_unemp_benefit-self.timestep)
                p[alku+'saa_ansiopaivarahaa'] = 0
                p[alku+'tyoelake'] = tyoelake/12 # ove

                # karenssi??
            else:
                p[alku+'tyoton'] = 0 # ei oikeutta työttömyysturvaan
                p[alku+'t'] = 0
                p[alku+'vakiintunutpalkka'] = 0
                p[alku+'saa_ansiopaivarahaa'] = 0
                p[alku+'tyoelake'] = tyoelake/12 # ove
        elif employment_state==3: # tk
            p[alku+'t'] = 0
            p[alku+'elakkeella'] = 1
            p[alku+'tyoelake'] = tyoelake/12 # ove
            p[alku+'kansanelake'] = kansanelake/12
            p[alku+'disabled'] = 1
        elif employment_state==4: # työttömyysputki
            if ika<self.max_unemploymentbenefitage:
                p[alku+'tyoton'] = 1
                p[alku+'t'] = 0
                p[alku+'vakiintunutpalkka'] = benefitbasis/12
                p[alku+'saa_ansiopaivarahaa'] = 1
                p[alku+'tyottomyyden_kesto'] = 12*21.5*(used_unemp_benefit-self.timestep)
                p[alku+'tyoelake'] = tyoelake/12 # ove
            else:
                p[alku+'tyoton'] = 0 # ei oikeutta työttömyysturvaan
                p[alku+'t'] = 0
                p[alku+'vakiintunutpalkka'] = 0
                p[alku+'saa_ansiopaivarahaa'] = 0
                p[alku+'tyoelake'] = tyoelake/12 # ove
        elif employment_state==5: # ansiosidonnainen vanhempainvapaa, äidit
            #if self.plotdebug:
            #    print('****** ',wage,time_in_state)
            p[alku+'aitiysvapaalla'] = 1
            p[alku+'aitiysvapaa_kesto'] = time_in_state
            if time_in_state<0.25:
                p[alku+'t'] = wage/12 # palkkaa maksetaan 3 kk tässä mallissa
            else:
                p[alku+'t'] = 0
            p[alku+'vakiintunutpalkka'] = benefitbasis/12
            p[alku+'saa_ansiopaivarahaa'] = 1
        elif employment_state==6: # ansiosidonnainen vanhempainvapaa, isät
            p[alku+'isyysvapaalla'] = 1
            p[alku+'isyysvapaa_kesto'] = time_in_state
            if time_in_state<0.25:
                p[alku+'t'] = wage/12 # palkkaa maksetaan 3 kk tässä mallissa
            else:
                p[alku+'t'] = 0
            p[alku+'vakiintunutpalkka'] = benefitbasis/12
            p[alku+'saa_ansiopaivarahaa'] = 1
        elif employment_state==7: # hoitovapaa
            p[alku+'kotihoidontuella'] = 1
            p[alku+'kotihoidontuki_kesto'] = time_in_state
            p[alku+'t'] = 0
            p[alku+'vakiintunutpalkka'] = benefitbasis/12
            p[alku+'tyoelake'] = tyoelake/12
        elif employment_state==2: # vanhuuseläke
            if ika >= self.min_retirementage:
                p[alku+'t'] = 0
                p[alku+'elakkeella'] = 1
                p[alku+'tyoelake'] = tyoelake/12
                p[alku+'kansanelake'] = kansanelake/12
            else:
                p[alku+'t'] = 0
                p[alku+'elakkeella'] = 0
                p[alku+'tyoelake'] = 0
        elif employment_state==8: # ve+työ
            p[alku+'t'] = wage/12
            p[alku+'elakkeella'] = 1
            p[alku+'tyoelake'] = tyoelake/12
            p[alku+'kansanelake'] = kansanelake/12
            if pt_state==0:
                p[alku+'tyoaika'] = 32
            elif pt_state==1:
                p[alku+'tyoaika'] = 40
            else:
                p[alku+'tyoaika'] = 48
        elif employment_state==9: # ve+osatyö
            p[alku+'t'] = wage/12
            p[alku+'elakkeella'] = 1
            p[alku+'tyoelake'] = tyoelake/12
            p[alku+'kansanelake'] = kansanelake/12
            if pt_state==0:
                p[alku+'tyoaika'] = 8
            elif pt_state==1:
                p[alku+'tyoaika'] = 16
            else:
                p[alku+'tyoaika'] = 24
        elif employment_state==10: # osa-aikatyö
            p[alku+'t'] = wage/12
            p[alku+'tyoelake'] = tyoelake/12 # ove
            if pt_state==0:
                p[alku+'tyoaika'] = 8
            elif pt_state==1:
                p[alku+'tyoaika'] = 16
            else:
                p[alku+'tyoaika'] = 24
        elif employment_state==11: # työelämän ulkopuolella
            p[alku+'toimeentulotuki_vahennys'] = 0 # oletetaan että ei kieltäytynyt työstä
            p[alku+'t'] = 0
            p[alku+'tyoelake'] = tyoelake/12 # ove
        elif employment_state==12: # opiskelija
            p[alku+'opiskelija'] = 1
            p[alku+'t'] = 0
            p[alku+'tyoelake'] = tyoelake/12 # ove
        elif employment_state==14: # sv-päiväraha
            if time_in_state<0.25:
                p[alku+'t'] = wage/12 # palkkaa maksetaan 3 kk tässä mallissa
                p[alku+'tyoelake'] = tyoelake/12 # ove
                p[alku+'vakiintunutpalkka'] = benefitbasis/12
                p[alku+'sairauspaivarahalla'] = 1
            else:
                p[alku+'t'] = 0
                p[alku+'tyoelake'] = tyoelake/12 # ove
                p[alku+'vakiintunutpalkka'] = benefitbasis/12
                p[alku+'sairauspaivarahalla'] = 1
        elif employment_state==15: # kuollut
            p[alku+'t'] = 0
            p[alku+'tyoelake'] = 0 # ove
        else:
            print('Unknown employment_state ',employment_state)

        p[alku+'elake_maksussa'] = p[alku+'tyoelake']+p[alku+'kansanelake']

        # if self.year>=2024:
        #     p['ansiopvrahan_suojaosa'] = 1
        #     p['ansiopvraha_lapsikorotus'] = 1
        # else:
        #     p['ansiopvrahan_suojaosa'] = 1
        #     p['ansiopvraha_lapsikorotus'] = 1
        #     p['tyottomyysturva_suojaosa_taso'] = 300

        if puoliso>0 and employment_state!=15:
            p['aikuisia'] = 2
        elif alku!=puolisoalku:
            p['aikuisia'] = 1
            p[puolisoalku+'opiskelija'] = 0
            p[puolisoalku+'peruspaivarahalla'] = 0
            p[puolisoalku+'elakkeella'] = 0
            p[puolisoalku+'toimeentulotuki_vahennys'] = 0
            p[puolisoalku+'ika'] = ika
            p[puolisoalku+'tyoton'] = 0
            p[puolisoalku+'saa_ansiopaivarahaa'] = 0
            p[puolisoalku+'vakiintunutpalkka'] = 0
            p[puolisoalku+'tyottomyyden_kesto'] = 0
            p[puolisoalku+'isyysvapaalla'] = 0
            p[puolisoalku+'aitiysvapaalla'] = 0
            p[puolisoalku+'kotihoidontuella'] = 0
            p[puolisoalku+'tyoelake'] = 0
            p[puolisoalku+'kansanelake'] = 0
            p[puolisoalku+'elakkeella'] = 0
            p[puolisoalku+'sairauspaivarahalla'] = 0
            p[puolisoalku+'disabled'] = 0
            p[puolisoalku+'t'] = 0

        return p

    def setup_asumismenot(self,employment_state: int,puoliso: int,spouse_empstate: int,children_under18: int,p : dict) -> None:
        puolisokerroin=0.75
        if employment_state==12: # opiskelija
            if puoliso>0:
                p['asumismenot_toimeentulo'] = self.opiskelija_asumismenot_toimeentulo*puolisokerroin+children_under18*self.muu_asumismenot_lisa
                p['asumismenot_asumistuki'] = self.opiskelija_asumismenot_asumistuki*puolisokerroin+children_under18*self.muu_asumismenot_lisa
                if spouse_empstate==12:
                    p['asumismenot_toimeentulo'] += self.opiskelija_asumismenot_toimeentulo*puolisokerroin
                    p['asumismenot_asumistuki'] += self.opiskelija_asumismenot_asumistuki*puolisokerroin
                elif spouse_empstate in set([2,3,8,9]):
                    p['asumismenot_toimeentulo'] += self.elakelainen_asumismenot_toimeentulo*puolisokerroin
                    p['asumismenot_asumistuki'] += self.elakelainen_asumismenot_asumistuki*puolisokerroin
                else:
                    p['asumismenot_toimeentulo'] += self.muu_asumismenot_toimeentulo*puolisokerroin
                    p['asumismenot_asumistuki'] += self.muu_asumismenot_asumistuki*puolisokerroin
            else:
                p['asumismenot_toimeentulo'] = self.opiskelija_asumismenot_toimeentulo+children_under18*self.muu_asumismenot_lisa
                p['asumismenot_asumistuki'] = self.opiskelija_asumismenot_asumistuki+children_under18*self.muu_asumismenot_lisa

        elif employment_state in set([2,3,8,9]): # eläkeläinen
            if puoliso>0:
                p['asumismenot_toimeentulo'] = self.elakelainen_asumismenot_toimeentulo*puolisokerroin+children_under18*self.muu_asumismenot_lisa
                p['asumismenot_asumistuki'] = self.elakelainen_asumismenot_asumistuki*puolisokerroin+children_under18*self.muu_asumismenot_lisa
                if spouse_empstate==12:
                    p['asumismenot_toimeentulo'] += self.opiskelija_asumismenot_toimeentulo*puolisokerroin
                    p['asumismenot_asumistuki'] += self.opiskelija_asumismenot_asumistuki*puolisokerroin
                elif spouse_empstate in set([2,3,8,9]):
                    p['asumismenot_toimeentulo'] += self.elakelainen_asumismenot_toimeentulo*puolisokerroin
                    p['asumismenot_asumistuki'] += self.elakelainen_asumismenot_asumistuki*puolisokerroin
                else:
                    p['asumismenot_toimeentulo'] += self.muu_asumismenot_toimeentulo*puolisokerroin
                    p['asumismenot_asumistuki'] += self.muu_asumismenot_asumistuki*puolisokerroin
            else:
                p['asumismenot_toimeentulo'] = self.elakelainen_asumismenot_toimeentulo*puolisokerroin+children_under18*self.muu_asumismenot_lisa
                p['asumismenot_asumistuki'] = self.elakelainen_asumismenot_asumistuki*puolisokerroin+children_under18*self.muu_asumismenot_lisa
        elif employment_state==15:
            p['asumismenot_toimeentulo'] = 0
            p['asumismenot_asumistuki'] = 0
        else: # muu
            if puoliso>0:
                p['asumismenot_toimeentulo'] = self.muu_asumismenot_toimeentulo*puolisokerroin+children_under18*self.muu_asumismenot_lisa
                p['asumismenot_asumistuki'] = self.muu_asumismenot_asumistuki*puolisokerroin+children_under18*self.muu_asumismenot_lisa
                if spouse_empstate==12:
                    p['asumismenot_toimeentulo'] += self.opiskelija_asumismenot_toimeentulo*puolisokerroin
                    p['asumismenot_asumistuki'] += self.opiskelija_asumismenot_asumistuki*puolisokerroin
                elif spouse_empstate in set([2,3,8,9]):
                    p['asumismenot_toimeentulo'] += self.elakelainen_asumismenot_toimeentulo*puolisokerroin
                    p['asumismenot_asumistuki'] += self.elakelainen_asumismenot_asumistuki*puolisokerroin
                else:
                    p['asumismenot_toimeentulo'] += self.muu_asumismenot_toimeentulo*puolisokerroin
                    p['asumismenot_asumistuki'] += self.muu_asumismenot_asumistuki*puolisokerroin
            else:
                p['asumismenot_toimeentulo'] = self.muu_asumismenot_toimeentulo+children_under18*self.muu_asumismenot_lisa
                p['asumismenot_asumistuki'] = self.muu_asumismenot_asumistuki+children_under18*self.muu_asumismenot_lisa

    def setup_contrafactual(self,empstate: int,wage: float,pot_wage: float,unempwage_basis: float,jasen: int,unempwage: float): #,ptact: int):
        '''
        setup contrafactual state
        returns empstate0, empstate1, empwage0, empwage1, benefitbasis0, benefitbasis1
        '''
        if empstate in set([0,4]):
            return empstate,1,0,pot_wage,max(unempwage,unempwage_basis),0
        elif empstate in set([5,6,7,11,12,13,14]):
            return empstate,1,0,pot_wage,max(unempwage,unempwage_basis),0
        elif empstate in set([1,10]):
            if jasen:
                return 0,empstate,0,wage,max(unempwage,unempwage_basis),0
            else:
                return 13,empstate,0,wage,0,0
        elif empstate in set([3,2]):
            return empstate,empstate,0,0,0,0
        elif empstate in {8,9}:
            return 2,empstate,0,wage,0,0
        else:
            return 15,15,0,0,0,0

    def setup_couples(self,age: int,wage: float,benefitbasis: float,main_kansanelake: float,main_tyoelake: float,
            main_employment_state: int,main_pt_state: int,main_time_in_state: float,main_used_unemp_benefit: float,
            children_under3: int,children_under7: int,children_under18: int,irtisanottu: int,karenssia_jaljella: float,
            spouse_wage: float,spouse_benefitbasis: float,spouse_kansanelake: float,puoliso_tyoelake: float,
            spouse_empstate: int,spouse_pt_state: int,spouse_time_in_state: float,puoliso_used_unemp_benefit: float,
            puoliso_irtisanottu: float,puoliso_karenssia_jaljella: float):

        p = self.setup_benefits(wage,benefitbasis,main_kansanelake,main_tyoelake,main_employment_state,main_pt_state,main_time_in_state,age,main_used_unemp_benefit,
            children_under3,children_under7,children_under18,puoliso=1,
            irtisanottu=irtisanottu,karenssia_jaljella=karenssia_jaljella,alku='')

        p = self.setup_benefits(spouse_wage,spouse_benefitbasis,spouse_kansanelake,puoliso_tyoelake,spouse_empstate,spouse_pt_state,spouse_time_in_state,age,puoliso_used_unemp_benefit,
                  children_under3,children_under7,children_under18,puoliso=1,
                  irtisanottu=puoliso_irtisanottu,karenssia_jaljella=puoliso_karenssia_jaljella,
                  alku='puoliso_',p2=p)

        self.setup_asumismenot(main_employment_state,1,spouse_empstate,children_under18,p)
        self.setup_children(p,1,main_employment_state,spouse_empstate,children_under3,children_under7,children_under18,children_under18)
        p['aikuisia'] = 2

        return p

    def comp_benefits(self,wage: float,benefitbasis: float,main_kansanelake: float,tyoelake: float,employment_state: int,pt_state: int,
                    time_in_state: float,children_under3: int,children_under7: int,children_under18: int,ika: float,
                    puoliso: int,spouse_empstate: int,spouse_pt_state: int,spouse_wage: float,spouse_kansanelake: float,puoliso_tyoelake: float,
                    puoliso_benefitbasis: float,spouse_time_in_state: float,
                    used_unemp_benefit: float, puoliso_used_unemp_benefit: float,
                    g: int,p_g: int,
                    irtisanottu: int = 0,puoliso_irtisanottu: int = 0,karenssia_jaljella: float = 0,
                    puoliso_karenssia_jaljella: float = 0, ove: bool = False,debug: bool = False,
                    potential_wage: float = 0,potential_spouse_wage: float = 0,unempwage_basis: float = 0,
                    puoliso_unempwage_basis: float = 0, kassanjasen: int =0,unempwage: float = 0,puoliso_unempwage: float = 0):
        '''
        Kutsuu fin_benefits-modulia, jonka avulla lasketaan etuudet ja huomioidaan verotus
        Laske etuuksien arvo, kun
            wage on palkka
            benefitbasis on etuuksien perusteena oleva ansio
            pension on maksettavan eläkkeen määrä
            employment_state on töissä olo (0)/työttämyys (1)/eläkkeellä olo (2) jne.
            time_in_state on kesto tilassa
            ika on henkilän ikä
        '''

        if puoliso>0 and not (employment_state==15 or spouse_empstate==15): # pariskunta
            p = self.setup_couples(ika,wage,benefitbasis,main_kansanelake,tyoelake,employment_state,pt_state,time_in_state,used_unemp_benefit,
                    children_under3,children_under7,children_under18,irtisanottu,karenssia_jaljella,
                    spouse_wage,puoliso_benefitbasis,spouse_kansanelake,puoliso_tyoelake,spouse_empstate,spouse_pt_state,spouse_time_in_state,puoliso_used_unemp_benefit,
                    puoliso_irtisanottu,puoliso_karenssia_jaljella)

            netto,benefitq = self.ben.laske_tulot_v3(p,include_takuuelake = self.include_takuuelake,omatalku='',puolisoalku='puoliso_',
                split_costs=True,set_equal=True,add_kansanelake=False) # kansaneläke already included
            benefitq['netto'] = netto
            netto=netto*12

            #if children_under18>0: # FIXME
            #    print(f"ika {ika:.2f} c18 {children_under18} q {benefitq['lapsilisa']}")

            if self.include_emtr:
                emtr_tilat=set([0,1,4,5,6,7,8,9,10,11,12,13,14])
                if employment_state in emtr_tilat:
                    #if employment_state == 1:
                    #    printtaa = True
                    #else:
                    #    printtaa = False
                    printtaa=True

                    e0,e1,w0,w1,ow0,ow1 = self.setup_contrafactual(employment_state,wage,potential_wage,unempwage_basis,kassanjasen,unempwage)

                    p0 = self.setup_couples(ika,w0,ow0,main_kansanelake,tyoelake,e0,pt_state,0,used_unemp_benefit,
                            children_under3,children_under7,children_under18,irtisanottu,0,
                            spouse_wage,puoliso_benefitbasis,spouse_kansanelake,puoliso_tyoelake,spouse_empstate,spouse_pt_state,spouse_time_in_state,puoliso_used_unemp_benefit,
                            puoliso_irtisanottu,0)
                    p1 = self.setup_couples(ika,w1,ow1,main_kansanelake,tyoelake,e1,pt_state,0,used_unemp_benefit,
                            children_under3,children_under7,children_under18,irtisanottu,0,
                            spouse_wage,puoliso_benefitbasis,spouse_kansanelake,puoliso_tyoelake,spouse_empstate,spouse_pt_state,spouse_time_in_state,puoliso_used_unemp_benefit,
                            puoliso_irtisanottu,0)
                    _,_,tvax = self.marg.comp_emtr(p0,p1,w1,dt=1200)
                    _,effx,_ = self.marg.comp_emtr(p,p,wage,dt=1200)
                    benefitq['omat_emtr'] = effx
                    benefitq['omat_tva'] = tvax
                    if w1<1.0 or w1<w0+1.0:
                        print('co omat',employment_state,w0,w1)
                        self.render(render_puoliso=False,render_omat=True)
                    if tvax<1.0:
                        print('co omat tvax',employment_state,w0,w1,tvax)
                        self.render(render_puoliso=False,render_omat=True)
                else:
                    benefitq['omat_emtr'] = np.nan
                    benefitq['omat_tva'] = np.nan

                if spouse_empstate in emtr_tilat:
                    printtaa=True
                    pe0,pe1,pw0,pw1,pow0,pow1 = self.setup_contrafactual(spouse_empstate,spouse_wage,potential_spouse_wage,puoliso_unempwage_basis,kassanjasen,puoliso_unempwage)
                    if pw1<1.0 or pw1<pw0+1.0:
                        print('co puoliso',spouse_empstate,pw0,pw1,ika,spouse_wage)
                        self.render(render_puoliso=True,render_omat=False)
                    pp0 = self.setup_couples(ika,wage,benefitbasis,main_kansanelake,tyoelake,employment_state,pt_state,time_in_state,used_unemp_benefit,
                            children_under3,children_under7,children_under18,irtisanottu,0,
                            pw0,pow0,spouse_kansanelake,puoliso_tyoelake,pe0,spouse_pt_state,0,puoliso_used_unemp_benefit,
                            puoliso_irtisanottu,0)
                    pp1 = self.setup_couples(ika,wage,benefitbasis,main_kansanelake,tyoelake,employment_state,pt_state,time_in_state,used_unemp_benefit,
                            children_under3,children_under7,children_under18,irtisanottu,0,
                            pw1,pow1,spouse_kansanelake,puoliso_tyoelake,pe1,spouse_pt_state,0,puoliso_used_unemp_benefit,
                            puoliso_irtisanottu,0)
                    nettox2,_,tvax2 = self.marg.comp_emtr(pp0,pp1,pw1,dt=1200,alku='puoliso_')
                    _,effx2,_ = self.marg.comp_emtr(p,p,spouse_wage,dt=1200,alku='puoliso_')
                    if tvax2<1.0:
                        print('puoliso tvax',spouse_empstate,pw0,pw1,tvax2)
                        self.render(render_puoliso=True,render_omat=False)
                    benefitq['puoliso_emtr'] = effx2
                    benefitq['puoliso_tva'] = tvax2
                else:
                    benefitq['puoliso_emtr'] = np.nan
                    benefitq['puoliso_tva'] = np.nan
            else:
                benefitq['omat_emtr'] = np.nan
                benefitq['puoliso_emtr'] = np.nan
                benefitq['omat_tva'] = np.nan
                benefitq['puoliso_tva'] = np.nan

            benefitq['omat_potential_wage'] = potential_wage
            benefitq['puoliso_potential_wage'] = potential_spouse_wage

            if benefitq['omat_netto']<1e-6 or benefitq['puoliso_netto']<1e-6:
                if benefitq['omat_netto']<1e-6:
                    benefitq['omat_netto'] = 1.0
                    if benefitq['puoliso_netto']>2.0:
                        benefitq['puoliso_netto']-=1.0
                else:
                    benefitq['puoliso_netto'] = 1.0
                    if benefitq['omat_netto']>2.0:
                        benefitq['omat_netto']-=1.0

            netto_omat=benefitq['omat_netto']*12
            netto_puoliso=benefitq['puoliso_netto']*12
        else: # ei pariskunta
            if employment_state in [5,6,7]: # vanhempainvapaalla
                if spouse_empstate not in [5,6,7] or (spouse_empstate in [5,6,7] and g>p_g):
                    c3=children_under3
                    c7=children_under7
                    c18=children_under18
                    p = self.setup_benefits(wage,benefitbasis,main_kansanelake,tyoelake,employment_state,pt_state,time_in_state,ika,used_unemp_benefit,
                        c3,c7,c18,puoliso=0,irtisanottu=irtisanottu,karenssia_jaljella=karenssia_jaljella,alku='')
                    self.setup_children(p,puoliso,employment_state,-1,c3,c7,c18,children_under18)
                    self.setup_asumismenot(employment_state,0,-1,children_under18,p)
                else: # molemmat ei vanhempainvapaalla yhtä aikaa niin että saavat lapsilisät
                    # lapset 0 tässä, yksinkertaistus
                    p = self.setup_benefits(wage,benefitbasis,main_kansanelake,tyoelake,employment_state,pt_state,time_in_state,ika,used_unemp_benefit,
                        0,0,0,puoliso=0,irtisanottu=irtisanottu,karenssia_jaljella=karenssia_jaljella,alku='')
                    c3,c7,c18=0,0,0
                    self.setup_children(p,puoliso,employment_state,spouse_empstate,c3,c7,c18,children_under18)
                    self.setup_asumismenot(employment_state,0,-1,0,p)
            elif employment_state!=15 and (g>p_g or spouse_empstate==15) and spouse_empstate not in [5,6,7]:
                # lapset itsellä, ei puolisolla. tässä epäsymmetria
                c3=children_under3
                c7=children_under7
                c18=children_under18
                p = self.setup_benefits(wage,benefitbasis,main_kansanelake,tyoelake,employment_state,pt_state,time_in_state,ika,used_unemp_benefit,
                    c3,c7,c18,puoliso=0,irtisanottu=irtisanottu,karenssia_jaljella=karenssia_jaljella,alku='')
                self.setup_children(p,puoliso,employment_state,-1,children_under3,children_under7,children_under18,children_under18)
                self.setup_asumismenot(employment_state,0,-1,children_under18,p)
            else:
                # lapset 0 tässä, yksinkertaistus
                p = self.setup_benefits(wage,benefitbasis,main_kansanelake,tyoelake,employment_state,pt_state,time_in_state,ika,used_unemp_benefit,
                    0,0,0,puoliso=0,irtisanottu=irtisanottu,karenssia_jaljella=karenssia_jaljella,alku='')
                c3,c7,c18=0,0,0
                self.setup_children(p,puoliso,employment_state,spouse_empstate,0,0,0,children_under18)
                self.setup_asumismenot(employment_state,0,-1,0,p)

            netto1,benefitq1 = self.ben.laske_tulot_v3(p,include_takuuelake = self.include_takuuelake,split_costs=True,add_kansanelake=False,set_equal=False) # kansaneläke jo mukana

            if spouse_empstate in [5,6,7]:
                if employment_state not in [5,6,7] or (employment_state in [5,6,7] and p_g>g):
                    pc3=children_under3
                    pc7=children_under7
                    pc18=children_under18
                    p2 = self.setup_benefits(spouse_wage,puoliso_benefitbasis,spouse_kansanelake,puoliso_tyoelake,spouse_empstate,spouse_pt_state,spouse_time_in_state,ika,puoliso_used_unemp_benefit,
                              children_under3,children_under7,children_under18,puoliso=0,irtisanottu=puoliso_irtisanottu,
                              karenssia_jaljella=puoliso_karenssia_jaljella,alku='')
                    self.setup_children(p2,puoliso,spouse_empstate,-1,pc3,pc7,pc18,pc18)
                    self.setup_asumismenot(spouse_empstate,0,-1,pc18,p2)
                else:
                    p2 = self.setup_benefits(spouse_wage,puoliso_benefitbasis,spouse_kansanelake,puoliso_tyoelake,spouse_empstate,spouse_pt_state,spouse_time_in_state,ika,puoliso_used_unemp_benefit,
                              0,0,0,puoliso=0,irtisanottu=puoliso_irtisanottu,karenssia_jaljella=puoliso_karenssia_jaljella,alku='')
                    self.setup_children(p2,puoliso,spouse_empstate,-1,0,0,0,children_under18)
                    self.setup_asumismenot(spouse_empstate,0,-1,0,p2)
                    pc3,pc7,pc18 = 0,0,0
            elif spouse_empstate!=15 and (g<p_g or employment_state==15) and employment_state not in [5,6,7]:
                # lapsilisat maksetaan puolisolle
                pc3=children_under3
                pc7=children_under7
                pc18=children_under18
                p2 = self.setup_benefits(spouse_wage,puoliso_benefitbasis,spouse_kansanelake,puoliso_tyoelake,spouse_empstate,spouse_pt_state,spouse_time_in_state,ika,puoliso_used_unemp_benefit,
                          pc3,pc7,pc18,puoliso=0,irtisanottu=puoliso_irtisanottu,karenssia_jaljella=puoliso_karenssia_jaljella,alku='')
                self.setup_children(p2,puoliso,spouse_empstate,-1,pc3,pc7,pc18,pc18)
                self.setup_asumismenot(spouse_empstate,0,-1,pc18,p2)
                pc3=children_under3
                pc7=children_under7
                pc18=children_under18
            else:
                p2 = self.setup_benefits(spouse_wage,puoliso_benefitbasis,spouse_kansanelake,puoliso_tyoelake,spouse_empstate,spouse_pt_state,spouse_time_in_state,ika,puoliso_used_unemp_benefit,
                          0,0,0,puoliso=0,irtisanottu=puoliso_irtisanottu,karenssia_jaljella=puoliso_karenssia_jaljella,alku='') 
                self.setup_children(p2,puoliso,spouse_empstate,-1,0,0,0,children_under18)
                self.setup_asumismenot(spouse_empstate,0,-1,0,p2)
                pc3,pc7,pc18 = 0,0,0

            netto2,benefitq2 = self.ben.laske_tulot_v3(p2,include_takuuelake = self.include_takuuelake,omat='puoliso_',puoliso='omat_',omatalku='',puolisoalku='puoliso_',split_costs=True,add_kansanelake=False,set_equal=False) # switch order
            netto = netto1+netto2

            #if c18>0 or pc18>0: # FIXME
            #    print(f'ika {ika:.2f} c18 {c18} pc18',pc18,'q1',benefitq1['lapsilisa'],'q2',benefitq2['lapsilisa'])

            if self.include_emtr:
                emtr_tilat=set([0,1,4,5,6,7,10,11,13,14])
                if employment_state in emtr_tilat:
                    printtaa=True
                    e0,e1,w0,w1,ow0,ow1 = self.setup_contrafactual(employment_state,wage,potential_wage,unempwage_basis,kassanjasen,unempwage)

                    p0 = self.setup_benefits(w0,ow0,main_kansanelake,tyoelake,e0,pt_state,0,ika,used_unemp_benefit,
                        c3,c7,c18,puoliso=0,irtisanottu=irtisanottu,karenssia_jaljella=0,alku='')
                    self.setup_children(p0,0,e0,employment_state,c3,c7,c18,c18)
                    self.setup_asumismenot(e0,0,employment_state,c18,p0)

                    p1 = self.setup_benefits(w1,ow1,main_kansanelake,tyoelake,e1,pt_state,time_in_state,ika,used_unemp_benefit,
                        c3,c7,c18,puoliso=0,irtisanottu=irtisanottu,karenssia_jaljella=0,alku='')
                    self.setup_children(p1,0,e1,employment_state,c3,c7,c18,c18)
                    self.setup_asumismenot(e1,0,employment_state,c18,p1)
                    if w1<1.0 or w1<w0+1.0:
                        print('omat',employment_state,w0,w1)
                        self.render()

                    nettox,_,tvax = self.marg.comp_emtr(p0,p1,w1,dt=1200)#,display=printtaa)
                    _,effx,_ = self.marg.comp_emtr(p,p,wage,dt=1200)
                    benefitq1['omat_emtr'] = effx
                    benefitq1['omat_tva'] = tvax
                else:
                    benefitq1['omat_emtr'] = np.nan
                    benefitq1['omat_tva'] = np.nan

                if spouse_empstate in emtr_tilat:
                    pe0,pe1,pw0,pw1,pow0,pow1 = self.setup_contrafactual(spouse_empstate,spouse_wage,potential_spouse_wage,puoliso_unempwage_basis,kassanjasen,puoliso_unempwage)
                    pp0 = self.setup_benefits(pw0,pow0,spouse_kansanelake,puoliso_tyoelake,pe0,spouse_pt_state,0,ika,puoliso_used_unemp_benefit,
                              pc3,pc7,pc18,puoliso=0,irtisanottu=puoliso_irtisanottu,karenssia_jaljella=0,alku='')
                    self.setup_children(pp0,0,pe0,spouse_empstate,pc3,pc7,pc18,pc18)
                    self.setup_asumismenot(pe0,0,spouse_empstate,pc18,pp0)
                    if pw1<1.0 or pw1<pw0+1.0:
                        print('puoliso',spouse_empstate,pw0,pw1)
                        self.render()

                    pp1 = self.setup_benefits(pw1,pow1,spouse_kansanelake,puoliso_tyoelake,pe1,spouse_pt_state,spouse_time_in_state,ika,puoliso_used_unemp_benefit,
                              pc3,pc7,pc18,puoliso=0,irtisanottu=puoliso_irtisanottu,karenssia_jaljella=0,alku='')
                    self.setup_children(pp1,0,pe1,spouse_empstate,pc3,pc7,pc18,pc18)
                    self.setup_asumismenot(pe1,0,spouse_empstate,pc18,pp1)

                    nettox2,_,tvax2 = self.marg.comp_emtr(pp0,pp1,pw1,dt=1200,alku='')
                    _,effx2,_ = self.marg.comp_emtr(p2,p2,spouse_wage,dt=1200,alku='')

                    benefitq2['puoliso_emtr'] = effx2
                    benefitq2['puoliso_tva'] = tvax2
                else:
                    benefitq2['puoliso_emtr'] = np.nan
                    benefitq2['puoliso_tva'] = np.nan
            else:
                benefitq1['omat_emtr'] = np.nan
                benefitq2['puoliso_emtr'] = np.nan
                benefitq1['omat_tva'] = np.nan
                benefitq2['puoliso_tva'] = np.nan

            benefitq1['omat_potential_wage'] = potential_wage
            benefitq2['puoliso_potential_wage'] = potential_spouse_wage

            if netto1<1 and debug:
                print(f'netto<1, omat tila {employment_state}',wage,benefitbasis,main_kansanelake,tyoelake,time_in_state,ika,children_under3,children_under7,children_under18)
            if netto2<1 and puoliso>0 and debug:
                print(f'netto<1, spouse {spouse_empstate}',spouse_wage,puoliso_benefitbasis,spouse_kansanelake,puoliso_tyoelake,spouse_time_in_state,ika)

            if (benefitq1['omat_netto']<1e-6 or benefitq2['puoliso_netto']<1e-6) and debug:
                print(f'omat netto {employment_state}: ',benefitq1['omat_netto'])
                print(f'puoliso netto {spouse_empstate}: ',benefitq2['puoliso_netto'])

            netto_omat=benefitq1['omat_netto']*12
            netto_puoliso=benefitq2['puoliso_netto']*12

            benefitq = self.ben.add_q(benefitq1,benefitq2)
            benefitq['netto'] = netto
            netto=netto*12

        return netto,benefitq,netto_omat,netto_puoliso

    def seed(self, seed: int =None):
    #    '''
    #    Open AI interfacen mukainen seed-funktio, joka alustaa satunnaisluvut
    #    '''
        #self.np_random, seed = seeding.np_random(seed)

        self.env_seed(init_seed=seed)

        return [seed]

    def env_seed(self, move_seed=None,init_seed=None):
        '''
        Alustetaan numpy.random enviä varten
        '''
        if move_seed is not None:
            np.random.seed(move_seed)
            print('move_seed',move_seed)

        if init_seed is not None:
            random.seed(init_seed)
            print('init_seed',init_seed)

    def scale_pension(self,pension: float,age: float,scale: bool = True,unemp_after_ra: float = 0,elinaikakerroin: bool = True):
        '''
        Elinaikakertoimen ja lykkäyskorotuksen huomiointi
        '''
        if elinaikakerroin:
            eak = self.elinaikakerroin
        else:
            eak=1

        if scale:
            if self.plotdebug:
                print('scale pension','eak',eak,'pension',pension,'age',age,'unemp_after_ra',unemp_after_ra,'kerroin',eak*pension*self.elakeindeksi*(1+0.048*(age-self.min_retirementage-unemp_after_ra)) )

            return eak*pension*self.elakeindeksi*(1+0.048*(age-self.min_retirementage-unemp_after_ra))
        else:
            return eak*pension*self.elakeindeksi


#############################
#
# STATE TRANSITIONS
#
#############################

    def move_to_parttime(self,raw_wage: float,pt_action: int,pension: float,tyoelake: float,
                        age: float,tyoura: float,time_in_state: float,
                        has_spouse: bool = False,is_spouse: bool = False):
        '''
        Siirtymä osa-aikaiseen työskentelyyn
        '''
        employment_status = 10 # switch to part-time work
        paid_wage,pt_factor,_ = self.get_paid_wage(raw_wage,employment_status,pt_action)
        tyoura += self.timestep
        time_in_state = self.timestep
        pension = self.pension_accrual(age,paid_wage,pension,state=employment_status)
        pinkslip = 0
        tyoelake = tyoelake * self.elakeindeksi
        basis_wage=0

        return employment_status,pension,tyoelake,paid_wage,time_in_state,tyoura,pinkslip,basis_wage

    def move_to_work(self,raw_wage: float,pt_action: int,pension: float,tyoelake: float,age: float,time_in_state: float,tyoura: float,pinkslip: int,
                    has_spouse:bool =False,is_spouse:bool =False):
        '''
        Siirtymä täysiaikaiseen työskentelyyn
        '''
        employment_status = 1 # töihin
        pinkslip = 0
        paid_wage,main_pt_factor,_ = self.get_paid_wage(raw_wage,employment_status,pt_action)
        tyoura += self.timestep
        pension = self.pension_accrual(age,paid_wage,pension,state=employment_status)
        time_in_state = self.timestep
        tyoelake=tyoelake*self.elakeindeksi
        basis_wage=0

        return employment_status,pension,tyoelake,paid_wage,time_in_state,tyoura,pinkslip,basis_wage

    def move_to_oa_fulltime(self,wage: float,pt_action: int,pension: float,age: float,kansanelake: float,
            tyoelake: float,employment_status: int,unemp_after_ra: float,scale_acc:bool =True,has_spouse: int=0,is_spouse:bool =False):
        '''
        Siirtymä vanhuuseläkkeelle
        '''
        basis_wage=0
        # if age >= self.max_retirementage: # move to retirement state 2
        #     if employment_status in set([2,8,9]): # ve, ve+työ, ve+osatyö
        #         # ei lykkäyskorotusta
        #         tyoelake = self.elakeindeksi*tyoelake+self.scale_pension(pension,age,scale=False,unemp_after_ra=0)
        #         kansanelake = kansanelake * self.kelaindeksi
        #         pension=0
        #         employment_status = 9
        #         unemp_after_ra = 0
        #     else: # ei vielä eläkkeellä
        #         # lykkäyskorotus
        #         tyoelake = tyoelake*self.elakeindeksi + pension
        #         if self.include_kansanelake:
        #             kansanelake = self.ben.laske_kansanelake(age,tyoelake*self.elinaikakerroin/12,1-has_spouse)*12 # ben-modulissa palkat kk-tasolla
        #         else:
        #             kansanelake = 0
        #         # lykkäys ei vähennä kansaneläkettä
        #         tyoelake += (self.scale_pension(pension,age,scale=scale_acc,unemp_after_ra=unemp_after_ra) - pension)
        #         paid_pension = tyoelake + kansanelake
        #         pension=0
        #         employment_status = 9
        #         unemp_after_ra = 0

        #     time_in_state = self.timestep
        #     alkanut_ansiosidonnainen=0
        #     ove_paid=0
        #     #paid_wage=0
        # el
        if age >= self.min_retirementage:
            if employment_status in set([2,8,9]): # ve, ve+työ, ve+osatyö
                tyoelake = self.elakeindeksi*tyoelake
                kansanelake = kansanelake * self.kelaindeksi
                employment_status = 9
                paid_wage,pt_factor,_ = self.get_paid_wage(wage,employment_status,pt_action)
            elif employment_status==3:
                print('error: moved from disability to oa_fulltime')
            elif employment_status==4:
                # lykkäyskorotus
                tyoelake = tyoelake*self.elakeindeksi + pension
                if self.include_kansanelake:
                    # vähentämätön kansaneläke
                    kansanelake = self.ben.laske_kansanelake(age,tyoelake*self.elinaikakerroin/12,1-has_spouse,disability=True)*12 # ben-modulissa palkat kk-tasolla
                    if self.plotdebug:
                        print('tyoelake',tyoelake,'kansanelake',kansanelake)
                else:
                    kansanelake = 0
                # lykkäys ei vähennä kansaneläkettä
                tyoelake += (self.scale_pension(pension,age,scale=scale_acc,unemp_after_ra=unemp_after_ra) - pension)
                pension=0
                unemp_after_ra = 0
                employment_status = 9
                paid_wage,pt_factor,_ = self.get_paid_wage(wage,employment_status,pt_action)
            else:
                # lykkäyskorotus
                tyoelake = tyoelake*self.elakeindeksi + pension
                if self.include_kansanelake:
                    kansanelake = self.ben.laske_kansanelake(age,tyoelake*self.elinaikakerroin/12,1-has_spouse)*12 # ben-modulissa palkat kk-tasolla
                    if self.plotdebug:
                        print('tyoelake',tyoelake,'kansanelake',kansanelake)
                else:
                    kansanelake = 0
                # lykkäys ei vähennä kansaneläkettä
                tyoelake += (self.scale_pension(pension,age,scale=scale_acc,unemp_after_ra=unemp_after_ra) - pension)
                pension=0
                unemp_after_ra = 0
                employment_status = 9
                paid_wage,pt_factor,_ = self.get_paid_wage(wage,employment_status,pt_action)

            if paid_wage<1.0:
                print(f'state {employment_status} wage {wage} paid_wage {paid_wage} age {age} is_spouse {is_spouse}')
                self.render()

            time_in_state = self.timestep
            alkanut_ansiosidonnainen = 0
            pension = self.pension_accrual(age,paid_wage,pension,state=employment_status)
            ove_paid=0
        else: # työvoiman ulkopuolella
            time_in_state=0
            employment_status = 2
            pension = pension * self.palkkakerroin
            wage=0
            time_in_state += self.timestep
            ove_paid=0
            paid_wage=0

        return employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,ove_paid,basis_wage

    def move_to_student(self,wage: float,pt_action: int,pension: float,tyoelake: float,age: float,time_in_state: float,tyoura: float,pinkslip: int,g: int):
        '''
        Siirtymä opiskelijaksi
        Tässä ei muuttujien päivityksiä, koska se tehdään jo muualla!
        '''
        basis_wage=0
        employment_status = 12
        time_in_state = self.timestep
        pinkslip = 0
        tyoelake=tyoelake*self.elakeindeksi
        until_student = self.comp_time_to_study(employment_status,age,g)
        karenssia_jaljella=0

        paid_wage,pt_factor,_ = self.get_paid_wage(wage,employment_status,pt_action)
        pension = self.pension_accrual(age,paid_wage,pension,state=employment_status)
        #if pt_factor>0:
        #    tyoura += self.timestep

        return employment_status,paid_wage,pension,tyoelake,time_in_state,pinkslip,basis_wage,until_student,karenssia_jaljella,pt_factor#,tyoura

    def move_to_oa_parttime(self,wage: float,pt_action: int,pension: float,age: float,kansanelake: float,tyoelake: float,employment_status: float,
            unemp_after_ra: float,scale_acc: bool=True,has_spouse: int =0,is_spouse: bool =False):
        '''
        Siirtymä vanhuuseläkkeelle

        pension on tuleva eläke
        tyoelake on maksettava eläke
        kansaneläke on maksettava eläke, muuten se on 0
        '''
        basis_wage=0
        # if age >= self.max_retirementage: # move to retirement state 2
        #     if employment_status in set([2,8,9]): # ve, ve+työ, ve+osatyö
        #         # ei lykkäyskorotusta
        #         tyoelake = self.elakeindeksi*tyoelake+self.scale_pension(pension,age,scale=False,unemp_after_ra=0)
        #         kansanelake = kansanelake * self.kelaindeksi
        #         pension=0
        #         employment_status = 8
        #         unemp_after_ra = 0
        #     else: # ei vielä eläkkeellä
        #         # lykkäyskorotus
        #         tyoelake = tyoelake*self.elakeindeksi + pension
        #         if self.include_kansanelake:
        #             kansanelake = self.ben.laske_kansanelake(age,tyoelake*self.elinaikakerroin/12,1-has_spouse)*12 # ben-modulissa palkat kk-tasolla
        #         else:
        #             kansanelake = 0

        #         # lykkäys ei vähennä kansaneläkettä
        #         tyoelake += (self.scale_pension(pension,age,scale=scale_acc,unemp_after_ra=unemp_after_ra) - pension)
        #         pension=0
        #         unemp_after_ra = 0
        #         employment_status = 8

        #     time_in_state = self.timestep
        #     alkanut_ansiosidonnainen=0
        #     ove_paid=0
        #     #paid_wage=0
        # el
        if age >= self.min_retirementage:
            if employment_status in set([2,8,9]): # ve, ve+työ, ve+osatyö
                tyoelake = self.elakeindeksi * tyoelake
                kansanelake = kansanelake * self.kelaindeksi
                employment_status = 8
                paid_wage,pt_factor,_ = self.get_paid_wage(wage,employment_status,pt_action)
            elif employment_status==3:
                print('error: moved from disability to oa_parttime')
            elif employment_status==4:
                # lykkäyskorotus
                tyoelake = tyoelake*self.elakeindeksi + pension
                if self.include_kansanelake:
                    # ei lykkäyskorotusta
                    kansanelake = self.ben.laske_kansanelake(age,tyoelake*self.elinaikakerroin/12,1-has_spouse,disability=True)*12 # ben-modulissa palkat kk-tasolla
                    if self.plotdebug:
                        print('tyoelake',tyoelake,'kansanelake',kansanelake)
                else:
                    kansanelake = 0
                # lykkäys ei vähennä kansaneläkettä
                tyoelake += (self.scale_pension(pension,age,scale=scale_acc,unemp_after_ra=unemp_after_ra) - pension)
                pension=0
                employment_status = 8
                paid_wage,pt_factor,_ = self.get_paid_wage(wage,employment_status,pt_action)
                unemp_after_ra = 0
            else:
                # lykkäyskorotus
                tyoelake = tyoelake*self.elakeindeksi + pension
                if self.include_kansanelake:
                    # lykkäyskorotus
                    kansanelake = self.ben.laske_kansanelake(age,tyoelake*self.elinaikakerroin/12,1-has_spouse)*12 # ben-modulissa palkat kk-tasolla
                    if self.plotdebug:
                        print('tyoelake',tyoelake,'kansanelake',kansanelake)
                else:
                    kansanelake = 0
                # lykkäys ei vähennä kansaneläkettä
                tyoelake += (self.scale_pension(pension,age,scale=scale_acc,unemp_after_ra=unemp_after_ra) - pension)
                pension=0
                employment_status = 8
                unemp_after_ra = 0
                paid_wage,pt_factor,_ = self.get_paid_wage(wage,employment_status,pt_action)

            if paid_wage<1.0:
                print(f'state {employment_status} wage {wage} paid_wage {paid_wage} age {age} is_spouse {is_spouse}')
                self.render()

            time_in_state = self.timestep
            alkanut_ansiosidonnainen = 0
            pension = self.pension_accrual(age,paid_wage,pension,state=employment_status)
            ove_paid=0
        else: # työvoiman ulkopuolella
            time_in_state=0
            employment_status = 2
            wage=0
            time_in_state += self.timestep
            paid_wage=0
            ove_paid=0

        return employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,ove_paid,basis_wage

    def move_to_ove(self,employment_status: int,pension: float,tyoelake: float,ove_paid: float,age: float,unemp_after_ra: float):
        if not self.include_ove:
            return pension,tyoelake,0

        if ove_paid:
            print('Moving to OVE twice')
            error('failure')
            exit()

        if employment_status in set([2,3,8,9]): # ei eläkettä maksuun
            print('move_to_ove: From incorrect state',employment_status)
            error('failure')
            exit()
        else:
            tyoelake = tyoelake + self.scale_pension(self.ove_ratio*pension,age,scale=True,unemp_after_ra=unemp_after_ra)/self.elakeindeksi # ei eläkeindeksi tässä
            pension = (1-self.ove_ratio)*pension # *self.palkkakerroin, tässä ei indeksoida, koska pension_accrual hoitaa tämän
            ove_paid = 1

        return pension,tyoelake,ove_paid

    def move_to_retirement(self,pension: float,age: float,kansanelake: float,tyoelake: float,employment_status: float,unemp_after_ra: float,
        all_acc: bool =True,scale_acc: bool =True,has_spouse: int=0,is_spouse: bool =False):
        '''
        Moving to retirement
        '''
        basis_wage=0
        if age >= self.max_retirementage:
            paid_wage=0
            if employment_status in set([2,8,9]): # ve, ve+työ, ve+osatyö
                # ei lykkäyskorotusta
                tyoelake = self.elakeindeksi*tyoelake+self.scale_pension(pension,age,scale=False,unemp_after_ra=unemp_after_ra)
                if self.include_kansanelake:
                    kansanelake = kansanelake * self.kelaindeksi
                else:
                    kansanelake = 0
                pension = 0
                employment_status = 2
            elif employment_status==3: # tk
                # do nothing
                employment_status=3
                tyoelake = self.elakeindeksi*tyoelake+self.scale_pension(pension,age,scale=False,unemp_after_ra=unemp_after_ra)
                if self.include_kansanelake:
                    kansanelake = kansanelake * self.kelaindeksi
                else:
                    kansanelake = 0
                pension = 0
            else: # ei vielä eläkkeellä
                # lykkäyskorotus
                tyoelake = tyoelake*self.elakeindeksi + pension
                if self.include_kansanelake:
                    kansanelake = self.ben.laske_kansanelake(age,tyoelake*self.elinaikakerroin/12,1-has_spouse)*12 # ben-modulissa palkat kk-tasolla
                    if self.plotdebug:
                        print('tyoelake',tyoelake,'kansanelake',kansanelake)
                else:
                    kansanelake = 0
                # lykkäys ei vähennä kansaneläkettä
                tyoelake += (self.scale_pension(pension,age,scale=scale_acc,unemp_after_ra=unemp_after_ra) - pension)
                pension = 0
                employment_status = 2

            time_in_state = self.timestep
            alkanut_ansiosidonnainen = 0
            ove_paid=0
            until_student,until_outsider=100,100
        elif age >= self.min_retirementage or (age >= self.min_retirementage_putki and employment_status == 4):
            paid_wage=0
            if all_acc:
                if employment_status in set([2,8,9]): # ve, ve+työ, ve+osatyö
                    tyoelake = self.elakeindeksi*tyoelake
                    kansanelake = kansanelake * self.kelaindeksi
                    pension=pension*self.palkkakerroin
                    employment_status = 2
                elif employment_status==3: # tk
                    # do nothing
                    employment_status = 3
                    ove_paid=0
                    pension = pension * self.palkkakerroin
                    tyoelake = self.elakeindeksi * tyoelake
                    kansanelake = kansanelake * self.kelaindeksi
                elif employment_status==4: # putki
                    # lykkäyskorotus
                    tyoelake = tyoelake * self.elakeindeksi + pension
                    if self.include_kansanelake: # ei varhennnusvähennystä putken tapauksessa
                        kansanelake = self.ben.laske_kansanelake(age,tyoelake*self.elinaikakerroin/12,1-has_spouse,disability=True)*12 # ben-modulissa palkat kk-tasolla
                        if self.plotdebug:
                            print('tyoelake',tyoelake,'kansanelake',kansanelake)
                    else:
                        kansanelake = 0
                    # lykkäys ei vähennä kansaneläkettä
                    tyoelake += (self.scale_pension(pension,age,scale=scale_acc,unemp_after_ra=unemp_after_ra) - pension)
                    pension=0
                    ove_paid=0
                    employment_status = 2
                else:
                    # lykkäyskorotus
                    tyoelake = tyoelake * self.elakeindeksi + pension
                    if self.include_kansanelake:
                        kansanelake = self.ben.laske_kansanelake(age,tyoelake*self.elinaikakerroin/12,1-has_spouse)*12 # ben-modulissa palkat kk-tasolla
                        if self.plotdebug:
                            print('tyoelake',tyoelake,'kansanelake',kansanelake)
                    else:
                        kansanelake = 0
                    # lykkäys ei vähennä kansaneläkettä
                    tyoelake += (self.scale_pension(pension,age,scale=scale_acc,unemp_after_ra=unemp_after_ra) - pension)
                    pension=0
                    ove_paid=0
                    employment_status = 2

            elif employment_status in {8,9}: # ve, ve+työ, ve+osatyö
                tyoelake = self.elakeindeksi * tyoelake
                kansanelake = kansanelake * self.kelaindeksi
                pension = pension * self.palkkakerroin
                employment_status = 2
            else:
                print('error 289')

            time_in_state = self.timestep
            alkanut_ansiosidonnainen=0
            ove_paid=0
            until_student,until_outsider=100,100
        else: # työvoiman ulkopuolella
            paid_wage=0
            ove_paid=0
            time_in_state = 0
            employment_status = 2
            wage = 0
            time_in_state += self.timestep
            until_student,until_outsider=100,100
            print('retired before retirement age!!!!')

        #if kansanelake>10_000:
        #    print('kansanelake',kansanelake,age,tyoelake*self.elinaikakerroin/12,1-has_spouse)

        return employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,ove_paid,basis_wage,until_student,until_outsider

    def move_to_retdisab(self,pension: float,age: float,time_in_state: float,kansanelake: float,tyoelake: float,unemp_after_ra: float):
        '''
        Siirtymä vanhuuseläkkeelle, jossa ei voi tehdä työtä
        '''

        if age >= self.max_retirementage:
            # ei mene täsmälleen oikein
            tyoelake = self.elakeindeksi*tyoelake+self.scale_pension(pension,age,scale=False,unemp_after_ra=unemp_after_ra)
            kansanelake = kansanelake * self.kelaindeksi
            pension = 0
        else:
            tyoelake = self.elakeindeksi*tyoelake
            kansanelake = kansanelake * self.kelaindeksi
            pension = self.palkkakerroin*pension

        basis_wage = 0
        employment_status = 3
        paid_wage = 0
        time_in_state = self.timestep
        #wage_reduction=0.9
        alkanut_ansiosidonnainen = 0
        until_student,until_outsider = 100,100

        return employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,kansanelake,tyoelake,basis_wage,until_student,until_outsider

    def tyossaoloehto(self,toe: float,tyoura: float,age: float):
        '''
        täyttyykä työssäoloehto
        '''
        if toe >= self.ansiopvraha_toe:
            return True
        else:
            return False

    def setup_unempdays_left(self,porrastus: bool =False):
        '''
        valmistaudu toen porrastukseen
        '''
        if porrastus:
            self.comp_unempdays_left = self.comp_unempdays_left_porrastus
            self.paivarahapaivia_jaljella = self.paivarahapaivia_jaljella_porrastus
            self.comp_toe_wage = self.comp_toe_wage_porrastus
        else:
            self.comp_unempdays_left = self.comp_unempdays_left_nykytila
            self.paivarahapaivia_jaljella = self.paivarahapaivia_jaljella_nykytila
            self.comp_toe_wage = self.comp_toe_wage_nykytila

    def comp_unempdays_left_nykytila(self,kesto: float,tyoura: float,age: float,toe: float,emp: int,alkanut_ansiosidonnainen: int,toe58: int,old_toe: float,printti: bool =False):
        '''
        Nykytilan mukainen jäljellä olevien työttämyyspäivärahapäivien laskenta
        '''
        if emp in set([2,3,8,9,13]):
            return 0

        if self.get_kassanjasenyys()<1:
            return 0

        toe_tayttyy = self.tyossaoloehto(toe,tyoura,age)

        if self.include_putki and (
            emp==4
            or (emp==0 and age >= self.min_tyottputki_ika and tyoura >= self.tyohistoria_tyottputki)
            or (emp in set ([1,10]) and age >= self.min_tyottputki_ika and tyoura >= self.tyohistoria_tyottputki and toe_tayttyy)
            ):
            return max(0,self.max_unemploymentbenefitage-age)

        if (not toe_tayttyy) and alkanut_ansiosidonnainen<1:
            return 0

        if toe_tayttyy:
            kesto=0

        if tyoura >= self.tyohistoria_vaatimus500 and age >= self.minage_500 and toe58>0:
            toekesto=max(0,self.apvkesto500-kesto)
        elif tyoura >= self.tyohistoria_vaatimus:
            toekesto=max(0,self.apvkesto400-kesto)
        else:
            toekesto=max(0,self.apvkesto300-kesto)

        return max(0,min(toekesto,self.max_unemploymentbenefitage-age))

    def paivarahapaivia_jaljella_nykytila(self,kesto: float,tyoura: float,age: float,toe58: int,toe: int):
        '''
        Onko työttämyyspäivärahapäiviä jäljellä?
        '''
        if age >= self.max_unemploymentbenefitage:
            return False

        if self.get_kassanjasenyys()<1:
            return False

        if ((tyoura >= self.tyohistoria_vaatimus500 and kesto >= self.apvkesto500 and age >= self.minage_500 and toe58>0) \
            or (tyoura >= self.tyohistoria_vaatimus and kesto >= self.apvkesto400 and (age<self.minage_500 or tyoura<self.tyohistoria_vaatimus500 or toe58<1)) \
            or (tyoura<self.tyohistoria_vaatimus and kesto >= self.apvkesto300)):
            return False
        else:
            return True

    def comp_unempdays_left_porrastus(self,kesto: float,tyoura: float,age: float,toe: float,emp: int,alkanut_ansiosidonnainen: int,toe58: int,old_toe: float,printti: bool=False):
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

        if tyoura >= self.tyohistoria_vaatimus500 and age >= self.minage_500 and toe58>0 and not self.porrastus500:
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

    def toe_porrastus_kesto(self,kesto: float,toe: float,tyoura: float):
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

    def paivarahapaivia_jaljella_porrastus(self,kesto: float,tyoura: float,age: float,toe58: float,toe: float):
        if age >= self.max_unemploymentbenefitage:
            return False

        if (tyoura >= self.tyohistoria_vaatimus500 and kesto >= self.apvkesto500 and age >= self.minage_500 and toe58>0 and not self.porrastus500) \
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

    def move_to_unemp(self,pension: bool,old_wage: bool,age: bool,kansanelake: bool,tyoelake: bool,toe: bool,toekesto: bool,irtisanottu: int,tyoura: bool,
                    used_unemp_benefit: bool,unemp_after_ra: bool,unempwage: bool,unempwage_basis: bool,alkanut_ansiosidonnainen: int,toe58: float,
                    ove_paid: int,has_spouse: bool,is_spouse: int):
        '''
        Siirtymä työttämyysturvalle
        '''
        basis_wage=0
        if age >= self.min_retirementage: # ei uusia työttämiä enää alimman ve-iän jälkeen, vanhat jatkavat
            pinkslip = 0
            employment_status=0
            unempwage_basis=0
            alkanut_ansiosidonnainen = 0
            used_unemp_benefit = 0
            karenssia_jaljella=0
            paid_wage=0

            employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,ove_paid,basis_wage,until_student,until_outsider=\
                self.move_to_retirement(pension,age,kansanelake,tyoelake,employment_status,
                    unemp_after_ra,all_acc=True,scale_acc=True,has_spouse=has_spouse,is_spouse=is_spouse)

            return employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,\
                   used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,\
                   alkanut_ansiosidonnainen,karenssia_jaljella,ove_paid,basis_wage
        else:
            tehto = self.tyossaoloehto(toe,tyoura,age)
            if tehto or alkanut_ansiosidonnainen>0:
                if tehto:
                    kesto=0
                    used_unemp_benefit=0
                    if age>=58 and self.suojasaanto_toe58: # suojasääntö 58v täyttäneille
                        unempwage_basis=max(unempwage,self.update_unempwage_basis(unempwage_basis,unempwage,False))
                    else:
                        if self.infostate_check_aareset(age,is_spouse=is_spouse):
                            unempwage_basis = self.update_unempwage_basis(unempwage_basis,unempwage,True)
                        else:
                            unempwage_basis = self.update_unempwage_basis(unempwage_basis,unempwage,False)
                    self.infostate_set_enimmaisaika(age,is_spouse=is_spouse) # resetoidaan enimmäisaika

                    jaljella = self.paivarahapaivia_jaljella(kesto,tyoura,age,toe58,toekesto)
                else:
                    kesto=used_unemp_benefit
                    if self.porrasta_toe:
                        jaljella = self.paivarahapaivia_jaljella(kesto,tyoura,age,toe58,self.comp_oldtoe(spouse=is_spouse))
                    else:
                        jaljella = self.paivarahapaivia_jaljella(kesto,tyoura,age,toe58,toekesto) # toe ei vaikuta

                if jaljella:
                    employment_status  = 0 # siirto ansiosidonnaiselle
                    #if alkanut_ansiosidonnainen<1:
                    if irtisanottu>0 or alkanut_ansiosidonnainen>0: # muuten ei oikeutta ansiopäivärahaan karenssi vuoksi
                        used_unemp_benefit += self.timestep
                        karenssia_jaljella=0.0
                    else:
                        karenssia_jaljella=0.25 # 90 pv
                    alkanut_ansiosidonnainen = 1
                else:
                    if self.include_putki and age >= self.min_tyottputki_ika and tyoura >= self.tyohistoria_tyottputki:
                        employment_status = 4 # siirto lisäpäiville
                        used_unemp_benefit += self.timestep
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

            #time_in_state=0
            paid_wage=0

            if karenssia_jaljella>0:
                pension=pension*self.palkkakerroin
            else:
                if tyoelake>0: # ove maksussa, ei karttumaa
                    pension = self.pension_accrual(age,unempwage_basis,pension,state=employment_status,ove_paid=1)
                else:
                    pension = self.pension_accrual(age,unempwage_basis,pension,state=employment_status)

            time_in_state = self.timestep

            # Tässä ei tehdä karenssia_jaljella -muuttujasta tilamuuttujaa, koska karenssin kesto on lyhyempi kuin aika-askeleen
            # Samoin karenssia ei ole tm-tuessa, koska toimeentulotuki on suurempi

            pinkslip=irtisanottu

        return employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,\
               used_unemp_benefit,pinkslip,unemp_after_ra,\
               unempwage_basis,alkanut_ansiosidonnainen,karenssia_jaljella,ove_paid,basis_wage

    def update_karenssi(self,karenssia_jaljella: float):
        karenssia_jaljella=max(0,karenssia_jaljella-self.timestep)
        return karenssia_jaljella

    def move_to_outsider(self,pension: float,tyoelake: float,age: float, g: int,moved: bool=False):
        '''
        Siirtymä työvoiman ulkopuolelle
        '''
        employment_status = 11 # switch
        time_in_state = self.timestep
        pinkslip,paid_wage,basis_wage,karenssia_jaljella = 1,0,0,0  # ei karenssia jos siirtyy työttömäksi
        until_outsider = self.comp_time_to_outsider(employment_status,age,g)

        if not moved:
            pension=pension*self.palkkakerroin
            tyoelake=tyoelake*self.elakeindeksi

        return employment_status,pension,tyoelake,paid_wage,time_in_state,pinkslip,basis_wage,until_outsider,karenssia_jaljella

    def move_to_svraha(self,old_paid_wage: float,age: float,pension: float,tyoelake: float,kansanelake: float,ove_paid: int,is_spouse: bool,life_left: float):
        '''
        Siirtymä sairaspäivärahalle aktiivista, ei eläkkeeltä
        '''
        employment_status = 14 # sv-päiväraha
        kansanelake = kansanelake * self.kelaindeksi
        basis_wage = self.infostate_comp_svpaivaraha_1v(is_spouse=is_spouse)
        paid_wage,pt_factor,_ = self.get_paid_wage(0,employment_status,0,old_paid_wage,0)
        pension = self.pension_accrual(age,paid_wage,pension,state=5)
        tyoelake = tyoelake*self.elakeindeksi
        time_in_state = self.timestep
        pinkslip=1 # ei karenssia jos siirty työttömäksi
        until_disab = 1.0 # self.comp_until_disab(g,age,state=employment_status)

        return employment_status,pension,kansanelake,tyoelake,paid_wage,time_in_state,ove_paid,basis_wage,until_disab,life_left,pinkslip

    def move_to_disab(self,pension: float,old_paid_wage: float,age: float,unemp_after_ra: float,kansanelake: float,tyoelake: float,ove_paid: int,has_spouse: int,children_under18: int,is_spouse: bool,life_left: float):
        '''
        Siirtymä työkyvyttämyyseläkkeelle aktiivista, ei eläkkeeltä
        '''
        if age<self.max_svbenefitage: # ensisijaisuusaika sv-päivärahaa
            return self.move_to_svraha(old_paid_wage,age,pension,tyoelake,kansanelake,ove_paid,is_spouse,life_left)
        else:
            return self.move_to_disab_state(pension,old_paid_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,has_spouse,children_under18,is_spouse,life_left)

    def move_to_disab_state(self,pension: float,old_wage: float,age: float,unemp_after_ra: float,kansanelake: float,
            tyoelake: float,ove_paid: float,has_spouse: int,children_under18: int,is_spouse: bool,life_left: float):
        '''
        Siirtymä työkyvyttämyyseläkkeelle aktiivista, ei eläkkeeltä
        '''
        employment_status = 3 # tk
        paid_wage=0
        basis_wage=0
        if age<self.min_retirementage:
            wage5y,realwage = self.infostate_comp_5y_ave_wage(is_spouse=is_spouse)
            if is_spouse:
                self.spouse_dis_wage5y=realwage/5
            else:
                self.main_dis_wage5y=realwage/5

            if realwage>self.min_disab_tulevaaika: # oikeus tulevaan aikaan, tosin tässä 5v ajalta laskettuna
                basis_wage = self.elinaikakerroin*(pension+self.acc/self.timestep*wage5y*max(0,self.min_retirementage-age))
                tyoelake=(tyoelake+basis_wage)*self.elakeindeksi
            else:
                tyoelake=(tyoelake+self.elinaikakerroin*pension)*self.elakeindeksi

            if self.include_kansanelake:
                kansanelake = self.ben.laske_kansanelake(age,tyoelake/12,1-has_spouse,disability=True,lapsia=children_under18)*12 # ben-modulissa palkat kk-tasolla
                if self.plotdebug:
                    print('tyoelake',tyoelake,'kansanelake',kansanelake)
            else:
                kansanelake = 0

            pension=0
            alkanut_ansiosidonnainen=0
            time_in_state = self.timestep
            ove_paid=0 # ?? oikein??
        else:
            # siirtymä vanhuuseläkkeelle, lykkäyskorotus, ei tulevaa aikaa
            tyoelake = tyoelake*self.elakeindeksi + pension
            if self.include_kansanelake:
                kansanelake = self.ben.laske_kansanelake(age,tyoelake*self.elinaikakerroin/12,1-has_spouse,lapsia=children_under18)*12 # ben-modulissa palkat kk-tasolla
                if self.plotdebug:
                    print('tyoelake',tyoelake,'kansanelake',kansanelake)
            else:
                kansanelake = 0
            # lykkäys ei vähennä kansaneläkettä
            tyoelake += (self.scale_pension(pension,age,scale=True,unemp_after_ra=unemp_after_ra) - pension)
            pension=0

            time_in_state = self.timestep
            alkanut_ansiosidonnainen=0
            ove_paid=0
            #wage_reduction=0.60 # vastaa määritelmää

        until_disab=100.0
        until_student=100.0
        until_outsider=100.0
        life_left = life_left # should be self.comp_life_left(group,age,3) but group is not known
        pinkslip = 0

        # tässä pitäisi päivittää myös elinaika
        return employment_status,pension,kansanelake,tyoelake,paid_wage,time_in_state,ove_paid,basis_wage,until_disab,life_left,pinkslip #,until_student,until_outsider

    def comp_familypension(self,puoliso: float,emp_state: int,spouse_empstate: int,tyoelake: float,pension: float,age: float,
                            puoliso_tyoelake: float,spouse_pension: float,children_under18: int,has_spouse: bool,is_spouse: bool):
        '''
        Siirtymä työkyvyttämyyseläkkeelle aktiivista, ei eläkkeeltä
        '''
        if spouse_empstate==15:
            return 0

        if has_spouse<1:
            return puoliso_tyoelake

        if emp_state in set([2,3,8,9]):
            add_pension=0.5*tyoelake
        elif age<self.min_retirementage:
            wage5y,_ = self.infostate_comp_5y_ave_wage(is_spouse=not is_spouse)
            dis_pension=(tyoelake+self.elinaikakerroin*(pension+self.acc/self.timestep*wage5y*max(0,self.min_retirementage-age)))*self.elakeindeksi
            add_pension=0.5*dis_pension
        else:
            tyoelake = tyoelake*self.elakeindeksi + self.scale_pension(pension,age,scale=True,unemp_after_ra=0)
            add_pension=0.5*tyoelake

        if spouse_empstate in set([2,3,8,9]):
            omaelake=puoliso_tyoelake
        elif age<self.min_retirementage:
            wage5y,_ = self.infostate_comp_5y_ave_wage(is_spouse=is_spouse)
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

    def move_to_deceiced(self):
        '''
        Siirtymä tilaan kuollut
        '''
        employment_status = 15 # deceiced
        wage = 0
        pension = 0
        tyoelake_maksussa = 0
        time_in_state = 0
        alkanut_ansiosidonnainen = 0
        puoliso = 0
        basis_wage = 0
        kansanelake = 0

        if self.mortplot:
            self.plotdebug=True

        return employment_status,pension,wage,time_in_state,puoliso,tyoelake_maksussa,kansanelake,basis_wage

    def move_to_kht(self,pension: float,tyoelake: float,old_wage: float,age: float):
        '''
        Siirtymä kotihoidontuelle
        '''
        employment_status = 7 # kotihoidontuelle
        pension = self.pension_accrual(age,old_wage,pension,state=7)
        tyoelake=tyoelake*self.elakeindeksi
        pinkslip = 0 # karenssi jos siirtyy työttömäksi
        paid_wage=0
        basis_wage=0

        time_in_state = self.timestep

        return employment_status,pension,tyoelake,paid_wage,time_in_state,pinkslip,basis_wage

    def move_to_fatherleave(self,pension: float,tyoelake: float,age: float,is_spouse : bool,old_paid_wage: float):
        '''
        Siirtymä isyysvapaalle
        '''

        #self.infostate_add_child(age) # only for the mother
        employment_status = 6 # isyysvapaa
        time_in_state = 0
        basis_wage = self.infostate_comp_svpaivaraha_1v(is_spouse=is_spouse)
        paid_wage,pt_factor,_ = self.get_paid_wage(0,employment_status,0,old_paid_wage,0)
        pension = self.pension_accrual(age,paid_wage,pension,state=6)
        tyoelake = tyoelake*self.elakeindeksi
        time_in_state += self.timestep
        pinkslip = 0 # karenssi jos siirtyy työttömäksi
        paid_wage=0

        return employment_status,pension,tyoelake,paid_wage,time_in_state,pinkslip,basis_wage

    def move_to_motherleave(self,pension: float,tyoelake: float,age: float,is_spouse : bool,old_paid_wage: float):
        '''
        Siirtymä äitiysvapaalle
        '''
        employment_status = 5 # äitiysvapaa
        time_in_state = 0
        basis_wage = self.infostate_comp_svpaivaraha_1v(is_spouse=is_spouse)
        paid_wage,pt_factor,_ = self.get_paid_wage(0,employment_status,0,old_paid_wage,0)
        pension = self.pension_accrual(age,paid_wage,pension,state=5)
        tyoelake = tyoelake*self.elakeindeksi
        time_in_state += self.timestep
        pinkslip = 0 # karenssi jos siirtyy työttömäksi

        return employment_status,pension,tyoelake,paid_wage,time_in_state,pinkslip,basis_wage

    def under_unemp_ft_reemp_rate(self,intage: int,g: int,career: float,s: float):
        vrt = self.unemp_reemp_ft_prob[intage,g]
        if intage<50:
            if career>2.0:
                vrt += 0.04            
            elif career>15.0:
                vrt += 0.08
        elif intage<60:
            if career>10.0:
                vrt += 0.04
        else:
            if career>20.0:
                vrt += 0.02

        if s<vrt:
            return True
        else:
            return False

    def under_unemp_pt_reemp_rate(self,intage: int,g: int,career: float,s: float):
        vrt = self.unemp_reemp_pt_prob[intage,g]
        if intage<25:
            if career>1.0:
                vrt += 0.05            
            elif career>5.0:
                vrt += 0.10
        elif intage<55:
            if career>2.0:
                vrt += 0.05            
            elif career>15.0:
                vrt += 0.10
        elif intage<65:
            if career>10.0:
                vrt += 0.05
        else:
            if career>20.0:
                vrt += 0.05

        if s<vrt:
            return True
        else:
            return False


    def stay_unemployed(self,raw_wage: float,paid_wage: float,pt_action: int,employment_status: int,
                        kansanelake: float,tyoelake: float,pension: float,time_in_state: float,toe: float,toekesto: float,
                        tyoura: float,used_unemp_benefit: float,pinkslip: int,unemp_after_ra: float,old_paid_wage: float,unempwage: float,
                        unempwage_basis: float,action: int,age: float,sattuma,intage: int,g: int ,alkanut_ansiosidonnainen: int,
                        toe58: int,ove_paid: int,children_under3: int,children_under18: int,basis_wage: float,has_spouse: int,is_spouse: bool,
                        until_disab: float,until_student: float,until_outsider: float,life_left: float):
        '''
        Pysy tilassa työtön (0)
        '''
        time_in_state += self.timestep
        karenssia_jaljella=0

        # if the aim is to be employed, there is a definite age-dependent probability that a person is reemployed
        if  self.unemp_limit_reemp and self.randomness:
            if (action == 1):
                if not self.under_unemp_ft_reemp_rate(intage,g,tyoura,sattuma[7]): #sattuma[7]>self.unemp_reemp_ft_prob[intage,g]:
                    if self.under_unemp_pt_reemp_rate(intage,g,tyoura,sattuma[7]): #sattuma[7]<self.unemp_reemp_pt_prob[intage,g]:
                        action = 3
                    else:
                        action = 0
            elif (action == 3 or (action==2 and age < self.min_retirementage) or action == 4):
                if not self.under_unemp_pt_reemp_rate(intage,g,tyoura,sattuma[7]): #sattuma[7]>self.unemp_reemp_pt_prob[intage,g]:
                    action = 0

        if age >= self.max_unemploymentbenefitage:
            # karttuma vanhan palkan mukaan myös tässä
            employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,ove_paid,basis_wage,until_student,until_outsider\
                 = self.move_to_retirement(pension,age,kansanelake,tyoelake,employment_status,
                    unemp_after_ra,all_acc=True,scale_acc=True,has_spouse=has_spouse,is_spouse=is_spouse)
        elif action == 0:# or action == 5:
            employment_status = 0 # unchanged

            if self.porrasta_toe:
                oldtoe = self.comp_oldtoe(spouse=spouse_value)
            else:
                oldtoe=0

            #if action == 5 and (not ove_paid) and (age >= self.min_ove_age):
            #    pension,tyoelake,ove_paid = self.move_to_ove(employment_status,pension,tyoelake,ove_paid,age,unemp_after_ra)

            kesto=used_unemp_benefit
            tyoelake=tyoelake*self.elakeindeksi

            if not self.paivarahapaivia_jaljella(kesto,tyoura,age,toe58,oldtoe):
                if self.include_putki and age >= self.min_tyottputki_ika and tyoura >= self.tyohistoria_tyottputki:
                    employment_status = 4 # siirto lisäpäiville
                    pension = self.pension_accrual(age,unempwage_basis,pension,state=4,ove_paid=ove_paid)
                    used_unemp_benefit += self.timestep
                else:
                    employment_status = 13 # siirto työmarkkinatuelle
                    alkanut_ansiosidonnainen=0
                    pension = self.pension_accrual(age,old_paid_wage,pension,state=13)
            else:
                pension = self.pension_accrual(age,unempwage_basis,pension,state=0,ove_paid=ove_paid)
                used_unemp_benefit += self.timestep # sic!

            if age >= self.min_retirementage:
                unemp_after_ra += self.timestep

        elif action == 1: # kokoaikatyö
            employment_status,pension,tyoelake,paid_wage,time_in_state,tyoura,pinkslip,basis_wage=\
                self.move_to_work(raw_wage,pt_action,pension,tyoelake,age,time_in_state,tyoura,pinkslip)
        elif action == 3: # osatyö 50%
            employment_status,pension,tyoelake,paid_wage,time_in_state,tyoura,pinkslip,basis_wage=\
                self.move_to_parttime(raw_wage,pt_action,pension,tyoelake,age,tyoura,time_in_state)
        elif action == 2:
            if age >= self.min_retirementage: # ve
                employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,ove_paid,basis_wage,until_student,until_outsider=\
                    self.move_to_retirement(pension,age,kansanelake,tyoelake,employment_status,unemp_after_ra,scale_acc=True,has_spouse=has_spouse,is_spouse=is_spouse)
            elif children_under3>0:
                employment_status,pension,tyoelake,paid_wage,time_in_state,pinkslip,basis_wage = self.move_to_kht(pension,tyoelake,old_paid_wage,age)
            else:
                employment_status,pension,tyoelake,paid_wage,time_in_state,tyoura,pinkslip,basis_wage=\
                    self.move_to_parttime(raw_wage,pt_action,pension,tyoelake,age,tyoura,time_in_state)
        elif action == 4: # osatyö 50% + ve
            if age >= self.min_retirementage: # ve
                employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,ove_paid,basis_wage=\
                    self.move_to_oa_parttime(raw_wage,pt_action,pension,age,kansanelake,tyoelake,employment_status,
                            unemp_after_ra,scale_acc=True,has_spouse=has_spouse,is_spouse=is_spouse)
            else:
                # no ove here to keep code simple
                employment_status,pension,tyoelake,paid_wage,time_in_state,tyoura,pinkslip,basis_wage=\
                    self.move_to_parttime(raw_wage,pt_action,pension,tyoelake,age,tyoura,time_in_state)
        elif action == 11: # tk
            employment_status,pension,kansanelake,tyoelake,paid_wage,time_in_state,ove_paid,basis_wage,until_disab,life_left,pinkslip=\
                self.move_to_disab(pension,old_paid_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,has_spouse,children_under18,is_spouse,life_left)
        else:
            print('error 17')

        return employment_status,kansanelake,tyoelake,pension,time_in_state,\
            pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
            alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella,basis_wage,until_disab,\
            until_student,until_outsider,life_left

    def stay_tyomarkkinatuki(self,raw_wage: float,paid_wage: float,pt_action: int,employment_status: int,
                        kansanelake: float,tyoelake: float,pension: float,time_in_state: float,toe: float,toekesto: float,
                        tyoura: float,used_unemp_benefit: float,pinkslip: int,unemp_after_ra: float,old_paid_wage: float,unempwage: float,
                        unempwage_basis: float,action: int,age: float,sattuma,intage: int,g: int ,alkanut_ansiosidonnainen: int,
                        toe58: int,ove_paid: int,children_under3: int,children_under18: int,basis_wage: float,has_spouse: int,is_spouse: bool,
                        until_disab: float,until_student: float,until_outsider: float,life_left: float):
        '''
        Pysy tilassa työmarkkinatuki (13)
        '''
        time_in_state += self.timestep
        karenssia_jaljella=0

        if  self.unemp_limit_reemp and self.randomness:
            if (action == 1):
                if not self.under_unemp_ft_reemp_rate(intage,g,tyoura,sattuma[7]): #sattuma[7]>self.unemp_reemp_ft_prob[intage,g]:
                    if self.under_unemp_pt_reemp_rate(intage,g,tyoura,sattuma[7]): #sattuma[7]<self.unemp_reemp_pt_prob[intage,g]:
                        action = 3
                    else:
                        action = 0
            elif (action == 3 or (action==2 and age < self.min_retirementage) or action == 4):
                if not self.under_unemp_pt_reemp_rate(intage,g,tyoura,sattuma[7]): #sattuma[7]>self.unemp_reemp_pt_prob[intage,g]:
                    action = 0

        if age >= self.max_unemploymentbenefitage:
            employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,ove_paid,basis_wage,until_student,until_outsider=\
                self.move_to_retirement(pension,age,kansanelake,tyoelake,employment_status,
                    unemp_after_ra,all_acc=True,scale_acc=True,has_spouse=has_spouse,is_spouse=is_spouse)
        elif action == 0:# or action == 5:
            employment_status = 13 # unchanged

            #if action == 5 and (not ove_paid) and (age >= self.min_ove_age):
            #    pension,tyoelake,ove_paid = self.move_to_ove(employment_status,pension,tyoelake,ove_paid,age,unemp_after_ra)

            tyoelake=tyoelake*self.elakeindeksi
            pension = self.pension_accrual(age,0,pension,state=13)

            if age >= self.min_retirementage:
                unemp_after_ra += self.timestep

        elif action == 1: #
            employment_status,pension,tyoelake,paid_wage,time_in_state,tyoura,pinkslip,basis_wage=\
                self.move_to_work(raw_wage,pt_action,pension,tyoelake,age,time_in_state,tyoura,pinkslip)
        elif action == 3: # osatyö 50%
            employment_status,pension,tyoelake,paid_wage,time_in_state,tyoura,pinkslip,basis_wage=\
                self.move_to_parttime(raw_wage,pt_action,pension,tyoelake,age,tyoura,time_in_state)
        elif action == 2:
            if age >= self.min_retirementage: # ve
                employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,ove_paid,basis_wage,until_student,until_outsider=\
                    self.move_to_retirement(pension,age,kansanelake,tyoelake,employment_status,
                        unemp_after_ra,scale_acc=True,has_spouse=has_spouse,is_spouse=is_spouse)
            elif children_under3>0:
                employment_status,pension,tyoelake,paid_wage,time_in_state,pinkslip,basis_wage = self.move_to_kht(pension,tyoelake,old_paid_wage,age)
            else: # move to ove, do no change state
                if (not ove_paid) and (age >= self.min_ove_age):
                    pension,tyoelake,ove_paid = self.move_to_ove(employment_status,pension,tyoelake,ove_paid,age,unemp_after_ra)

                tyoelake=tyoelake*self.elakeindeksi
                pension = self.pension_accrual(age,0,pension,state=13)

                if age >= self.min_retirementage:
                    unemp_after_ra += self.timestep

                #employment_status,pension,tyoelake,paid_wage,time_in_state,tyoura,pinkslip,basis_wage=\
                #    self.move_to_parttime(raw_wage,pt_action,pension,tyoelake,age,tyoura,time_in_state)
        elif action == 4: # osatyö 50% + ve
            if age >= self.min_retirementage: # ve
                employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,ove_paid,basis_wage=\
                    self.move_to_oa_parttime(raw_wage,pt_action,pension,age,kansanelake,tyoelake,employment_status,
                            unemp_after_ra,scale_acc=False,has_spouse=has_spouse,is_spouse=is_spouse)
            else:
                employment_status,pension,tyoelake,paid_wage,time_in_state,tyoura,pinkslip,basis_wage=\
                    self.move_to_parttime(raw_wage,pt_action,pension,tyoelake,age,tyoura,time_in_state)
        elif action == 11: # tk
            employment_status,pension,kansanelake,tyoelake,paid_wage,time_in_state,ove_paid,basis_wage,until_disab,life_left,pinkslip=\
                self.move_to_disab(pension,old_paid_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,has_spouse,children_under18,is_spouse,life_left)
        else:
            print('error 17')

        return employment_status,kansanelake,tyoelake,pension,time_in_state,\
               pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella,basis_wage,until_disab,\
               until_student,until_outsider,life_left

    def stay_pipeline(self,raw_wage: float,paid_wage: float,pt_action: int,employment_status: int,
                        kansanelake: float,tyoelake: float,pension: float,time_in_state: float,toe: float,toekesto: float,
                        tyoura: float,used_unemp_benefit: float,pinkslip: int,unemp_after_ra: float,old_paid_wage: float,unempwage: float,
                        unempwage_basis: float,action: int,age: float,sattuma,intage: int,g: int ,alkanut_ansiosidonnainen: int,
                        toe58: int,ove_paid: int,children_under3: int,children_under18: int,basis_wage: float,has_spouse: int,is_spouse: bool,
                        until_disab: float,until_student: float,until_outsider: float,life_left: float):
        '''
        Pysy tilassa työttämyysputki (4)
        '''
        time_in_state += self.timestep
        karenssia_jaljella=0


        if self.unemp_limit_reemp and self.randomness:
            if (action == 1):
                if not self.under_unemp_ft_reemp_rate(intage,g,tyoura,sattuma[7]): #sattuma[7]>self.unemp_reemp_ft_prob[intage,g]:
                    if self.under_unemp_pt_reemp_rate(intage,g,tyoura,sattuma[7]): #sattuma[7]<self.unemp_reemp_pt_prob[intage,g]:
                        action = 3
                    else:
                        action = 0
            elif (action == 3 or (action==2 and age < self.min_retirementage) or action == 4):
                if not self.under_unemp_pt_reemp_rate(intage,g,tyoura,sattuma[7]): #sattuma[7]>self.unemp_reemp_pt_prob[intage,g]:
                    action = 0


        #if action>0:
        #    if (action == 1) and self.unemp_limit_reemp:
        #        if sattuma[7]>self.unemp_reemp_ft_prob[intage,g] and self.randomness:
        ##            if sattuma[7]>self.unemp_reemp_pt_prob[intage,g] and self.randomness:
        #                action = 3
        ##            else:
        #                action = 0
        #    elif (action == 3 or (action==2 and age < self.min_retirementage) or action == 4) and self.unemp_limit_reemp:
        #        if sattuma[7]>self.unemp_reemp_pt_prob[intage,g] and self.randomness:
        #            action = 0

        if age >= self.max_unemploymentbenefitage:
            employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,ove_paid,basis_wage,until_student,until_outsider=\
                self.move_to_retirement(pension,age,kansanelake,tyoelake,employment_status,
                    unemp_after_ra,all_acc=True,scale_acc=True,has_spouse=has_spouse,is_spouse=is_spouse)
        elif action == 0: # or action == 5:
            employment_status  = 4 # unchanged
            pension = self.pension_accrual(age,unempwage_basis,pension,state=4,ove_paid=ove_paid)

            #if action == 5 and (not ove_paid) and (age >= self.min_ove_age):
            #    pension,tyoelake,ove_paid = self.move_to_ove(employment_status,pension,tyoelake,ove_paid,age,unemp_after_ra)

            tyoelake=tyoelake*self.elakeindeksi
            used_unemp_benefit += self.timestep
            if age >= self.min_retirementage:
                unemp_after_ra += self.timestep
        elif action == 1:
            employment_status,pension,tyoelake,paid_wage,time_in_state,tyoura,pinkslip,basis_wage=\
                self.move_to_work(raw_wage,pt_action,pension,tyoelake,age,time_in_state,tyoura,pinkslip)
        elif action == 3:
            employment_status,pension,tyoelake,paid_wage,time_in_state,tyoura,pinkslip,basis_wage=\
                self.move_to_parttime(raw_wage,pt_action,pension,tyoelake,age,tyoura,time_in_state)
        elif action==2:
            if age >= self.min_retirementage_putki:
                if age < self.min_retirementage:
                    employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,ove_paid,basis_wage,until_student,until_outsider=\
                        self.move_to_retirement(pension,age,kansanelake,tyoelake,employment_status,
                            unemp_after_ra,all_acc=True,scale_acc=False,has_spouse=has_spouse,is_spouse=is_spouse)
                else:
                    employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,ove_paid,basis_wage,until_student,until_outsider=\
                        self.move_to_retirement(pension,age,kansanelake,tyoelake,employment_status,
                            unemp_after_ra,all_acc=True,scale_acc=True,has_spouse=has_spouse,is_spouse=is_spouse)
                pinkslip = 0
            else:
                # stay in state, but get ove
                employment_status  = 4 # unchanged
                pension = self.pension_accrual(age,unempwage_basis,pension,state=4,ove_paid=ove_paid)

                if (not ove_paid) and (age >= self.min_ove_age):
                    pension,tyoelake,ove_paid = self.move_to_ove(employment_status,pension,tyoelake,ove_paid,age,unemp_after_ra)

                tyoelake=tyoelake*self.elakeindeksi
                used_unemp_benefit += self.timestep
                if age >= self.min_retirementage:
                    unemp_after_ra += self.timestep
        elif action == 4: # osatyö 50% + ve
            if age >= self.min_retirementage: # ve
                employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,ove_paid,basis_wage=\
                    self.move_to_oa_parttime(raw_wage,pt_action,pension,age,kansanelake,tyoelake,employment_status,
                        unemp_after_ra,scale_acc=True,has_spouse=has_spouse,is_spouse=is_spouse)
            else:
                # no ove here to keep code simple
                employment_status,pension,tyoelake,paid_wage,time_in_state,tyoura,pinkslip,basis_wage=\
                    self.move_to_parttime(raw_wage,pt_action,pension,tyoelake,age,tyoura,time_in_state)
        elif action == 11: # tk
            employment_status,pension,kansanelake,tyoelake,paid_wage,time_in_state,ove_paid,basis_wage,until_disab,life_left,pinkslip=\
                self.move_to_disab(pension,old_paid_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,has_spouse,children_under18,is_spouse,life_left)
        else:
            print('error 1: ',action)

        return employment_status,kansanelake,tyoelake,pension,time_in_state,\
               pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella,basis_wage,until_disab,\
               until_student,until_outsider,life_left

    def stay_employed(self,raw_wage: float,paid_wage: float,pt_action: int,employment_status: int,
                        kansanelake: float,tyoelake: float,pension: float,time_in_state: float,toe: float,toekesto: float,
                        tyoura: float,used_unemp_benefit: float,pinkslip: int,unemp_after_ra: float,old_paid_wage: float,unempwage: float,
                        unempwage_basis: float,action: int,age: float,sattuma,intage: int,g: int ,alkanut_ansiosidonnainen: int,
                        toe58: int,ove_paid: int,children_under3: int,children_under18: int,basis_wage: float,has_spouse: int,is_spouse: bool,
                        until_disab: float,until_student: float,until_outsider: float,life_left: float):
        '''
        Pysy tilassa töissä (1)
        '''
        time_in_state += self.timestep
        karenssia_jaljella=0
        if sattuma[1]<self.pinkslip_intensity[g,intage]:
            if age<self.min_retirementage:
                pinkslip = 1
                action = 1 # unemp
            else:
                pinkslip = 0
                action = 2 # ve
        else:
            pinkslip = 0

        #if action == 3: # or (action == 4 and age < self.min_ove_age):
        #    if sattuma[7]>self.fulltime_pt_prob and self.randomness:
        #        action = 0

        if action == 0: # or action == 5:
            employment_status = 1 # unchanged

            #if action == 5 and (not ove_paid) and (age>self.min_ove_age):
            #    pension,tyoelake,ove_paid = self.move_to_ove(employment_status,pension,tyoelake,ove_paid,age,unemp_after_ra)

            tyoelake=tyoelake*self.elakeindeksi
            tyoura += self.timestep
            pension = self.pension_accrual(age,paid_wage,pension,state=1)
        elif action == 1: # työttömäksi
            employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,\
                used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,alkanut_ansiosidonnainen,karenssia_jaljella,ove_paid,basis_wage=\
                self.move_to_unemp(pension,old_paid_wage,age,kansanelake,tyoelake,toe,toekesto,pinkslip,tyoura,
                    used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,alkanut_ansiosidonnainen,toe58,ove_paid,has_spouse,is_spouse)
        elif action == 3: # osatyö
            employment_status,pension,tyoelake,paid_wage,time_in_state,tyoura,pinkslip,basis_wage=\
                self.move_to_parttime(raw_wage,pt_action,pension,tyoelake,age,tyoura,0)
        elif action==2:
            if age >= self.min_retirementage:
                employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,ove_paid,basis_wage,until_student,until_outsider=\
                    self.move_to_retirement(pension,age,kansanelake,tyoelake,
                        employment_status,unemp_after_ra,has_spouse=has_spouse,is_spouse=is_spouse,all_acc=True,scale_acc=True)
            elif children_under3>0:
                employment_status,pension,tyoelake,paid_wage,time_in_state,pinkslip,basis_wage = self.move_to_kht(pension,tyoelake,old_paid_wage,age)
            else:
                employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,\
                    used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,alkanut_ansiosidonnainen,\
                    karenssia_jaljella,ove_paid,basis_wage=\
                    self.move_to_unemp(pension,old_paid_wage,age,kansanelake,tyoelake,toe,toekesto,pinkslip,tyoura,
                        used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,alkanut_ansiosidonnainen,toe58,ove_paid,has_spouse,is_spouse)
        elif action == 4: # ve / ove
            if age >= self.min_retirementage:
                employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,ove_paid,basis_wage=\
                    self.move_to_oa_fulltime(raw_wage,pt_action,pension,age,kansanelake,tyoelake,employment_status,
                        unemp_after_ra,scale_acc=True,has_spouse=has_spouse,is_spouse=is_spouse)
            else:
                if (not ove_paid) and (age>self.min_ove_age):
                    pension,tyoelake,ove_paid = self.move_to_ove(employment_status,pension,tyoelake,ove_paid,age,unemp_after_ra)

                employment_status = 1 # unchanged

                tyoelake=tyoelake*self.elakeindeksi
                tyoura += self.timestep
                pension = self.pension_accrual(age,paid_wage,pension,state=1)
        elif action == 11: # tk
            employment_status,pension,kansanelake,tyoelake,paid_wage,time_in_state,ove_paid,basis_wage,until_disab,life_left,pinkslip=\
                self.move_to_disab(pension,old_paid_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,has_spouse,children_under18,is_spouse,life_left)
        else:
            print('error 12')

        return employment_status,kansanelake,tyoelake,pension,time_in_state,\
              pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
              alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella,basis_wage,until_disab,\
              until_student,until_outsider,life_left

    def stay_disabled(self,raw_wage: float,paid_wage: float,pt_action: int,employment_status: int,
                        kansanelake: float,tyoelake: float,pension: float,time_in_state: float,toe: float,toekesto: float,
                        tyoura: float,used_unemp_benefit: float,pinkslip: int,unemp_after_ra: float,old_paid_wage: float,unempwage: float,
                        unempwage_basis: float,action: int,age: float,sattuma,intage: int,g: int ,alkanut_ansiosidonnainen: int,
                        toe58: int,ove_paid: int,children_under3: int,children_under18: int,basis_wage: float,has_spouse: int,is_spouse: bool,
                        until_disab: float,until_student: float,until_outsider: float,life_left: float):

        '''
        Pysy tilassa työkyvytön (4)
        '''
        time_in_state += self.timestep
        karenssia_jaljella = 0
        if age >= self.min_retirementage:
            employment_status = 3 # ve
        else:
            employment_status = 3 # unchanged

        tyoelake = tyoelake*self.elakeindeksi

        if math.isclose(time_in_state,5.0) and age<55.0:
            # kertakorotus
            if age<31:
                korotus = 1.25
            else:
                korotus = 1.25-0.01*max(0,age-31.0)
            tyoelake = tyoelake*korotus

        if pension>0:
            if age >= self.max_retirementage:
                tyoelake = tyoelake+self.scale_pension(pension,age,scale=False)/self.elakeindeksi
                pension = 0
            else:
                pension = pension*self.palkkakerroin

        #wage=0
        kansanelake = kansanelake*self.kelaindeksi

        return employment_status,kansanelake,tyoelake,pension,time_in_state,\
               pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella,basis_wage,until_disab,\
               until_student,until_outsider,life_left

    def stay_retired(self,raw_wage: float,paid_wage: float,pt_action: int,employment_status: int,
                        kansanelake: float,tyoelake: float,pension: float,time_in_state: float,toe: float,toekesto: float,
                        tyoura: float,used_unemp_benefit: float,pinkslip: int,unemp_after_ra: float,old_paid_wage: float,unempwage: float,
                        unempwage_basis: float,action: int,age: float,sattuma,intage: int,g: int ,alkanut_ansiosidonnainen: int,
                        toe58: int,ove_paid: int,children_under3: int,children_under18: int,basis_wage: float,has_spouse: int,is_spouse: bool,
                        until_disab: float,until_student: float,until_outsider: float,life_left: float):
        '''
        Pysy tilassa vanhuuseläke (2)
        '''
        karenssia_jaljella=0
        #if age >= self.min_retirementage: # ve
        time_in_state += self.timestep

        if self.randomness and self.unemp_limit_reemp:
            if (action in set([1,2])):
                if not self.under_unemp_pt_reemp_rate(intage,g,tyoura,sattuma[7]): #sattuma[7]>self.unemp_reemp_pt_prob[intage,g] and self.randomness:
                    action = 0
            elif (action in set([3,4])):
                if not self.under_unemp_ft_reemp_rate(intage,g,tyoura,sattuma[7]): #sattuma[7]>self.unemp_reemp_ft_prob[intage,g] and self.randomness:
                    if self.under_unemp_pt_reemp_rate(intage,g,tyoura,sattuma[7]): #sattuma[7]<self.unemp_reemp_pt_prob[intage,g]:
                        action = 1
                    else:
                        action = 0

        if age >= self.max_retirementage and pension>0:
            employment_status = 2 # unchanged
            tyoelake = tyoelake*self.elakeindeksi+self.scale_pension(pension,age,scale=False,unemp_after_ra=unemp_after_ra)
            pension = 0
            kansanelake = kansanelake * self.kelaindeksi

        if action == 0:
            employment_status = 2 # unchanged
            tyoelake = self.elakeindeksi*tyoelake
            kansanelake = kansanelake * self.kelaindeksi
            pension = pension*self.palkkakerroin
        elif action in (1,2):
            #if paid_wage<1.0:
            #    print(f'ret state {employment_status} raw_wage {raw_wage} paid_wage {paid_wage} age {age} is_spouse {is_spouse}')
            #    self.render()

            employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,ove_paid,basis_wage=\
                self.move_to_oa_parttime(raw_wage,pt_action,pension,age,kansanelake,tyoelake,employment_status,0,
                    scale_acc=False,has_spouse=has_spouse,is_spouse=is_spouse)
        elif action in (3,4):
            #if paid_wage<1.0:
            #    print(f'ret state {employment_status} raw_wage {raw_wage} paid_wage {paid_wage} age {age} is_spouse {is_spouse}')
            #    self.render()

            employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,ove_paid,basis_wage=\
                self.move_to_oa_fulltime(raw_wage,pt_action,pension,age,kansanelake,tyoelake,employment_status,0,
                    scale_acc=False,has_spouse=has_spouse,is_spouse=is_spouse)
        elif action == 11:
            employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,kansanelake,tyoelake,basis_wage,until_student,until_outsider=\
                self.move_to_retdisab(pension,age,time_in_state,kansanelake,tyoelake,unemp_after_ra)
        else:
            print('error 221, action {} age {}'.format(action,age))
        # else:
        #     # työvoiman ulkopuolella
        #     time_in_state += self.timestep
        #     if action == 0:
        #         employment_status = 2
        #         tyoelake = self.elakeindeksi*tyoelake
        #         kansanelake = kansanelake * self.kelaindeksi
        #         pension=pension*self.palkkakerroin
        #     elif action == 1: # työttömäksi
        #         employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,\
        #             used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,\
        #             alkanut_ansiosidonnainen,karenssia_jaljella,ove_paid,basis_wage=\
        #             self.move_to_unemp(pension,old_paid_wage,age,kansanelake,tyoelake,toe,toekesto,0,tyoura,
        #                 used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,alkanut_ansiosidonnainen,toe58,ove_paid,has_spouse,is_spouse)
        #     elif action == 2: # täihin
        #         employment_status,pension,tyoelake,paid_wage,time_in_state,tyoura,pinkslip,basis_wage=\
        #             self.move_to_work(raw_wage,pt_action,pension,tyoelake,age,time_in_state,tyoura,pinkslip)
        #     elif action == 3: # osatyö 50%
        #         employment_status,pension,tyoelake,paid_wage,time_in_state,tyoura,pinkslip,basis_wage=\
        #             self.move_to_parttime(raw_wage,pt_action,pension,tyoelake,age,tyoura,time_in_state)
        #     elif action == 11: # tk
        #         employment_status,pension,kansanelake,tyoelake,paid_wage,time_in_state,ove_paid,basis_wage,until_disab=\
        #             self.move_to_disab(pension,old_paid_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,has_spouse,children_under18,is_spouse,life_left)
        #     else:
        #         print('error 12')

        return employment_status,kansanelake,tyoelake,pension,time_in_state,\
               pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella,basis_wage,until_disab,\
               until_student,until_outsider,life_left

    def stay_motherleave(self,raw_wage: float,paid_wage: float,pt_action: int,employment_status: int,
                        kansanelake: float,tyoelake: float,pension: float,time_in_state: float,toe: float,toekesto: float,
                        tyoura: float,used_unemp_benefit: float,pinkslip: int,unemp_after_ra: float,old_paid_wage: float,unempwage: float,
                        unempwage_basis: float,action: int,age: float,sattuma,intage: int,g: int ,alkanut_ansiosidonnainen: int,
                        toe58: int,ove_paid: int,children_under3: int,children_under18: int,basis_wage: float,has_spouse: int,is_spouse: bool,
                        until_disab: float,until_student: float,until_outsider: float,life_left: float):
        '''
        Pysy tilassa äitiysvapaa (5)
        '''
        #exit_prb=np.random.uniform(0,80_000)
        karenssia_jaljella=0
        if time_in_state >= self.aitiysvapaa_kesto or sattuma[5]<self.aitiysvapaa_pois:
            if time_in_state >= self.aitiysvapaa_kesto and sattuma[7]<self.nainen_jatkaa_kotihoidontuelle:
                action = 3

            until_outsider = self.comp_time_to_outsider(0,age,g) # state 0 is not 11
            until_student = self.comp_time_to_study(0,age,g)

            pinkslip = 0 # karenssi, jos eroaa
            if action == 0:
                employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,\
                    used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,alkanut_ansiosidonnainen,\
                    karenssia_jaljella,ove_paid,basis_wage =\
                    self.move_to_unemp(pension,old_paid_wage,age,kansanelake,tyoelake,toe,toekesto,pinkslip,tyoura,
                        used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,alkanut_ansiosidonnainen,toe58,ove_paid,has_spouse,is_spouse)
            elif action == 1: #
                employment_status,pension,tyoelake,paid_wage,time_in_state,tyoura,pinkslip,basis_wage=\
                    self.move_to_work(raw_wage,pt_action,pension,tyoelake,age,time_in_state,tyoura,pinkslip)
            elif action in (2,3): #
                employment_status,pension,tyoelake,paid_wage,time_in_state,pinkslip,basis_wage = self.move_to_kht(pension,tyoelake,old_paid_wage,age)
            elif action in (4,5):
                employment_status,pension,tyoelake,paid_wage,time_in_state,tyoura,pinkslip,basis_wage=\
                    self.move_to_parttime(raw_wage,pt_action,pension,tyoelake,age,tyoura,time_in_state)
            elif action == 11: # tk
                employment_status,pension,kansanelake,tyoelake,paid_wage,time_in_state,ove_paid,basis_wage,until_disab,life_left,pinkslip=\
                    self.move_to_disab(pension,old_paid_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,has_spouse,children_under18,is_spouse,life_left)
                pinkslip=0
            else:
                print('Error 21')
        else:
            pension = self.pension_accrual(age,old_paid_wage,pension,state=5)
            tyoelake=tyoelake*self.elakeindeksi
            time_in_state += self.timestep

        return employment_status,kansanelake,tyoelake,pension,time_in_state,\
               pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella,basis_wage,until_disab,\
               until_student,until_outsider,life_left

    def stay_fatherleave(self,raw_wage: float,paid_wage: float,pt_action: int,employment_status: int,
                        kansanelake: float,tyoelake: float,pension: float,time_in_state: float,toe: float,toekesto: float,
                        tyoura: float,used_unemp_benefit: float,pinkslip: int,unemp_after_ra: float,old_paid_wage: float,unempwage: float,
                        unempwage_basis: float,action: int,age: float,sattuma,intage: int,g: int ,alkanut_ansiosidonnainen: int,
                        toe58: int,ove_paid: int,children_under3: int,children_under18: int,basis_wage: float,has_spouse: int,is_spouse: bool,
                        until_disab: float,until_student: float,until_outsider: float,life_left: float):
        '''
        Pysy tilassa isyysvapaa (6)
        '''
        karenssia_jaljella=0
        if time_in_state >= self.isyysvapaa_kesto:
            until_outsider = self.comp_time_to_outsider(0,age,g) # state 0 is not 11
            until_student = self.comp_time_to_study(0,age,g)

            if sattuma[7]<self.mies_jatkaa_kotihoidontuelle:
                action = 3

            pinkslip = 0 # karenssi, jos eroaa
            if action == 0:
                employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,\
                    used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,\
                    alkanut_ansiosidonnainen,karenssia_jaljella,ove_paid,basis_wage=\
                    self.move_to_unemp(pension,old_paid_wage,age,kansanelake,tyoelake,toe,toekesto,pinkslip,tyoura,
                        used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,alkanut_ansiosidonnainen,toe58,ove_paid,has_spouse,is_spouse)
            elif action == 1: #
                # ei vaikutusta palkkaan
                employment_status,pension,tyoelake,paid_wage,time_in_state,tyoura,pinkslip,basis_wage=\
                    self.move_to_work(raw_wage,pt_action,pension,tyoelake,age,0,tyoura,pinkslip)
            elif action in (2,3): #
                employment_status,pension,tyoelake,paid_wage,time_in_state,pinkslip,basis_wage = self.move_to_kht(pension,tyoelake,old_paid_wage,age)
            elif action in (4,5):
                employment_status,pension,tyoelake,paid_wage,time_in_state,tyoura,pinkslip,basis_wage=\
                    self.move_to_parttime(raw_wage,pt_action,pension,tyoelake,age,tyoura,0)
            elif action == 11: # tk
                employment_status,pension,kansanelake,tyoelake,paid_wage,time_in_state,ove_paid,basis_wage,until_disab,life_left,pinkslip=\
                    self.move_to_disab(pension,old_paid_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,has_spouse,children_under18,is_spouse,life_left)
            else:
                print('Error 23')
        else:
            pension = self.pension_accrual(age,old_paid_wage,pension,state=6)
            tyoelake=tyoelake*self.elakeindeksi
            time_in_state += self.timestep

        return employment_status,kansanelake,tyoelake,pension,time_in_state,\
               pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella,basis_wage,until_disab,\
               until_student,until_outsider,life_left

    def stay_kht(self,raw_wage: float,paid_wage: float,pt_action: int,employment_status: int,
                kansanelake: float,tyoelake: float,pension: float,time_in_state: float,toe: float,toekesto: float,
                tyoura: float,used_unemp_benefit: float,pinkslip: int,unemp_after_ra: float,old_paid_wage: float,unempwage: float,
                unempwage_basis: float,action: int,age: float,sattuma,intage: int,g: int ,alkanut_ansiosidonnainen: int,
                toe58: int,ove_paid: int,children_under3: int,children_under18: int,basis_wage: float,has_spouse: int,is_spouse: bool,
                until_disab: float,until_student: float,until_outsider: float,life_left: float):
        '''
        Pysy tilassa kotihoidontuki (0)
        '''
        karenssia_jaljella=0

        if action == 2 and self.unemp_limit_reemp:
            if sattuma[7]<self.unemp_reemp_ft_prob[intage,g] and self.randomness:
                action = 2
            elif sattuma[7]<self.unemp_reemp_pt_prob[intage,g] and self.randomness:
                action = 3
            else:
                action = 1
        elif action == 3 and self.unemp_limit_reemp:
            if sattuma[7]>self.unemp_reemp_pt_prob[intage,g] and self.randomness:
                action = 1

        if (action == 0) and (time_in_state>self.kht_kesto or children_under3<1) and not self.perustulo: # jos etuus loppuu ja yritys jäädä etuudelle, siirtymä satunnaisesti
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

        if age >= self.min_retirementage: # ve
            employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,ove_paid,basis_wage,until_student,until_outsider=\
                self.move_to_retirement(pension,age,kansanelake,tyoelake,employment_status,
                    unemp_after_ra,all_acc=True,scale_acc=True,has_spouse=has_spouse,is_spouse=is_spouse)
        elif (action == 0) and ((time_in_state <= self.kht_kesto and children_under3>0) or (self.perustulo and children_under18>0)): # jos perustulo, ei aikarajoitetta
            employment_status  = 7 # stay
            time_in_state += self.timestep
            pension = self.pension_accrual(age,0,pension,state=7)
            tyoelake=tyoelake*self.elakeindeksi
        elif action in (1,4): #
            pinkslip = 0 # karenssia
            employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,\
                used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,\
                alkanut_ansiosidonnainen,karenssia_jaljella,ove_paid,basis_wage=\
                self.move_to_unemp(pension,old_paid_wage,age,kansanelake,tyoelake,toe,toekesto,pinkslip,tyoura,
                    used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,alkanut_ansiosidonnainen,toe58,ove_paid,has_spouse,is_spouse)
        elif action in (2,5): #
            employment_status,pension,tyoelake,paid_wage,time_in_state,tyoura,pinkslip,basis_wage=\
                self.move_to_work(raw_wage,pt_action,pension,tyoelake,age,time_in_state,tyoura,pinkslip)
        elif action == 3: #
            employment_status,pension,tyoelake,paid_wage,time_in_state,tyoura,pinkslip,basis_wage=\
                self.move_to_parttime(raw_wage,pt_action,pension,tyoelake,age,tyoura,time_in_state)
        elif action == 11: # tk
            employment_status,pension,kansanelake,tyoelake,paid_wage,time_in_state,ove_paid,basis_wage,until_disab,life_left,pinkslip=\
                self.move_to_disab(pension,old_paid_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,has_spouse,children_under18,is_spouse,life_left)
        else:
            print('Error 25')

        return employment_status,kansanelake,tyoelake,pension,time_in_state,\
               pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella,basis_wage,until_disab,\
               until_student,until_outsider,life_left

    def stay_student(self,raw_wage: float,paid_wage: float,pt_action: int,employment_status: int,
                        kansanelake: float,tyoelake: float,pension: float,time_in_state: float,toe: float,toekesto: float,
                        tyoura: float,used_unemp_benefit: float,pinkslip: int,unemp_after_ra: float,old_paid_wage: float,unempwage: float,
                        unempwage_basis: float,action: int,age: float,sattuma,intage: int,g: int ,alkanut_ansiosidonnainen: int,
                        toe58: int,ove_paid: int,children_under3: int,children_under18: int,basis_wage: float,has_spouse: int,is_spouse: bool,
                        until_disab: float,until_student: float,until_outsider: float,life_left: float):
        '''
        Pysy tilassa opiskelija (12)
        '''

        pinkslip=1 # ei karenssia
        karenssia_jaljella=0

        if age >= self.min_retirementage:
            employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,ove_paid,basis_wage,until_student,until_outsider=\
                self.move_to_retirement(pension,age,kansanelake,tyoelake,employment_status,
                    unemp_after_ra,all_acc=True,scale_acc=True,has_spouse=has_spouse,is_spouse=is_spouse)
            unempwage_basis,alkanut_ansiosidonnainen=0,0
        else:
            # after at most 10 years, move out of being a student
            #if sattuma[5] >= self.student_outrate[intage,g] and time_in_state<10:
            if until_student > 0 and time_in_state < 10:
                employment_status = 12 # unchanged
                time_in_state += self.timestep
                pension = self.pension_accrual(age,0,pension,state=12)
                tyoelake = tyoelake*self.elakeindeksi
            else:
                if (action == 0) and self.unemp_limit_reemp: # and intage<40:
                    if sattuma[7]>self.student_reemp_ft_prob[intage,g] and self.randomness:
                        if sattuma[7]<self.student_reemp_pt_prob[intage,g]:
                            action = 3
                        else:
                            action = 2
                elif (action in {3,4,5}) and self.unemp_limit_reemp: # and intage<40:
                    if sattuma[7]>self.student_reemp_pt_prob[intage,g] and self.randomness:
                        action = 2

                until_student = self.comp_time_to_study(0,age,g) # state 0 is not state 11
                if action == 0: #
                    employment_status,pension,tyoelake,paid_wage,time_in_state,tyoura,pinkslip,basis_wage=\
                        self.move_to_work(raw_wage,pt_action,pension,tyoelake,age,0,tyoura,pinkslip)
                elif action == 1: #
                    if children_under3>0:
                        employment_status,pension,tyoelake,paid_wage,time_in_state,pinkslip,basis_wage = self.move_to_kht(pension,tyoelake,old_paid_wage,age)
                    else:
                        employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,\
                            used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,\
                            alkanut_ansiosidonnainen,karenssia_jaljella,ove_paid,basis_wage=\
                            self.move_to_unemp(pension,old_paid_wage,age,kansanelake,tyoelake,toe,toekesto,pinkslip,tyoura,
                                used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,alkanut_ansiosidonnainen,toe58,ove_paid,has_spouse,is_spouse)
                elif action == 2:
                    employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,\
                        used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,\
                        alkanut_ansiosidonnainen,karenssia_jaljella,ove_paid,basis_wage=\
                        self.move_to_unemp(pension,old_paid_wage,age,kansanelake,tyoelake,toe,toekesto,pinkslip,tyoura,
                            used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,alkanut_ansiosidonnainen,toe58,ove_paid,has_spouse,is_spouse)
                elif action in (3,4,5):
                    employment_status,pension,tyoelake,paid_wage,time_in_state,tyoura,pinkslip,basis_wage=\
                        self.move_to_parttime(raw_wage,pt_action,pension,tyoelake,age,tyoura,time_in_state)
                elif action == 11: # tk
                    employment_status,pension,kansanelake,tyoelake,paid_wage,time_in_state,ove_paid,basis_wage,until_disab,life_left,pinkslip=\
                        self.move_to_disab(pension,old_paid_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,has_spouse,children_under18,is_spouse,life_left)
                else:
                    print('error 29: ',action)

        return employment_status,kansanelake,tyoelake,pension,time_in_state,\
               pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella,basis_wage,until_disab,\
               until_student,until_outsider,life_left

    def stay_oa_parttime(self,raw_wage: float,paid_wage: float,pt_action: int,employment_status: int,
                        kansanelake: float,tyoelake: float,pension: float,time_in_state: float,toe: float,toekesto: float,
                        tyoura: float,used_unemp_benefit: float,pinkslip: int,unemp_after_ra: float,old_paid_wage: float,unempwage: float,
                        unempwage_basis: float,action: int,age: float,sattuma,intage: int,g: int ,alkanut_ansiosidonnainen: int,
                        toe58: int,ove_paid: int,children_under3: int,children_under18: int,basis_wage: float,has_spouse: int,is_spouse: bool,
                        until_disab: float,until_student: float,until_outsider: float,life_left: float):
        '''
        Pysy tilassa ve+(osa-aikatyö) (0)
        '''

        karenssia_jaljella=0

        # irtisanominen
        if sattuma[1]<self.pinkslip_intensity[g,intage]:
            action=4 # ve:lle
            pinkslip=1

        if (action in {2,3}) and self.unemp_limit_reemp:
            if sattuma[7]>self.parttime_fullemp_prob[intage,g] and self.randomness: # FIXME
                action = 0

        #if age >= self.max_retirementage:
        #    employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,ove_paid,basis_wage,until_student,until_outsider=\
        #        self.move_to_retirement(pension,age,kansanelake,tyoelake,employment_status,
        #            unemp_after_ra,all_acc=True,scale_acc=False,has_spouse=has_spouse,is_spouse=is_spouse)
        #el
        if action == 0: # jatkaa osa-aikatöissä, ei voi saada työttämyyspäivärahaa
            employment_status = 8 # unchanged
            time_in_state += self.timestep
            pension = self.pension_accrual(age,paid_wage,pension,state=employment_status)
            tyoelake = self.elakeindeksi * tyoelake
            kansanelake = kansanelake * self.kelaindeksi
        elif action in (2,3): # jatkaa täysin töissä, ei voi saada työttämyyspäivärahaa
            employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,ove_paid,basis_wage=\
                self.move_to_oa_fulltime(raw_wage,pt_action,pension,age,kansanelake,tyoelake,employment_status,
                        0,scale_acc=False,has_spouse=has_spouse,is_spouse=is_spouse)
        elif action in (1,4): # eläkkeelle, eläkeaikana karttunutta eläkettä ei vielä maksuun
            employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,ove_paid,basis_wage,until_student,until_outsider=\
                self.move_to_retirement(pension,age,kansanelake,tyoelake,employment_status,
                    0,all_acc=False,scale_acc=False,has_spouse=has_spouse,is_spouse=is_spouse)
        elif action == 11:
            # no more working, move to "disab" with no change in paid_pension
            employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,kansanelake,tyoelake,basis_wage,until_student,until_outsider=\
                self.move_to_retdisab(pension,age,time_in_state,kansanelake,tyoelake,unemp_after_ra)
        else:
            print('error 14, action {} age {}'.format(action,age))

        return employment_status,kansanelake,tyoelake,pension,time_in_state,\
               pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella,basis_wage,until_disab,\
               until_student,until_outsider,life_left

    def stay_oa_fulltime(self,raw_wage: float,paid_wage: float,pt_action: int,employment_status: int,
                        kansanelake: float,tyoelake: float,pension: float,time_in_state: float,toe: float,toekesto: float,
                        tyoura: float,used_unemp_benefit: float,pinkslip: int,unemp_after_ra: float,old_paid_wage: float,unempwage: float,
                        unempwage_basis: float,action: int,age: float,sattuma,intage: int,g: int ,alkanut_ansiosidonnainen: int,
                        toe58: int,ove_paid: int,children_under3: int,children_under18: int,basis_wage: float,has_spouse: int,is_spouse: bool,
                        until_disab: float,until_student: float,until_outsider: float,life_left: float):
        '''
        Pysy tilassa ve+työ (9)
        '''

        karenssia_jaljella=0

        # irtisanominen
        if sattuma[1]<self.pinkslip_intensity[g,intage]:
            action=4 # ve:lle
            pinkslip=1

        if (action in {1,2}) and self.unemp_limit_reemp:
            if sattuma[7]>self.fulltime_pt_prob and self.randomness:
                action = 0

        #if age >= self.max_retirementage:
        #    employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,ove_paid,basis_wage,until_student,until_outsider=\
        #        self.move_to_retirement(pension,age,kansanelake,tyoelake,employment_status,
        #            unemp_after_ra,all_acc=True,scale_acc=False,has_spouse=has_spouse,is_spouse=is_spouse)
        #el
        if action == 0: # jatkaa töissä, ei voi saada työttämyyspäivärahaa
            employment_status = 9 # unchanged
            time_in_state += self.timestep
            pension = self.pension_accrual(age,paid_wage,pension,state=employment_status)
            tyoelake = tyoelake * self.elakeindeksi
            kansanelake = kansanelake * self.kelaindeksi
        elif action in (1,2): # jatkaa osa-aikatöissä, ei voi saada työttämyyspäivärahaa
            employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,ove_paid,basis_wage=\
                self.move_to_oa_parttime(raw_wage,pt_action,pension,age,kansanelake,tyoelake,employment_status,0,
                    scale_acc=False,has_spouse=has_spouse,is_spouse=is_spouse)
        elif action in (3,4): # eläkkeelle, eläkeaikana karttunutta eläkettä ei vielä maksuun
            employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,ove_paid,basis_wage,until_student,until_outsider=\
                self.move_to_retirement(pension,age,kansanelake,tyoelake,employment_status,0,
                    all_acc=False,scale_acc=False,has_spouse=has_spouse,is_spouse=is_spouse)
        elif action == 11:
            # no more working, move to "disab" with no change in paid_pension
            employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,kansanelake,tyoelake,basis_wage,until_student,until_outsider=\
                self.move_to_retdisab(pension,age,time_in_state,kansanelake,tyoelake,unemp_after_ra)
        else:
            print('error 14, action {} age {}'.format(action,age))

        return employment_status,kansanelake,tyoelake,pension,time_in_state,\
               pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella,basis_wage,until_disab,\
               until_student,until_outsider,life_left

    def stay_parttime(self,raw_wage: float,paid_wage: float,pt_action: int,employment_status: int,
                        kansanelake: float,tyoelake: float,pension: float,time_in_state: float,toe: float,toekesto: float,
                        tyoura: float,used_unemp_benefit: float,pinkslip: int,unemp_after_ra: float,old_paid_wage: float,unempwage: float,
                        unempwage_basis: float,action: int,age: float,sattuma,intage: int,g: int,alkanut_ansiosidonnainen: int,
                        toe58: int,ove_paid: int,children_under3: int,children_under18: int,basis_wage: float,has_spouse: int,is_spouse: bool,
                        until_disab: float,until_student: float,until_outsider: float,life_left: float):
        '''
        Pysy tilassa osa-aikatyö (0)
        '''

        time_in_state += self.timestep
        karenssia_jaljella=0

        # irtisanominen
        if sattuma[1]<self.pinkslip_intensity[g,intage]:
            if age<self.min_retirementage:
                action=1 # unemp
                pinkslip=1
            else:
                action=2 # ve
                pinkslip=1
        else:
            pinkslip = 0

        if (action == 3): # or (action == 4 and age<self.min_retirementage):
            if sattuma[7]>self.parttime_fullemp_prob[intage,g] and self.randomness:
                action = 0

        if action == 0: 
            employment_status = 10 # unchanged
            tyoura += self.timestep

            #if action == 5 and (not ove_paid) and (age >= self.min_ove_age):
            #    pension,tyoelake,ove_paid = self.move_to_ove(employment_status,pension,tyoelake,ove_paid,age,unemp_after_ra)

            tyoelake=tyoelake*self.elakeindeksi
            pension = self.pension_accrual(age,paid_wage,pension,state=10)
        elif action == 1: # työttömäksi
            employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,\
                used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,\
                alkanut_ansiosidonnainen,karenssia_jaljella,ove_paid,basis_wage=\
                self.move_to_unemp(pension,old_paid_wage,age,kansanelake,tyoelake,toe,toekesto,pinkslip,tyoura,
                    used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,
                    alkanut_ansiosidonnainen,toe58,ove_paid,has_spouse,is_spouse)
        elif action==3:
            employment_status,pension,tyoelake,paid_wage,time_in_state,tyoura,pinkslip,basis_wage=\
                self.move_to_work(raw_wage,pt_action,pension,tyoelake,age,0,tyoura,pinkslip)
        elif action==2:
            if age >= self.min_retirementage:
                employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,ove_paid,basis_wage,until_student,until_outsider=\
                    self.move_to_retirement(pension,age,kansanelake,tyoelake,employment_status,
                        unemp_after_ra,all_acc=True,scale_acc=True,has_spouse=has_spouse,is_spouse=is_spouse)
            elif children_under3>0:
                employment_status,pension,tyoelake,paid_wage,time_in_state,pinkslip,basis_wage=\
                    self.move_to_kht(pension,tyoelake,old_paid_wage,age)
            else:
                employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,\
                    used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,\
                    alkanut_ansiosidonnainen,karenssia_jaljella,ove_paid,basis_wage=\
                    self.move_to_unemp(pension,old_paid_wage,age,kansanelake,tyoelake,toe,toekesto,pinkslip,tyoura,
                        used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,alkanut_ansiosidonnainen,toe58,ove_paid,has_spouse,is_spouse)
        elif action==4: # move to oa_work or ove_paid
            if age >= self.min_retirementage:
                employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,ove_paid,basis_wage=\
                    self.move_to_oa_parttime(raw_wage,pt_action,pension,age,kansanelake,tyoelake,employment_status,
                        unemp_after_ra,scale_acc=True,has_spouse=has_spouse,is_spouse=is_spouse)
            else:
                if (not ove_paid) and (age >= self.min_ove_age):
                    pension,tyoelake,ove_paid = self.move_to_ove(employment_status,pension,tyoelake,ove_paid,age,unemp_after_ra)

                employment_status = 10 # unchanged
                tyoura += self.timestep

                tyoelake=tyoelake*self.elakeindeksi
                pension = self.pension_accrual(age,paid_wage,pension,state=10)

                #employment_status,pension,tyoelake,paid_wage,time_in_state,tyoura,pinkslip,basis_wage=\
                #    self.move_to_work(wage,pt_action,pension,tyoelake,age,0,tyoura,pinkslip)
        elif action == 11: # tk
            employment_status,pension,kansanelake,tyoelake,paid_wage,time_in_state,ove_paid,basis_wage,until_disab,life_left,pinkslip=\
                self.move_to_disab(pension,old_paid_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,has_spouse,children_under18,is_spouse,life_left)
        else:
            print('error 12')

        return employment_status,kansanelake,tyoelake,pension,time_in_state,\
               pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella,basis_wage,until_disab,\
               until_student,until_outsider,life_left

    def stay_outsider(self,raw_wage: float,paid_wage: float,pt_action: int,employment_status: int,
                        kansanelake: float,tyoelake: float,pension: float,time_in_state: float,toe: float,toekesto: float,
                        tyoura: float,used_unemp_benefit: float,pinkslip: int,unemp_after_ra: float,old_paid_wage: float,unempwage: float,
                        unempwage_basis: float,action: int,age: float,sattuma,intage: int,g: int,alkanut_ansiosidonnainen: int,
                        toe58: int,ove_paid: int,children_under3: int,children_under18: int,basis_wage: float,has_spouse: int,is_spouse: bool,
                        until_disab: float,until_student: float,until_outsider: float,life_left: float):
        '''
        Pysy tilassa työvoiman ulkopuolella (11)
        '''
        karenssia_jaljella=0

        if age >= self.max_retirementage:
            employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,ove_paid,basis_wage,until_student,until_outsider=\
                self.move_to_retirement(pension,age,kansanelake,tyoelake,employment_status,
                    unemp_after_ra,all_acc=True,scale_acc=True,has_spouse=has_spouse,is_spouse=is_spouse)
        #elif sattuma[5] >= self.outsider_outrate[intage,g]:
        elif until_outsider > 0.0:
            time_in_state += self.timestep
            employment_status = 11 # unchanged
            pension = pension * self.palkkakerroin
            tyoelake = tyoelake * self.elakeindeksi
        else:
            until_outsider = self.comp_time_to_outsider(0,age,g) # state 0 is not 11
            until_student = self.comp_time_to_study(0,age,g)

            if (action in {0,1}) and self.unemp_limit_reemp:
                if sattuma[7]>self.unemp_reemp_ft_prob[intage,g] and self.randomness:
                    if sattuma[7]<self.unemp_reemp_pt_prob[intage,g] and action == 1:
                        action = 4
                    else:
                        action = 2
            elif (action in {4,5}) and self.unemp_limit_reemp:
                if sattuma[7]>self.unemp_reemp_pt_prob[intage,g] and self.randomness:
                    action = 2

            if action in (0,1): #
                employment_status,pension,tyoelake,paid_wage,time_in_state,tyoura,pinkslip,basis_wage=\
                    self.move_to_work(raw_wage,pt_action,pension,tyoelake,age,time_in_state,tyoura,pinkslip)
            elif action == 3:
                if children_under3>0:
                    employment_status,pension,tyoelake,paid_wage,time_in_state,pinkslip,basis_wage = self.move_to_kht(pension,tyoelake,old_paid_wage,age)
                else:
                    pinkslip=1  # ei karenssia
                    employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,\
                        used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,\
                        alkanut_ansiosidonnainen,karenssia_jaljella,ove_paid,basis_wage=\
                        self.move_to_unemp(pension,old_paid_wage,age,kansanelake,tyoelake,toe,toekesto,pinkslip,tyoura,
                            used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,alkanut_ansiosidonnainen,toe58,ove_paid,has_spouse,is_spouse)
            elif action == 2: #
                pinkslip=1  # ei karenssia
                employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,\
                    used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,\
                    alkanut_ansiosidonnainen,karenssia_jaljella,ove_paid,basis_wage=\
                    self.move_to_unemp(pension,old_paid_wage,age,kansanelake,tyoelake,toe,toekesto,pinkslip,tyoura,
                        used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,alkanut_ansiosidonnainen,toe58,ove_paid,has_spouse,is_spouse)
            elif action in (4,5): #
                employment_status,pension,tyoelake,paid_wage,time_in_state,tyoura,pinkslip,basis_wage=\
                    self.move_to_parttime(raw_wage,pt_action,pension,tyoelake,age,tyoura,time_in_state)
            elif action == 11: # tk
                employment_status,pension,kansanelake,tyoelake,paid_wage,time_in_state,ove_paid,basis_wage,until_disab,life_left,pinkslip=\
                    self.move_to_disab(pension,old_paid_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,has_spouse,children_under18,is_spouse,life_left)
            else:
                print('error 19: ',action)

        return employment_status,kansanelake,tyoelake,pension,time_in_state,\
               pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella,basis_wage,until_disab,\
               until_student,until_outsider,life_left

    def stay_svpaivaraha(self,raw_wage: float,paid_wage: float,pt_action: int,employment_status: int,
                        kansanelake: float,tyoelake: float,pension: float,time_in_state: float,toe: float,toekesto: float,
                        tyoura: float,used_unemp_benefit: float,pinkslip: int,unemp_after_ra: float,old_paid_wage: float,unempwage: float,
                        unempwage_basis: float,action: int,age: float,sattuma,intage: int,g: int,alkanut_ansiosidonnainen: int,
                        toe58: int,ove_paid: int,children_under3: int,children_under18: int,basis_wage: float,has_spouse: int,is_spouse: bool,
                        until_disab: float,until_student: float,until_outsider: float,life_left: float):
        '''
        Pysy sairauspäivärahalla (14)
        '''
        karenssia_jaljella=0

        if time_in_state<0.3:
           if sattuma[8]<self.svpaivaraha_short3m[intage,g]: #0.5:
               exit=True
           else:
               exit=False
        else:
           exit=False

        if age >= self.max_svbenefitage and age >= self.min_retirementage:
                employment_status,pension,kansanelake,tyoelake,paid_wage,time_in_state,ove_paid,basis_wage,until_disab,life_left,pinkslip=\
                    self.move_to_disab_state(pension,old_paid_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,has_spouse,children_under18,is_spouse,life_left)
        elif time_in_state<1.0 and not exit:
            if age >= self.max_svbenefitage and action==1:
                employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,ove_paid,basis_wage,until_student,until_outsider=\
                    self.move_to_retirement(pension,age,kansanelake,tyoelake,employment_status,
                        unemp_after_ra,all_acc=True,scale_acc=True,has_spouse=has_spouse,is_spouse=is_spouse)
            #elif action == 11: # ei siirretä henkilöitä sv-päivärahalta tk:lle
            #    employment_status,pension,kansanelake,tyoelake,paid_wage,time_in_state,ove_paid,basis_wage=\
            #        self.move_to_disab_state(pension,old_paid_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,has_spouse,children_under18,is_spouse)
            #    pinkslip = 0
            else:
                # tähän myös siirtymä takaisin töihin?
                time_in_state += self.timestep
                employment_status = 14 # unchanged
                tyoelake = tyoelake * self.elakeindeksi
                pension = self.pension_accrual(age,old_paid_wage,pension,state=14)
        else:
            if (not exit and sattuma[5]<self.svpaivaraha_disabilityrate[intage,g]) or action == 11:
                employment_status,pension,kansanelake,tyoelake,paid_wage,time_in_state,ove_paid,basis_wage,until_disab,life_left,pinkslip=\
                    self.move_to_disab_state(pension,old_paid_wage,age,unemp_after_ra,kansanelake,tyoelake,ove_paid,has_spouse,children_under18,is_spouse,life_left)
            else:
                until_outsider = self.comp_time_to_outsider(0,age,g) # state 0 is not 11
                until_student = self.comp_time_to_study(0,age,g)

                if (action in {0}) and self.unemp_limit_reemp:
                    if sattuma[7]>self.unemp_reemp_pt_prob[intage,g]: # here we use part-time probability to compensate for the lack of knowledge about employment status
                        action = 2
                elif (action in (1,5)) and self.unemp_limit_reemp:
                    if sattuma[7]>self.unemp_reemp_pt_prob[intage,g] and self.randomness:
                        action = 2

                if action == 0: #
                    employment_status,pension,tyoelake,paid_wage,time_in_state,tyoura,pinkslip,basis_wage=\
                        self.move_to_work(raw_wage,pt_action,pension,tyoelake,age,time_in_state,tyoura,pinkslip)
                elif action == 3:
                    if age >= self.min_retirementage:
                        employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,ove_paid,basis_wage,until_student,until_outsider=\
                            self.move_to_retirement(pension,age,kansanelake,tyoelake,employment_status,
                                unemp_after_ra,all_acc=True,scale_acc=True,has_spouse=has_spouse,is_spouse=is_spouse)
                    elif children_under3>0:
                        employment_status,pension,tyoelake,paid_wage,time_in_state,pinkslip,basis_wage=\
                         self.move_to_kht(pension,tyoelake,old_paid_wage,age)
                    else:
                        pinkslip=1  # ei karenssia
                        employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,\
                            used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,\
                            alkanut_ansiosidonnainen,karenssia_jaljella,ove_paid,basis_wage=\
                            self.move_to_unemp(pension,old_paid_wage,age,kansanelake,tyoelake,toe,toekesto,pinkslip,tyoura,
                                used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,alkanut_ansiosidonnainen,toe58,ove_paid,has_spouse,is_spouse)
                elif action == 2: #
                    pinkslip=1  # ei karenssia
                    employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,\
                        used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,\
                        alkanut_ansiosidonnainen,karenssia_jaljella,ove_paid,basis_wage=\
                        self.move_to_unemp(pension,old_paid_wage,age,kansanelake,tyoelake,toe,toekesto,pinkslip,tyoura,
                            used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,alkanut_ansiosidonnainen,toe58,ove_paid,has_spouse,is_spouse)
                elif action == 4:
                    if age >= self.min_retirementage:
                        employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,ove_paid,basis_wage,until_student,until_outsider=\
                            self.move_to_retirement(pension,age,kansanelake,tyoelake,employment_status,
                                unemp_after_ra,all_acc=True,scale_acc=True,has_spouse=has_spouse,is_spouse=is_spouse)
                    else:
                        pinkslip=1 # ei karenssia
                        employment_status,kansanelake,tyoelake,pension,paid_wage,time_in_state,\
                            used_unemp_benefit,pinkslip,unemp_after_ra,unempwage_basis,\
                            alkanut_ansiosidonnainen,karenssia_jaljella,ove_paid,basis_wage=\
                            self.move_to_unemp(pension,old_paid_wage,age,kansanelake,tyoelake,toe,toekesto,pinkslip,tyoura,
                                used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,alkanut_ansiosidonnainen,toe58,ove_paid,has_spouse,is_spouse)
                elif action == 1: #
                    employment_status,pension,tyoelake,paid_wage,time_in_state,tyoura,pinkslip,basis_wage=\
                        self.move_to_parttime(raw_wage,pt_action,pension,tyoelake,age,tyoura,time_in_state)
                else:
                    print('error 19: ',action)
                until_disab = self.comp_until_disab(g,age,employment_status)

        return employment_status,kansanelake,tyoelake,pension,time_in_state,\
               pinkslip,unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella,basis_wage,until_disab,\
               until_student,until_outsider,life_left

    def get_benefits(self,empstate: int,pt_state: int,wage: float,kansanelake: float,tyoelake: float,pension: float,time_in_state: float,pinkslip: int,unempwage: float,
                        unempwage_basis: float,karenssia_jaljella: float,age: float,children_under3: int,children_under7: int,children_under18: int,ove_paid: int,used_unemp_benefit: float,
                        puoliso: int,spouse_empstate: int,spouse_pt_state: int,spouse_wage: float,spouse_kansanelake: float,puoliso_tyoelake: float,
                        puoliso_pinkslip: int,puoliso_karenssia_jaljella: float,spouse_time_in_state: float,
                        puoliso_unempwage: float,puoliso_unempwage_basis: float,puoliso_used_unemp_benefit: float,
                        main_basis_wage: float,spouse_basis_wage: float,
                        g: int,p_g: int,potential_wage: float=0,potential_spouse_wage: float=0,kassanjasen: int=0):
        '''
        This could be handled better
        '''
        tis=0
        pot_wage=potential_wage
        if empstate==0:
            wage=0
            benefitbasis=unempwage_basis
        elif empstate==1:
            #wage=wage
            pot_wage=wage
            benefitbasis=0
        elif empstate==2:
            wage=0
            benefitbasis=0
        elif empstate==3:
            wage=0
            benefitbasis=0
        elif empstate==4:
            wage=0
            benefitbasis=unempwage_basis
        elif empstate==5:
            #wage=wage
            benefitbasis=main_basis_wage
        elif empstate==6:
            #wage=wage
            benefitbasis=main_basis_wage
        elif empstate==7:
            wage=0
            benefitbasis=0
        elif empstate==8:
            wage=wage #parttimewage
            pot_wage=wage
            benefitbasis=0
        elif empstate==9:
            #wage=wage
            pot_wage=wage
            benefitbasis=0
        elif empstate==10:
            #wage=wage #parttimewage
            pot_wage=wage
            benefitbasis=0
        elif empstate==11:
            wage=0
            benefitbasis=0
        elif empstate==12:
            wage=0
            benefitbasis=0
        elif empstate==13:
            wage=0
            benefitbasis=unempwage_basis
        elif empstate==14:
            #wage=wage
            benefitbasis=main_basis_wage
        elif empstate==15:
            wage=0
            benefitbasis=0
        else:
            print('unknown state',empstate)

        puoliso_tis=0
        pot_spouse_wage=potential_spouse_wage
        if spouse_empstate==0:
            puoliso_palkka=0
            spouse_benefitbasis=puoliso_unempwage_basis
        elif spouse_empstate==1:
            puoliso_palkka=spouse_wage
            pot_spouse_wage=spouse_wage
            spouse_benefitbasis=0
        elif spouse_empstate==2:
            puoliso_palkka=0
            spouse_benefitbasis=0
        elif spouse_empstate==3:
            puoliso_palkka=0
            spouse_benefitbasis=0
        elif spouse_empstate==4:
            puoliso_palkka=0
            spouse_benefitbasis=puoliso_unempwage_basis
        elif spouse_empstate==5:
            puoliso_palkka=spouse_wage
            spouse_benefitbasis=spouse_basis_wage
        elif spouse_empstate==6:
            puoliso_palkka=spouse_wage
            spouse_benefitbasis=spouse_basis_wage
        elif spouse_empstate==7:
            puoliso_palkka=0
            spouse_benefitbasis=0
        elif spouse_empstate==8:
            puoliso_palkka=spouse_wage
            pot_spouse_wage=spouse_wage
            spouse_benefitbasis=0
        elif spouse_empstate==9:
            puoliso_palkka=spouse_wage
            pot_spouse_wage=spouse_wage
            spouse_benefitbasis=0
        elif spouse_empstate==10:
            puoliso_palkka=spouse_wage
            pot_spouse_wage=spouse_wage
            spouse_benefitbasis=spouse_wage
            puoliso_old_wage=0
        elif spouse_empstate==11:
            puoliso_palkka=0
            spouse_benefitbasis=0
        elif spouse_empstate==12:
            puoliso_palkka=0
            spouse_benefitbasis=0
        elif spouse_empstate==13:
            puoliso_palkka=0
            spouse_benefitbasis=puoliso_unempwage_basis
        elif spouse_empstate==14:
            puoliso_palkka=spouse_wage
            spouse_benefitbasis=spouse_basis_wage
        elif spouse_empstate==15:
            puoliso_palkka=0
            spouse_benefitbasis=0
        else:
            print('unknown state',spouse_empstate)

        paid_pension=kansanelake+tyoelake
        puoliso_paid_pension=spouse_kansanelake+puoliso_tyoelake

        netto,benq,netto_omat,netto_puoliso = self.comp_benefits(wage,benefitbasis,kansanelake,tyoelake,empstate,pt_state,tis,
                                children_under3,children_under7,children_under18,age,
                                puoliso,spouse_empstate,spouse_pt_state,puoliso_palkka,spouse_kansanelake,puoliso_tyoelake,spouse_benefitbasis,
                                spouse_time_in_state,used_unemp_benefit,puoliso_used_unemp_benefit,
                                g,p_g,potential_wage=pot_wage,potential_spouse_wage=pot_spouse_wage,
                                irtisanottu=pinkslip,karenssia_jaljella=karenssia_jaljella,
                                puoliso_irtisanottu=puoliso_pinkslip,puoliso_karenssia_jaljella=puoliso_karenssia_jaljella,
                                unempwage_basis=unempwage_basis,puoliso_unempwage_basis=puoliso_unempwage_basis,
                                kassanjasen=kassanjasen,unempwage=unempwage,puoliso_unempwage=puoliso_unempwage)

        return netto,benq,netto_omat,netto_puoliso

    def pension_accrual(self,age: float,wage: float,pension: float,state: int=1,ove_paid: int=0):
        '''
        Eläkkeen karttumisrutiini
        '''
        if age>=52 and age<63:
            acc = self.acc_over_52
        else:
            acc = self.acc

        if state in {0,4}:
            if age>=52 and age<63:
                acc = self.acc_unemp_over_52
            else:
                acc = self.acc_unemp

            if ove_paid>0:
                acc=0

            if age<self.min_retirementage:
                pension = pension*self.palkkakerroin+acc*wage
            else: # muuten ei karttumaa
                pension = pension*self.palkkakerroin
        elif state in {1,10}:
            if age<self.max_retirementage:
                pension = pension*self.palkkakerroin+acc*wage
            else:
                pension = pension*self.palkkakerroin
        elif state in {16}:
            if age<self.max_retirementage:
                pension = pension*self.palkkakerroin+acc*wage
            else:
                pension = pension*self.palkkakerroin
        elif state in {5,6}:
            if age>=52 and age<63:
                acc = self.acc_family_over_52
            else:
                acc = self.acc_family

            if age<self.max_retirementage:
                pension = pension*self.palkkakerroin+acc*max(wage,self.accbasis_kht)
            else:
                pension = pension*self.palkkakerroin
        elif state == 7:
            if age<self.max_retirementage:
                pension = pension*self.palkkakerroin+acc*self.accbasis_kht
            else:
                pension = pension*self.palkkakerroin
        elif state == 14:
            if age<self.max_retirementage:
                pension = pension*self.palkkakerroin+acc*wage*self.acc_sv
            else:
                pension = pension*self.palkkakerroin
        elif state in {8,9}:
            acc = self.acc # ei korotettua
            if age<self.max_retirementage:
                pension = pension*self.palkkakerroin+acc*wage
            else:
                pension = pension*self.palkkakerroin
        elif state == 13: # tm-tuki
            pension=pension*self.palkkakerroin # ei karttumaa!
        else: # 2,3,11,12,14,15 # ei karttumaa
            pension = pension*self.palkkakerroin # vastainen eläke, ei alkanut, ei karttumaa

        return pension

    def update_wage_reduction_baseline(self,state: int,wage_reduction: float,pinkslip: int,time_in_state: float,
            g: int,age: float,initial_reduction: bool=False, student_increase: bool=False):
        '''
        Pidetään kirjaa siitä, kuinka paljon palkkaa alennetaan työttämyyden keston suhteen,
        ja miten siitä palaudutaan
        '''
        if initial_reduction:
            wage_reduction=max(min_reduction,1.0-(1.0-self.wage_initial_reduction)*(1.0-wage_reduction))
        if student_increase:
            wage_reduction=max(min_reduction,wage_reduction-self.salary_const_student_final)

        if state in set([1,10]): # töissä
            wage_reduction=max(0,wage_reduction-self.salary_const_up)
        if state in {8,9}: # ve+töissä, reduction ei parane enää, vaan jää eläkkeellejääntihetken tasoon
            wage_reduction=max(0,wage_reduction) #-self.salary_const_up)
        elif state==12: # opiskelee
            wage_reduction=max(0,wage_reduction-self.salary_const_student)
        elif state in set([0,4,13,11]): # työtön tai työelämän ulkopuolella
            wage_reduction=min(1.0,wage_reduction+self.salary_const)
        elif state in set([5,6]): # äitiys-, isyys- tai vanhempainvapaa
            #wage_reduction += self.salary_const
            wage_reduction=min(1.0,wage_reduction+self.salary_const)
        elif state in set([3]):
            #wage_reduction=0.60 # vastaa määritelmää
            #wage_reduction=wage_reduction # jäädytetään
            if self.update_disab_wage_reduction:
                wage_reduction=min(1.0,wage_reduction+self.salary_const)
        elif state in set([7]): # kotihoidontuki tai ve tai tk
            wage_reduction=min(1.0,wage_reduction+self.salary_const)
        elif state in set([2]): # kotihoidontuki tai ve tai tk
            wage_reduction=min(1.0,wage_reduction+self.salary_const_retirement)
        elif state in {14}: # sairaspäiväraha
            wage_reduction=min(1.0,wage_reduction+self.salary_const_svpaiva[g])
        elif state in {15}: # ei muutosta
            wage_reduction=wage_reduction
        else: # ylivuoto, ei tiloja
            wage_reduction=wage_reduction

        return wage_reduction

    def step_wage_reduction_sigma(self,min_reduction: float,const: float,wr: float) -> float:
        return max(min_reduction,1.0-(1.0-const)*(1.0-wr))

    def update_wage_reduction_sigma(self,state: int,wage_reduction: float,pinkslip: int,time_in_state: float,
            g: int,age: float,initial_reduction: bool=False, student_increase: bool=False) -> float:
        '''
        Pidetään kirjaa siitä, kuinka paljon palkkaa alennetaan työttämyyden keston suhteen,
        ja miten siitä palaudutaan

        Tämä malli ei mene koskaan nollaan.

        Irtisanottuja työttömiä ei reduktio koske ensimmäisenä vuonna
        '''

        if g < 3:
            min_reduction=-0.01 # This is used to calibrate the modeled wages to match the observed wages
        else: # naisilla suurempi
            min_reduction=-0.05 # This is used to calibrate the modeled wages to match the observed wages

        if initial_reduction:
            wage_reduction=max(min_reduction,1.0-(1.0-self.wage_initial_reduction)*(1.0-wage_reduction))
        if student_increase:
            wage_reduction=max(min_reduction,wage_reduction-self.salary_const_student_final)

        if state in {1}: # töissä
            if age<40:
                wage_reduction=max(min_reduction,wage_reduction-self.salary_const_up)
            elif age<50:
                wage_reduction=max(min_reduction,wage_reduction-self.salary_const_up_40)
            elif age<60:
                wage_reduction=max(min_reduction,wage_reduction-self.salary_const_up_50)
            else:
                wage_reduction=max(min_reduction,wage_reduction-self.salary_const_up_60)
        elif state in {10}: # osatöissä
            if age<40:
                wage_reduction=max(min_reduction,wage_reduction-self.salary_const_up_osaaika)
            elif age<50:
                wage_reduction=max(min_reduction,wage_reduction-self.salary_const_up_osaaika40)
            elif age<60:
                wage_reduction=max(min_reduction,wage_reduction-self.salary_const_up_osaaika50)
            else:
                wage_reduction=max(min_reduction,wage_reduction-self.salary_const_up_osaaika60)
        elif state in {8,9}: # ve+töissä, reduction ei parane enää, vaan jää eläkkeellejääntihetken tasoon
            wage_reduction=max(min_reduction,wage_reduction) #-self.salary_const_up)
        elif state==12: # opiskelee
            wage_reduction=max(min_reduction,wage_reduction-self.salary_const_student)
        elif state in {11}: # työelämän ulkopuolella
            wage_reduction = self.step_wage_reduction_sigma(min_reduction,self.salary_const,wage_reduction) #max(min_reduction,1.0-(1.0-self.salary_const)*(1.0-wage_reduction))
        elif state in {0,4,13}: # työtön
            #if pinkslip<1 or (pinkslip>0 and time_in_state>0.49): # time_in_state ei ole ihan oikein tässä
            wage_reduction=max(min_reduction,1.0-(1.0-self.salary_const)*(1.0-wage_reduction))
        elif state in {5,6}: # isyys-, äitiys- tai vanhempainvapaa
            wage_reduction=max(min_reduction,1.0-(1.0-self.salary_const_ml)*(1.0-wage_reduction))
        elif state in {3}:
            #wage_reduction=0.60 # vastaa määritelmää
            #wage_reduction=wage_reduction # jäädytys
            if self.update_disab_wage_reduction:
                wage_reduction=max(min_reduction,1.0-(1.0-self.salary_const)*(1.0-wage_reduction))
        elif state in {7}: # kotihoidontuki
            wage_reduction=max(min_reduction,1.0-(1.0-self.salary_const_khh)*(1.0-wage_reduction))
        elif state in {2}: # ve
            wage_reduction=max(min_reduction,1.0-(1.0-self.salary_const_retirement)*(1.0-wage_reduction))
        elif state in {14}: # sairaspäiväraha
            wage_reduction=max(min_reduction,1.0-(1.0-self.salary_const_svpaiva[g])*(1.0-wage_reduction))
        elif state in {15}: # ei muutosta
            wage_reduction=wage_reduction
        else: # ylivuoto, ei tiloja
            wage_reduction=wage_reduction
            print('Error in sigma reduction: state {state}')

        return wage_reduction

    def update_family(self,puoliso: int,age: float,employment_status: int,spouse_empstate: int,sattuma) -> int:
        '''
        Päivitä puolison/potentiaalisen puolison tila & palkka
        Päivitä avioliitto/avoliitto
        '''
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

        if employment_status==15 or spouse_empstate==15:
            puoliso=0

        return puoliso

    def move_to_mort(self,age: float,children_under3: int,children_under7: int,children_under18: int,g: int,spouse_g: int,puoliso: int,prefnoise: int):
        #time_in_state += self.timestep
        if not self.include_mort:
            print('mort not included but emp state 15')

        employment_status,spouse_empstate=15,15

        wage=0
        nextwage=0
        toe=0
        if self.mortstop:
            done=True
        else:
            age=age+self.timestep
            done = age >= self.max_age
            done = bool(done)

        pension,raw_wage,nextwage,time_in_state=0,0,0,0
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

        spouse_wage,spouse_pension=0,0
        spouse_wage_reduction=0
        spouse_tyoelake_maksussa=0
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
        spouse_time_in_state=0
        puoliso_pinkslip = 0
        puoliso_ove_paid=0
        kansanelake=0
        spouse_kansanelake=0
        main_pt_action=0
        spouse_pt_action=0
        main_wage_basis=0
        spouse_wage_basis=0
        main_paid_wage=0
        spouse_paid_wage=0
        main_life_left=0
        spouse_life_left=0
        main_until_disab=0
        spouse_until_disab=0
        time_to_marriage=0
        time_to_divorce=0
        until_child=0
        main_until_student=0
        spouse_until_student=0
        main_until_outsider=0
        spouse_until_outsider=0

        self.state = self.states.state_encode(employment_status,g,spouse_g,pension,raw_wage,age,
                        time_in_state,tyoelake_maksussa,pinkslip,toe,toekesto,tyoura,nextwage,
                        used_unemp_benefit,wage_reduction,unemp_after_ra,unempwage,unempwage_basis,
                        children_under3,children_under7,children_under18,
                        0,alkanut_ansiosidonnainen,toe58,ove_paid,0,
                        puoliso,spouse_empstate,spouse_wage,spouse_pension,
                        spouse_wage_reduction,spouse_tyoelake_maksussa,puoliso_next_wage,
                        puoliso_used_unemp_benefit,puoliso_unemp_benefit_left,
                        puoliso_unemp_after_ra,puoliso_unempwage,puoliso_unempwage_basis,
                        puoliso_alkanut_ansiosidonnainen,puoliso_toe58,
                        puoliso_toe,puoliso_toekesto,puoliso_tyoura,spouse_time_in_state,puoliso_pinkslip,puoliso_ove_paid,
                        kansanelake,spouse_kansanelake,
                        main_paid_wage,spouse_paid_wage,
                        main_pt_action,spouse_pt_action,
                        main_wage_basis,spouse_wage_basis,
                        main_life_left,spouse_life_left,
                        main_until_disab,spouse_until_disab,
                        time_to_marriage,time_to_divorce,until_child,main_until_student,spouse_until_student,main_until_outsider,spouse_until_outsider,
                        prefnoise)

        if self.plotdebug:
            self.render()

        netto,benq,netto_omat,netto_puoliso = self.get_benefits(15,0,0,0,0,0,0,0,0,
                        0,0,0,children_under3,children_under7,children_under18,0,0,
                        0,15,0,0,0,0,
                        0,0,0,0,0,0,
                        0,0,0,3,potential_wage=0,potential_spouse_wage=0)

        benq['omat_eq'] = 0
        benq['puoliso_eq'] = 0
        benq['eq'] = 0

        reward=0
        equivalent=0
        return np.array(self.state), reward, done, benq

    def get_paid_wage(self,raw_wage,empstate,pt_act,old_wage=0,time_in_state=0):
        if empstate in {14,5,6} and time_in_state<0.25:
            paid_wage = old_wage
            main_pt_factor = 0 #self.parttime_actions[empstate,pt_act]
            pot_main_wage = paid_wage
        if empstate in {12}:
            main_pt_factor = self.parttime_actions[empstate,pt_act]
            paid_wage = main_pt_factor*raw_wage
            pot_main_wage = paid_wage
        if empstate in {1,10,8,9}:
            main_pt_factor = self.parttime_actions[empstate,pt_act]
            paid_wage = main_pt_factor*raw_wage
            pot_main_wage = paid_wage
        else:
            main_pt_factor = self.parttime_actions[empstate,pt_act]
            paid_wage = main_pt_factor*raw_wage
            pot_main_wage = raw_wage

        return paid_wage,main_pt_factor,pot_main_wage

    def check_empwages(self,main_empstate,main_wage,spouse_empstate,spouse_wage):
        if main_empstate not in set([3,15]) and main_wage<1.0:
            print(f'main: emp {main_empstate} w {main_wage}')
            self.render()

        if spouse_empstate not in set([3,15]) and spouse_wage<1.0:
            print(f'spouse: emp {spouse_empstate} w {spouse_wage}')
            self.render()

#############################
#
# STEP
#
#############################


    def step(self, action: int, dynprog: bool=False, debug: bool=False):
        '''
        Open AI interfacen mukainen step-funktio, joka tekee askeleen eteenpäin
        toiminnon action mukaan

        Keskeinen funktio simuloinnissa
        '''
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        emp_action,emp_savaction,main_pt_action,spouse_action,spouse_savaction,spouse_pt_action = self.get_actions(action)

        if self.plotdebug and False:
            print(f'act {emp_action} s_act {spouse_action}')

        main_empstate,g,spouse_g,main_pension,main_old_paid_wage,age,time_in_state,main_paid_pension,pinkslip,toe,\
            toekesto,tyoura,used_unemp_benefit,main_wage_reduction,unemp_after_ra,unempwage,\
            unempwage_basis,prefnoise,children_under3,children_under7,children_under18,\
            unemp_left,alkanut_ansiosidonnainen,toe58,ove_paid,jasen,\
            puoliso,spouse_empstate,spouse_old_paid_wage,spouse_pension,spouse_wage_reduction,\
            puoliso_paid_pension,puoliso_next_wage,puoliso_used_unemp_benefit,\
            puoliso_unemp_benefit_left,puoliso_unemp_after_ra,puoliso_unempwage,\
            puoliso_unempwage_basis,puoliso_alkanut_ansiosidonnainen,puoliso_toe58,\
            puoliso_toe,puoliso_toekesto,puoliso_tyoura,spouse_time_in_state,puoliso_pinkslip,\
            puoliso_ove_paid,kansanelake,spouse_kansanelake,tyoelake_maksussa,\
            spouse_tyoelake_maksussa,main_next_wage,\
            main_paid_wage,spouse_paid_wage,\
            pt_act,sp_pt_act,main_basis_wage,spouse_basis_wage,\
            main_life_left,spouse_life_left,main_until_disab,spouse_until_disab,\
            time_to_marriage,time_to_divorce,until_birth,\
            main_until_student,spouse_until_student,main_until_outsider,spouse_until_outsider\
                 = self.states.state_decode(self.state)

        main_wage = main_next_wage
        spouse_wage = puoliso_next_wage

        self.check_empwages(main_empstate,main_wage,spouse_empstate,spouse_wage)
        main_paid_wage,main_pt_factor,pot_main_wage = self.get_paid_wage(main_wage,main_empstate,main_pt_action,main_old_paid_wage,time_in_state)
        spouse_paid_wage,spouse_pt_factor,pot_spouse_wage = self.get_paid_wage(spouse_wage,spouse_empstate,spouse_pt_action,spouse_old_paid_wage,spouse_time_in_state)

        #pot_main_wage=main_wage
        #pot_spouse_wage=spouse_wage

        intage=int(math.floor(age))
        t=round((age-self.min_age)/self.timestep)
        main_moved=False
        spouse_moved=False

        if self.randomness: # can be disabled for testing purposes
            # kaikki satunnaisuus kerralla
            sattuma = np.random.uniform(size=9)
            sattuma2 = np.random.uniform(size=9)

            if self.include_spouses:
                #puoliso = self.update_family(puoliso,age,main_empstate,spouse_empstate,sattuma)
                if puoliso>0:
                    if time_to_divorce <= 0 and main_empstate not in {15} and spouse_empstate not in {15}:
                        puoliso = 0
                        time_to_marriage,time_to_divorce = self.comp_time_to_marriage(puoliso,age,g,spouse_g)
                    else:
                        time_to_divorce -= self.timestep
                else:
                    if time_to_marriage <= 0 and main_empstate not in {15} and spouse_empstate not in {15}:
                        puoliso = 1
                        time_to_marriage,time_to_divorce = self.comp_time_to_marriage(puoliso,age,g,spouse_g)
                    else:
                        time_to_marriage -= self.timestep
            else:
                puoliso=0

            if until_birth <= 0.0: # vanhempainvapaa
                # ikä valittu äidin iän mukaan. oikeastaan tämä ei mene ihan oikein miehille
                if spouse_empstate not in {3,15}: # äiti ei ole tiloissa 3 tai 15
                    until_birth = self.comp_until_birth(spouse_g,age)
                    self.infostate_add_child(age)
                    spouse_empstate,spouse_pension,tyoelake_maksussa,spouse_paid_wage,spouse_time_in_state,puoliso_pinkslip,spouse_basis_wage=\
                        self.move_to_motherleave(spouse_pension,spouse_tyoelake_maksussa,age,True,spouse_old_paid_wage)
                    puoliso_karenssia_jaljella = 0
                    spouse_wage_reduction = self.update_wage_reduction(spouse_empstate,spouse_wage_reduction,puoliso_pinkslip,spouse_time_in_state,spouse_g,age)
                    spouse_moved = True
                    if sattuma[4]<0.39 and main_empstate not in set([3,15]):
                        main_empstate,main_pension,tyoelake_maksussa,main_paid_wage,time_in_state,pinkslip,main_basis_wage=\
                            self.move_to_fatherleave(main_pension,tyoelake_maksussa,age,False,main_old_paid_wage)
                        karenssia_jaljella=0
                        wage_reduction = self.update_wage_reduction(main_empstate,main_wage_reduction,pinkslip,time_in_state,g,age)
                        main_moved = True
            # siirtymät
            if main_empstate!=15:
                if main_life_left <= 0 and self.include_mort:
                    if puoliso>0: # avo- tai avioliitossa
                        # huomioi vain maksussa olevan puolison eläkkeen. Ei taida olla oikein.
                        spouse_tyoelake_maksussa = self.comp_familypension(puoliso,main_empstate,spouse_empstate,
                            tyoelake_maksussa,main_pension,age,spouse_tyoelake_maksussa,
                            spouse_pension,children_under18,puoliso,False)
                    main_empstate,main_pension,main_wage,time_in_state,puoliso,tyoelake_maksussa,kansanelake,basis_wage=\
                        self.move_to_deceiced()
                    main_moved = True
                elif main_until_disab <= 0 and main_empstate not in {3,14,15}:
                    emp_action=11 # disability
                elif main_until_student <= 0.0 and main_empstate in {0,1,4,10,13} and age<self.min_retirementage and not main_moved: # opiskelu sisään, not in set([2,3,5,6,7,8,9,11,12,15]):
                    main_empstate,main_paid_wage,main_pension,tyoelake_maksussa,time_in_state,pinkslip,main_basis_wage,main_until_student,karenssia_jaljella,main_pt_factor=\
                        self.move_to_student(main_wage,main_pt_action,main_pension,tyoelake_maksussa,age,time_in_state,tyoura,pinkslip,g)
                    main_wage_reduction = self.update_wage_reduction(main_empstate,main_wage_reduction,pinkslip,time_in_state,g,age)
                    main_moved = True
                elif main_until_outsider <= 0.0 and main_empstate in {0,1,4,10,12,13} and age<self.max_retirementage and not main_moved: # not in set([2,3,5,6,7,8,9,11,15]), outsider sisään
                    main_empstate,main_pension,tyoelake_maksussa,main_paid_wage,time_in_state,pinkslip,main_basis_wage,main_until_outsider,karenssia_jaljella=\
                        self.move_to_outsider(main_pension,tyoelake_maksussa,age,g,moved=main_moved)
                    if not main_moved:
                        main_wage_reduction = self.update_wage_reduction(main_empstate,main_wage_reduction,pinkslip,time_in_state,g,age)
                    main_moved = True

            if spouse_empstate!=15:
                if spouse_life_left <= 0 and spouse_empstate!=15 and self.include_mort:
                    if puoliso>0: # avo- tai avioliitossa
                        # huomioi vain maksussa olevan puolison eläkkeen. Ei taida olla oikein.
                        tyoelake_maksussa = self.comp_familypension(puoliso,spouse_empstate,main_empstate,
                            spouse_tyoelake_maksussa,spouse_pension,age,tyoelake_maksussa,
                            main_pension,children_under18,puoliso,True)
                    spouse_empstate,spouse_pension,spouse_wage,spouse_time_in_state,puoliso,spouse_tyoelake_maksussa,spouse_kansanelake,puoliso_basis_wage=\
                        self.move_to_deceiced()
                    spouse_moved = True
                elif spouse_until_disab <= 0 and spouse_empstate not in {3,14,15}:
                    spouse_action=11 # disability
                elif spouse_until_student <= 0.0 and spouse_empstate in {0,1,4,10,13} and age<self.min_retirementage and not spouse_moved: # opiskelu sisään, not in set([2,3,5,6,7,8,9,11,12,15]):
                    spouse_empstate,spouse_paid_wage,spouse_pension,spouse_tyoelake_maksussa,spouse_time_in_state,puoliso_pinkslip,spouse_basis_wage,spouse_until_student,puoliso_karenssia_jaljella,spouse_pt_factor=\
                        self.move_to_student(spouse_wage,spouse_pt_action,spouse_pension,spouse_tyoelake_maksussa,age,spouse_time_in_state,puoliso_tyoura,puoliso_pinkslip,spouse_g)
                    spouse_wage_reduction = self.update_wage_reduction(spouse_empstate,spouse_wage_reduction,puoliso_pinkslip,spouse_time_in_state,spouse_g,age)
                    spouse_moved = True
                elif spouse_until_outsider <= 0.0 and spouse_empstate in {0,1,4,10,12,13} and age<self.max_retirementage and not spouse_moved: # outsider sisään, not in set([2,3,5,6,7,8,9,11,12,15]):
                    spouse_empstate,spouse_pension,spouse_tyoelake_maksussa,spouse_paid_wage,spouse_time_in_state,puoliso_pinkslip,puoliso_basis_wage,spouse_until_outsider,puoliso_karenssia_jaljella=\
                        self.move_to_outsider(spouse_pension,spouse_tyoelake_maksussa,age,spouse_g,moved=spouse_moved)
                    if not spouse_moved:
                        spouse_wage_reduction = self.update_wage_reduction(spouse_empstate,spouse_wage_reduction,puoliso_pinkslip,spouse_time_in_state,spouse_g,age)
                    spouse_moved = True
        else:
            # tn ei ole koskaan alle rajan, jos tämä on 1
            sattuma = np.ones(9)
            sattuma2 = np.ones(9)

        if main_empstate==15 and spouse_empstate==15: # both deceiced
            return self.move_to_mort(age,children_under3,children_under7,children_under18,g,spouse_g,puoliso,prefnoise)

        karenssia_jaljella,puoliso_karenssia_jaljella=0,0 # karenssi max 3m
        if age >= self.max_retirementage and main_empstate not in set([2,3,15]):
            if sattuma[2]<0.8:
                main_empstate,kansanelake,tyoelake_maksussa,main_pension,main_paid_wage,time_in_state,ove_paid,main_basis_wage,main_until_student,main_until_outsider\
                    = self.move_to_retirement(main_pension,age,kansanelake,tyoelake_maksussa,
                        main_empstate,unemp_after_ra,all_acc=True,scale_acc=True,has_spouse=puoliso,is_spouse=False)
                pinkslip = 0
                main_wage_reduction = self.update_wage_reduction(main_empstate,main_wage_reduction,pinkslip,time_in_state,g,age)
                karenssia_jaljella=0
                main_moved = True

        if age >= self.max_retirementage and spouse_empstate not in set([2,3,15]):
            if sattuma2[2]<0.8:
                spouse_empstate,spouse_kansanelake,spouse_tyoelake_maksussa,spouse_pension,spouse_paid_wage,spouse_time_in_state,puoliso_ove_paid,spouse_basis_wage,spouse_until_student,spouse_until_outsider\
                    = self.move_to_retirement(spouse_pension,age,spouse_kansanelake,spouse_tyoelake_maksussa,spouse_empstate,\
                        puoliso_unemp_after_ra,all_acc=True,scale_acc=True,has_spouse=puoliso,is_spouse=True)
                puoliso_pinkslip = 0
                spouse_wage_reduction = self.update_wage_reduction(spouse_empstate,spouse_wage_reduction,puoliso_pinkslip,spouse_time_in_state,spouse_g,age)
                puoliso_karenssia_jaljella=0
                spouse_moved = True

        if (not main_moved) and main_empstate != 15:
            # hoidetaan tilasiirtymät ja -pysymiset alirutiineilla, joita kutsutaan mäppäämällä tila funktioksi,
            # jota sitten kutsutaan
            is_spouse=False
            main_empstate,kansanelake,tyoelake_maksussa,main_pension,time_in_state,pinkslip,\
            unemp_after_ra,tyoura,used_unemp_benefit,unempwage_basis,alkanut_ansiosidonnainen,ove_paid,karenssia_jaljella,\
            main_basis_wage,main_until_disab,main_until_student,main_until_outsider,main_life_left\
                = self.map_stays[main_empstate](main_wage,main_paid_wage,main_pt_action,main_empstate,kansanelake,tyoelake_maksussa,
                                main_pension,time_in_state,toe,toekesto,tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,
                                main_old_paid_wage,unempwage,unempwage_basis,emp_action,age,sattuma,intage,g,alkanut_ansiosidonnainen,
                                toe58,ove_paid,children_under3,children_under18,main_basis_wage,puoliso,is_spouse,\
                                main_until_disab,main_until_student,main_until_outsider,main_life_left)
            main_wage_reduction = self.update_wage_reduction(main_empstate,main_wage_reduction,pinkslip,time_in_state,g,age)

        if (not spouse_moved) and spouse_empstate != 15:
            # hoidetaan tilasiirtymät ja -pysymiset alirutiineilla, joita kutsutaan mäppäämällä tila funktioksi,
            # jota sitten kutsutaan
            is_spouse=True
            spouse_empstate,spouse_kansanelake,spouse_tyoelake_maksussa,spouse_pension,spouse_time_in_state,puoliso_pinkslip,\
            puoliso_unemp_after_ra,puoliso_tyoura,puoliso_used_unemp_benefit,puoliso_unempwage_basis,puoliso_alkanut_ansiosidonnainen,puoliso_ove_paid,\
            puoliso_karenssia_jaljella,spouse_basis_wage,spouse_until_disab,spouse_until_student,spouse_until_outsider,spouse_life_left\
                = self.map_stays[spouse_empstate](spouse_wage,spouse_paid_wage,spouse_pt_action,spouse_empstate,spouse_kansanelake,spouse_tyoelake_maksussa,spouse_pension,spouse_time_in_state,puoliso_toe,puoliso_toekesto,
                               puoliso_tyoura,puoliso_used_unemp_benefit,puoliso_pinkslip,puoliso_unemp_after_ra,spouse_old_paid_wage,puoliso_unempwage,puoliso_unempwage_basis,
                               spouse_action,age,sattuma2,intage,spouse_g,puoliso_alkanut_ansiosidonnainen,puoliso_toe58,puoliso_ove_paid,children_under3,children_under18,spouse_basis_wage,
                               puoliso,is_spouse,spouse_until_disab,spouse_until_student,spouse_until_outsider,spouse_life_left)
            spouse_wage_reduction = self.update_wage_reduction(spouse_empstate,spouse_wage_reduction,puoliso_pinkslip,spouse_time_in_state,spouse_g,age)

        main_paid_wage,main_pt_factor,pot_main_wage = self.get_paid_wage(main_wage,main_empstate,main_pt_action,main_old_paid_wage,time_in_state-0.25)
        spouse_paid_wage,spouse_pt_factor,pot_spouse_wage = self.get_paid_wage(spouse_wage,spouse_empstate,spouse_pt_action,spouse_old_paid_wage,spouse_time_in_state-0.25)

        netto,benq,netto_omat,netto_puoliso = self.get_benefits(main_empstate,main_pt_action,main_paid_wage,kansanelake,tyoelake_maksussa,main_pension,
                    time_in_state,pinkslip,unempwage,unempwage_basis,karenssia_jaljella,age,
                    children_under3,children_under7,children_under18,ove_paid,used_unemp_benefit,
                    puoliso,spouse_empstate,spouse_pt_action,spouse_paid_wage,spouse_kansanelake,spouse_tyoelake_maksussa,
                    puoliso_pinkslip,puoliso_karenssia_jaljella,spouse_time_in_state,puoliso_unempwage,puoliso_unempwage_basis,
                    puoliso_used_unemp_benefit,main_basis_wage,spouse_basis_wage,
                    g,spouse_g,potential_wage=pot_main_wage,potential_spouse_wage=pot_spouse_wage,kassanjasen=jasen)

        # after this, preparing for the next step
        age = age+self.timestep
        done = age >= self.max_age
        done = bool(done)

        until_birth,main_life_left,spouse_life_left,main_until_disab,spouse_until_disab,main_until_student,spouse_until_student,main_until_outsider,spouse_until_outsider = \
            self.update_times(age,main_empstate,spouse_empstate,until_birth,main_life_left,spouse_life_left,main_until_disab,spouse_until_disab,
                main_until_student,spouse_until_student,main_until_outsider,spouse_until_outsider)

        # test this!
        used_unemp_benefit,alkanut_ansiosidonnainen,unempwage_basis,pvr_jaljella,\
         toe,toekesto,unempwage,children_under3,children_under7,children_under18 = \
            self.update_toes(t,age,toe,tyoura,toe58,main_empstate,main_paid_wage,main_basis_wage,unempwage_basis,False,
                    used_unemp_benefit,alkanut_ansiosidonnainen,unempwage)

        puoliso_used_unemp_benefit,puoliso_alkanut_ansiosidonnainen,puoliso_unempwage_basis,puoliso_pvr_jaljella,\
         puoliso_toe,puoliso_toekesto,puoliso_unempwage,_,_,_ = \
            self.update_toes(t,age,puoliso_toe,puoliso_tyoura,puoliso_toe58,spouse_empstate,spouse_paid_wage,spouse_basis_wage,
                    puoliso_unempwage_basis,True,puoliso_used_unemp_benefit,puoliso_alkanut_ansiosidonnainen,puoliso_unempwage)

        kassanjasenyys = self.get_kassanjasenyys()

        #self.render_infostate()

        if not done:
            reward_omat,omat_equivalent = self.log_utility(netto_omat,main_empstate,age,g=g,pinkslip=pinkslip,pt_factor=main_pt_factor,children_under_3y=children_under3)
            reward_puoliso,spouse_equivalent = self.log_utility(netto_puoliso,spouse_empstate,age,g=spouse_g,pinkslip=puoliso_pinkslip,pt_factor=spouse_pt_factor,children_under_3y=children_under3)
            reward=reward_omat+reward_puoliso
            equivalent=omat_equivalent+spouse_equivalent

            if not np.isfinite(reward_omat):
                print('omat',netto_omat,reward_omat)
            if not np.isfinite(reward_puoliso):
                print('puoliso',netto_puoliso,reward_puoliso)

            benq['omat_eq'] = omat_equivalent
            benq['puoliso_eq'] = spouse_equivalent
            benq['omat_dis_wage5y'] = self.main_dis_wage5y
            benq['puoliso_dis_wage5y'] = self.spouse_dis_wage5y
            benq['eq'] = equivalent
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0

            main_paid_pension += self.elinaikakerroin*main_pension # ei ihan oikein! lykkäyskorotus puuttuu, optimoijan pitäisi huomata, joten ei vaikutusta?
            main_pension=0.0
            puoliso_paid_pension += self.elinaikakerroin*spouse_pension # ei ihan oikein! lykkäyskorotus puuttuu, optimoijan pitäisi huomata, joten ei vaikutusta?
            spouse_pension=0.0

            netto,benq,netto_omat,netto_puoliso = self.get_benefits(main_empstate,main_pt_action,main_paid_wage,kansanelake,tyoelake_maksussa,main_pension,time_in_state,pinkslip,
                unempwage,unempwage_basis,karenssia_jaljella,age,children_under3,children_under7,children_under18,ove_paid,used_unemp_benefit,
                puoliso,spouse_empstate,spouse_pt_action,spouse_paid_wage,spouse_kansanelake,spouse_tyoelake_maksussa,puoliso_pinkslip,puoliso_karenssia_jaljella,
                spouse_time_in_state,puoliso_unempwage,puoliso_unempwage_basis,puoliso_used_unemp_benefit,0,0,g,spouse_g,potential_wage=pot_main_wage,potential_spouse_wage=pot_spouse_wage)

            if main_empstate in {2,3,8,9}: # retired
                reward_omat,omat_equivalent = self.log_utility(netto_omat,main_empstate,age,g=g,pinkslip=pinkslip,children_under_3y=children_under3)
                k=round(main_life_left/self.timestep)
                reward_omat *= self.npv_gpension[k] # discounting with pension index & gamma
                npv0 = k
                npv = self.npv_pension[k]
            else:
                # giving up the pension
                reward_omat,omat_equivalent=0.0,0.0
                npv,npv0 = 0.0,0.0

            if spouse_empstate in {2,3,8,9}: # retired
                reward_puoliso,spouse_equivalent = self.log_utility(netto_puoliso,int(spouse_empstate),age,g=spouse_g,pinkslip=puoliso_pinkslip,children_under_3y=children_under3)
                k=round(spouse_life_left/self.timestep)
                reward_puoliso *= self.npv_gpension[k] # discounting with pension index & gamma
                p_npv0 = k
                p_npv = self.npv_pension[k]
            else:
                # giving up the pension
                reward_puoliso,spouse_equivalent=0.0,0.0
                p_npv,p_npv0=0.0,0.0

            benq['omat_eq'] = omat_equivalent
            benq['puoliso_eq'] = spouse_equivalent

            # updates benq directly
            self.scale_q(npv,npv0,p_npv,p_npv0,benq,age)

            # total reward is
            reward=reward_omat+reward_puoliso
            equivalent=omat_equivalent+spouse_equivalent
            benq['eq'] = equivalent
            benq['omat_dis_wage5y'] = self.main_dis_wage5y
            benq['puoliso_dis_wage5y'] = self.spouse_dis_wage5y
            pinkslip = 0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0
            equivalent = 0.0
            omat_equivalent = 0.0
            spouse_equivalent = 0.0

            benq['omat_eq'] = omat_equivalent
            benq['puoliso_eq'] = spouse_equivalent
            benq['omat_dis_wage5y'] = 0
            benq['puoliso_dis_wage5y'] = 0
            benq['eq'] = equivalent

        # seuraava palkka tiedoksi valuaatioapproksimaattorille
        if main_empstate in {15}:
            main_next_wage=0
        else:
            main_next_wage = self.get_wage(age,main_wage_reduction)

        if spouse_empstate in {15}:
            puoliso_next_wage=0
        else:
            puoliso_next_wage = self.get_spousewage(age,spouse_wage_reduction)

        #self.infostate_print_ages(age)
        #print(f'1.0: c3 {children_under3} c7 {children_under7} c18 {children_under18}')

        self.state = self.states.state_encode(main_empstate,g,spouse_g,main_pension,main_wage,age,time_in_state,
                                tyoelake_maksussa,pinkslip,toe,toekesto,tyoura,main_next_wage,used_unemp_benefit,
                                main_wage_reduction,unemp_after_ra,unempwage,unempwage_basis,
                                children_under3,children_under7,children_under18,
                                pvr_jaljella,alkanut_ansiosidonnainen,toe58,ove_paid,kassanjasenyys,
                                puoliso,spouse_empstate,spouse_wage,spouse_pension,
                                spouse_wage_reduction,spouse_tyoelake_maksussa,puoliso_next_wage,
                                puoliso_used_unemp_benefit,puoliso_pvr_jaljella,
                                puoliso_unemp_after_ra,puoliso_unempwage,puoliso_unempwage_basis,
                                puoliso_alkanut_ansiosidonnainen,puoliso_toe58,
                                puoliso_toe,puoliso_toekesto,puoliso_tyoura,spouse_time_in_state,
                                puoliso_pinkslip,puoliso_ove_paid,kansanelake,spouse_kansanelake,
                                main_paid_wage,spouse_paid_wage,
                                main_pt_action,spouse_pt_action,
                                main_basis_wage,spouse_basis_wage,
                                main_life_left,spouse_life_left,
                                main_until_disab,spouse_until_disab,
                                time_to_marriage,time_to_divorce,until_birth,
                                main_until_student,spouse_until_student,main_until_outsider,spouse_until_outsider,
                                prefnoise)

        #self.check_cstate()

        if self.plotdebug:
            self.render(done=done,reward=reward,netto=netto,benq=benq,netto_omat=netto_omat,netto_puoliso=netto_puoliso)

        return np.array(self.state,dtype=np.float32), reward, done, benq

    def check_cstate(self):
        state = np.array(self.state,dtype=np.float32)
        main_empstate,g,spouse_g,main_pension,main_old_paid_wage,age,time_in_state,main_paid_pension,pinkslip,toe,\
            toekesto,tyoura,used_unemp_benefit,main_wage_reduction,unemp_after_ra,unempwage,\
            unempwage_basis,prefnoise,children_under3,children_under7,children_under18,\
            unemp_left,alkanut_ansiosidonnainen,toe58,ove_paid,jasen,\
            puoliso,spouse_empstate,spouse_old_paid_wage,spouse_pension,spouse_wage_reduction,\
            puoliso_paid_pension,puoliso_next_wage,puoliso_used_unemp_benefit,\
            puoliso_unemp_benefit_left,puoliso_unemp_after_ra,puoliso_unempwage,\
            puoliso_unempwage_basis,puoliso_alkanut_ansiosidonnainen,puoliso_toe58,\
            puoliso_toe,puoliso_toekesto,puoliso_tyoura,spouse_time_in_state,puoliso_pinkslip,\
            puoliso_ove_paid,kansanelake,spouse_kansanelake,tyoelake_maksussa,\
            spouse_tyoelake_maksussa,main_next_wage,\
            main_paid_wage,spouse_paid_wage,\
            pt_act,sp_pt_act,main_basis_wage,spouse_basis_wage,\
            main_life_left,spouse_life_left,main_until_disab,spouse_until_disab,\
            time_to_marriage,time_to_divorce,until_birth,\
            main_until_student,spouse_until_student,main_until_outsider,spouse_until_outsider\
                 = self.states.state_decode(state)
        print(f'2.0: c3 {children_under3} c7 {children_under7} c18 {children_under18}')

    def update_toes(self,t,age,toe,tyoura,toe58,main_empstate,main_paid_wage,main_basis_wage,unempwage_basis,is_spouse,
                    used_unemp_benefit,alkanut_ansiosidonnainen,unempwage_old):
        toe58 = self.check_toe58(age,toe,tyoura,toe58)
        #puoliso_toe58 = self.check_toe58(age,puoliso_toe,puoliso_tyoura,puoliso_toe58)

        work={1,10}
        retired={2,8,9}
        self.update_infostate(t,main_empstate,main_paid_wage,main_basis_wage,unempwage_basis,is_spouse=is_spouse)
        toe,toekesto,unempwage,children_under3,children_under7,children_under18 = self.comp_infostats(age,is_spouse=is_spouse)
        #if is_spouse:
        #    self.infostate_print_ages(age)
        if age >= 58 and self.suojasaanto_toe58: # suojasääntö
            unempwage = max(unempwage,unempwage_old)

        if main_empstate in work and self.tyossaoloehto(toe,tyoura,age):
            used_unemp_benefit=0
            alkanut_ansiosidonnainen=0
            #if alkanut_ansiosidonnainen>0:
            #    if not self.infostate_check_aareset(age):
            #        alkanut_ansiosidonnainen=0
        elif main_empstate in retired:
            alkanut_ansiosidonnainen=0
        if alkanut_ansiosidonnainen<1 or age>self.max_unemploymentbenefitage:
            unempwage_basis=0

        if main_empstate not in set([2,3,8,9,15]) and age<self.max_unemploymentbenefitage:
            if self.porrasta_toe and (main_empstate in set([0,4]) or alkanut_ansiosidonnainen>0):
                old_toe = self.comp_oldtoe(spouse=False)
                pvr_jaljella = self.comp_unempdays_left(used_unemp_benefit,tyoura,age,toe,main_empstate,alkanut_ansiosidonnainen,toe58,old_toe,printti=False)
            else:
                pvr_jaljella = self.comp_unempdays_left(used_unemp_benefit,tyoura,age,toe,main_empstate,alkanut_ansiosidonnainen,toe58,toekesto)
        else:
            pvr_jaljella=0

        return used_unemp_benefit,alkanut_ansiosidonnainen,unempwage_basis,pvr_jaljella,toe,toekesto,unempwage,\
               children_under3,children_under7,children_under18

    def update_times(self,age,main_empstate,spouse_empstate,until_birth,main_life_left,spouse_life_left,main_until_disab,spouse_until_disab,
                     main_until_student,spouse_until_student,main_until_outsider,spouse_until_outsider):

        if age<=55 and spouse_empstate not in {3,15}:
            until_birth -= self.timestep

        if main_empstate != 15:
            main_life_left -= self.timestep

        if spouse_empstate != 15:
            spouse_life_left -= self.timestep

        if main_empstate not in {3,14,15}:
            main_until_disab -= self.timestep
            main_until_student -= self.timestep
            main_until_outsider -= self.timestep

        if spouse_empstate not in {3,14,15}:
            spouse_until_disab -= self.timestep
            spouse_until_student -= self.timestep
            spouse_until_outsider -= self.timestep

        return until_birth,main_life_left,spouse_life_left,main_until_disab,spouse_until_disab,main_until_student,spouse_until_student,main_until_outsider,spouse_until_outsider

    def scale_q(self,npv: float,npv0: float,p_npv: float,p_npv0: float,benq: dict,age: float) -> None:
        '''
        Scaling the incomes etc by a discounted nominal present value
        '''
        omat='omat_'
        puoliso='puoliso_'

        v_pens={omat: npv, puoliso: p_npv}
        v0_pens={omat: npv0, puoliso: p_npv0}
        for alku in set([omat,puoliso]):
            p1=v_pens[alku]
            p2=v0_pens[alku]
            benq[alku+'verot'] *= p1
            benq[alku+'etuustulo_brutto'] *= p1
            benq[alku+'ylevero'] *= p1
            benq[alku+'alv'] *= p1
            benq[alku+'valtionvero'] *= p1
            benq[alku+'kunnallisvero'] *= p1
            benq[alku+'asumistuki'] *= p1   # eläkeläisen asumistuki puuttuu??
            benq[alku+'tyotvakmaksu'] *= p1 # tätä ei oikeastaan tarvita, mutta ei haittaa
            benq[alku+'sairausvakuutusmaksu'] *= p1 # sairaanhoitomaksu maksetaan myös eläketulosta
            benq[alku+'elake_maksussa'] *= p1
            benq[alku+'tyoelake'] *= p1
            benq[alku+'kansanelake'] *= p1 # p.o. kelaindeksillä
            benq[alku+'takuuelake'] *= p1 # p.o. kelaindeksillä
            benq[alku+'kokoelake'] *= p1
            benq[alku+'perustulo'] *= p1
            benq[alku+'toimeentulotuki'] *= p1
            benq[alku+'netto'] *= p1
            benq[alku+'etuustulo_netto'] *= p1
            benq[alku+'eq'] *= p1
            benq[alku+'multiplier'] = p2

        benq['verot'] = benq[omat+'verot']+benq[puoliso+'verot']
        benq['etuustulo_brutto'] = benq[omat+'etuustulo_brutto']+benq[puoliso+'etuustulo_brutto']
        benq['ylevero'] = benq[omat+'ylevero']+benq[puoliso+'ylevero']
        benq['alv'] = benq[omat+'alv']+benq[puoliso+'alv']
        benq['valtionvero'] = benq[omat+'valtionvero']+benq[puoliso+'valtionvero']
        benq['kunnallisvero'] = benq[omat+'kunnallisvero']+benq[puoliso+'kunnallisvero']
        benq['asumistuki'] = benq[omat+'asumistuki']+benq[puoliso+'asumistuki']
        benq['tyotvakmaksu'] = benq[omat+'tyotvakmaksu']
        benq['sairausvakuutusmaksu'] = benq[omat+'sairausvakuutusmaksu']+benq[puoliso+'sairausvakuutusmaksu']
        benq['elake_maksussa'] = benq[omat+'elake_maksussa']+benq[puoliso+'elake_maksussa']
        benq['tyoelake'] = benq[omat+'tyoelake']+benq[puoliso+'tyoelake']
        benq['kansanelake'] = benq[omat+'kansanelake']+benq[puoliso+'kansanelake']
        benq['takuuelake'] = benq[omat+'takuuelake']+benq[puoliso+'takuuelake']
        benq['kokoelake'] = benq[omat+'kokoelake']+benq[puoliso+'kokoelake']
        #delta = benq['kokoelake'] - benq['kansanelake'] - benq['takuuelake'] - benq['tyoelake']
        #if abs(delta)>0.01:
        #    print(delta,'kokoelake',benq['kokoelake'],'kansanelake',benq['kansanelake'],'takuuelake',benq['takuuelake'],'tyoelake',benq['tyoelake'])
        benq['perustulo'] = benq[omat+'perustulo']+benq[puoliso+'perustulo']
        benq['toimeentulotuki'] = benq[omat+'toimeentulotuki']+benq[puoliso+'toimeentulotuki']
        benq['netto'] = benq[omat+'netto']+benq[puoliso+'netto']
        benq['etuustulo_netto'] = benq[omat+'etuustulo_netto']+benq[puoliso+'etuustulo_netto']
        benq['eq'] = benq[omat+'eq']+benq[puoliso+'eq']
        #benq['multiplier'] = (benq[omat+'multiplier']+benq[puoliso+'multiplier'])/2

    #  Perussetti, tuottaa korkean elastisuuden

    ##################
    ###
    ###     UTILITY
    ###
    ##################

    def log_utility_mort_ove_det_params(self):
        #
        # OVE - YES
        # MORT - YES
        # DETERMINISTIC
        #

        self.salary_const=0.045*self.timestep # 0.0425*self.timestep # 0.038 työttämyydestä palkka alenee tämän verran vuodessa
        self.salary_const_khh=0.025*self.timestep # 0.0425*self.timestep # 0.038 työttämyydestä palkka alenee tämän verran vuodessa
        self.salary_const_ml=0.0*self.timestep # vanhempainvapaasta palkka alenee tämän verran vuodessa; palkkapenalty on jo mukana keskipalkoissa, joten tässä ei
        self.salary_const_retirement=0.10*self.timestep # vanhuuseläkkeellä muutos nopeampaa
        self.salary_const_svpaiva=np.array([0.30,0.25,0.25,0.30,0.25,0.20])*self.timestep # [0.20,0.16,0.12,0.20,0.15,0.12] pitkällä svpäivärahalla muutos nopeaa fyysisissä töissä
        self.salary_const_up=0.030*self.timestep # 0.04 työssäolo palauttaa ansioita tämän verran vuodessa
        self.salary_const_up_40=1.0*self.salary_const_up # 0.04 työssäolo palauttaa ansioita tämän verran vuodessa
        self.salary_const_up_50=1.0*self.salary_const_up # 0.04 työssäolo palauttaa ansioita tämän verran vuodessa
        self.salary_const_up_60=1.0*self.salary_const_up # 0.04 työssäolo palauttaa ansioita tämän verran vuodessa
        self.salary_const_up_osaaika=0.025*self.timestep # 0.04 osa-aikainen työssäolo palauttaa ansioita tämän verran vuodessa
        self.salary_const_up_osaaika40=1.0*self.salary_const_up_osaaika # 0.04 työssäolo palauttaa ansioita tämän verran vuodessa
        self.salary_const_up_osaaika50=1.0*self.salary_const_up_osaaika # 0.04 työssäolo palauttaa ansioita tämän verran vuodessa
        self.salary_const_up_osaaika60=1.0*self.salary_const_up_osaaika # 0.04 työssäolo palauttaa ansioita tämän verran vuodessa
        self.salary_const_student=0.025*self.timestep # 0.05 opiskelu pienentää leikkausta tämän verran vuodessa
        self.salary_const_student_final=0.05 # 0.05 opiskelu pienentää leikkausta tämän verran vuodessa

        self.max_mu_age = self.min_retirementage+15.0 #

        self.men_mu_scale_kokoaika_before=0.075 # 0.075 # 0.130 # 0.0595 # 0.065 # how much penalty is associated with work increase with age after mu_age
        self.men_mu_scale_kokoaika_after=0.025 # 0.035 # 0.130 # 0.0595 # 0.065 # how much penalty is associated with work increase with age after mu_age
        #self.men_mu_scale_osaaika=0.085 #0.010 # how much penalty is associated with work increase with age after mu_age
        self.men_mu_age = self.min_retirementage - 6.0 # - 5.0
        self.men_kappa_hoitovapaa=0.005 # hyöty henkilölle hoitovapaalla olosta
        self.men_kappa_under_3y=0.005
        self.men_kappa_ve=0.0
        self.men_kappa_pinkslip_young=0.25
        self.men_kappa_pinkslip_middle=0.15
        self.men_kappa_pinkslip_elderly=0.10
        self.men_kappa_pinkslip_putki=0.25
        self.men_kappa_param= np.array([-0.360, -0.390, -0.450, -0.550, -0.705, -1.400]) # osa-aika 8h, 16h, 24h, kokoaika 32h, 40h, 48h
                            # np.array([-0.350, -0.385, -0.435, -0.535, -0.690, -1.200]) # osa-aika 8h, 16h, 24h, kokoaika 32h, 40h, 48h
                            # delta      0.020   0.090   0.75  0.125  0.300
        self.men_student_kappa_param=np.array([0.05, 0.0, -0.05, -100.0, -100.0, -100.0]) # osa-aika 8h, 16h, 24h, kokoaika 32h, 40h, 48h

        self.women_mu_scale_kokoaika_before=0.065 # 0.065  #0.150 # 0.0785 #0.085 # how much penalty is associated with work increase with age after mu_age
        self.women_mu_scale_kokoaika_after=0.015  #0.150 # 0.0785 #0.085 # how much penalty is associated with work increase with age after mu_age
        #self.women_mu_scale_osaaika=0.105 # 0.020 # how much penalty is associated with work increase with age after mu_age
        self.women_mu_age = self.min_retirementage - 4.0 # - 3.0
        self.women_kappa_hoitovapaa=0.050 # 0.170 # 0.27
        self.women_kappa_under_3y=0.010
        self.women_kappa_ve=0.0
        self.women_kappa_pinkslip_young=0.10
        self.women_kappa_pinkslip_middle=0.40
        self.women_kappa_pinkslip_elderly=0.10
        self.women_kappa_pinkslip_putki=0.20
        self.women_kappa_param = np.array([-0.270, -0.320, -0.345, -0.365, -0.490, -1.400]) # osa-aika 8h, 16h, 24h, kokoaika 32h, 40h, 48h
                                # np.array([-0.270, -0.320, -0.345, -0.350, -0.480, -1.400]) # osa-aika 8h, 16h, 24h, kokoaika 32h, 40h, 48h
                                # delta      0.005   0.060   0.105   0.115    0.220
        self.women_student_kappa_param=np.array([0.05, 0.0, -0.05, -100.0, -100.0, -100.0]) # osa-aika 8h, 16h, 24h, kokoaika 32h, 40h, 48h

        self.kappa_student = 0.0
        self.kappa_svpaivaraha = 0.5

    def map_pt_kappa(self,pt_factor,nu,div):
        return (1-nu)/nu*math.log(1-pt_factor/div)

    def map_pt_kappa_v2(self,pt_factor,g):
        n=round(pt_factor*4.0)-1
        print(n,pt_factor,'v2')
        if g<3:
            return self.men_kappa_param[n]
        else:
            return self.women_kappa_param[n]

    def map_pt_kappa_v3(self,pt_factor,g):
        map_of_reduction=np.array([8/40,16/40,24/40,32/40,40/40,48/40])
        n = (np.abs(map_of_reduction - pt_factor)).argmin()
        if g<3:
            return self.men_kappa_param[n]
        else:
            return self.women_kappa_param[n]

    def map_pt_kappa_v3_student(self,pt_factor,g):
        map_of_reduction=np.array([0/40,8/40,16/40,32/40,40/40,48/40])
        n = (np.abs(map_of_reduction - pt_factor)).argmin()
        if g<3:
            return self.men_student_kappa_param[n]
        else:
            return self.women_student_kappa_param[n]

    def map_pt_kappa_TU(self,pt_factor,g):
        '''
        Trabandt-Uhlig

        self.x_kappa_fii is Frisch's elasticity for gender x
        self.x_kappa_pt is weight of effort for gender x
        '''
        if g<3:
            return self.men_kappa_pt*pt_factor^(1.0+1.0/self.men_kappa_fii)
        else:
            return self.women_kappa_pt*pt_factor^(1.0+1.0/self.women_kappa_fii)

    def log_get_kappa(self,age: float,g: int,employment_state: int,pinkslip: int,
                        pt_factor: float, children_under_3y: int):
        # kappa tells how much person values free-time
        if g<3: # miehet
            #kappa_kokoaika = self.men_kappa_fulltime
            mu_scale_work_before = self.men_mu_scale_kokoaika_before * pt_factor
            mu_scale_work_after = self.men_mu_scale_kokoaika_after * pt_factor

            mu_age = self.men_mu_age
            # lognormaali
            #if self.include_preferencenoise:
            #    kappa_kokoaika += prefnoise

            if employment_state in {1,10,8,9}:
                kappa_tyo = self.map_pt_kappa_v3(pt_factor,g)
            elif employment_state in {12}:
                kappa_tyo = self.map_pt_kappa_v3_student(pt_factor,g)
            else:
                kappa_tyo = 0

            if children_under_3y>0 and employment_state in {1,10}:
                kappa_tyo -= self.men_kappa_under_3y

            kappa_hoitovapaa = self.men_kappa_hoitovapaa
            kappa_ve = self.men_kappa_ve
            kappa_pinkslip_putki = self.men_kappa_pinkslip_putki
            if age>=51:
                kappa_pinkslip = self.men_kappa_pinkslip_elderly
            elif age>28:
                kappa_pinkslip = self.men_kappa_pinkslip_middle
            else:
                kappa_pinkslip = self.men_kappa_pinkslip_young
        else: # naiset
            #kappa_kokoaika = self.wo
            # men_kappa_fulltime
            mu_scale_work_before = self.women_mu_scale_kokoaika_before * pt_factor
            mu_scale_work_after = self.women_mu_scale_kokoaika_after * pt_factor
            mu_age = self.women_mu_age

            # lognormaali
            #if self.include_preferencenoise:
            #    kappa_kokoaika += prefnoise

            if employment_state in {1,10,8,9}:
                kappa_tyo = self.map_pt_kappa_v3(pt_factor,g)
            elif employment_state in {12}:
                kappa_tyo = self.map_pt_kappa_v3_student(pt_factor,g)
            else:
                kappa_tyo = 0

            if children_under_3y>0 and employment_state in {1,10}:
                kappa_tyo -= self.women_kappa_under_3y * pt_factor

            kappa_hoitovapaa = self.women_kappa_hoitovapaa
            kappa_ve = self.women_kappa_ve
            kappa_pinkslip_putki = self.women_kappa_pinkslip_putki
            if age>=51:
                kappa_pinkslip = self.women_kappa_pinkslip_elderly
            elif age>28:
                kappa_pinkslip = self.women_kappa_pinkslip_middle
            else:
                kappa_pinkslip = self.women_kappa_pinkslip_young

        if pinkslip>0: # irtisanottu
            kappa_pinkslip = 0 # irtisanotuille ei vaikutuksia

        #kappa_osaaika = kappa_osaaika*kappa_kokoaika
        if age>mu_age and employment_state in {1,8,9,10}:
            mage_after=max(0,age-self.min_retirementage)
            mage_before=max(0,min(self.min_retirementage,age)-min(self.min_retirementage,mu_age))
            if employment_state in {1,9}:
                #kappa_tyo += -mu_scale_kokoaika*mage
                kappa_tyo += -mu_scale_work_before * mage_before -mu_scale_work_after * mage_after
            elif employment_state in {8,10}:
                #kappa_tyo += -mu_scale_osaaika*mage
                kappa_tyo += -mu_scale_work_before * mage_before -mu_scale_work_after * mage_after

            #print(age,':',mu_age,'after',mage_after,'before',mage_before,'kappa',kappa_tyo,'employment_state',employment_state,'mu_scale_work_before',mu_scale_work_before,'mu_scale_work_after',mu_scale_work_after)

        if employment_state in {1,9}:
            kappa = kappa_tyo
        elif employment_state in {8,10}:
            kappa = kappa_tyo
        elif employment_state in {0}:
            kappa = -kappa_pinkslip
        elif employment_state in {4}:
            kappa = -kappa_pinkslip_putki
        elif employment_state in {13}:
            kappa = -kappa_pinkslip
        elif employment_state == 2:
            kappa = kappa_ve
        elif employment_state == 7:
            kappa = kappa_hoitovapaa
        elif employment_state == 11:
            kappa = 0
        elif employment_state == 12:
            kappa = self.kappa_student + kappa_tyo
        elif employment_state == 14:
            kappa = -self.kappa_svpaivaraha
        else: # states 3, 5, 6, 15
            kappa = 0

        return kappa

    def log_utility(self,income: float,employment_state: int,age: float,g: int=0,pinkslip: int=0,prefnoise: int=0,spouse:int=0,debug: bool=False,pt_factor: float=0,children_under_3y: int=0):
        '''
        Log-utiliteettifunktio muokattuna lähteestä Määttänen, 2013 & Hakola & Määttänen, 2005

        Tulot _income_ ovat vuositasolla, jotta askelpituuden muutos ei vaikuta vapaa-aika-vakioihin
        Tämä versio on parametrisoitu optimoijaa varten

        nettotulot skaalataan palkkojen kasvulla
        tällöin eri vuosien utiliteetit ovat samassa skaalassa
        '''

        if employment_state==15:
            return 0,0

        kappa = self.log_get_kappa(age,g,employment_state,pinkslip,pt_factor,children_under_3y)

        # hyäty/score
        if self.include_preferencenoise:
            # normaali
            u=math.log(prefnoise*income/self.salaryinflationfactor)+kappa
            equ=(income/self.salaryinflationfactor)*math.exp(kappa)
        else:
            u=math.log(income/self.salaryinflationfactor)+kappa
            equ=(income/self.salaryinflationfactor)*math.exp(kappa)

        if u is np.inf and debug:
            print(f'inf: state {employment_state} spouse {spouse} sp_state {spouse_empstate} netto {income}')

        if income<1 and debug:
            print(f'inf: state {employment_state} spouse {spouse} sp_state {spouse_empstate} netto {income}')

        #return u/2.0-4.5,equ # tulot ovat vuositasolla, skaalataan hyöty välille -1,1
        #return (u/2.0-4.5)/50.0,equ # tulot ovat vuositasolla, skaalataan hyöty välille -1,1
        #return (u/2.0-4.5)/25.0,equ # tulot ovat vuositasolla, skaalataan hyöty välille -1,1
        #return (u-3.0)/1500.0,equ # tulot ovat vuositasolla, skaalataan hyöty välille 0,1
        return u/1500.0,equ # tulot ovat vuositasolla, skaalataan hyöty välille 0,1


    def CRRA(self,net: float):
        '''
        Isoelastic utility function
        A common setting is self.CRRA_eta=1.5
        When CRRA_eta==1.0, it reduces to log utility
        '''
        return ((net)**(1.0-self.CRRA_eta)-1.0)/(1-self.CRRA_eta)

    def CRRA_utility(self,income: float,employment_state: int,age,g=0,pinkslip = 0,prefnoise=0,spouse=0,debug=False):
        '''
        CRRA-utiliteettifunktio

        Tulot _income_ ovat vuositasolla, jotta askelpituuden muutos ei vaikuta vapaa-aika-vakioihin
        Tämä versio on parametrisoitu optimoijaa varten
        '''

        if employment_state==15:
            return 0,0

        kappa = self.log_get_kappa(age,g,employment_state,pinkslip)

        # hyäty/score
        if self.include_preferencenoise:
            # normaali
            u = self.CRRA(prefnoise*income/self.salaryinflationfactor)+kappa
            equ=(income/self.salaryinflationfactor)*math.exp(kappa)
        else:
            u = self.CRRA(income/self.salaryinflationfactor)+kappa
            equ=(income/self.salaryinflationfactor)*math.exp(kappa)

        if u is np.inf and debug:
            print(f'inf: state {employment_state} spouse {spouse} sp_state {spouse_empstate} netto {income}')

        if income<1 and debug:
            print(f'inf: state {employment_state} spouse {spouse} sp_state {spouse_empstate} netto {income}')

        return u/20,equ # tulot ovat vuositasolla, mutta skaalataan hyäty

    def set_parameters(self,**kwargs):
        if 'kwargs' in kwargs:
            kwarg=kwargs['kwargs']
        else:
            kwarg=kwargs

        kwarg['include_joustavahoitoraha']=True

        for key, value in kwarg.items():
            if key=='ben':
                if value is not None:
                    if value=='benefitsHO':
                        self.custom_ben = fin_benefits.BenefitsHO
                        self.suojasaanto_toe58 = False
                        self.ansiopvraha_toe = 1.0
                    elif value=='BenefitsPuoliväliriihi2025':
                        self.custom_ben = fin_benefits.BenefitsPuoliväliriihi2025
                        self.suojasaanto_toe58 = False
                        self.ansiopvraha_toe = 1.0
                    elif value=='benefits':
                        self.custom_ben=fin_benefits.Benefits
                    elif value=='basicIncomeBenefits':
                        self.custom_ben = fin_benefits.BasicIncomeBenefits
                    elif value=='benefitsYleistuki':
                        self.custom_ben = fin_benefits.BenefitsYleistuki
                        self.suojasaanto_toe58 = False
                        self.ansiopvraha_toe = 1.0
                    else:
                        print('ERROR: unknown ben')
            if key=='step':
                if value is not None:
                    self.timestep=value
            elif key=='mortplot':
                if value is not None:
                    self.mortplot=value
            elif key=='silent':
                if value is not None:
                    self.silent=value
            elif key=='include_emtr':
                if value is not None:
                    self.include_emtr=value
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

        if self.custom_ben is not None:
            self.ben = self.custom_ben(**kwargs)
        elif self.perustulo:
            self.ben = fin_benefits.BasicIncomeBenefits(**kwargs)
        elif self.universalcredit:
            self.ben = fin_benefits.BenefitsUC(**kwargs)
        else:
            self.ben = fin_benefits.Benefits(**kwargs)

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

    def map_age(self,age: float,start_zero: bool=False):
        if start_zero:
            return round((age)*self.inv_timestep)
        else:
            return round((age-self.min_age)*self.inv_timestep)


    ##############
    ### RESET ####
    ##############

    def reset(self,init=None):
        '''
        Open AI-interfacen mukainen reset-funktio, joka nollaa laskennan alkutilaan
        '''
        # We need the following line to seed self.np_random
        #super().reset()

        self.init_state()
        self.steps_beyond_done = None

        if self.mortplot:
            self.plotdebug=False

        if self.plotdebug:
            self.render()

        if True:
            statenames = self.states.get_state_name()
            if not np.logical_and(self.state >= self.low, self.state <= self.high).all():
                print('reset FAILED')
                for k in range(self.state.shape[0]):
                    if np.logical_or(self.state[k] < self.low[k], self.state[k] > self.high[k]):
                        aa='*'
                    else:
                        aa=''
                    name=statenames[k]
                    print(f'{aa} {k} {name}:',self.state[k],'dl =',self.state[k]-self.low[k],'dh =',self.high[k] - self.state[k],'[',self.low[k],';',self.high[k],']')

        return np.array(self.state,dtype=np.float32)


    #############
    ### INITIAL
    #############

    def get_initial_state(self,puoliso: int,is_spouse:bool =False,init_g: int=-1):
        '''
        Alusta tila

        tässä satunnaisuus peräisin random -kirjastosta, kaikkialla muualla np.random -kirjastosta
        random-alustus koskee siten vain alkutilaa, np.random-alustus muuta
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
        main_pt_action=1
        main_paid_wage=0
        basis_wage=0

        if is_spouse:
            gender=1
            q = self.group_weights[1,:]
            group = random.choices(np.array([3,4,5],dtype=int),weights=q)[0]

            #group=random.choices(np.array([3,4,5],dtype=int),weights=self.group_weights[1,:])[0]
        else:
            gender=0 # random.choices(np.array([0,1],dtype=int),weights=[0.5,0.5])[0]
            g=random.choices(np.array([0,1,2],dtype=int),weights=self.group_weights[0,:])[0]
            group=round(g+gender*3)

        employment_state=random.choices(np.array([13,0,1,10,3,11,12],dtype=int),
                weights = self.initial_weights[group,:])[0]

        self.init_infostate(age=age,spouse=is_spouse)

        initial_salary=None
        reset_exp=False

        # set up salary for the entire career
        if is_spouse:
            self.wages_spouse.compute_salary(group=group,initial_salary=initial_salary)
        else:
            self.wages_main.compute_salary(group=group,initial_salary=initial_salary)

        maxred=-0.02

        if employment_state==0:
            wage_reduction=random.uniform(maxred,0.30)
        elif employment_state==13:
            wage_reduction=random.uniform(0.10,0.40) # 20-70
        elif employment_state==1:
            wage_reduction=random.uniform(maxred,0.15) # 20-70
        elif employment_state in {5,6,7}:
            wage_reduction=random.uniform(maxred,0.30) # 20-70
            pink=1
        elif employment_state==10:
            wage_reduction=random.uniform(maxred,0.15) # 20-70
        elif employment_state==12:
            wage_reduction=random.uniform(maxred,0.35)
            pink=1
        elif employment_state==11:
            wage_reduction=random.uniform(0.10,0.60) # 15-50
            pink=1
        elif employment_state==3:
            wage_reduction=0.60
            pension=0
        elif employment_state==2:
            pension=0
            wage_reduction=0.60

        if employment_state==10:
            main_pt_action=1 # alussa suurin osa opiskelijoita
        elif employment_state==1:
            main_pt_action=1
        else:
            main_pt_action=0

        if is_spouse:
            old_wage = self.get_spousewage(self.min_age,wage_reduction)
        else:
            old_wage = self.get_wage(self.min_age,wage_reduction)

        next_wage=old_wage

        main_paid_wage,main_pt_factor,_ = self.get_paid_wage(old_wage,employment_state,main_pt_action,old_wage,0)

        if not reset_exp:
            if employment_state==0:
                tyohist=1.0
                toe=0.0
                toekesto=1.0
                used_unemp_benefit=0.0
                unempwage_basis=old_wage
                alkanut_ansiosidonnainen=1
                unempwage=0
            elif employment_state==1:
                tyohist=1.0
                toe=1.0
                used_unemp_benefit=0.0
            elif employment_state==10:
                tyohist=1.0
                toe=1.0
                used_unemp_benefit=0.0
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
                if self.plotdebug:
                    print('tyoelake',tyoelake_maksussa,'kansanelake',kansanelake)
                pension=0
            elif employment_state==2:
                wage5y=next_wage
                tyoelake_maksussa=pension
                # takuueläke voidaan huomioida jo tässä
                kansanelake = self.ben.laske_kansanelake(age,tyoelake_maksussa/12,1-puoliso)*12 # ben-modulissa palkat kk-tasolla
                if self.plotdebug:
                    print('tyoelake',tyoelake_maksussa,'kansanelake',kansanelake)
                pension=0
        else:
            if employment_state==0:
                tyohist=random.uniform(0.0,age-18)
                toe=random.uniform(0.0,28/12)
                toekesto=toe
                used_unemp_benefit=random.uniform(0.0,2.0)
                unempwage_basis=old_wage
                alkanut_ansiosidonnainen=1
                unempwage=random.uniform(0.0,90_000.0)
            elif employment_state==13:
                tyohist=random.uniform(0.0,age-18)
                toe=0.0
                toekesto=toe
                used_unemp_benefit=2.0
            elif employment_state==10:
                tyohist=random.uniform(0.0,age-18)
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
                tyoelake_maksussa=random.uniform(0.0,30_000)
                # takuueläke voidaan huomioida jo tässä
                kansanelake = self.ben.laske_kansanelake(age,tyoelake_maksussa/12,1-puoliso,disability=True)*12 # ben-modulissa palkat kk-tasolla
                if self.plotdebug:
                    print('tyoelake',tyoelake_maksussa,'kansanelake',kansanelake)
                pension=0
            elif employment_state==2:
                wage5y=next_wage
                tyoelake_maksussa=random.uniform(0.0,40_000)
                # takuueläke voidaan huomioida jo tässä
                kansanelake = self.ben.laske_kansanelake(age,tyoelake_maksussa/12,1-puoliso)*12 # ben-modulissa palkat kk-tasolla
                if self.plotdebug:
                    print('tyoelake',tyoelake_maksussa,'kansanelake',kansanelake)
                pension=0

        if employment_state in {1,10}:
            unempwage=old_wage

        unemp_benefit_left = self.comp_unempdays_left(used_unemp_benefit,tyohist,age,toe,employment_state,alkanut_ansiosidonnainen,toe58,toe)

        if employment_state in set([0,4]):
            self.set_kassanjasenyys(1)
            kassanjasenyys=1
        else:
            kassanjasenyys = self.get_kassanjasenyys()

        lleft = self.comp_life_left(group,age)
        until_disab = self.comp_until_disab(group,age,employment_state)
        until_student = self.comp_time_to_study(employment_state,age,group)
        until_outsider = self.comp_time_to_outsider(employment_state,age,group)

        return employment_state,group,pension,old_wage,age,time_in_state,paid_pension,pink,toe,toekesto,tyohist,next_wage,\
            used_unemp_benefit,wage_reduction,unemp_after_ra,unempwage,unempwage_basis,\
            children_under3,children_under7,children_under18,unemp_benefit_left,alkanut_ansiosidonnainen,toe58,\
            ove_paid,kassanjasenyys,kansanelake,tyoelake_maksussa,main_pt_action,main_paid_wage,basis_wage,\
            lleft,until_disab,until_student,until_outsider


    def init_state(self):

        if self.randomness:
            rn = random.uniform(0,1)
        else:
            rn = 0

        if self.rates.get_initial_marriage_ratio()>rn:
            puoliso=1
        else:
            puoliso=0

        employment_state,group,pension,old_wage,age,time_in_state,paid_pension,pink,toe,toekesto,tyohist,next_wage,\
            used_unemp_benefit,wage_reduction,unemp_after_ra,unempwage,unempwage_basis,\
            children_under3,children_under7,children_under18,unemp_benefit_left,alkanut_ansiosidonnainen,toe58,\
            ove_paid,kassanjasenyys,kansanelake,tyoelake_maksussa,\
            main_pt_action,main_paid_wage,main_basis_wage,main_life_left,main_until_disab,\
            main_until_student,main_until_outsider\
             = self.get_initial_state(puoliso)

        self.main_dis_wage5y=0
        self.spouse_dis_wage5y=0

        #spouse_g = self.states.get_spouse_g(group)
        spouse_empstate,spouse_g,spouse_pension,spouse_old_paid_wage,_,spouse_time_in_state,puoliso_paid_pension,puoliso_pink,puoliso_toe,\
            puoliso_toekesto,puoliso_tyohist,puoliso_next_wage,\
            puoliso_used_unemp_benefit,spouse_wage_reduction,puoliso_unemp_after_ra,puoliso_unempwage,puoliso_unempwage_basis,\
            _,_,_,puoliso_unemp_benefit_left,puoliso_alkanut_ansiosidonnainen,puoliso_toe58,\
            puoliso_ove_paid,puoliso_kassanjasenyys,spouse_kansanelake,spouse_tyoelake_maksussa,\
            spouse_pt_action,spouse_paid_wage,spouse_basis_wage,spouse_life_left,spouse_until_disab,\
            spouse_until_student,spouse_until_outsider\
             = self.get_initial_state(puoliso,is_spouse=True,init_g=group)

        until_child = self.comp_until_birth(spouse_g,age,initial=True)
        time_to_marriage,time_to_divorce = self.comp_time_to_marriage(puoliso,age,group,spouse_g)

        # tarvitseeko alkutilassa laskea muita tietoja uusiksi? ei kait

        if self.plotdebug:
            print(f'emp {employment_state} g {group} old_wage {old_wage} next_wage {next_wage} age {age} kassanjäsen {kassanjasenyys}')
            print(f'emp {spouse_empstate} g {spouse_g} old_wage {spouse_old_paid_wage} next_wage {puoliso_next_wage} age {age} kassanjäsen {kassanjasenyys}')
            print(f'main lf {main_life_left} years; ud {main_until_disab} y')
            print(f'spouse lf {spouse_life_left} years; ud {spouse_until_disab} y')
            print(f'until child {until_child} until marriage {time_to_marriage} until divorce {time_to_divorce}')

        if self.include_preferencenoise:
            # lognormaali
            #prefnoise=np.random.normal(loc=-0.5*self.preferencenoise_std*self.preferencenoise_std,scale = self.preferencenoise_std,size=1)[0]
            # normaali
            prefnoise=min(2.0,max(1e-6,np.random.normal(loc=1.0,scale = self.preferencenoise_std,size=1)[0]))
        else:
            prefnoise=0

        self.state = self.states.state_encode(employment_state,group,spouse_g,pension,old_wage,age,
                                       time_in_state,tyoelake_maksussa,pink,toe,toekesto,tyohist,next_wage,
                                       used_unemp_benefit,wage_reduction,unemp_after_ra,
                                       unempwage,unempwage_basis,
                                       children_under3,children_under7,children_under18,
                                       unemp_benefit_left,alkanut_ansiosidonnainen,toe58,
                                       ove_paid,kassanjasenyys,
                                       puoliso,spouse_empstate,spouse_old_paid_wage,spouse_pension,
                                       spouse_wage_reduction,spouse_tyoelake_maksussa, puoliso_next_wage,
                                       puoliso_used_unemp_benefit,puoliso_unemp_benefit_left,
                                       puoliso_unemp_after_ra,
                                       puoliso_unempwage,puoliso_unempwage_basis,
                                       puoliso_alkanut_ansiosidonnainen,puoliso_toe58,puoliso_toe,
                                       puoliso_toekesto,puoliso_tyohist,spouse_time_in_state,
                                       puoliso_pink,puoliso_ove_paid,
                                       kansanelake,spouse_kansanelake,
                                       main_paid_wage,spouse_paid_wage,
                                       main_pt_action,spouse_pt_action,
                                       main_basis_wage,spouse_basis_wage,
                                       main_life_left,spouse_life_left,
                                       main_until_disab,spouse_until_disab,
                                       time_to_marriage,time_to_divorce,until_child,
                                       main_until_student,spouse_until_student,main_until_outsider,spouse_until_outsider,
                                       prefnoise)

    #############
    ### RENDER
    ##############

    def render(self,mode: str='human',close:bool =False,done:bool =False,reward: float=None,netto: float=None,render_omat: bool=True,
                render_puoliso: bool=False,benq: dict=None,netto_omat: float=None,netto_puoliso: float=None,p=None):
        '''
        Tulostus-rutiini
        '''
        if p is None:
            emp,g,spouse_g,pension,old_paid_wage,age,time_in_state,paid_pension,pink,toe,toekesto,tyohist,used_unemp_benefit,\
                wage_red,unemp_after_ra,unempwage,unempwage_basis,prefnoise,c3,c7,c18,\
                unemp_left,oikeus,toe58,ove_paid,jasen,\
                puoliso,spouse_empstate,spouse_old_paid_wage,spouse_pension,\
                spouse_wage_reduction,puoliso_paid_pension,spouse_next_wage,\
                puoliso_used_unemp_benefit,puoliso_unemp_benefit_left,\
                puoliso_unemp_after_ra,puoliso_unempwage,puoliso_unempwage_basis,\
                puoliso_alkanut_ansiosidonnainen,puoliso_toe58,puoliso_toe,\
                puoliso_toekesto,puoliso_tyohist,spouse_time_in_state,puoliso_pink,puoliso_ove_paid,\
                kansanelake,spouse_kansanelake,tyoelake_maksussa,spouse_tyoelake_maksussa,\
                next_wage,main_paid_wage,spouse_paid_wage,pt_act,s_pt_act,main_wage_basis,spouse_wage_basis,\
                main_life_left,spouse_life_left,main_until_disab,spouse_until_disab,\
                time_to_marriage,time_to_divorce,until_child,main_until_student,spouse_until_student,main_until_outsider,spouse_until_outsider\
                    = self.states.state_decode(self.state)
        else:
            emp,g,spouse_g,pension,old_paid_wage,age,time_in_state,paid_pension,pink,toe,toekesto,tyohist,used_unemp_benefit,\
                wage_red,unemp_after_ra,unempwage,unempwage_basis,prefnoise,c3,c7,c18,\
                unemp_left,oikeus,toe58,ove_paid,jasen,\
                puoliso,spouse_empstate,spouse_old_paid_wage,spouse_pension,\
                spouse_wage_reduction,puoliso_paid_pension,spouse_next_wage,\
                puoliso_used_unemp_benefit,puoliso_unemp_benefit_left,\
                puoliso_unemp_after_ra,puoliso_unempwage,puoliso_unempwage_basis,\
                puoliso_alkanut_ansiosidonnainen,puoliso_toe58,puoliso_toe,\
                puoliso_toekesto,puoliso_tyohist,spouse_time_in_state,puoliso_pink,puoliso_ove_paid,\
                kansanelake,spouse_kansanelake,tyoelake_maksussa,spouse_tyoelake_maksussa,\
                next_wage,main_paid_wage,spouse_paid_wage,pt_act,s_pt_act,main_wage_basis,spouse_wage_basis,\
                main_life_left,spouse_life_left,main_until_disab,spouse_until_disab,\
                time_to_marriage,time_to_divorce,until_child,main_until_student,spouse_until_student,main_until_outsider,spouse_until_outsider\
                    = self.states.state_decode(p)
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

        if emp in set([0,1,10,4,5,6,7,13,14]):
            var=f' wb {main_wage_basis:.0f}'
        elif emp in set([2,3,8,9]):
            var=f' pd_k {kansanelake:.0f}'
        else:
            var=''

        main_paid_wage,pt_factor,pot_main_wage = self.get_paid_wage(next_wage,emp,pt_act,old_paid_wage,time_in_state) # CHECK!
        kappa = self.log_get_kappa(age,g,emp,pink,pt_factor,c3)

        if False:
            uo=f' ud {main_until_disab:.1f} uo {main_until_outsider:.1f} us {main_until_student:.1f}'
            ue = ''
        else:
            uo=''
            ue= f' toe {toe:.2f}({toe58:.0f}){kassassa} tk {toekesto:.2f} ura {tyohist:.2f} uew {unempwage:.0f}'+\
                f' (b {unempwage_basis:.0f}) ult {unemp_left:.2f} aa {oikeus:.0f} ove {ove_paid:.0f}'

        out=f'{m}s{onpuoliso} {emp:2d} g {g:d} a {age:.2f} w {old_paid_wage:.0f} ({main_paid_wage:.0f} #{pt_act:d}) nw {next_wage:.0f}'+\
            f' red {wage_red:.2f} tis {time_in_state:.2f}'+\
            f' pen {pension:.0f} pd_e {tyoelake_maksussa:.0f}{var} ueb {used_unemp_benefit:.2f}'+ue+\
            f' pk{pink:d} lf{main_life_left:.1f}{uo} c{c3:.0f}/{c7:.0f}/{c18:.0f} k {kappa:.2f}'

        if reward is not None:
            out+=f' r {reward:.4f}'
        if netto_omat is not None:
            out+=f' n {netto_omat:.0f}'

        if render_omat:
            print(out)

        if spouse_empstate==15:
            m='*'
        else:
            m=''

        spouse_paid_wage,s_pt_factor,pot_spouse_wage = self.get_paid_wage(spouse_next_wage,spouse_empstate,s_pt_act,spouse_old_paid_wage,spouse_time_in_state) # CHECK!
        kappa = self.log_get_kappa(age,spouse_g,spouse_empstate,puoliso_pink,s_pt_factor,c3)

        if spouse_empstate in set([0,1,4,10,5,6,7,13,14]):
            var=f' wb {spouse_wage_basis:.0f}'
        elif spouse_empstate in set([2,3,8,9]):
            var=f' pd_k {spouse_kansanelake:.0f}'
        else:
            var=''

        if False:
            uo=f' ud {spouse_until_disab:.1f} uo {spouse_until_outsider:.1f} us {spouse_until_student:.1f}'
            ue=''
        else:
            uo=''
            ue =f' toe {puoliso_toe:.2f}({puoliso_toe58:d}){kassassa} tk {puoliso_toekesto:.2f} ura {puoliso_tyohist:.2f} uew {puoliso_unempwage:.0f} (b {puoliso_unempwage_basis:.0f})'+\
                f' ult {puoliso_unemp_benefit_left:.2f} aa {puoliso_alkanut_ansiosidonnainen:d}'

        puoliso=f'{m}ps{onpuoliso} {spouse_empstate:d} g {spouse_g:d} a {age:.2f} w {spouse_old_paid_wage:.0f} ({spouse_paid_wage:.0f} #{s_pt_act:d}) nw {spouse_next_wage:.0f} red {spouse_wage_reduction:.2f} tis {spouse_time_in_state:.2f}'+\
                f' pen {spouse_pension:.0f} pd_e {spouse_tyoelake_maksussa:.0f}{var} ueb {puoliso_used_unemp_benefit:.2f}'+ue+\
                f' pk {puoliso_pink:d} lf{spouse_life_left:.1f}{uo} k {kappa:.2f}'

        if reward is not None:
            puoliso+=f' r {reward:.4f}'
        if netto_puoliso is not None:
            puoliso+=f' n {netto_puoliso:.0f}'

        if render_puoliso:
            print(puoliso)

        if done:
            print('-------------------------------------------------------------------------------------------------------')

    def __str___(self):
        '''
        Tulostus-rutiini
        '''
        emp,g,pension,wage,age,time_in_state,paid_pension,pink,toe,toekesto,tyohist,used_unemp_benefit,\
            wage_red,unemp_after_ra,unempwage,unempwage_basis,prefnoise,c3,c7,c18,\
            unemp_left,oikeus,toe58,ove_paid,jasen,\
            puoliso,spouse_empstat,spouse_g,puoliso_old_wage,spouse_pension,\
            spouse_wage_reduction,puoliso_paid_pension,puoliso_next_wage,\
            puoliso_used_unemp_benefit,puoliso_unemp_benefit_left,\
            puoliso_unemp_after_ra,puoliso_unempwage,puoliso_unempwage_basis,\
            puoliso_alkanut_ansiosidonnainen,puoliso_toe58,puoliso_toe,\
            puoliso_toekesto,puoliso_tyohist,spouse_time_in_state,puoliso_pink,puoliso_ove_paid,\
            kansanelake,spouse_kansanelake,tyoelake_maksussa,spouse_tyoelake_maksussa,\
            next_wage,main_paid_wage,spouse_paid_wage,pt_act,s_pt_act,main_wage_basis,spouse_wage_basis,\
            main_until_disab,spouse_until_disab,time_to_marriage,time_to_divorce,until_child,main_until_student,spouse_until_student,main_until_outsider,spouse_until_outsider\
                 = self.states.state_decode(self.state)

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

        if emp in set([0,1,10,4,5,6,7,13,14]):
            var=f'wbas {main_wage_basis:.0f}'
        elif emp in set([2,3,8,9]):
            var=f'paid_k {kansanelake:.0f}'
        else:
            var=''

        main_paid_wage,pt_factor,_ = self.get_paid_wage(wage,emp,pt_act,wage,time_in_state) # CHECK!
        kappa = self.log_get_kappa(age,g,emp,pink,pt_factor)

        out=f'{m}s{onpuoliso} {emp:2d} g {g:d} a {age:.2f} w {wage:.0f} (wp {main_paid_wage:.0f} pt {pt_act:d}) nw {next_wage:.0f}'+\
            f' red {wage_red:.2f} tis {time_in_state:.2f}'+\
            f' pen {pension:.0f} paid_e {tyoelake_maksussa:.0f} {var} ueb {used_unemp_benefit:.2f}'+\
            f' toe {toe:.2f}{kassassa} tk{toekesto:.2f} ura {tyohist:.2f} uew {unempwage:.0f}'+\
            f' (b {unempwage_basis:.0f}) uleft {unemp_left:.2f} aa {oikeus:.0f} 58 {toe58:.0f} ove {ove_paid:.0f}'+\
            f' pink {pink:d} c{c3:.0f}/{c7:.0f}/{c18:.0f} k {kappa:.2f}'

        if spouse_empstat==15:
            m='*'
        else:
            m=''

        spouse_paid_wage,s_pt_factor,_ = self.get_paid_wage(puoliso_old_wage,spouse_empstat,s_pt_act) # CHECK!
        kappa = self.log_get_kappa(age,spouse_g,spouse_empstat,puoliso_pink,s_pt_factor)

        if spouse_empstat in set([0,1,4,10,5,6,7,13,14]):
            var=f'wbas {spouse_wage_basis:.0f}'
        elif emp in set([2,3,8,9]):
            var=f'paid_k {spouse_kansanelake:.0f}'
        else:
            var=''

        puoliso=f'{m}ps{onpuoliso} {spouse_empstat:d} g {spouse_g:d} a {age:.2f} w {puoliso_old_wage:.0f} (wp {spouse_paid_wage:.0f} pt {s_pt_act:d}) nw {puoliso_next_wage:.0f} red {spouse_wage_reduction:.2f} tis {spouse_time_in_state:.2f}'+\
                f' pen {spouse_pension:.0f} paid_e {spouse_tyoelake_maksussa:.0f} {var} ueb {puoliso_used_unemp_benefit:.2f}'\
                f' toe {puoliso_toe:.2f}{kassassa} tk {puoliso_toekesto:.2f} ura {puoliso_tyohist:.2f} uew {puoliso_unempwage:.0f} (b {puoliso_unempwage_basis:.0f})'+\
                f' uleft {puoliso_unemp_benefit_left:.2f} 58 {puoliso_toe58:d} aa {puoliso_alkanut_ansiosidonnainen:d}'+\
                f' pink {puoliso_pink:d} k {kappa:.2f}'

        return 'omat: '+out+'\npuoliso: '+puoliso

    def close(self):
        '''
        Ei käytässä
        '''

        self.infostate = None

        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def explain(self):
        '''
        Tulosta laskennan parametrit
        '''
        print('Parameters of life cycle:\ntimestep {}\ngamma {} ({} per anno)\nmin_age {}\nmax_age {}\nmin_retirementage {}\n'.format(self.timestep,self.gamma,self.gamma**(1.0/self.timestep),self.min_age,self.max_age,self.min_retirementage)+\
            f'max_retirementage {self.max_retirementage}\nansiopvraha_kesto300 {self.ansiopvraha_kesto300}\nansiopvraha_kesto400 {self.ansiopvraha_kesto400}\nansiopvraha_kesto500 {self.ansiopvraha_kesto500}\nansiopvraha_toe {self.ansiopvraha_toe}\n'+\
            f'karenssi_kesto {self.karenssi_kesto}\nmortality {self.include_mort}\nrandomness {self.randomness}\n'+\
            f'include_putki {self.include_putki}\ninclude_pinkslip {self.include_pinkslip}\n'+\
            f'perustulo {self.perustulo}\nsigma_reduction {self.use_sigma_reduction}\nplotdebug {self.plotdebug}\n'+\
            f'additional_tyel_premium {self.additional_tyel_premium}\nscale_tyel_accrual {self.scale_tyel_accrual}\ninclude_ove {self.include_ove}\n'+\
            f'unemp_limit_reemp {self.unemp_limit_reemp}\nmin_salary {self.min_salary}\n')

    def unempright_left(self,emp: int,tis: float,bu: float,ika: float,tyohistoria: float):
        '''
        Tilastointia varten lasketaan jäljellä olevat ansiosidonnaiset työttämyysturvapäivät
        '''
        if ika >= self.minage_500 and tyohistoria >= self.tyohistoria_vaatimus500:
            kesto = self.apvkesto500 #ansiopvraha_kesto500
        elif tyohistoria >= self.tyohistoria_vaatimus:
            kesto = self.apvkesto400 #ansiopvraha_kesto400
        else:
            kesto = self.apvkesto300 #ansiopvraha_kesto300

        #kesto=kesto/(12*21.5)
        #if irtisanottu<1 and time_in_state<self.karenssi_kesto: # karenssi, jos ei irtisanottu

        if emp==13:
            return tis
        else:
            return kesto-bu

    ###############################################
    ###
    ###    INFOSTATE
    ###
    ###############################################

    def init_inforate(self):
        self.kassanjasenyys_joinrate,self.kassanjasenyys_rate = self.rates.get_kassanjasenyys_rate()

    def init_infostate(self,lapsia: int=0,lasten_iat=np.zeros(15),lapsia_paivakodissa: int=0,age: int=18,spouse:bool =False):
        '''
        Alustaa infostate-dictorionaryn
        Siihen talletetaan tieto aiemmista tiloista, joiden avulla lasketaan statistiikkoja
        '''
        self.infostate={}
        states,latest,enimaika,palkka,voc_unempbasis,member,voc_basiswage = self.infostate_vocabulary(is_spouse=False)

        self.infostate[states] = np.zeros(self.n_time)-1
        self.infostate[palkka] = np.zeros(self.n_time)-1
        self.infostate[voc_unempbasis] = np.zeros(self.n_time)-1
        self.infostate[member] = np.zeros(self.n_time,dtype=np.int8)
        self.infostate[latest] = 0
        self.infostate['children_n'] = 0
        self.infostate['children_date'] = np.zeros(15)
        self.infostate[enimaika] = 0
        self.infostate[voc_basiswage] = np.zeros(self.n_time)-1

        states,latest,enimaika,palkka,voc_unempbasis,member,voc_basiswage = self.infostate_vocabulary(is_spouse=True)
        self.infostate[states] = np.zeros(self.n_time)-1
        self.infostate[palkka] = np.zeros(self.n_time)-1
        self.infostate[voc_unempbasis] = np.zeros(self.n_time)-1
        self.infostate[member] = np.zeros(self.n_time,dtype=np.int8)
        self.infostate[latest] = 0
        self.infostate[enimaika] = 0
        self.infostate[voc_basiswage] = np.zeros(self.n_time)-1
        sattuma = random.uniform(0,1)
        t=round((age-self.min_age)/self.timestep)

        if sattuma<self.kassanjasenyys_rate[t]:
            self.set_kassanjasenyys(1) #self.infostate['kassanjasen'] = 1
        else:
            self.set_kassanjasenyys(0) # self.infostate['kassanjasen'] = 0

    def infostate_add_child(self,age: float):
        if self.infostate['children_n']<14:
            self.infostate['children_date'][self.infostate['children_n']] = age
            self.infostate['children_n'] = self.infostate['children_n']+1

    def infostate_set_enimmaisaika(self,age: float,is_spouse:bool =False):
        states,latest,enimaika,palkka,voc_unempbasis,member,voc_wagebasis = self.infostate_vocabulary(is_spouse=is_spouse)
        t = round((age-self.min_age)/self.timestep)
        self.infostate[enimaika] = t

    def update_infostate(self,t: int,state: int,paid_wage: float,basiswage: float,unempbasis: float,is_spouse:bool =False):
        states,latest,enimaika,palkka,voc_unempbasis,member,voc_wagebasis = self.infostate_vocabulary(is_spouse=is_spouse)

        self.infostate[states][t] = state
        self.infostate[latest] = int(t)
        self.infostate[voc_unempbasis][t] = unempbasis
        self.infostate[member][t] = self.infostate['kassanjasen']
        if state==1:
            self.infostate[palkka][t] = paid_wage
        elif state==10:
            self.infostate[palkka][t] = paid_wage
        elif state in set([5,6,14]):
            self.infostate[palkka][t] = basiswage
            self.infostate[voc_wagebasis][t] = basiswage
        else:
            self.infostate[palkka][t] = 0

    def render_infostate(self):
        print('states {}'.format(self.infostate['states']))

    def get_kassanjasenyys(self):
        return self.infostate['kassanjasen']

    def set_kassanjasenyys(self,value: int):
        self.infostate['kassanjasen'] = value

    def infostate_kassanjasenyys_update(self,age: float):
        if self.infostate['kassanjasen']<1:
            sattuma = random.uniform(0,1)
            if sattuma<self.kassanjasenyys_joinrate[self.map_age(age)] and self.randomness:
                self.set_kassanjasenyys(1)

    def comp_toe_wage_nykytila(self,is_spouse:bool =False):
        states,latest,enimaika,palkka,voc_unempbasis,member,voc_wagebasis = self.infostate_vocabulary(is_spouse=is_spouse)
        lstate=round(self.infostate[states][self.infostate[latest]])
        toes=0
        wage=0
        n_toe=int(np.floor(self.max_toe/self.timestep))
        emp_states={1,10}
        unemp_states={0,4}
        #family_states={5,6,7,14}
        accepted_states={5,6,7,12,14}
        ret_states={2,3,8,9}

        if self.infostate['kassanjasen']>0:
            if lstate not in ret_states:
                if lstate in accepted_states: #family_states:
                    # laskee, onko ollut riittävä toe ansiosidonnaiseen, ei onko päiviä jäljellä
                    t2 = self.infostate[latest]
                    nt=0
                    while nt<n_toe and t2>=0:
                        emps = self.infostate[states][t2]
                        if self.infostate[member][t2]<1:
                            nt=nt+1
                        elif emps in accepted_states:
                            pass
                        elif emps in emp_states:
                            w = self.infostate[palkka][t2]
                            if w>self.min_toewage:
                                toes += self.timestep
                                wage += w*self.timestep
                            #elif self.include_halftoe and w>self.min_halftoewage: # and emps==10:
                            #    toes+=0.5*self.timestep
                            #    wage+=w*self.timestep
                            nt=nt+1
                        elif emps in unemp_states:
                            nt=nt+1
                        else:
                            nt=nt+1
                        t2=t2-1
                else:
                    # laskee, onko toe täyttynyt viimeisimmän ansiosidonnaisen alkamisen jälkeen
                    t2 = self.infostate[latest]
                    nt=0
                    t0 = self.infostate[enimaika]
                    while nt<n_toe and t2>=t0:
                        emps = self.infostate[states][t2]
                        if self.infostate[member][t2]<1:
                            nt=nt+1
                        elif emps in accepted_states:
                            pass
                        elif emps in emp_states:
                            w = self.infostate[palkka][t2]
                            if w>self.min_toewage:
                                toes += self.timestep
                                wage+=w*self.timestep
                            nt=nt+1
                        elif emps in unemp_states:
                            nt=nt+1
                        else:
                            nt=nt+1
                        t2=t2-1
                if toes >= self.ansiopvraha_toe and toes>0:
                    wage=wage/toes
                else:
                    wage=0
        else:
            wage=0
            toes=0

        toekesto=toes

        return toes,toekesto,wage

    def comp_toe_wage_porrastus(self,is_spouse:bool =False):
        states,latest,enimaika,palkka,voc_unempbasis,member,voc_wagebasis = self.infostate_vocabulary(is_spouse=is_spouse)
        lstate=round(self.infostate[states][self.infostate[latest]])
        toes=0
        toekesto=0
        wage=0
        n_toe=int(np.floor(self.max_toe/self.timestep))
        emp_states={1,10}
        unemp_states={0,4}
        #family_states={5,6,7,14}
        accepted_states={5,6,7,12,14}
        ret_states={2,3,8,9}

        if self.infostate['kassanjasen']>0 and lstate not in ret_states:
            # laskee, onko toe täyttynyt viimeisimmän ansiosidonnaisen alkamisen jälkeen
            t2 = self.infostate[latest]
            nt=0
            t0 = self.infostate[enimaika]
            while nt<n_toe and t2>=t0:
                emps = self.infostate[states][t2]
                if self.infostate[member][t2]<1:
                    nt=nt+1
                elif emps in accepted_states:
                    pass
                elif emps in emp_states:
                    w = self.infostate[palkka][t2]
                    if w>self.min_toewage:
                        toes += self.timestep
                    elif self.include_halftoe and w>self.min_halftoewage: # and emps==10:
                        toes+=0.5*self.timestep
                    nt=nt+1
                #elif emps in unemp_states:
                #    nt=nt+1
                else:
                    nt=nt+1
                t2=t2-1

            # laskee, onko ollut riittävä toe ansiosidonnaiseen, ei onko päiviä jäljellä
            t2 = self.infostate[latest]
            nt=0
            while nt<n_toe and t2>=0:
                emps = self.infostate[states][t2]
                if self.infostate[member][t2]<1:
                    nt=nt+1
                elif emps in accepted_states:
                    pass
                elif emps in emp_states:
                    w = self.infostate[palkka][t2]
                    if w>self.min_toewage:
                        toekesto += self.timestep
                        wage+=w*self.timestep
                    elif self.include_halftoe and w>self.min_halftoewage: # and emps==10:
                        toekesto+=0.5*self.timestep
                        wage+=w*self.timestep
                    nt=nt+1
                elif emps in unemp_states:
                    nt=nt+1
                else:
                    nt=nt+1
                t2=t2-1

            if toekesto >= self.ansiopvraha_toe and toekesto>0:
                wage=wage/toekesto
            else:
                wage=0

            if lstate in accepted_states: #family_states:
                toes=toekesto
        else:
            wage=0
            toes=0
            toekesto=0

        return toes,toekesto,wage

    def comp_infostats(self,age: float,is_spouse:bool =False):
        # laske työssäoloehto tarkasti
        # laske työttömyysturvaan vaikuttavat lasten määrät

        if not is_spouse:
            self.infostate_kassanjasenyys_update(age)

        toes,toekesto,wage = self.comp_toe_wage(is_spouse=is_spouse)

        children_under18=0
        children_under7=0
        children_under3=0
        # tässä <=, koska halutaan että koko ikä 0-3, 0-7 tai 0-18 huomioidaan
        for k in range(self.infostate['children_n']):
            c_age = age-self.infostate['children_date'][k]
            if c_age <= 18:
                children_under18 += 1
                if c_age <= 7:
                    children_under7 += 1
                    if c_age <= 3:
                        children_under3 += 1

        return toes,toekesto,wage,children_under3,children_under7,children_under18

    def infostate_print_ages(self,age):
        toes,toekesto,wage,c3,c7,c18 = self.comp_infostats(age)
        if self.infostate['children_n']>0:
            first = True
            for k in range(self.infostate['children_n']):
                c_age = age-self.infostate['children_date'][k]
                if c_age<=20:
                    if first:
                        print(f'age {age:.2f} child {k}/{self.infostate["children_n"]}: age {c_age:.2f} c3: {c3} c7: {c7} c18: {c18}')
                        first = False
                    else:
                        print(f'age {age:.2f} child {k}/{self.infostate["children_n"]}: age {c_age:.2f}')

    def infostate_comp_5y_ave_wage(self,is_spouse:bool =False,render:bool =False):
        emp_states={1,10}
        unemp_states={0,4}
        family_states={5,6}
        muu_states={7,12,13}
        sv_state={14}

        states,latest,enimaika,voc_wage,voc_unempbasis,member,voc_wagebasis = self.infostate_vocabulary(is_spouse=is_spouse)

        lstate=round(self.infostate[latest])+1
        n=int(np.ceil(5/self.timestep))
        wage=0
        truewage=0
        for x in range(lstate-n,lstate):
            if x<0:
                pass
            else:
                empstate = self.infostate[states][x]
                if empstate in emp_states:
                    value = self.infostate[voc_wage][x]
                    w=value
                elif empstate in family_states:
                    value=1.21*self.infostate[voc_wagebasis][x] # FIXME!
                    w=0
                elif empstate in sv_state:
                    value=0.62*self.infostate[voc_wagebasis][x] # FIXME!
                    w=0
                elif empstate in unemp_states:
                    value=0.75*self.infostate[voc_unempbasis][x]
                    w=0
                elif empstate in muu_states:
                    value = self.disabbasis_tmtuki
                    w=0
                else:
                    value=0
                    w=0

                if render:
                    print(f'{empstate}: {value:.2f}')

                wage += value*self.timestep/5
                truewage += w*self.timestep

        return wage,truewage

    def infostate_comp_svpaivaraha_1v(self,is_spouse:bool =False,render:bool =False):
        emp_states={1,10}
        unemp_states={0,4}
        family_states={5,6}
        muu_states={7,12,13}
        sv_states={14}

        states,latest,enimaika,voc_wage,voc_unempbasis,member,voc_wagebasis = self.infostate_vocabulary(is_spouse=is_spouse)

        lstate=round(self.infostate[latest])+1
        n=int(np.ceil(1/self.timestep))
        wage=0
        truewage=0
        for x in range(lstate-n,lstate):
            if x<0:
                pass
            else:
                empstate = self.infostate[states][x]
                if empstate in emp_states:
                    value = self.infostate[voc_wage][x]
                    w=value
                elif empstate in family_states:
                    value = self.infostate[voc_wagebasis][x] # FIXME!
                    w=0
                elif empstate in muu_states:
                    value = self.disabbasis_tmtuki
                    w=0
                elif empstate in sv_states:
                    value = self.infostate[voc_wagebasis][x] # FIXME!
                    w=0
                elif empstate in unemp_states:
                    value = self.infostate[voc_unempbasis][x]
                    w=0
                else:
                    value=0
                    w=0

                if render:
                    print(f'{empstate}: {value:.2f}')

                wage += value*self.timestep
                #truewage += w*self.timestep

        return wage#,truewage

    def infostate_can_have_children(self,age: float):
        children_under1=0
        for k in range(self.infostate['children_n']):
            if age-self.infostate['children_date'][k] < 1.0:
                children_under1=1
                break

        if children_under1>0:
            return False
        else:
            return True

    def infostate_vocabulary(self,is_spouse:bool =False):
        if is_spouse:
            states='spouse_states'
            latest='spouse_latest'
            enimaika='spouse_enimmaisaika_alkaa'
            palkka='spouse_wage'
            unempbasis='spouse_unempbasis'
            wagebasis='spouse_wagebasis'
            jasen='spouse_unempmember'
        else:
            states='main_states'
            latest='main_latest'
            enimaika='main_enimmaisaika_alkaa'
            palkka='main_wage'
            unempbasis='main_unempbasis'
            wagebasis='main_wagebasis'
            jasen='main_unempmember'

        return states,latest,enimaika,palkka,unempbasis,jasen,wagebasis

    def infostate_check_aareset(self,age,is_spouse:bool =False):
        '''
        Tarkasta, onko edellisestä uudelleenmäärittelystä alle vuosi aikaa
        '''
        states,latest,enimaika,palkka,voc_unempbasis,member,voc_wagebasis = self.infostate_vocabulary(is_spouse=is_spouse)

        t = round((age-self.min_age)/self.timestep)
        ed_t = self.infostate[enimaika]
        if (t-ed_t)*self.timestep < 1.0: 
            return True
        else:
            return False

    def comp_oldtoe(self,printti:bool =False,is_spouse:bool =False):
        '''
        laske työttämyysjaksoa edeltävä työssäoloehto tarkasti
        '''
        toes=0
        n_toe=int(np.floor(self.max_toe/self.timestep))
        emp_states={1,10}
        unemp_states={0,4,13}
        family_states={5,6,7,12,14}
        ret_states={2,3,8,9}
        wage=0

        states,latest,enimaika,palkka,voc_unempbasis,member,voc_wagebasis = self.infostate_vocabulary(is_spouse=is_spouse)

        lstate=round(self.infostate[states][self.infostate[latest]])

        nt=0
        t2=max(0,self.infostate[enimaika]-1)
        emp_states={1,10}
        unemp_states={0,4}
        family_states={5,6,7,12,14}
        while nt<n_toe:
            emps = self.infostate[states][t2]
            if printti:
                print('emps {} t2 {} toes {}'.format(emps,t2,toes))
            if emps in family_states:
                pass
            elif emps in emp_states:
                if self.infostate[palkka][t2]>self.min_toewage:
                    toes += self.timestep
                elif self.include_halftoe and self.infostate[palkka][t2] >= self.min_halftoewage: # and emps==10:
                    toes+=0.5*self.timestep
                nt=nt+1
            elif emps in unemp_states:
                nt=nt+1
            else:
                nt=nt+1
            t2=t2-1

        return toes

    def comp_svperuste(self,printti:bool=False,is_spouse:bool=False):
        '''
        laske sairauspäivärahan perustepalkka
        '''
        toes=0
        n_svp=int(np.floor(1/self.timestep))
        emp_states={1,10}
        unemp_states={0,4,13}
        family_states={5,6,7,14}
        ret_states={2,3,8,9}
        wage=0

        states,latest,enimaika,palkka,voc_unempbasis,member,voc_wagebasis = self.infostate_vocabulary(is_spouse=is_spouse)

        lstate=round(self.infostate[states][self.infostate[latest]])

        nt=0
        t2=max(0,self.infostate[enimaika]-1)
        emp_states={1,10}
        unemp_states={0,4}
        family_states={5,6,7,14}
        while nt<n_svp:
            emps = self.infostate[states][t2]
            if printti:
                print(f'emps {emps} t2 {t2} toes {toes}')
            if emps in family_states:
                pass
            elif emps in emp_states:
                if self.infostate[palkka][t2]:
                    toes += self.infostate[palkka][t2]
                nt=nt+1
            elif emps in unemp_states:
                if self.infostate[palkka][t2]:
                    toes += self.infostate[palkka][t2]
                nt=nt+1
            else:
                nt=nt+1
            t2=t2-1

        return toes

    def check_toe58(self,age: float,toe: float,tyoura: float,toe58: int,is_spouse:bool =False):
        '''
        laske työttämyysjaksoa edeltävä työssäoloehto tarkasti
        '''
        states,latest,enimaika,palkka,voc_unempbasis,member,voc_wagebasis = self.infostate_vocabulary(is_spouse=is_spouse)

        toes=0
        n_toe=int(np.floor(self.max_toe/self.timestep))
        emp_states={1,10}
        unemp_states={0,4}
        family_states={5,6,7,12,14}
        ret_states={2,3,8,9}
        lstate=round(self.infostate[states][self.infostate[latest]])

        if age<self.minage_500 or lstate in ret_states:
            return 0

        t = self.map_age(age)
        t58 = self.map_age(58)

        #if lstate!=0:
        #    return 0

        nt=0
        if lstate in unemp_states:
            t2=max(0,self.infostate[enimaika]-1)
        else:
            t2=max(0,self.infostate[latest])

        while nt<n_toe and nt<t-t58:
            emps = self.infostate[states][t2]
            if emps in family_states:
                pass
            elif emps in emp_states:
                if self.infostate[palkka][t2]>self.min_toewage:
                    toes += self.timestep
                elif self.include_halftoe and self.infostate[palkka][t2] >= self.min_halftoewage: # and emps==10:
                    toes+=0.5*self.timestep
                nt=nt+1
            elif emps in unemp_states:
                nt=nt+1
            else:
                nt=nt+1
            t2=t2-1

        if self.tyossaoloehto(toes,tyoura,age) and tyoura >= self.tyohistoria_vaatimus500:
            return 1
        else:
            return 0

    ###############################################

    def test_swap(self,minage: float=18,maxage: float=70,n: int=100):
        r = self.randomness
        self.randomness=False
        self.silent=True

        print('**** Test swap')

        for k in range(n):
            self.plotdebug=False
            self.reset()
            self.plotdebug=True
            self.steps_beyond_done=None
            vec0 = self.states.random_init_state(minage=minage,maxage=maxage)
            self.state=vec0
            action0=random.randint(0,self.n_actions-1)
            action1=random.randint(0,self.n_actions-1)
            ptaction0=random.randint(0,2)
            ptaction1=random.randint(0,2)
            a=np.array([action0,action1,ptaction0,ptaction1])
            print(a)
            _,r0,_,q0 = self.step(a)

            vec1 = self.states.swap_spouses(vec0)
            a=np.array([action1,action0,ptaction1,ptaction0])

            self.state=vec1
            self.steps_beyond_done=None
            _,r1,_,q1 = self.step(a)

            if not math.isclose(r0,r1):
                self.render(render_omat=True,render_puoliso=True)
                print('!!!! ERROR in swap, rewards not identical:')
                print(r0,r1)

            vec2 = self.states.swap_spouses(vec1)
            self.states.check_state_vec(vec0,vec2)
            if not math.isclose(r0,r1):
                self.render(render_omat=True,render_puoliso=True)
                crosscheck_print(q0,q1)
                print('\n----------')

        print('**** End swap')
        self.randomness=r

    #########################
    ### STATE ROUTINES
    #########################

    def get_minimal(self):
        return False

    def get_timestep(self):
        return self.timestep

    def get_statenames(self):
        return ['Unemployed, earnings-related','Employed','Retired','Disabled','Unemployed, pipe','Motherleave','Fatherleave','Home care support','Retired, part-time work',
                'Retired, full-time work','Part-time word','Outsider','Student','Uneemployed, basic','Sickleave','Deceiced']

    def dim_action(self,state):
        emp,g,pension,raw_wage,age,time_in_state,paid_pension,pink,toe,toekesto,tyohist,used_unemp_benefit,\
            wage_red,unemp_after_ra,unempwage,unempwage_basis,prefnoise,c3,c7,c18,\
            unemp_left,oikeus,toe58,ove_paid,jasen,\
            puoliso,puoliso_tila,spouse_g,puoliso_old_wage,spouse_pension,\
            spouse_wage_reduction,puoliso_paid_pension,puoliso_next_wage,\
            puoliso_used_unemp_benefit,puoliso_unemp_benefit_left,\
            puoliso_unemp_after_ra,puoliso_unempwage,puoliso_unempwage_basis,\
            puoliso_alkanut_ansiosidonnainen,puoliso_toe58,puoliso_toe,\
            puoliso_toekesto,puoliso_tyohist,spouse_time_in_state,puoliso_pink,puoliso_ove_paid,\
            kansanelake,spouse_kansanelake,tyoelake_maksussa,spouse_tyoelake_maksussa,\
            next_wage,main_paid_wage,spouse_paid_wage,pt_act,s_pt_act,main_wage_basis,spouse_wage_basis,\
            main_until_disab,spouse_until_disab,time_to_marriage,time_to_divorce,until_child\
                 = self.states.state_decode(state)

        if emp in set([1,8,9,10]):
            n_acts=4
            n_pt_acts=3
        else:
            n_acts=4
            n_pt_acts=1

        if puoliso_tila in set([1,8,9,10]):
            n_sp_acts=4
            n_sp_pt_acts=3
        else:
            n_sp_acts=4
            n_sp_pt_acts=1

        return (n_acts,n_sp_acts,n_pt_acts,n_sp_pt_acts)

    def set_state(self,state):
        self.state=state

    def get_state(self):
        return self.state

    def get_retirementage(self) -> float:
        return self.min_retirementage

    def get_n_states(self):
        '''
        returns number of the employment state & number of actions
        '''
        return self.n_empl,[self.n_actions,3,self.n_actions,3]

    def get_lc_version(self) -> int:
        '''
        returns the version of life-cycle model's episodestate used
        '''
        return 9

    def get_mortstate(self) -> int:
        return 15

    def get_actions(self,action):
        emp_action=int(action[0])
        spouse_action=int(action[1])

        if self.include_savings:
            emp_savaction=int(action[4])
            spouse_savaction=int(action[5])
        else:
            emp_savaction=0
            spouse_savaction=0

        if self.include_parttimeactions:
            main_pt_action=int(action[2])
            spouse_pt_action=int(action[3])
        else:
            main_pt_action=0
            spouse_pt_action=0

        return emp_action,emp_savaction,main_pt_action,spouse_action,spouse_savaction,spouse_pt_action