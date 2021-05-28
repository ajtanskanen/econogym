"""

    unemployment_v3


    Gym module implementing the Finnish social security including earnings-related components,
    e.g., the unemployment benefit
    
"""

import math
import gym
from gym import spaces, logger, utils, error
from gym.utils import seeding   
import numpy as np
import fin_benefits
import random
from scipy.interpolate import interp1d
from . rates import Rates

# class StayDict(dict):
#     '''
#     Apuluokka, jonka avulla tehdään virheenkorjausta 
#     '''
#     def __missing__(self, key):
#         return 'Unknown state '+key


class UnemploymentLargeEnv_v3(gym.Env):
    """
    Description:
        The Finnish Unemployment Pension Scheme 

    Source:
        This environment corresponds to the environment of the Finnish Social Security

    Observation: 
        Type: Box(17)  UPDATE!
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
       17    Preferenssikohina         

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

        self.ansiopvraha_toe=0.5 # = 6kk
        self.karenssi_kesto=0.25 #0.25 # = 3kk
        self.isyysvapaa_kesto=0.25 # = 3kk
        self.aitiysvapaa_kesto=0.75 # = 9kk ml vanhempainvapaa
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

        self.include_mort=True # onko kuolleisuus mukana laskelmissa
        self.include_npv_mort=False # onko kuolleisuus mukana laskelmissa
        self.include_preferencenoise=False # onko työllisyyspreferenssissä hajonta mukana 
        self.perustulo=False # onko Kelan perustulo laskelmissa
        self.randomness=True # onko stokastiikka mukana
        self.mortstop=True # pysäytä kuolleisuuden jälkeen
        self.include_putki=True # työttömyysputki mukana
        self.include_pinkslip=True # irtisanomiset mukana
        self.use_sigma_reduction=True # kumpi palkkareduktio
        self.include_children=False # yksinhuoltajat ja lapset mukana mallissa
        self.include_kansanelake=True
        self.include_takuuelake=True
        self.preferencenoise_std=0.1
        
        self.additional_income_tax=0
        self.additional_income_tax_high=0
        self.additional_tyel_premium=0
        self.additional_kunnallisvero=0
        self.scale_tyel_accrual=True
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

        # parametrejä
        if self.porrasta_toe:
            self.max_toe=35/12
            self.min_toewage=844*12 # vuoden 2019 luku tilanteessa, jossa ei tessiä
            self.min_halftoewage=422*12 # vuoden 2019 luku tilanteessa, jossa ei tessiä
        else:
            self.max_toe=28/12
            self.min_toewage=1211*12 # vuoden 2019 luku tilanteessa, jossa ei tessiä
            self.min_halftoewage=800*12 # vuoden 2019 luku tilanteessa, jossa ei tessiä
        self.setup_unempdays_left(porrastus=self.porrasta_toe)

        self.accbasis_kht=719.0*12
        self.accbasis_tmtuki=0 # 1413.75*12
        self.disabbasis_tmtuki=1413.75*12
        
        self.log_utility_default_params()

        self.n_age=self.max_age-self.min_age+1

        # male low income, male mid, male high, female low, female mid, female high income
        self.n_groups=6

        # käytetäänkö exp/log-muunnosta tiloissa vai ei?
        self.log_transform=False
        self.eps=1e-20

        # ryhmäkohtaisia muuttujia
        self.rates=Rates(year=self.year,max_age=self.max_age,n_groups=self.n_groups,timestep=self.timestep)
        self.setup_salaries_v3()

        self.disability_intensity=self.rates.get_eff_disab_rate()
        self.pinkslip_intensity=self.rates.get_pinkslip_rate()*self.timestep
        self.birth_intensity=self.rates.get_birth_rate()
        self.mort_intensity=self.rates.get_mort_rate()
        self.student_inrate,self.student_outrate=self.rates.get_student_rate() # myös armeijassa olevat tässä
        self.outsider_inrate,self.outsider_outrate=self.rates.get_outsider_rate()

        self.npv,self.npv0,self.npv_pension=self.comp_npv()
        self.initial_weights=self.get_initial_weights()

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
            
        self.setup_state_encoding()

        self.n_actions=5 # valittavien toimenpiteiden määrä, kasvata viiteen ja korjaa siirtymät 4 & 5

        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        
        if self.use_sigma_reduction:
            self.update_wage_reduction=self.update_wage_reduction_sigma
        else:
            self.update_wage_reduction=self.update_wage_reduction_baseline

        #self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None
        
        # normitetaan työttömyysjaksot askelvälin monikerroiksi
        scale=21.5*12
        self.apvkesto300=np.round(self.ansiopvraha_kesto300/scale/self.timestep)*self.timestep
        self.apvkesto400=np.round(self.ansiopvraha_kesto400/scale/self.timestep)*self.timestep
        self.apvkesto500=np.round(self.ansiopvraha_kesto500/scale/self.timestep)*self.timestep
                
        if self.perustulo:
            self.ben = fin_benefits.BasicIncomeBenefits(**kwargs)
        else:
            #self.ben = fin_benefits.CyBenefits(**kwargs)
            self.ben = fin_benefits.Benefits(**kwargs)
            
        self.ben.set_year(self.year)
        
        self.init_inforate()
        
        self.explain()
        
        if self.plotdebug:
            self.unit_test_code_decode()

    def get_n_states(self):
        '''
        returns number of the employment state & number of actions
        '''
        return self.n_empl,self.n_actions
        
    def get_lc_version(self):
        '''
        returns the version of life-cycle model's episodestate used
        '''
        return 3
        
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
                cpsum_pension=m*1+(1-m)*(1+cpsum_pension*self.elakeindeksi)
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

    def comp_benefits(self,wage,old_wage,pension,employment_state,time_in_state,children_under3,children_under7,children_under18,ika,
                      irtisanottu=0,tyohistoria=0,karenssia_jaljella=0,retq=True,ove=False):
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
        p={}

        if self.perustulo:
            p['perustulo']=1
        else:
            p['perustulo']=0
        
        p['opiskelija']=0
        p['toimeentulotuki_vahennys']=0
        p['ika']=ika
        p['tyoton']=0
        p['saa_ansiopaivarahaa']=0
        p['vakiintunutpalkka']=0
        
        if self.include_children:
            p['lapsia']=children_under18
            p['lapsia_paivahoidossa']=children_under7
            p['lapsia_alle_kouluikaisia']=children_under7
            p['lapsia_alle_3v']=children_under3
        else:
            p['lapsia']=children_under3
            p['lapsia_paivahoidossa']=children_under3
            p['lapsia_alle_kouluikaisia']=children_under3
            p['lapsia_alle_3v']=children_under3
        p['aikuisia']=1
        p['veromalli']=0
        p['kuntaryhma']=3
        p['lapsia_kotihoidontuella']=0
        p['tyottomyyden_kesto']=1
        p['puoliso_tyottomyyden_kesto']=10
        p['isyysvapaalla']=0
        p['aitiysvapaalla']=0
        p['kotihoidontuella']=0
        p['tyoelake']=0
        p['elakkeella']=0
        p['sairauspaivarahalla']=0
        p['disabled']=0
        
        if employment_state==1:
            p['tyoton']=0 # voisi olla työtön siinä mielessä, että oikeutettu soviteltuun päivärahaan
            p['t']=wage/12
            p['vakiintunutpalkka']=wage/12
            p['saa_ansiopaivarahaa']=0
            p['tyoelake']=pension/12 # ove
        elif employment_state==0: # työtön, ansiopäivärahalla
            if ika<65:
                #self.render()
                p['tyoton']=1
                p['t']=0
                p['vakiintunutpalkka']=old_wage/12
                p['saa_ansiopaivarahaa']=1
                p['tyottomyyden_kesto']=12*21.5*time_in_state
                    
                if irtisanottu<1 and karenssia_jaljella>0:
                    p['saa_ansiopaivarahaa']=0
                    p['tyoton']=0
                    
                p['tyoelake']=pension/12 # ove
            else:
                p['tyoton']=0 # ei oikeutta työttömyysturvaan
                p['t']=0
                p['vakiintunutpalkka']=0
                p['saa_ansiopaivarahaa']=0
                p['tyoelake']=pension/12 # ove
        elif employment_state==13: # työmarkkinatuki
            if ika<65:
                p['tyoton']=1
                p['t']=0
                p['vakiintunutpalkka']=0
                p['tyottomyyden_kesto']=12*21.5*time_in_state
                p['saa_ansiopaivarahaa']=0
                p['tyoelake']=pension/12 # ove
            else:
                p['tyoton']=0 # ei oikeutta työttömyysturvaan
                p['t']=0
                p['vakiintunutpalkka']=0
                p['saa_ansiopaivarahaa']=0
                p['tyoelake']=pension/12 # ove
        elif employment_state==3: # tk
            p['t']=0
            p['elakkeella']=1 
            #p['elake']=pension
            p['tyoelake']=pension/12
            p['disabled']=1
        elif employment_state==4: # työttömyysputki
            if ika<65:
                p['tyoton']=1
                p['t']=0
                p['vakiintunutpalkka']=old_wage/12
                p['saa_ansiopaivarahaa']=1
                p['tyottomyyden_kesto']=12*21.5*time_in_state
                p['tyoelake']=pension/12 # ove
            else:
                p['tyoton']=0 # ei oikeutta työttömyysturvaan
                p['t']=0
                p['vakiintunutpalkka']=0
                p['saa_ansiopaivarahaa']=0
                p['tyoelake']=pension/12 # ove
        elif employment_state==5: # ansiosidonnainen vanhempainvapaa, äidit
            p['aitiysvapaalla']=1
            p['aitiysvapaa_kesto']=0
            p['t']=0
            p['vakiintunutpalkka']=old_wage/12
            p['saa_ansiopaivarahaa']=1
        elif employment_state==6: # ansiosidonnainen vanhempainvapaa, isät
            p['isyysvapaalla']=1
            p['t']=0
            p['vakiintunutpalkka']=old_wage/12
            p['saa_ansiopaivarahaa']=1
        elif employment_state==7: # hoitovapaa
            p['kotihoidontuella']=1
            if self.include_children:
                p['lapsia_paivahoidossa']=0
                p['lapsia_kotihoidontuella']=children_under7
            else:
                p['lapsia_paivahoidossa']=0
                p['lapsia_kotihoidontuella']=children_under3
            p['kotihoidontuki_kesto']=time_in_state
            p['t']=0
            p['vakiintunutpalkka']=old_wage/12
        elif employment_state==2: # vanhuuseläke
            if ika>=self.min_retirementage:
                p['t']=0
                p['elakkeella']=1  
                p['tyoelake']=pension/12
            else:
                p['t']=0
                p['elakkeella']=0
                p['tyoelake']=0
        elif employment_state in set([8,9]): # ve+osatyö
            p['t']=wage/12
            p['elakkeella']=1  
            p['tyoelake']=pension/12
        elif employment_state==10: # osa-aikatyö
            p['t']=wage/12
            p['tyoelake']=pension/12 # ove
        elif employment_state==11: # työelämän ulkopuolella
            p['toimeentulotuki_vahennys']=0 # oletetaan että ei kieltäytynyt työstä
            p['t']=0
            p['tyoelake']=pension/12 # ove
        elif employment_state==12: # opiskelija
            p['opiskelija']=1
            p['t']=0
            p['tyoelake']=pension/12 # ove
        elif employment_state==14: # kuollut
            p['t']=0
        else:
            print('Unknown employment_state ',employment_state)

        # tarkastellaan yksinasuvia henkilöitä
        if employment_state==12: # opiskelija
            p['asumismenot_toimeentulo']=250
            p['asumismenot_asumistuki']=250
        elif employment_state in set([2,8,9]): # eläkeläinen
            p['asumismenot_toimeentulo']=200
            p['asumismenot_asumistuki']=200
        else: # muu
            p['asumismenot_toimeentulo']=320 # per hlö, ehkä 500 e olisi realistisempi, mutta se tuottaa suuren asumistukimenon
            p['asumismenot_asumistuki']=320

        p['ansiopvrahan_suojaosa']=1
        p['ansiopvraha_lapsikorotus']=1
        p['puoliso_tulot']=0
        p['puoliso_tyoton']=0  
        p['puoliso_vakiintunutpalkka']=0  
        p['puoliso_saa_ansiopaivarahaa']=0

        netto,benefitq=self.ben.laske_tulot(p,include_takuuelake=self.include_takuuelake)
        #netto=max(0,netto-p['asumismenot_asumistuki']) # netotetaan asumismenot pois käteenjäävästä
        #netto=max(0,netto) # ei netotusta
        netto=netto*12
        
        if retq:
            return netto,benefitq
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
            tyoton=0.019
            m1,m2,m3=get_wees(1.27,1.0,tyoton)
            #print(m1,m2,m3)
            tyoton=0.016
            w1,w2,w3=get_wees(1.2,1.0,tyoton)
            om=0.686+0.257+0.029+0.019 # miehet töissä + opiskelija
            om1,om2,om3=get_wees(1.25,1.0,0.257)
            #print(om1,om2,om3)
            ow=0.607+0.340+0.025+0.016
            ow1,ow2,ow3=get_wees(1.2,1.0,0.340)
            tyovoimanulkop=0.029
            um1,um2,um3=get_wees(1.3,1.0,tyovoimanulkop)
            #print(um1,um2,um3)
            tyovoimanulkop=0.025
            uw1,uw2,uw3=get_wees(1.2,1.0,tyovoimanulkop)
            md=0.009 # 0.014339
            wd=0.00730# 0.0121151

            initial_weight[0,:]=[m1*4/5,m1*1/5,0.68*om1,0.32*om1,md,um1,om-om1-um1-m1]
            initial_weight[1,:]=[m2*4/5,m2*1/5,0.68*om2,0.32*om2,md,um2,om-om2-um2-m2]
            initial_weight[2,:]=[m3*4/5,m3*1/5,0.68*om3,0.32*om3,md,um3,om-om3-um3-m3]
            initial_weight[3,:]=[w1*4/5,w1*1/5,0.44*ow1,0.56*ow1,wd,uw1,ow-ow1-uw1-w1]
            initial_weight[4,:]=[w2*4/5,w2*1/5,0.44*ow2,0.56*ow2,wd,uw2,ow-ow2-uw2-w2]
            initial_weight[5,:]=[w3*4/5,w3*1/5,0.44*ow3,0.56*ow3,wd,uw3,ow-ow3-uw3-w3]
        elif self.year==2019:
            # tilat [13,0,1,10,3,11,12]

            tyoton=0.021
            m1,m2,m3=get_wees(1.27,1.0,tyoton)
            #print(m1,m2,m3)
            tyoton=0.017
            w1,w2,w3=get_wees(1.2,1.0,tyoton)
            om=0.679+0.261+0.030+0.021
            om1,om2,om3=get_wees(1.25,1.0,0.261)
            #print(om1,om2,om3)
            ow=0.587+0.360+0.027+0.017
            ow1,ow2,ow3=get_wees(1.2,1.0,0.360)
            tyovoimanulkop=0.030
            um1,um2,um3=get_wees(1.3,1.0,tyovoimanulkop)
            #print(um1,um2,um3)
            tyovoimanulkop=0.027
            uw1,uw2,uw3=get_wees(1.2,1.0,tyovoimanulkop)
            md=0.009 # 0.010315
            wd=0.006 # 0.0071558

            initial_weight[0,:]=[m1*4/5,m1*1/5,0.68*om1,0.32*om1,md,um1,om-om1-um1-m1]
            initial_weight[1,:]=[m2*4/5,m2*1/5,0.68*om2,0.32*om2,md,um2,om-om2-um2-m2]
            initial_weight[2,:]=[m3*4/5,m3*1/5,0.68*om3,0.32*om3,md,um3,om-om3-um3-m3]
            initial_weight[3,:]=[w1*4/5,w1*1/5,0.44*ow1,0.56*ow1,wd,uw1,ow-ow1-uw1-w1]
            initial_weight[4,:]=[w2*4/5,w2*1/5,0.44*ow2,0.56*ow2,wd,uw2,ow-ow2-uw2-w2]
            initial_weight[5,:]=[w3*4/5,w3*1/5,0.44*ow3,0.56*ow3,wd,uw3,ow-ow3-uw3-w3]
        
        else:
            error(999)
            
        return initial_weight

    def scale_pension(self,pension,age,scale=True,unemp_after_ra=0):
        '''
        Elinaikakertoimen ja lykkäyskorotuksen huomiointi
        '''
        if scale:
            return self.elinaikakerroin*pension*self.elakeindeksi*(1+0.048*(age-self.min_retirementage-unemp_after_ra)) 
        else:
            return self.elinaikakerroin*pension*self.elakeindeksi
        
    def move_to_parttime(self,pension,old_wage,age,tyoura,time_in_state,wage_reduction,children_under3,children_under7,children_under18,paid_pension):
        '''
        Siirtymä osa-aikaiseen työskentelyyn
        '''
        employment_status = 10 # switch to part-time work
        wage=self.get_wage(age,wage_reduction)
        parttimewage=0.5*wage
        tyoura += self.timestep
        time_in_state=0
        old_wage=0
        pension=self.pension_accrual(age,parttimewage,pension,state=employment_status)
        netto,benq=self.comp_benefits(parttimewage,old_wage,paid_pension,employment_status,time_in_state,children_under3,children_under7,children_under18,age,retq=True)
        pinkslip=0
        time_in_state=self.timestep
        wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)        

        return employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq

    def move_to_work(self,pension,old_wage,age,time_in_state,tyoura,pinkslip,wage_reduction,children_under3,children_under7,children_under18,paid_pension):
        '''
        Siirtymä täysiaikaiseen työskentelyyn
        '''
        employment_status = 1 # töihin
        pinkslip=0
        wage=self.get_wage(age,wage_reduction)
        time_in_state=0
        old_wage=0
        tyoura+=self.timestep
        pension=self.pension_accrual(age,wage,pension,state=employment_status)
        netto,benq=self.comp_benefits(wage,old_wage,paid_pension,employment_status,time_in_state,children_under3,children_under7,children_under18,age)
        time_in_state=self.timestep
        wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)        

        return employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq


    #def move_to_oa_fulltime(self,pension,old_wage,age,time_in_state,paid_pension,wage_reduction,children_under3,children_under7,children_under18):
    def move_to_oa_fulltime(self,pension,old_wage,age,paid_pension,employment_status,
            wage_reduction,unemp_after_ra,children_under3,children_under7,children_under18,
            all_acc=True,scale_acc=True):
        '''
        Siirtymä vanhuuseläkkeelle
        '''
        if age>=self.max_retirementage: # move to retirement state 2
            if employment_status in set([2,8,9]): # ve, ve+työ, ve+osatyö
                # ei lykkäyskorotusta
                paid_pension = self.elakeindeksi*paid_pension+self.scale_pension(pension,age,scale=False,unemp_after_ra=unemp_after_ra)
                pension=0
                employment_status = 2
            elif employment_status==3: # tk
                # do nothing
                employment_status=3
                paid_pension = self.elakeindeksi*paid_pension+self.scale_pension(pension,age,scale=False,unemp_after_ra=unemp_after_ra)
                pension=0
            else: # ei vielä eläkkeellä
                # lykkäyskorotus
                paid_pension = self.elakeindeksi*paid_pension+self.scale_pension(pension,age,scale=scale_acc,unemp_after_ra=unemp_after_ra)
                if self.include_kansanelake:
                    paid_pension += self.ben.laske_kansanelake(age,paid_pension/12,1)*12 # ben-modulissa palkat kk-tasolla
                pension=0
                employment_status = 2
                
            time_in_state=self.timestep
            wage=self.get_wage(age,wage_reduction)
            alkanut_ansiosidonnainen=0
            netto,benq=self.comp_benefits(0,0,paid_pension,employment_status,0,children_under3,children_under7,children_under18,age)
            wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)        
            ove_paid=0
        elif age>=self.min_retirementage:
            if all_acc:
                if employment_status in set([2,8,9]): # ve, ve+työ, ve+osatyö
                    paid_pension = self.elakeindeksi*paid_pension
                    employment_status = 9
                elif employment_status==3: # tk
                    # do nothing
                    employment_status=3
                    paid_pension = self.elakeindeksi*paid_pension
                else:
                    # lykkäyskorotus
                    paid_pension = paid_pension*self.elakeindeksi+self.scale_pension(pension,age,scale=scale_acc,unemp_after_ra=unemp_after_ra)
                    if self.include_kansanelake:
                        paid_pension += self.ben.laske_kansanelake(age,paid_pension/12,1)*12 # ben-modulissa palkat kk-tasolla
                    pension=0
                    employment_status = 9
            elif employment_status in set([2,8,9]): # ve, ve+työ, ve+osatyö
                paid_pension = self.elakeindeksi*paid_pension
                pension = pension*self.palkkakerroin
                employment_status = 9

            time_in_state=self.timestep
            alkanut_ansiosidonnainen=0
            wage=self.get_wage(age,wage_reduction)
            pension=self.pension_accrual(age,wage,pension,state=employment_status)
            #print(paid_pension,employment_status,0,children_under3,children_under7,children_under18,age)
            netto,benq=self.comp_benefits(wage,0,paid_pension,employment_status,0,children_under3,children_under7,children_under18,age)
            #self.print_q(benq)
            wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)        
            ove_paid=0
        else: # työvoiman ulkopuolella
            time_in_state=0
            employment_status = 2 
            wage=old_wage
            netto,benq=self.comp_benefits(0,0,paid_pension,employment_status,0,children_under3,children_under7,children_under18,age)
            time_in_state+=self.timestep
            wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)        
            ove_paid=0

        return employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq

#     def move_to_oa_fulltime(self,pension,old_wage,age,time_in_state,paid_pension,wage_reduction,children_under3,children_under7,children_under18):
#         '''
#         Siirtymä vanhuuseläkkeellä työskentelyyn
#         '''
#         employment_status = 9
#         wage=self.get_wage(age,wage_reduction)
#         paid_pension=paid_pension*self.elakeindeksi
#         pension=self.pension_accrual(age,wage,pension,state=employment_status)
#         alkanut_ansiosidonnainen=0
#         time_in_state=0
#         netto,benq=self.comp_benefits(wage,0,paid_pension,employment_status,time_in_state,children_under3,children_under7,children_under18,age)
#         time_in_state+=self.timestep
#         wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)        
# 
#         return employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,benq

    def move_to_student(self,pension,old_wage,age,time_in_state,tyoura,pinkslip,wage_reduction,children_under3,children_under7,children_under18,paid_pension):
        '''
        Siirtymä opiskelijaksi
        Tässä ei muuttujien päivityksiä, koska se tehdään jo muualla!
        '''
        employment_status = 12
        time_in_state=0
        netto,benq=self.comp_benefits(0,0,paid_pension,employment_status,time_in_state,children_under3,children_under7,children_under18,age)
        time_in_state+=self.timestep
        wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)        
        pinkslip=0
        wage=self.get_wage(age,wage_reduction)

        return employment_status,pension,wage,time_in_state,netto,pinkslip,wage_reduction,benq

#     def move_to_oa_parttime(self,pension,old_wage,age,time_in_state,paid_pension,wage_reduction,children_under3,children_under7,children_under18):
#         '''
#         Siirtymä osa-aikaiseen vanhuuseläkkeellä työskentelyyn
#         '''
#         employment_status = 8 # unchanged
#         wage=self.get_wage(age,wage_reduction)
#         ptwage=0.5*wage
#         alkanut_ansiosidonnainen=0
#         paid_pension=paid_pension*self.elakeindeksi
#         pension=self.pension_accrual(age,ptwage,pension,state=employment_status)
#         time_in_state=0
#         netto,benq=self.comp_benefits(ptwage,0,paid_pension,employment_status,time_in_state,children_under3,children_under7,children_under18,age)
#         time_in_state+=self.timestep
#         wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)        
# 
#         return employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,benq
        
    def move_to_oa_parttime(self,pension,old_wage,age,paid_pension,employment_status,
            wage_reduction,unemp_after_ra,children_under3,children_under7,children_under18,
            all_acc=True,scale_acc=True):
        '''
        Siirtymä vanhuuseläkkeelle
        '''
        if age>=self.max_retirementage: # move to retirement state 2
            if employment_status in set([2,8,9]): # ve, ve+työ, ve+osatyö
                # ei lykkäyskorotusta
                paid_pension = self.elakeindeksi*paid_pension+self.scale_pension(pension,age,scale=False,unemp_after_ra=unemp_after_ra)
                pension=0
                employment_status = 2
            elif employment_status==3: # tk
                # do nothing
                employment_status=3
                paid_pension = self.elakeindeksi*paid_pension+self.scale_pension(pension,age,scale=False,unemp_after_ra=unemp_after_ra)
                pension=0
            else: # ei vielä eläkkeellä
                # lykkäyskorotus
                paid_pension = self.elakeindeksi*paid_pension+self.scale_pension(pension,age,scale=scale_acc,unemp_after_ra=unemp_after_ra)
                if self.include_kansanelake:
                    paid_pension += self.ben.laske_kansanelake(age,paid_pension/12,1)*12 # ben-modulissa palkat kk-tasolla
                pension=0
                employment_status = 2
                
            time_in_state=self.timestep
            wage=self.get_wage(age,wage_reduction)
            alkanut_ansiosidonnainen=0
            netto,benq=self.comp_benefits(0,0,paid_pension,employment_status,0,children_under3,children_under7,children_under18,age)
            wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)        
            ove_paid=0
        elif age>=self.min_retirementage:
            if all_acc:
                if employment_status in set([2,8,9]): # ve, ve+työ, ve+osatyö
                    paid_pension = self.elakeindeksi*paid_pension
                    employment_status = 8
                elif employment_status==3: # tk
                    # do nothing
                    employment_status=3
                    paid_pension = self.elakeindeksi*paid_pension
                else:
                    # lykkäyskorotus
                    paid_pension = paid_pension*self.elakeindeksi+self.scale_pension(pension,age,scale=scale_acc,unemp_after_ra=unemp_after_ra)
                    if self.include_kansanelake:
                        paid_pension += self.ben.laske_kansanelake(age,paid_pension/12,1)*12 # ben-modulissa palkat kk-tasolla
                    pension=0
                    employment_status = 8
            elif employment_status in set([2,8,9]): # ve, ve+työ, ve+osatyö
                paid_pension = self.elakeindeksi*paid_pension
                pension = pension*self.palkkakerroin
                employment_status = 8

            time_in_state=self.timestep
            alkanut_ansiosidonnainen=0
            wage=self.get_wage(age,wage_reduction)
            ptwage=0.5*wage
            pension=self.pension_accrual(age,ptwage,pension,state=employment_status)
            #print(paid_pension,employment_status,0,children_under3,children_under7,children_under18,age)
            netto,benq=self.comp_benefits(ptwage,0,paid_pension,employment_status,0,children_under3,children_under7,children_under18,age)
            #self.print_q(benq)
            wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)        
            ove_paid=0
        else: # työvoiman ulkopuolella
            time_in_state=0
            employment_status = 2 
            wage=old_wage
            netto,benq=self.comp_benefits(0,0,paid_pension,employment_status,0,children_under3,children_under7,children_under18,age)
            time_in_state+=self.timestep
            wage_reduction=self.update_wage_reduction(employment_status,wage_reduction) 
            print(age,wage)

        return employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq        

    def move_to_ove(self,employment_status,pension,paid_pension,ove_paid,age,unemp_after_ra):
        if not self.include_ove:
            return pension,paid_pension,0
    
        if ove_paid:
            print('Moving to OVE twice')
            error('failure')
            exit()
            
        if employment_status in set([2,3,8,9]): # ei eläkettä maksuun
            print('Incorrect state',employment_status)
            error('failure')
            exit()
        else:
            paid_pension = self.scale_pension(self.ove_ratio*pension,age,scale=True,unemp_after_ra=unemp_after_ra)
            pension=(1-self.ove_ratio)*pension # *self.palkkakerroin, tässä ei indeksoida, koska pension_accrual hoitaa tämän
            ove_paid=1

        return pension,paid_pension,ove_paid

    def move_to_retirement(self,pension,old_wage,age,paid_pension,employment_status,
            wage_reduction,unemp_after_ra,children_under3,children_under7,children_under18,
            all_acc=True,scale_acc=True):
        '''
        Moving to retirement
        '''
        if age>=self.max_retirementage:
            if employment_status in set([2,8,9]): # ve, ve+työ, ve+osatyö
                # ei lykkäyskorotusta
                paid_pension = self.elakeindeksi*paid_pension+self.scale_pension(pension,age,scale=False,unemp_after_ra=unemp_after_ra)
                pension=0
                employment_status = 2 
            elif employment_status==3: # tk
                # do nothing
                employment_status=3
                paid_pension = self.elakeindeksi*paid_pension+self.scale_pension(pension,age,scale=False,unemp_after_ra=unemp_after_ra)
                pension=0
            else: # ei vielä eläkkeellä
                # lykkäyskorotus
                paid_pension = self.elakeindeksi*paid_pension+self.scale_pension(pension,age,scale=scale_acc,unemp_after_ra=unemp_after_ra)
                if self.include_kansanelake:
                    paid_pension += self.ben.laske_kansanelake(age,paid_pension/12,1)*12 # ben-modulissa palkat kk-tasolla
                pension=0
                employment_status = 2 
                
            time_in_state=self.timestep
            wage=self.get_wage(age,wage_reduction)
            alkanut_ansiosidonnainen=0
            netto,benq=self.comp_benefits(0,0,paid_pension,employment_status,0,children_under3,children_under7,children_under18,age)
            wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)        
            ove_paid=0
        elif age>=self.min_retirementage:
            if all_acc:
                if employment_status in set([2,8,9]): # ve, ve+työ, ve+osatyö
                    paid_pension = self.elakeindeksi*paid_pension
                    pension=pension*self.palkkakerroin
                    employment_status = 2 
                elif employment_status==3: # tk
                    # do nothing
                    employment_status = 3
                    pension=pension*self.palkkakerroin
                    paid_pension = self.elakeindeksi*paid_pension
                else:
                    # lykkäyskorotus
                    paid_pension = paid_pension*self.elakeindeksi+self.scale_pension(pension,age,scale=scale_acc,unemp_after_ra=unemp_after_ra)
                    if self.include_kansanelake:
                        paid_pension += self.ben.laske_kansanelake(age,paid_pension/12,1)*12 # ben-modulissa palkat kk-tasolla
                    pension=0
                    employment_status = 2 
            elif employment_status in set([8,9]): # ve, ve+työ, ve+osatyö
                paid_pension = self.elakeindeksi*paid_pension
                pension=pension*self.palkkakerroin
                employment_status = 2 

            time_in_state=self.timestep
            alkanut_ansiosidonnainen=0
            wage=self.get_wage(age,wage_reduction)
            netto,benq=self.comp_benefits(0,0,paid_pension,employment_status,0,children_under3,children_under7,children_under18,age)
            wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)        
            ove_paid=0
        else: # työvoiman ulkopuolella
            time_in_state=0
            employment_status = 2 
            wage=old_wage
            netto,benq=self.comp_benefits(0,0,paid_pension,employment_status,0,children_under3,children_under7,children_under18,age)
            time_in_state+=self.timestep
            wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)

        return employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq

    def move_to_retdisab(self,pension,old_wage,age,time_in_state,paid_pension,wage_reduction,
                         children_under3,children_under7,children_under18,unemp_after_ra):   
        '''
        Siirtymä vanhuuseläkkeelle, jossa ei voi tehdä työtä
        '''
        
        if age>=self.max_retirementage:
            # ei mene täsmälleen oikein
            paid_pension=self.elakeindeksi*paid_pension+self.scale_pension(pension,age,scale=False,unemp_after_ra=unemp_after_ra)
            pension=0                        
        else:
            paid_pension = self.elakeindeksi*paid_pension
            pension = self.palkkakerroin*pension

        employment_status = 3
        wage=0
        netto,benq=self.comp_benefits(0,0,paid_pension,employment_status,0,children_under3,children_under7,children_under18,age)
        time_in_state=self.timestep
        wage_reduction=0.9
        alkanut_ansiosidonnainen=0

        return employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,benq
        
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
            #if printti:
            #print('Ctoekesto {} toe {} kesto {}'.format(toekesto,old_toe,kesto))
                
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

    def move_to_unemp(self,pension,old_wage,age,paid_pension,toe,toekesto,irtisanottu,tyoura,wage_reduction,
                    used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,
                    children_under3,children_under7,children_under18,alkanut_ansiosidonnainen,toe58):
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
            
            employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq=\
                self.move_to_retirement(pension,old_wage,age,paid_pension,employment_status,
                    wage_reduction,unemp_after_ra,children_under3,children_under7,children_under18,all_acc=True)
                
            return employment_status,paid_pension,pension,wage,time_in_state,netto,\
                   wage_reduction,used_unemp_benefit,pinkslip,benq,unemp_after_ra,unempwage_basis,alkanut_ansiosidonnainen
        else:
            #if toe>=self.ansiopvraha_toe: # täyttyykö työssäoloehto
            tehto=self.tyossaoloehto(toe,tyoura,age)
            if tehto or alkanut_ansiosidonnainen>0:
                if tehto:
                    kesto=0
                    used_unemp_benefit=0
                    self.infostate_set_enimmaisaika(age) # resetoidaan enimmäisaika
                    if self.infostat_check_aareset(age):
                        unempwage_basis=self.update_unempwage_basis(unempwage_basis,unempwage,True)
                    else:
                        unempwage_basis=self.update_unempwage_basis(unempwage_basis,unempwage,False)
                    
                    jaljella=self.paivarahapaivia_jaljella(kesto,tyoura,age,toe58,toekesto)
                else:
                    kesto=used_unemp_benefit
                    if self.porrasta_toe:
                        jaljella=self.paivarahapaivia_jaljella(kesto,tyoura,age,toe58,self.comp_oldtoe())
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
            
            wage=self.get_wage(age,wage_reduction)
            if karenssia_jaljella>0:
                pension=pension*self.palkkakerroin
            else:
                if paid_pension>0: # ove maksussa, ei karttumaa
                    pension=self.pension_accrual(age,unempwage_basis,pension,state=employment_status,ove_paid=1)
                else:
                    pension=self.pension_accrual(age,unempwage_basis,pension,state=employment_status)

            # hmm, omavastuupäivät puuttuvat!
            # omavastuupäiviä on 5/(21.5*12*self.timestep), kerroin tällöin
            # 1-5/(21.5*12*self.timestep)
            netto,benq=self.comp_benefits(0,unempwage_basis,paid_pension,employment_status,used_unemp_benefit,children_under3,children_under7,children_under18,age,
                                     irtisanottu=irtisanottu,tyohistoria=tyoura,karenssia_jaljella=karenssia_jaljella)
            time_in_state=self.timestep
            karenssia_jaljella=max(0,karenssia_jaljella-self.timestep)
            #unemp_after_ra ei muutu
            wage_reduction=self.update_wage_reduction(employment_status,wage_reduction,initial_reduction=True)
            
            # Tässä ei tehdä karenssia_jaljella -muuttujasta tilamuuttujaa, koska karenssin kesto on lyhyempi kuin aika-askeleen
            # Samoin karenssia ei ole tm-tuessa, koska toimeentulotuki on suurempi
            
            pinkslip=irtisanottu

        return employment_status,paid_pension,pension,wage,time_in_state,netto,\
               wage_reduction,used_unemp_benefit,pinkslip,benq,unemp_after_ra,\
               unempwage_basis,alkanut_ansiosidonnainen

    def move_to_outsider(self,pension,old_wage,age,irtisanottu,wage_reduction,children_under3,children_under7,children_under18,paid_pension):
        '''
        Siirtymä työvoiman ulkopuolelle
        '''
        employment_status = 11 # switch
        time_in_state=0
        wage=self.get_wage(age,wage_reduction)
        pension=pension*self.palkkakerroin

        netto,benq=self.comp_benefits(0,0,paid_pension,employment_status,time_in_state,children_under3,children_under7,children_under18,age,irtisanottu=0)
        paid_pension=0
        time_in_state+=self.timestep
        wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)        
        pinkslip=0

        return employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,pinkslip,benq

    def move_to_disab(self,pension,old_wage,age,wage_reduction,children_under3,children_under7,children_under18,unemp_after_ra,paid_pension,ove_paid):
        '''
        Siirtymä työkyvyttömyyseläkkeelle
        '''
        employment_status = 3 # tk
        if age<self.min_retirementage:
            wage5y=self.infostat_comp_5y_ave_wage()
            #wage5y=old_wage
            paid_pension=(paid_pension+self.elinaikakerroin*(pension+self.acc/self.timestep*wage5y*max(0,self.min_retirementage-age)))*self.elakeindeksi
            #print(self.acc,wage5y,max(0,self.min_retirementage-age))
        
            # takuueläke voidaan huomioida jo tässä
            paid_pension=self.ben.laske_kokonaiselake(65,paid_pension/12,include_kansanelake=self.include_kansanelake,include_takuuelake=False,disability=True)*12
            pension=0
            alkanut_ansiosidonnainen=0
            time_in_state=0
            wage=0
            netto,benq=self.comp_benefits(0,0,paid_pension,employment_status,0,children_under3,children_under7,children_under18,age)
            time_in_state+=self.timestep
            wage_reduction=0.60 # vastaa määritelmää
        else:
            # siirtymä vanhuuseläkkeelle, lykkäyskorotus, ei tulevaa aikaa
            paid_pension = paid_pension*self.elakeindeksi + self.scale_pension(pension,age,scale=True,unemp_after_ra=unemp_after_ra)
            if self.include_kansanelake:
                paid_pension += self.ben.laske_kansanelake(age,paid_pension/12,1)*12 # ben-modulissa palkat kk-tasolla
            pension=0

            time_in_state=self.timestep
            alkanut_ansiosidonnainen=0
            employment_status = 3
            wage=self.get_wage(age,wage_reduction)
            netto,benq=self.comp_benefits(0,0,paid_pension,employment_status,0,children_under3,children_under7,children_under18,age)
            ove_paid=0
            wage_reduction=0.60 # vastaa määritelmää

        return employment_status,pension,paid_pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq

    def move_to_deceiced(self,pension,old_wage,age,children_under3,children_under7,children_under18):
        '''
        Siirtymä tilaan kuollut
        '''
        employment_status = 18 # deceiced
        wage=old_wage
        pension=pension
        netto=0
        time_in_state=0
        alkanut_ansiosidonnainen=0

        return employment_status,pension,wage,time_in_state,netto

    def move_to_kht(self,pension,old_wage,age,wage_reduction,children_under3,children_under7,children_under18):
        '''
        Siirtymä kotihoidontuelle
        '''
        employment_status = 7 # kotihoidontuelle
        wage=self.get_wage(age,wage_reduction)
        pension=self.pension_accrual(age,old_wage,pension,state=7)
        
        time_in_state=0
        netto,benq=self.comp_benefits(0,old_wage,0,employment_status,time_in_state,children_under3,children_under7,children_under18,age)
        time_in_state+=self.timestep
        wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)        

        return employment_status,pension,wage,time_in_state,netto,wage_reduction,benq

    def move_to_fatherleave(self,pension,old_wage,age,wage_reduction,children_under3,children_under7,children_under18):
        '''
        Siirtymä isyysvapaalle
        '''
        self.infostate_add_child(age)
        employment_status = 6 # isyysvapaa
        time_in_state=0
        wage=self.get_wage(age,wage_reduction)
        pension=self.pension_accrual(age,old_wage,pension,state=6)
        netto,benq=self.comp_benefits(0,old_wage,0,employment_status,0,children_under3,children_under7,children_under18,age)
        time_in_state+=self.timestep        
        pinkslip=0
        wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)        
        
        return employment_status,pension,wage,time_in_state,netto,pinkslip,wage_reduction,benq

    def move_to_motherleave(self,pension,old_wage,age,wage_reduction,children_under3,children_under7,children_under18):
        '''
        Siirtymä äitiysvapaalle
        '''
        self.infostate_add_child(age)
        employment_status = 5 # äitiysvapaa
        time_in_state=0
        wage=self.get_wage(age,wage_reduction)
        pension=self.pension_accrual(age,old_wage,pension,state=5)
        netto,benq=self.comp_benefits(0,old_wage,0,employment_status,0,children_under3,children_under7,children_under18,age)
        time_in_state+=self.timestep
        pinkslip=0
        wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)        

        return employment_status,pension,wage,time_in_state,netto,pinkslip,wage_reduction,benq

    def stay_unemployed(self,employment_status,paid_pension,pension,time_in_state,toe,toekesto,wage_reduction,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,
                        children_under3,children_under7,children_under18,alkanut_ansiosidonnainen,
                        toe58,ove_paid):
        '''
        Pysy tilassa työtön (0)
        '''
        time_in_state+=self.timestep
        
        if ove_paid:
            paid_pension=paid_pension*self.elakeindeksi
            
        if age>=65:
            employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq\
                =self.move_to_retirement(pension,0,age,paid_pension,employment_status,wage_reduction,
                        unemp_after_ra,children_under3,children_under7,children_under18,all_acc=True)
        elif action == 0 or ((action == 2 or action==4) and age < self.min_retirementage) or (action == 5):
            employment_status = 0 # unchanged
            wage=self.get_wage(age,wage_reduction)
            wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)
            
            if self.porrasta_toe:
                oldtoe=self.comp_oldtoe()
            else:
                oldtoe=0
                
            if (action == 1 or action == 3) and self.unemp_limit_reemp:
                if np.random.uniform()>self.unemp_reemp_prob:
                    action = 0
                    
            
            if action == 5 and (not ove_paid) and (age>=self.min_ove_age):
                pension,paid_pension,ove_paid=self.move_to_ove(employment_status,pension,paid_pension,ove_paid,age,unemp_after_ra)

            kesto=used_unemp_benefit
            #kesto=12*21.5*used_unemp_benefit
                
            if not self.paivarahapaivia_jaljella(kesto,tyoura,age,toe58,oldtoe):
                if self.include_putki and age>=self.min_tyottputki_ika and tyoura>=self.tyohistoria_tyottputki: 
                    employment_status = 4 # siirto lisäpäiville
                    pension=self.pension_accrual(age,unempwage_basis,pension,state=4,ove_paid=ove_paid)
                    netto,benq=self.comp_benefits(0,unempwage_basis,paid_pension,employment_status,
                            used_unemp_benefit,children_under3,children_under7,children_under18,age)
                    used_unemp_benefit+=self.timestep
                else:
                    employment_status = 13 # siirto työmarkkinatuelle
                    alkanut_ansiosidonnainen=0
                    pension=self.pension_accrual(age,old_wage,pension,state=13)
                    netto,benq=self.comp_benefits(0,unempwage_basis,paid_pension,employment_status,
                            used_unemp_benefit,children_under3,children_under7,children_under18,age)
            else:
                pension=self.pension_accrual(age,unempwage_basis,pension,state=0,ove_paid=ove_paid)                
                netto,benq=self.comp_benefits(0,unempwage_basis,paid_pension,employment_status,
                        used_unemp_benefit,children_under3,children_under7,children_under18,age)
                used_unemp_benefit+=self.timestep

            if age >= self.min_retirementage:
                unemp_after_ra+=self.timestep

        elif action == 1: # 
            employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                self.move_to_work(pension,old_wage,age,time_in_state,tyoura,pinkslip,wage_reduction,
                        children_under3,children_under7,children_under18,paid_pension)
        elif action==2:
            if age >= self.min_retirementage: # ve
                employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq=\
                    self.move_to_retirement(pension,old_wage,age,paid_pension,employment_status,
                        wage_reduction,unemp_after_ra,children_under3,children_under7,children_under18,scale_acc=True)
        elif action == 3: # osatyö 50%
            employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                self.move_to_parttime(pension,old_wage,age,tyoura,time_in_state,wage_reduction,
                    children_under3,children_under7,children_under18,paid_pension)
        elif action == 4: # osatyö 50% + ve
            employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq=\
                self.move_to_oa_parttime(pension,old_wage,age,paid_pension,employment_status,
                        wage_reduction,0,children_under3,children_under7,children_under18,all_acc=True,scale_acc=True)
        elif action==11: # tk
            employment_status,pension,paid_pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq=\
                self.move_to_disab(pension,old_wage,age,wage_reduction,children_under3,children_under7,children_under18,unemp_after_ra,paid_pension,ove_paid)
            pinkslip=0
        else:
            print('error 17')  
            
        return employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,\
            benq,pinkslip,unemp_after_ra,time_in_state,tyoura,used_unemp_benefit,unempwage_basis,\
            alkanut_ansiosidonnainen,ove_paid

    def stay_tyomarkkinatuki(self,employment_status,paid_pension,pension,time_in_state,toe,toekesto,wage_reduction,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,
                        unempwage,unempwage_basis,action,age,sattuma,intage,g,
                        children_under3,children_under7,children_under18,alkanut_ansiosidonnainen,
                        toe58,ove_paid):
        '''
        Pysy tilassa työmarkkinatuki (13)
        '''
        time_in_state+=self.timestep
        
        if ove_paid:
            paid_pension=paid_pension*self.elakeindeksi
        
        if (action == 1 or action == 3) and self.unemp_limit_reemp:
            if np.random.uniform()>self.unemp_reemp_prob:
                action = 0

        if age>=65:
            employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq=\
                self.move_to_retirement(pension,0,age,paid_pension,employment_status,wage_reduction,
                        unemp_after_ra,children_under3,children_under7,children_under18,all_acc=True)
        elif action == 0 or ((action == 2 or action == 4) and age < self.min_retirementage) or (action == 5):
            employment_status = 13 # unchanged
            wage=self.get_wage(age,wage_reduction)
            
            if action == 5 and (not ove_paid) and (age>=self.min_ove_age):
                pension,paid_pension,ove_paid=self.move_to_ove(employment_status,pension,paid_pension,ove_paid,age,unemp_after_ra)

            pension=self.pension_accrual(age,wage,pension,state=13)

            netto,benq=self.comp_benefits(0,old_wage,paid_pension,employment_status,used_unemp_benefit,
                    children_under3,children_under7,children_under18,age,tyohistoria=tyoura)
            wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)
            if age >= self.min_retirementage:
                unemp_after_ra+=self.timestep
        
        elif action == 1: # 
            employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                self.move_to_work(pension,old_wage,age,time_in_state,tyoura,pinkslip,wage_reduction,
                        children_under3,children_under7,children_under18,paid_pension)
        elif action == 2:
            if age >= self.min_retirementage: # ve
                employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq=\
                    self.move_to_retirement(pension,old_wage,age,paid_pension,employment_status,
                        wage_reduction,unemp_after_ra,children_under3,children_under7,children_under18,scale_acc=True)
        elif action == 3: # osatyö 50%
            employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                self.move_to_parttime(pension,old_wage,age,tyoura,time_in_state,wage_reduction,children_under3,children_under7,children_under18,paid_pension)
        elif action == 4: # osatyö 50% + ve
            employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq=\
                self.move_to_oa_parttime(pension,old_wage,age,paid_pension,employment_status,
                        wage_reduction,unemp_after_ra,children_under3,children_under7,children_under18,all_acc=True,scale_acc=False)
        elif action==11: # tk
            employment_status,pension,paid_pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq=\
                self.move_to_disab(pension,old_wage,age,wage_reduction,children_under3,children_under7,children_under18,unemp_after_ra,paid_pension,ove_paid)
        else:
            print('error 17')        
                
        return employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,\
               benq,pinkslip,unemp_after_ra,time_in_state,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid
                
    def stay_pipeline(self,employment_status,paid_pension,pension,time_in_state,toe,toekesto,wage_reduction,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,
                        children_under3,children_under7,children_under18,alkanut_ansiosidonnainen,
                        toe58,ove_paid):
        '''
        Pysy tilassa työttömyysputki (4)
        '''
        time_in_state+=self.timestep
        if ove_paid:
            paid_pension=paid_pension*self.elakeindeksi

        if (action == 1 or action == 3) and self.unemp_limit_reemp:
            if np.random.uniform()>self.unemp_reemp_prob:
                action = 0
        
        if age>=65:
            employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq=\
                self.move_to_retirement(pension,0,age,paid_pension,employment_status,
                    wage_reduction,unemp_after_ra,children_under3,children_under7,children_under18,all_acc=True)
        elif action == 0 or ((action == 2 or action == 4) and age < self.min_retirementage) or (action == 5):
            employment_status  = 4 # unchanged
            wage=self.get_wage(age,wage_reduction)
            pension=self.pension_accrual(age,unempwage_basis,pension,state=4,ove_paid=ove_paid)
            wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)
                
            if action == 5 and (not ove_paid) and (age>=self.min_ove_age):
                pension,paid_pension,ove_paid=self.move_to_ove(employment_status,pension,paid_pension,ove_paid,age,unemp_after_ra)
                
            netto,benq=self.comp_benefits(0,unempwage_basis,paid_pension,employment_status,used_unemp_benefit,
                    children_under3,children_under7,children_under18,age,tyohistoria=tyoura)
            used_unemp_benefit+=self.timestep
            if age >= self.min_retirementage:
                unemp_after_ra+=self.timestep
                
        elif action == 1: # 
            employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                self.move_to_work(pension,old_wage,age,time_in_state,tyoura,pinkslip,wage_reduction,
                        children_under3,children_under7,children_under18,paid_pension)
        elif action==2:
            employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq=\
                self.move_to_retirement(pension,old_wage,age,paid_pension,employment_status,
                    wage_reduction,unemp_after_ra,children_under3,children_under7,children_under18,all_acc=True,scale_acc=True)
            pinkslip=0
        elif action == 3: # 
            employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                self.move_to_parttime(pension,old_wage,age,tyoura,time_in_state,wage_reduction,
                    children_under3,children_under7,children_under18,paid_pension)
        elif action == 4: # osatyö 50% + ve
            employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq=\
                self.move_to_oa_parttime(pension,old_wage,age,paid_pension,employment_status,
                        wage_reduction,unemp_after_ra,children_under3,children_under7,children_under18,all_acc=True,scale_acc=True)
        elif action==11: # tk
            employment_status,pension,paid_pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq=\
                self.move_to_disab(pension,old_wage,age,wage_reduction,children_under3,children_under7,children_under18,unemp_after_ra,paid_pension,ove_paid)
        else:
            print('error 1: ',action)
            
        return employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,\
               benq,pinkslip,unemp_after_ra,time_in_state,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid
               
    def stay_employed(self,employment_status,paid_pension,pension,time_in_state,toe,toekesto,wage_reduction,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,
                        children_under3,children_under7,children_under18,alkanut_ansiosidonnainen,
                        toe58,ove_paid):
        '''
        Pysy tilassa töissä (1)
        '''
        time_in_state+=self.timestep
        if sattuma[1]<self.pinkslip_intensity[g]:
            if age<self.min_retirementage:
                pinkslip=1
                action=1 # unemp
            else:
                pinkslip=0
                action=2 # ve
        else:
            pinkslip=0

        if ove_paid:
            paid_pension=paid_pension*self.elakeindeksi

        if action == 0 or ((action == 2 or action == 4) and age < self.min_retirementage) or action == 5:
            employment_status = 1 # unchanged
            
            if action == 5 and (not ove_paid) and (age>=self.min_ove_age):
                pension,paid_pension,ove_paid=self.move_to_ove(employment_status,pension,paid_pension,ove_paid,age,unemp_after_ra)
                            
            wage=self.get_wage(age,wage_reduction)
            wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)
            tyoura+=self.timestep
            pension=self.pension_accrual(age,wage,pension,state=1)
            netto,benq=self.comp_benefits(wage,0,paid_pension,employment_status,time_in_state,children_under3,children_under7,children_under18,age)
        elif action == 1: # työttömäksi
            employment_status,paid_pension,pension,wage,time_in_state,netto,\
                wage_reduction,used_unemp_benefit,pinkslip,benq,unemp_after_ra,unempwage_basis,alkanut_ansiosidonnainen=\
                self.move_to_unemp(pension,old_wage,age,paid_pension,toe,toekesto,pinkslip,tyoura,
                    wage_reduction,used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,
                            children_under3,children_under7,children_under18,alkanut_ansiosidonnainen,toe58)
        elif action==2:
            if age >= self.min_retirementage: # ve
                employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq=\
                    self.move_to_retirement(pension,old_wage,age,paid_pension,employment_status,wage_reduction,
                        unemp_after_ra,children_under3,children_under7,children_under18) 
        elif action == 3: # osatyö 50%
            employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                self.move_to_parttime(pension,old_wage,age,tyoura,0,wage_reduction,children_under3,children_under7,children_under18,paid_pension)
        elif action == 4: # osatyö 50% + ve
            employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq=\
                self.move_to_oa_parttime(pension,old_wage,age,paid_pension,employment_status,
                        wage_reduction,unemp_after_ra,children_under3,children_under7,children_under18,all_acc=True,scale_acc=True)
        elif action==11: # tk
            employment_status,pension,paid_pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq=\
                self.move_to_disab(pension,old_wage,age,wage_reduction,children_under3,children_under7,children_under18,unemp_after_ra,paid_pension,ove_paid)
        else:
            print('error 12')    
            
        return employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,\
              benq,pinkslip,unemp_after_ra,time_in_state,tyoura,used_unemp_benefit,unempwage_basis,\
              alkanut_ansiosidonnainen,ove_paid
           
    def stay_disabled(self,employment_status,paid_pension,pension,time_in_state,toe,toekesto,wage_reduction,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,
                        children_under3,children_under7,children_under18,alkanut_ansiosidonnainen,
                        toe58,ove_paid):
            
        '''
        Pysy tilassa työkyvytön (4)
        '''
        time_in_state+=self.timestep
        if age >= self.min_retirementage:
            employment_status = 3 # ve
        else:
            employment_status = 3 # unchanged

        paid_pension=paid_pension*self.elakeindeksi
        pension=pension*self.palkkakerroin
        wage=0
        netto,benq=self.comp_benefits(0,0,paid_pension,employment_status,0,children_under3,children_under7,children_under18,age)

        return employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,\
               benq,pinkslip,unemp_after_ra,time_in_state,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid

    def stay_retired(self,employment_status,paid_pension,pension,time_in_state,toe,toekesto,wage_reduction,
                     tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                     unempwage_basis,action,age,sattuma,intage,g,
                     children_under3,children_under7,children_under18,alkanut_ansiosidonnainen,
                     toe58,ove_paid):
        '''
        Pysy tilassa vanhuuseläke (2)
        '''
        if age >= self.min_retirementage: # ve
            time_in_state+=self.timestep

            if age>=self.max_retirementage:
                paid_pension = paid_pension+self.scale_pension(pension,age,scale=False)/self.elakeindeksi # hack
                pension=0           

            if action == 0 or action == 1 or ((action == 2 or action == 3) and age>=self.max_retirementage) or (action == 5) or (action == 4):
                employment_status = 2 # unchanged

                paid_pension=paid_pension*self.elakeindeksi
                pension=pension*self.palkkakerroin
                wage=self.get_wage(age,wage_reduction)
                #print(paid_pension,employment_status,0,children_under3,children_under7,children_under18,age)
                netto,benq=self.comp_benefits(0,0,paid_pension,employment_status,0,children_under3,children_under7,children_under18,age)
                #self.print_q(benq)
                
                wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)
            elif action == 2 and age<self.max_retirementage:
                wage=self.get_wage(age,wage_reduction)
                employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq=\
                    self.move_to_oa_parttime(pension,wage,age,paid_pension,employment_status,
                            wage_reduction,0,children_under3,children_under7,children_under18,all_acc=False,scale_acc=False)
            elif action == 3 and age<self.max_retirementage:
                wage=self.get_wage(age,wage_reduction)
                employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq=\
                    self.move_to_oa_fulltime(pension,wage,age,paid_pension,employment_status,
                            wage_reduction,0,children_under3,children_under7,children_under18,all_acc=False,scale_acc=False)
            elif action == 11:
                employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,benq=\
                    self.move_to_retdisab(pension,old_wage,age,time_in_state,paid_pension,wage_reduction,
                            children_under3,children_under7,children_under18,unemp_after_ra)
            else:
                print('error 221, action {} age {}'.format(action,age))
        else:
            # työvoiman ulkopuolella
            time_in_state+=self.timestep
            if action == 0:
                employment_status = 2 # unchanged
                wage=old_wage
                pension=pension*self.palkkakerroin
                netto,benq=self.comp_benefits(0,0,paid_pension,employment_status,time_in_state,children_under3,children_under7,children_under18,age)
                wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)
            elif action == 1: # työttömäksi
                employment_status,paid_pension,pension,wage,time_in_state,netto,\
                    wage_reduction,used_unemp_benefit,pinkslip,benq,unemp_after_ra,unempwage_basis,alkanut_ansiosidonnainen=\
                    self.move_to_unemp(pension,old_wage,age,paid_pension,toe,toekesto,0,tyoura,
                        wage_reduction,used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,
                                children_under3,children_under7,children_under18,alkanut_ansiosidonnainen,toe58)
            elif action == 2: # töihin
                wage=self.get_wage(age,wage_reduction)
                employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                    self.move_to_work(pension,wage,age,time_in_state,tyoura,pinkslip,wage_reduction,
                            children_under3,children_under7,children_under18)
            elif action == 3: # osatyö 50%
                wage=self.get_wage(age,wage_reduction)
                employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                    self.move_to_parttime(pension,wage,age,tyoura,time_in_state,wage_reduction,
                            children_under3,children_under7,children_under18)
            elif action == 11: # tk
                employment_status,pension,paid_pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq=\
                    self.move_to_disab(pension,old_wage,age,wage_reduction,
                            children_under3,children_under7,children_under18,unemp_after_ra,paid_pension,ove_paid)
            else:
                print('error 12')
                
        return employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,\
               benq,pinkslip,unemp_after_ra,time_in_state,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid

    def stay_motherleave(self,employment_status,paid_pension,pension,time_in_state,toe,toekesto,wage_reduction,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,
                        children_under3,children_under7,children_under18,alkanut_ansiosidonnainen,
                        toe58,ove_paid):
        '''
        Pysy tilassa äitiysvapaa (5)
        '''
        #exit_prb=np.random.uniform(0,80_000)
        
        if time_in_state>=self.aitiysvapaa_kesto:
            pinkslip=0
            if action == 0:
                employment_status,paid_pension,pension,wage,time_in_state,netto,\
                    wage_reduction,used_unemp_benefit,pinkslip,benq,unemp_after_ra,unempwage_basis,alkanut_ansiosidonnainen=\
                    self.move_to_unemp(pension,old_wage,age,paid_pension,toe,toekesto,pinkslip,tyoura,
                        wage_reduction,used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,
                        children_under3,children_under7,children_under18,alkanut_ansiosidonnainen,toe58)
            elif action == 1 or action == 2: # 
                employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                    self.move_to_work(pension,old_wage,age,time_in_state,tyoura,pinkslip,wage_reduction,
                        children_under3,children_under7,children_under18,paid_pension)
            elif action == 3: # 
                employment_status,pension,wage,time_in_state,netto,wage_reduction,benq=\
                    self.move_to_kht(pension,old_wage,age,wage_reduction,children_under3,children_under7,children_under18)
            elif action == 4 or action == 5:
                employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                    self.move_to_parttime(pension,old_wage,age,tyoura,time_in_state,wage_reduction,
                        children_under3,children_under7,children_under18,paid_pension)
            elif action==11: # tk
                employment_status,pension,paid_pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq=\
                    self.move_to_disab(pension,old_wage,age,wage_reduction,children_under3,children_under7,children_under18,unemp_after_ra,paid_pension,ove_paid)
            else:
                print('Error 21')
        else:
            pension=self.pension_accrual(age,old_wage,pension,state=5)
            wage=self.get_wage(age,wage_reduction)
            netto,benq=self.comp_benefits(0,old_wage,0,employment_status,0,children_under3,children_under7,children_under18,age)
            time_in_state+=self.timestep
            wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)
                
        return employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,\
               benq,pinkslip,unemp_after_ra,time_in_state,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid

    def stay_fatherleave(self,employment_status,paid_pension,pension,time_in_state,toe,toekesto,wage_reduction,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,
                        children_under3,children_under7,children_under18,alkanut_ansiosidonnainen,
                        toe58,ove_paid):
        '''
        Pysy tilassa isyysvapaa (6)
        '''
        if time_in_state>=self.isyysvapaa_kesto:
            pinkslip=0
            if action == 0:
                employment_status,paid_pension,pension,wage,time_in_state,netto,\
                    wage_reduction,used_unemp_benefit,pinkslip,benq,unemp_after_ra,unempwage_basis,alkanut_ansiosidonnainen=\
                    self.move_to_unemp(pension,old_wage,age,paid_pension,toe,toekesto,pinkslip,tyoura,
                        wage_reduction,used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,
                        children_under3,children_under7,children_under18,alkanut_ansiosidonnainen,toe58)
            elif action == 1 or action == 2: # 
                # ei vaikutusta palkkaan
                employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                    self.move_to_work(pension,old_wage,age,0,tyoura,pinkslip,wage_reduction,
                        children_under3,children_under7,children_under18,paid_pension)
            elif action == 3: # 
                employment_status,pension,wage,time_in_state,netto,wage_reduction,benq=\
                    self.move_to_kht(pension,old_wage,age,wage_reduction,children_under3,children_under7,children_under18)
            elif action == 4 or action == 5:
                employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                    self.move_to_parttime(pension,old_wage,age,tyoura,0,wage_reduction,
                        children_under3,children_under7,children_under18,paid_pension)
            elif action==11: # tk
                employment_status,pension,paid_pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq=\
                    self.move_to_disab(pension,old_wage,age,wage_reduction,children_under3,children_under7,children_under18,unemp_after_ra,paid_pension,ove_paid)
            else:
                print('Error 23')
        else:
            pension=self.pension_accrual(age,old_wage,pension,state=6)
            wage=self.get_wage(age,wage_reduction)
            netto,benq=self.comp_benefits(0,old_wage,0,employment_status,0,children_under3,children_under7,children_under18,age)
            time_in_state+=self.timestep
            wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)

        return employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,\
               benq,pinkslip,unemp_after_ra,time_in_state,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid

    def stay_khh(self,employment_status,paid_pension,pension,time_in_state,toe,toekesto,wage_reduction,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,
                        children_under3,children_under7,children_under18,alkanut_ansiosidonnainen,
                        toe58,ove_paid):
        '''
        Pysy tilassa kotihoidontuki (0)
        '''
        time_in_state+=self.timestep

        if (action == 0) and (time_in_state>self.kht_kesto or children_under3<1): # jos etuus loppuu, siirtymä satunnaisesti
            s=np.random.uniform()
            if s<1/3:
                action=1
            elif s<2/3:
                action=2
            else:
                action=3

        if age >= self.min_retirementage: # ve
            employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq=\
                self.move_to_retirement(pension,old_wage,age,paid_pension,employment_status,
                    wage_reduction,unemp_after_ra,children_under3,children_under7,children_under18,all_acc=True)
        elif (action == 0) and ((time_in_state<=self.kht_kesto and children_under3>0) or self.perustulo): # jos perustulo, ei aikarajoitetta
            employment_status  = 7 # stay
            wage=self.get_wage(age,wage_reduction)
            pension=self.pension_accrual(age,wage,pension,state=7)
            netto,benq=self.comp_benefits(0,old_wage,paid_pension,employment_status,time_in_state,children_under3,children_under7,children_under18,age)
            wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)
        elif action == 1 or action == 4: # 
            pinkslip=0
            employment_status,paid_pension,pension,wage,time_in_state,netto,\
                wage_reduction,used_unemp_benefit,pinkslip,benq,unemp_after_ra,unempwage_basis,alkanut_ansiosidonnainen=\
                self.move_to_unemp(pension,old_wage,age,paid_pension,toe,toekesto,pinkslip,tyoura,
                    wage_reduction,used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,
                    children_under3,children_under7,children_under18,alkanut_ansiosidonnainen,toe58)
        elif action == 2 or action == 5: # 
            wage=self.get_wage(age,wage_reduction)
            employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                self.move_to_work(pension,wage,age,time_in_state,tyoura,pinkslip,wage_reduction,
                    children_under3,children_under7,children_under18,paid_pension)
        elif action == 3: # 
            wage=self.get_wage(age,wage_reduction)        
            employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                self.move_to_parttime(pension,wage,age,tyoura,time_in_state,wage_reduction,
                    children_under3,children_under7,children_under18,paid_pension)
        elif action==11: # tk
            employment_status,pension,paid_pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq=\
                self.move_to_disab(pension,old_wage,age,wage_reduction,children_under3,children_under7,children_under18,unemp_after_ra,paid_pension,ove_paid)
        else:
            print('Error 25')
            
        return employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,\
               benq,pinkslip,unemp_after_ra,time_in_state,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid

    def stay_student(self,employment_status,paid_pension,pension,time_in_state,toe,toekesto,wage_reduction,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,
                        children_under3,children_under7,children_under18,alkanut_ansiosidonnainen,
                        toe58,ove_paid):
        '''
        Pysy tilassa opiskelija (12)
        '''

        pinkslip=0
        if sattuma[5]>=self.student_outrate[intage,g]:
            employment_status = 12 # unchanged
            time_in_state+=self.timestep
            wage=self.get_wage(age,wage_reduction)
            pension=self.pension_accrual(age,0,pension,state=12)
            netto,benq=self.comp_benefits(0,0,paid_pension,employment_status,time_in_state,children_under3,children_under7,children_under18,age,tyohistoria=tyoura)
            # opiskelu parantaa tuloja
            wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)
        elif action == 0 or action == 1: # 
            wage=self.get_wage(age,wage_reduction)
            employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                self.move_to_work(pension,wage,age,0,tyoura,pinkslip,wage_reduction,
                    children_under3,children_under7,children_under18,paid_pension)
        elif action == 2:
            employment_status,paid_pension,pension,wage,time_in_state,netto,\
                wage_reduction,used_unemp_benefit,pinkslip,benq,unemp_after_ra,unempwage_basis,alkanut_ansiosidonnainen=\
                self.move_to_unemp(pension,old_wage,age,paid_pension,toe,toekesto,pinkslip,tyoura,
                    wage_reduction,used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,
                    children_under3,children_under7,children_under18,alkanut_ansiosidonnainen,toe58)
        elif action == 3 or action == 4 or action == 5:
            wage=self.get_wage(age,wage_reduction)            
            employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                self.move_to_parttime(pension,wage,age,tyoura,time_in_state,wage_reduction,
                    children_under3,children_under7,children_under18,paid_pension)
        elif action == 11: # tk
            employment_status,pension,paid_pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq=\
                self.move_to_disab(pension,old_wage,age,wage_reduction,children_under3,children_under7,children_under18,unemp_after_ra,paid_pension,ove_paid)
        else:
            print('error 29: ',action)
            
        return employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,\
               benq,pinkslip,unemp_after_ra,time_in_state,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid

    def stay_oa_parttime(self,employment_status,paid_pension,pension,time_in_state,toe,toekesto,wage_reduction,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,
                        children_under3,children_under7,children_under18,alkanut_ansiosidonnainen,
                        toe58,ove_paid):
        '''
        Pysy tilassa ve+(osa-aikatyö) (0)
        '''

        time_in_state+=self.timestep
        
        # irtisanominen
        if sattuma[1]<self.pinkslip_intensity[g]:
            action=2 # ve:lle

        if age>=self.max_retirementage:
            employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq=\
                self.move_to_retirement(self,pension,0,age,paid_pension,employment_status,
                    wage_reduction,unemp_after_ra,children_under3,children_under7,children_under18,all_acc=True,scale_acc=False)
        elif action == 0 or action == 1: # jatkaa osa-aikatöissä, ei voi saada työttömyyspäivärahaa
            employment_status = 8 # unchanged
            wage=self.get_wage(age,wage_reduction)
            parttimewage=0.5*wage
            pension=self.pension_accrual(age,parttimewage,pension,state=employment_status)
            paid_pension=paid_pension*self.elakeindeksi
            netto,benq=self.comp_benefits(parttimewage,0,paid_pension,employment_status,time_in_state,children_under3,children_under7,children_under18,age)
            wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)
        elif action==2 or action==3: # jatkaa täysin töissä, ei voi saada työttömyyspäivärahaa
            wage=self.get_wage(age,wage_reduction)
            #employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,benq=\
            #    self.move_to_oa_fulltime(pension,wage,age,0,paid_pension,wage_reduction,children_under3,children_under7,children_under18)
            employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq=\
                self.move_to_oa_fulltime(pension,wage,age,paid_pension,employment_status,
                        wage_reduction,0,children_under3,children_under7,children_under18,all_acc=False,scale_acc=False)
        elif action == 4 or action == 5: # eläkkeelle, eläkeaikana karttunutta eläkettä ei vielä maksuun
            wage=self.get_wage(age,wage_reduction)
            employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq=\
                self.move_to_retirement(pension,wage,age,paid_pension,employment_status,
                    wage_reduction,0,children_under3,children_under7,children_under18,all_acc=False,scale_acc=False)
        elif action == 11:
            # no more working, move to "disab" with no change in paid_pension
            employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,benq=\
                self.move_to_retdisab(pension,0,age,time_in_state,paid_pension,wage_reduction,children_under3,children_under7,children_under18,unemp_after_ra)
        else:
            print('error 14, action {} age {}'.format(action,age))

        return employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,\
               benq,pinkslip,unemp_after_ra,time_in_state,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid

    def stay_oa_fulltime(self,employment_status,paid_pension,pension,time_in_state,toe,toekesto,wage_reduction,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,
                        children_under3,children_under7,children_under18,alkanut_ansiosidonnainen,
                        toe58,ove_paid):
        '''
        Pysy tilassa ve+työ (0)
        '''

        time_in_state+=self.timestep        
        # irtisanominen
        if sattuma[1]<self.pinkslip_intensity[g]:
            action=2 # ve:lle

        if age>=self.max_retirementage:
            employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq=\
                self.move_to_retirement(self,pension,old_wage,age,paid_pension,employment_status,
                    wage_reduction,unemp_after_ra,children_under3,children_under7,children_under18,all_acc=True,scale_acc=False)
        elif action == 0 or action == 1: # jatkaa töissä, ei voi saada työttömyyspäivärahaa
            employment_status = 9 # unchanged
            wage=self.get_wage(age,wage_reduction)
            pension=self.pension_accrual(age,wage,pension,state=employment_status)
            
            paid_pension=paid_pension*self.elakeindeksi
            netto,benq=self.comp_benefits(wage,0,paid_pension,employment_status,time_in_state,children_under3,children_under7,children_under18,age)
            wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)
        elif action == 2: # jatkaa osa-aikatöissä, ei voi saada työttömyyspäivärahaa
            employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq=\
                self.move_to_oa_parttime(pension,old_wage,age,paid_pension,employment_status,
                        wage_reduction,0,children_under3,children_under7,children_under18,all_acc=False,scale_acc=False)
        elif action==3 or action == 4 or action == 5: # eläkkeelle, eläkeaikana karttunutta eläkettä ei vielä maksuun
            wage=self.get_wage(age,wage_reduction)
            employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq=\
                self.move_to_retirement(pension,wage,age,paid_pension,employment_status,
                    wage_reduction,0,children_under3,children_under7,children_under18,all_acc=False,scale_acc=False)
        elif action == 11:
            # no more working, move to "disab" with no change in paid_pension
            employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,benq=\
                self.move_to_retdisab(pension,old_wage,age,time_in_state,paid_pension,wage_reduction,
                    children_under3,children_under7,children_under18,unemp_after_ra)
        else:
            print('error 14, action {} age {}'.format(action,age))
            
        return employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,\
               benq,pinkslip,unemp_after_ra,time_in_state,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid

    def stay_parttime(self,employment_status,paid_pension,pension,time_in_state,toe,toekesto,wage_reduction,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,
                        children_under3,children_under7,children_under18,alkanut_ansiosidonnainen,
                        toe58,ove_paid):
        '''
        Pysy tilassa osa-aikatyö (0)
        '''

        time_in_state+=self.timestep
        
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
            paid_pension=paid_pension*self.elakeindeksi

        if action == 0 or ((action == 2 or action == 4) and age < self.min_retirementage) or (action == 5):
            employment_status = 10 # unchanged
            wage=self.get_wage(age,wage_reduction)
            parttimewage=0.5*wage
            tyoura+=self.timestep
            
            if action == 5 and (not ove_paid) and (age>=self.min_ove_age):
                pension,paid_pension,ove_paid=self.move_to_ove(employment_status,pension,paid_pension,ove_paid,age,unemp_after_ra)
            
            pension=self.pension_accrual(age,parttimewage,pension,state=10)
            netto,benq=self.comp_benefits(parttimewage,0,paid_pension,employment_status,time_in_state,children_under3,children_under7,children_under18,age)
            wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)
        elif action == 1: # työttömäksi
            employment_status,paid_pension,pension,wage,time_in_state,netto,\
                wage_reduction,used_unemp_benefit,pinkslip,benq,unemp_after_ra,unempwage_basis,alkanut_ansiosidonnainen=\
                self.move_to_unemp(pension,old_wage,age,paid_pension,toe,toekesto,pinkslip,tyoura,
                    wage_reduction,used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,
                    children_under3,children_under7,children_under18,alkanut_ansiosidonnainen,toe58)
        elif action==2:
            if age >= self.min_retirementage: # ve
                employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq=\
                    self.move_to_retirement(pension,old_wage,age,paid_pension,employment_status,
                        wage_reduction,unemp_after_ra,children_under3,children_under7,children_under18,all_acc=True)
        elif action==3:
            wage=self.get_wage(age,wage_reduction)
            employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                self.move_to_work(pension,wage,age,0,tyoura,pinkslip,wage_reduction,
                    children_under3,children_under7,children_under18,paid_pension)
        elif action==4: # move to oa_work
            wage=self.get_wage(age,wage_reduction)
            employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq=\
                self.move_to_oa_parttime(pension,wage,age,paid_pension,employment_status,
                        wage_reduction,unemp_after_ra,children_under3,children_under7,children_under18,all_acc=True,scale_acc=True)
#             employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
#                 self.move_to_work(pension,wage,age,0,tyoura,pinkslip,wage_reduction,
#                     children_under3,children_under7,children_under18,paid_pension)
        elif action==11: # tk
            employment_status,pension,paid_pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq=\
                self.move_to_disab(pension,old_wage,age,wage_reduction,children_under3,children_under7,children_under18,unemp_after_ra,paid_pension,ove_paid)
        else:
            print('error 12')
            
        return employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,\
               benq,pinkslip,unemp_after_ra,time_in_state,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid

    def stay_outsider(self,employment_status,paid_pension,pension,time_in_state,toe,toekesto,wage_reduction,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,
                        children_under3,children_under7,children_under18,alkanut_ansiosidonnainen,
                        toe58,ove_paid):
        '''
        Pysy tilassa työvoiman ulkopuolella (11)
        '''

        if age>=self.min_retirementage:
            employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq=\
                self.move_to_retirement(pension,old_wage,age,paid_pension,employment_status,
                wage_reduction,unemp_after_ra,children_under3,children_under7,children_under18,all_acc=True)
        elif sattuma[5]>=self.outsider_outrate[intage,g]:
            time_in_state+=self.timestep
            employment_status = 11 # unchanged
            wage=self.get_wage(age,wage_reduction)
            pension=self.pension_accrual(age,wage,pension,state=11)
            netto,benq=self.comp_benefits(0,old_wage,paid_pension,employment_status,time_in_state,children_under3,children_under7,children_under18,age,tyohistoria=tyoura)
            wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)
        elif action == 0 or action == 1: # 
            wage=self.get_wage(age,wage_reduction)
            employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                self.move_to_work(pension,wage,age,time_in_state,tyoura,pinkslip,wage_reduction,
                    children_under3,children_under7,children_under18,paid_pension)
        elif action == 2: # 
            pinkslip=0
            employment_status,paid_pension,pension,wage,time_in_state,netto,\
                wage_reduction,used_unemp_benefit,pinkslip,benq,unemp_after_ra,unempwage_basis,alkanut_ansiosidonnainen=\
                self.move_to_unemp(pension,old_wage,age,paid_pension,toe,toekesto,pinkslip,tyoura,wage_reduction,
                    used_unemp_benefit,unemp_after_ra,unempwage,unempwage_basis,children_under3,
                    children_under7,children_under18,alkanut_ansiosidonnainen,toe58)
        elif action == 3 or action == 4 or action == 5: # 
            wage=self.get_wage(age,wage_reduction)
            employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                self.move_to_parttime(pension,wage,age,tyoura,time_in_state,wage_reduction,
                    children_under3,children_under7,children_under18,paid_pension)
        elif action == 11: # tk
            employment_status,pension,paid_pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq=\
                self.move_to_disab(pension,old_wage,age,wage_reduction,children_under3,children_under7,children_under18,unemp_after_ra,paid_pension,ove_paid)
            pinkslip=0
        else:
            print('error 19: ',action)

        return employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,\
               benq,pinkslip,unemp_after_ra,time_in_state,tyoura,used_unemp_benefit,unempwage_basis,\
               alkanut_ansiosidonnainen,ove_paid

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
        elif state in set([7,2,3]): # kotihoidontuki tai ve tai tk
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
        elif state in set([7,2,3]): # kotihoidontuki tai ve
            #wage_reduction=max(0,1.0-((1-self.salary_const)**self.timestep)*(1-wage_reduction))
            wage_reduction=max(0,1.0-(1-self.salary_const)*(1-wage_reduction))
        elif state in set([14]): # ei muutosta
            wage_reduction=wage_reduction
        else: # ylivuoto, ei tiloja
            wage_reduction=wage_reduction
        
        return wage_reduction
        
    def step(self, action, dynprog=False, debug=False):
        '''
        Open AI interfacen mukainen step-funktio, joka tekee askeleen eteenpäin
        toiminnon action mukaan 

        Keskeinen funktio simuloinnissa
        '''
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        employment_status,g,pension,old_wage,age,time_in_state,paid_pension,pinkslip,toe,toekesto,\
            tyoura,used_unemp_benefit,wage_reduction,unemp_after_ra,\
            unempwage,unempwage_basis,prefnoise,children_under3,children_under7,children_under18,\
            unemp_left,alkanut_ansiosidonnainen,toe58,ove_paid,jasen\
                =self.state_decode(self.state)
            
        intage=int(np.floor(age))
        t=int((age-self.min_age)/self.timestep)
        moved=False
        use_func=True
        
        if self.randomness:
            # kaikki satunnaisuus kerralla
            sattuma = np.random.uniform(size=7)
            
            # siirtymät
            move_prob=self.disability_intensity[intage,g]+self.birth_intensity[intage,g]+self.student_inrate[intage,g]+self.outsider_inrate[intage,g]

            if sattuma[0]<move_prob:
            
                s1=self.disability_intensity[intage,g]
                s2=s1+self.birth_intensity[intage,g]
                s3=s2+self.student_inrate[intage,g]
                #s4=s3+self.outsider_inrate[intage,g]
            
                # tk-alkavuus, siisti kuntoon!
                if sattuma[2]<s1/move_prob: # age<self.min_retirementage and 
                    action=11 # disability
                elif sattuma[2]<s2/move_prob:
                    if self.infostat_can_have_children(age) and employment_status!=3: # lasten väli vähintään vuosi, ei työkyvyttömyyseläkkeellä
                        if g>2: # naiset
                            employment_status,pension,wage,time_in_state,netto,pinkslip,wage_reduction,benq=\
                                self.move_to_motherleave(pension,old_wage,age,wage_reduction,children_under3,children_under7,children_under18)
                            pinkslip=0
                            moved=True
                        else: # miehet
                            # ikä valittu äidin iän mukaan. oikeastaan tämä ei mene ihan oikein miehille
                            if sattuma[4]<0.35: # orig 0.5
                                employment_status,pension,wage,time_in_state,netto,pinkslip,wage_reduction,benq=\
                                    self.move_to_fatherleave(pension,old_wage,age,wage_reduction,children_under3,children_under7,children_under18)
                                moved=True
                elif sattuma[2]<s3/move_prob:
                    if employment_status not in set([2,3,5,6,7,8,9,11,12,18]): # and False:
                        employment_status,pension,wage,time_in_state,netto,pinkslip,wage_reduction,benq=\
                            self.move_to_student(pension,old_wage,age,time_in_state,tyoura,pinkslip,wage_reduction,
                                children_under3,children_under7,children_under18,paid_pension)
                        moved=True
                #elif sattuma[2]<s4/move_prob: # and False:
                else:
                    if employment_status not in set([2,3,5,6,7,8,9,11,12,18]):
                        employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,pinkslip,benq=\
                            self.move_to_outsider(pension,old_wage,age,pinkslip,wage_reduction,
                                children_under3,children_under7,children_under18,paid_pension)
                        moved=True

            # voi aiheuttaa epästabiilisuutta
            if sattuma[3]<self.mort_intensity[intage,g] and self.include_mort: 
                employment_status,pension,wage,time_in_state,netto=self.move_to_deceiced(pension,old_wage,age,children_under3,children_under7,children_under18)
        else:
            # tn ei ole koskaan alle rajan, jos tämä on 1
            sattuma = np.ones(7)
            
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
                            time_in_state,paid_pension,pinkslip,toe,tyoura,nextwage,
                            used_unemp_benefit,wage_reduction,unemp_after_ra,unempwage,unempwage_basis,
                            children_under3,children_under7,children_under18,
                            0,alkanut_ansiosidonnainen,toe58,ove_paid,0,toekesto,prefnoise)
                            
            netto,benq=self.comp_benefits(0,0,0,14,0,children_under3,children_under7,children_under18,age,retq=True)
                            
            reward=0
            equivalent=0
            return np.array(self.state), reward, done, benq
        elif age>=self.max_retirementage:
            employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,ove_paid,benq\
                =self.move_to_retirement(pension,0,age,paid_pension,employment_status,wage_reduction,unemp_after_ra,children_under3,children_under7,children_under18,all_acc=True)
        else:
             if not moved:
                # hoidetaan tilasiirtymät ja -pysymiset alirutiineilla, joita kutsutaan mäppäämällä tila funktioksi,
                # jota sitten kutsutaan
                map_stays={0: self.stay_unemployed,  1: self.stay_employed,         2: self.stay_retired,       3: self.stay_disabled,
                           4: self.stay_pipeline,    5: self.stay_motherleave,      6: self.stay_fatherleave,   7: self.stay_khh,
                           8: self.stay_oa_parttime, 9: self.stay_oa_fulltime,     10: self.stay_parttime,     11: self.stay_outsider,
                           12: self.stay_student,   13: self.stay_tyomarkkinatuki}
                employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,benq,pinkslip,\
                unemp_after_ra,time_in_state,tyoura,used_unemp_benefit,unempwage_basis,alkanut_ansiosidonnainen,ove_paid\
                    = map_stays[employment_status](employment_status,paid_pension,pension,time_in_state,toe,toekesto,wage_reduction,
                                   tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,unempwage_basis,
                                   action,age,sattuma,intage,g,children_under3,children_under7,children_under18,
                                   alkanut_ansiosidonnainen,toe58,ove_paid)

        done = age >= self.max_age
        done = bool(done)
        
        # after this, preparing for the next step
        age=age+self.timestep
        
        toe58=self.check_toe58(age,toe,tyoura,toe58)
        
        self.update_infostate(t,int(employment_status),wage,unempwage_basis)
        toe,toekesto,unempwage,children_under3,children_under7,children_under18=self.comp_infostats(age)
        work={1,10}
        retired={2,8,9}
        if employment_status in work and self.tyossaoloehto(toe,tyoura,age):
            used_unemp_benefit=0
            alkanut_ansiosidonnainen=0
            #if alkanut_ansiosidonnainen>0:
            #    if not self.infostat_check_aareset(age):
            #        alkanut_ansiosidonnainen=0
        elif employment_status in retired:
            alkanut_ansiosidonnainen=0
            
        if self.porrasta_toe and (employment_status in set([0,4]) or alkanut_ansiosidonnainen>0):
            old_toe=self.comp_oldtoe()
            pvr_jaljella=self.comp_unempdays_left(used_unemp_benefit,tyoura,age,toe,employment_status,alkanut_ansiosidonnainen,toe58,old_toe,printti=False)
        else:
            pvr_jaljella=self.comp_unempdays_left(used_unemp_benefit,tyoura,age,toe,employment_status,alkanut_ansiosidonnainen,toe58,toekesto)
            
        kassanjasenyys=self.get_kassanjasenyys()
        
        #self.render_infostate()

        if not done:
            reward,equivalent = self.log_utility(netto,int(employment_status),age,g=g,pinkslip=pinkslip)
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            
            paid_pension += self.elinaikakerroin*pension # ei ihan oikein! lykkäyskorotus puuttuu, optimoijan pitäisi huomata, joten ei vaikutusta?
            pension=0
            
            netto,benq=self.comp_benefits(0,old_wage,paid_pension,employment_status,time_in_state,children_under3,children_under7,children_under18,age)
            if employment_status in set([2,3,8,9]): # retired
                # pitäisi laskea tarkemmin, ei huomioi eläkkeen indeksointia!
                if self.include_npv_mort:
                    npv,npv0,npv_pension=self.comp_npv_simulation(g)
                    reward,equivalent = self.log_utility(netto,employment_status,age,pinkslip=0)
                    reward*=npv
                else:
                    npv,npv0,npv_pension=self.npv[g],self.npv0[g],self.npv_pension[g]
                    reward,equivalent = self.log_utility(netto,employment_status,age,pinkslip=0)
                    reward*=self.npv[g]
                
                # npv0 is undiscounted
                benq=self.scale_q(npv,npv0,npv_pension,benq)                
            else:
                # giving up the pension
                reward = 0.0 #-self.npv[g]*self.log_utility(netto,employment_status,age)
                equivalent = 0.0
                
            pinkslip=0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0
            equivalent = 0.0

        # seuraava palkka tiedoksi valuaatioapproksimaattorille
        if employment_status in set([3,14]):
            next_wage=0
        else:
            next_wage=self.get_wage(age,wage_reduction)
        
        self.state = self.state_encode(employment_status,g,pension,wage,age,time_in_state,
                                paid_pension,pinkslip,toe,toekesto,tyoura,next_wage,used_unemp_benefit,
                                wage_reduction,unemp_after_ra,unempwage,unempwage_basis,
                                children_under3,children_under7,children_under18,
                                pvr_jaljella,alkanut_ansiosidonnainen,toe58,ove_paid,kassanjasenyys,
                                prefnoise)

        if self.plotdebug:
            self.render(done=done,reward=reward, netto=netto)
            #if employment_status==0:
            #    print(benq)
            #self.render_infostate()

        benq['eq']=equivalent
        return np.array(self.state), reward, done, benq
        
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
            self.wage_initial_reduction=0.010 # työttömäksi siirtymisestä tuleva alennus tuleviin palkkoihin
            
            self.men_kappa_fulltime=0.730 # 0.675 #0.682 # 0.670 # vapaa-ajan menetyksestä rangaistus miehille
            self.men_mu_scale=0.035 #18 # 0.14 # 0.30 # 0.16 # how much penalty is associated with work increase with age after mu_age
            self.men_mu_age=self.min_retirementage-2.0 # P.O. 60??
            self.men_kappa_osaaika_young=0.485 # vapaa-ajan menetyksestä rangaistus miehille osa-aikatyön teosta, suhteessa kokoaikaan
            self.men_kappa_osaaika_middle=0.420 # vapaa-ajan menetyksestä rangaistus miehille osa-aikatyön teosta, suhteessa kokoaikaan
            self.men_kappa_osaaika_old=0.370 # vapaa-ajan menetyksestä rangaistus miehille osa-aikatyön teosta, suhteessa kokoaikaan, alle 35v
            self.men_kappa_hoitovapaa=0.10 # hyöty hoitovapaalla olosta
            self.men_kappa_ve=0.250 # 0.03 # ehkä 0.10?
            if self.perustulo:
                self.men_kappa_pinkslip_young=0.10
                self.men_kappa_pinkslip_middle=0.09
                self.men_kappa_pinkslip_elderly=0.12
            else:
                self.men_kappa_pinkslip_young=0.05
                self.men_kappa_pinkslip_middle=0.09
                self.men_kappa_pinkslip_elderly=0.12
            
            self.women_kappa_fulltime=0.620 # 0.605 # 0.640 # 0.620 # 0.610 # vapaa-ajan menetyksestä rangaistus naisille
            self.women_mu_scale=0.030 # 0.25 # how much penalty is associated with work increase with age after mu_age
            self.women_mu_age=self.min_retirementage-1.5 # 61 #5 P.O. 60??
            self.women_kappa_osaaika_young=0.390
            self.women_kappa_osaaika_middle=0.360
            self.women_kappa_osaaika_old=0.390
            self.women_kappa_hoitovapaa=0.10 # 0.08
            self.women_kappa_ve=0.280 # 0.03 # ehkä 0.10?
            if self.perustulo:
                self.women_kappa_pinkslip_young=0.10
                self.women_kappa_pinkslip_middle=0.14
                self.women_kappa_pinkslip_elderly=0.14
            else:
                self.women_kappa_pinkslip_young=0.03
                self.women_kappa_pinkslip_middle=0.06
                self.women_kappa_pinkslip_elderly=0.17

#     def log_utility_default_params(self):
#         # paljonko työstä poissaolo vaikuttaa palkkaan
#     
#         if self.include_mort:
#             if self.perustulo:
#                 self.men_perustulo_extra=0.10
#                 self.women_perustulo_extra=0.10
#             else:
#                 self.men_perustulo_extra=0.0
#                 self.women_perustulo_extra=0.0
# 
#             self.salary_const=0.07*self.timestep
#             self.salary_const_up=0.04*self.timestep # työssäolo palauttaa ansioita tämän verran vuodessa
#             self.salary_const_student=0.05*self.timestep # opiskelu nostaa leikkausta tämän verran vuodessa
#             self.wage_initial_reduction=0.5*self.salary_const
#             self.men_kappa_fulltime=0.705 # 0.635 # 0.665
#             self.men_mu_scale=0.130 # 0.14 # 0.30 # 0.16 # how much penalty is associated with work increase with age after mu_age
#             self.men_mu_age=59.0 # P.O. 60??
#             self.men_kappa_osaaika=0.365
#             self.men_kappa_hoitovapaa=0.05
#             self.men_kappa_ve=0.09 # ehkä 0.10?
#             self.men_kappa_pinkslip_young=0.035
#             self.men_kappa_pinkslip=0.05
#             self.women_kappa_fulltime=0.655 # 0.605 # 0.58
#             self.women_mu_scale=0.130 # 0.25 # 0.25 # 0.17 # how much penalty is associated with work increase with age after mu_age
#             self.women_mu_age=59.0 # 61 # P.O. 60??
#             self.women_kappa_osaaika=0.325
#             self.women_kappa_hoitovapaa=0.35
#             self.women_kappa_ve=0.09 # ehkä 0.10?
#             self.women_kappa_pinkslip_young=0.035
#             self.women_kappa_pinkslip=0.05
#         else:
#             
#             self.salary_const=0.05*self.timestep # työttömyydestä palkka alenee tämän verran aika-askeleessa
#             self.salary_const_up=0.03*self.timestep # työssäolo palauttaa ansioita tämän verran vuodessa
#             self.salary_const_student=0.10*self.timestep # opiskelu pienentää leikkausta tämän verran vuodessa
#             self.wage_initial_reduction=0.025 # työttömäksi siirtymisestä tuleva alennus tuleviin palkkoihin
#             
#             self.men_kappa_fulltime=0.734 # vapaa-ajan menetyksestä rangaistus miehille
#             self.men_mu_scale=0.13 #18 # 0.14 # 0.30 # 0.16 # how much penalty is associated with work increase with age after mu_age
#             self.men_mu_age=self.min_retirementage # P.O. 60??
#             self.men_kappa_osaaika=0.455 # vapaa-ajan menetyksestä rangaistus miehille osa-aikatyön teosta, suhteessa kokoaikaan
#             self.men_kappa_osaaika_old=0.410 # vapaa-ajan menetyksestä rangaistus miehille osa-aikatyön teosta, suhteessa kokoaikaan, alle 35v
#             self.men_kappa_hoitovapaa=0.03 # hyöty hoitovapaalla olosta
#             self.men_kappa_ve=0.08 # ehkä 0.10?
#             if self.perustulo:
#                 self.men_kappa_pinkslip_young=0.10 
#                 self.men_kappa_pinkslip=0.12
#             else:
#                 self.men_kappa_pinkslip_young=0.10
#                 self.men_kappa_pinkslip=0.12
#             
#             self.women_kappa_fulltime=0.675 # vapaa-ajan menetyksestä rangaistus naisille
#             self.women_mu_scale=0.13 # 0.25 # 0.25 # 0.17 # how much penalty is associated with work increase with age after mu_age
#             self.women_mu_age=self.min_retirementage # 61 #5 P.O. 60??
#             self.women_kappa_osaaika=0.420
#             self.women_kappa_osaaika_old=0.460
#             self.women_kappa_hoitovapaa=0.06
#             self.women_kappa_ve=0.08 # ehkä 0.10?
#             if self.perustulo:
#                 self.women_kappa_pinkslip_young=0.10
#                 self.women_kappa_pinkslip=0.17
#             else:
#                 self.women_kappa_pinkslip_young=0.10
#                 self.women_kappa_pinkslip=0.19

#     def log_utility_default_params_KAK(self):
#         if self.include_mort:
#             if self.perustulo:
#                 self.men_kappa_fulltime=0.620 # 0.635 # 0.665
#                 self.men_mu_scale=0.11 # 0.14 # 0.30 # 0.16 # how much penalty is associated with work increase with age after mu_age
#                 self.men_mu_age=60 # P.O. 60??
#                 self.men_kappa_osaaika=0.46
#                 self.men_kappa_hoitovapaa=0.015
#                 self.men_kappa_ve=0.22 # ehkä 0.10?
#                 self.women_kappa_fulltime=0.615 # 0.605 # 0.58
#                 self.women_mu_scale=0.09 # 0.25 # 0.25 # 0.17 # how much penalty is associated with work increase with age after mu_age
#                 self.women_mu_age=60 # 61 # P.O. 60??
#                 self.women_kappa_osaaika=0.40
#                 self.women_kappa_hoitovapaa=0.05
#                 self.women_kappa_ve=0.27 # ehkä 0.10?
#                 self.kappa_pinkslip=0.06
#                 self.kappa_pinkslip_young=0.06   
#             else:         
#                 self.men_kappa_fulltime=0.620 # 0.635 # 0.665
#                 self.men_mu_scale=0.11 # 0.14 # 0.30 # 0.16 # how much penalty is associated with work increase with age after mu_age
#                 self.men_mu_age=60 # P.O. 60??
#                 self.men_kappa_osaaika=0.46
#                 self.men_kappa_hoitovapaa=0.015
#                 self.men_kappa_ve=0.22 # ehkä 0.10?
#                 self.women_kappa_fulltime=0.615 # 0.605 # 0.58
#                 self.women_mu_scale=0.09 # 0.25 # 0.25 # 0.17 # how much penalty is associated with work increase with age after mu_age
#                 self.women_mu_age=60 # 61 # P.O. 60??
#                 self.women_kappa_osaaika=0.40
#                 self.women_kappa_hoitovapaa=0.05
#                 self.women_kappa_ve=0.27 # ehkä 0.10?
#                 self.kappa_pinkslip=0.06
#                 self.kappa_pinkslip_young=0.14            
#         else:
#             self.men_kappa_fulltime=0.620 # 0.635 # 0.665
#             self.men_mu_scale=0.11 # 0.14 # 0.30 # 0.16 # how much penalty is associated with work increase with age after mu_age
#             self.men_mu_age=60 # P.O. 60??
#             self.men_kappa_osaaika=0.46
#             self.men_kappa_hoitovapaa=0.015
#             self.men_kappa_ve=0.22 # ehkä 0.10?
#             self.women_kappa_fulltime=0.615 # 0.605 # 0.58
#             self.women_mu_scale=0.09 # 0.25 # 0.25 # 0.17 # how much penalty is associated with work increase with age after mu_age
#             self.women_mu_age=60 # 61 # P.O. 60??
#             self.women_kappa_osaaika=0.40
#             self.women_kappa_hoitovapaa=0.05
#             self.women_kappa_ve=0.27 # ehkä 0.10?
#             self.kappa_pinkslip=0.06
#             self.kappa_pinkslip_young=0.14       

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
            elif key=='year':
                if value is not None:
                    self.year=value  
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
                    
    def log_utility(self,income,employment_state,age,g=0,pinkslip=0,prefnoise=0):
        '''
        Log-utiliteettifunktio muokattuna lähteestä Määttänen, 2013 & Hakola & Määttänen, 2005

        Tulot _income_ ovat vuositasolla, jotta askelpituuden muutos ei vaikuta vapaa-aika-vakioihin
        Tämä versio on parametrisoitu optimoijaa varten
        '''

        # kappa tells how much person values free-time
        if g<3: # miehet
            kappa_kokoaika=self.men_kappa_fulltime
            mu_scale=self.men_mu_scale
            mu_age=self.men_mu_age
            # lognormaali
            #if self.include_preferencenoise:
            #    kappa_kokoaika += prefnoise
        
            if age<25: # ikääntyneet preferoivat osa-aikatyötä
                kappa_osaaika=self.men_kappa_osaaika_young*kappa_kokoaika
            elif age<56: # ikääntyneet preferoivat osa-aikatyötä
                kappa_osaaika=self.men_kappa_osaaika_middle*kappa_kokoaika
            else:
                kappa_osaaika=self.men_kappa_osaaika_old*kappa_kokoaika
                
            kappa_hoitovapaa=self.men_kappa_hoitovapaa
            kappa_ve=self.men_kappa_ve
            if age>56:
                kappa_pinkslip=self.men_kappa_pinkslip_elderly
            elif age>26:
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
        
            if age<25: # ikääntyneet preferoivat osa-aikatyötä
                kappa_osaaika=self.women_kappa_osaaika_young*kappa_kokoaika
            elif age<56: # ikääntyneet preferoivat osa-aikatyötä
                kappa_osaaika=self.women_kappa_osaaika_middle*kappa_kokoaika
            else:
                kappa_osaaika=self.women_kappa_osaaika_old*kappa_kokoaika
            kappa_hoitovapaa=self.women_kappa_hoitovapaa
            kappa_ve=self.women_kappa_ve
            if age>56:
                kappa_pinkslip=self.women_kappa_pinkslip_elderly
            elif age>26:
                kappa_pinkslip=self.women_kappa_pinkslip_middle
            else:
                kappa_pinkslip=self.women_kappa_pinkslip_young
                
        if pinkslip>0: # irtisanottu
            kappa_pinkslip = 0 # irtisanotuille ei vaikutuksia
        
        if age>mu_age:
            mage=min(self.min_retirementage+3,age)
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
        
        # hyöty/score
        tau_kulutusvero=0
        if self.include_preferencenoise:
            # normaali
            u=np.log(prefnoise*income/(1+tau_kulutusvero))+kappa
            equ=income*np.exp(-kappa)
        else:
            u=np.log(income/(1+tau_kulutusvero))+kappa
            equ=income*np.exp(-kappa)

        if u is np.inf:
            print('inf: state ',employment_state)

        if income<1:
            print('inf: state ',employment_state)

        return u/10,equ # tulot ovat vuositasolla, mutta skaalataan hyöty

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
        #group_sigmas=[0.08,0.10,0.15]
        #group_sigmas=[0.09,0.10,0.13]
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
        
    def setup_salaries_v3(self):
        self.get_wage=self.get_wage_step        
        
        # TK:n aineisto vuodelta 2018
        # iät 18-70
        if self.year==2018:
            self.palkat_ika_miehet=12.5*np.array([2339.01,2339.01,2339.01,2489.09,2571.40,2632.58,2718.03,2774.21,2884.89,2987.55,3072.40,3198.48,3283.81,3336.51,3437.30,3483.45,3576.67,3623.00,3731.27,3809.58,3853.66,3995.90,4006.16,4028.60,4104.72,4181.51,4134.13,4157.54,4217.15,4165.21,4141.23,4172.14,4121.26,4127.43,4134.00,4093.10,4065.53,4063.17,4085.31,4071.25,4026.50,4031.17,4047.32,4026.96,4028.39,4163.14,4266.42,4488.40,4201.40,4252.15,4443.96,3316.92,3536.03,3536.03])
            self.palkat_ika_naiset=12.5*np.array([2223.96,2223.96,2223.96,2257.10,2284.57,2365.57,2443.64,2548.35,2648.06,2712.89,2768.83,2831.99,2896.76,2946.37,2963.84,2993.79,3040.83,3090.43,3142.91,3159.91,3226.95,3272.29,3270.97,3297.32,3333.42,3362.99,3381.84,3342.78,3345.25,3360.21,3324.67,3322.28,3326.72,3326.06,3314.82,3303.73,3302.65,3246.03,3244.65,3248.04,3223.94,3211.96,3167.00,3156.29,3175.23,3228.67,3388.39,3457.17,3400.23,3293.52,2967.68,2702.05,2528.84,2528.84])
        elif self.year==2019:
            self.palkat_ika_miehet=12.5*np.array([2039,2334,2461,2498,2612,2683,2725,2833,2915,3014,3115,3196,3325,3389,3477,3540,3601,3690,3724,3863,3917,4018,4073,4150,4189,4212,4301,4273,4261,4323,4273,4225,4239,4188,4213,4174,4159,4165,4131,4136,4167,4135,4117,4134,4126,4101,4257,4368,4417,3916,4369,4045,3165,3923,2521,2982,3238,2898,3489,3063,6388,6388])
            self.palkat_ika_naiset=12.5*np.array([1926,2107,2188,2235,2283,2347,2417,2541,2639,2722,2796,2845,2907,2980,3032,3073,3109,3150,3225,3281,3333,3402,3467,3462,3492,3546,3593,3577,3603,3542,3525,3511,3515,3470,3484,3482,3477,3480,3408,3417,3428,3381,3361,3351,3292,3314,3461,3553,3583,3441,3039,2797,3320,2168,2360,3316,4525,4525,4525,2449,2449,1393])
        else:
            error(1001)
            
        def map_age18(x):
            return int(x-18)
            
        def ifunc(palkat,x1,x2):
            x = np.linspace(18, 72, num=54, endpoint=True)
            f = interp1d(x, palkat)
            n_time = int(np.round((x2-x1)*self.inv_timestep))+2
            palkat_x = np.linspace(x1, x2, num=n_time, endpoint=True)
        
            return f(palkat_x)        

        def gfunc(palkat,x1,x2):
            x = np.linspace(18, 73, num=55, endpoint=True)
            n_time = int(np.round((x2-x1)*self.inv_timestep))+2
            palkat_x = np.linspace(x1, x2, num=n_time, endpoint=True)
            g=np.zeros((n_time,3))
            for k in range(3):
                f = interp1d(x, palkat[:,k])
                g[:,k]=f(palkat_x)
        
            return g
        
        # filtteri
        m_age=int(self.min_retirementage-1)
        palkat_miehet_ve=self.palkat_ika_miehet[map_age18(m_age)]
        self.palkat_ika_miehet[map_age18(m_age):]=np.minimum(self.palkat_ika_miehet[map_age18(m_age):],palkat_miehet_ve)
        palkat_naiset_ve=self.palkat_ika_naiset[map_age18(m_age)]
        self.palkat_ika_naiset[map_age18(m_age):]=np.minimum(self.palkat_ika_naiset[map_age18(m_age):],palkat_naiset_ve)

        self.palkat_ika_miehet=ifunc(self.palkat_ika_miehet,self.min_age,self.max_age)
        self.palkat_ika_naiset=ifunc(self.palkat_ika_naiset,self.min_age,self.max_age)
        
        g_r=np.array([[0.81436,1.00992,1.2469],[0.81436,1.00992,1.2469],[0.81436,1.00992,1.2469],[0.8144725,1.011105,1.2677425],[0.814585,1.01229,1.288585],[0.8146975,1.013475,1.3094275],[0.81481,1.01466,1.33027],[0.81481,1.01466,1.33027],[0.798635,1.0116225,1.34756],[0.78246,1.008585,1.36485],[0.766285,1.0055475,1.38214],[0.75011,1.00251,1.39943],[0.75011,1.00251,1.39943],[0.74420478,0.99822499,1.40395702],[0.73829957,0.99393997,1.40848403],[0.73239435,0.98965496,1.41301105],[0.72648913,0.98536995,1.41753806],[0.72058391,0.98108493,1.42206508],[0.7146787,0.97679992,1.4265921],[0.70877348,0.9725149,1.43111911],[0.70286826,0.96822989,1.43564613],[0.69696304,0.96394488,1.44017314],[0.69105783,0.95965986,1.44470016],[0.68515261,0.95537485,1.44922718],[0.67924739,0.95108984,1.45375419],[0.67334218,0.94680482,1.45828121],[0.66743696,0.94251981,1.46280822],[0.66153174,0.93823479,1.46733524],[0.65562652,0.93394978,1.47186226],[0.64972131,0.92966477,1.47638927],[0.64381609,0.92537975,1.48091629],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433]])
        self.g_r=gfunc(g_r,self.min_age,self.max_age)

    def compute_salary_TK_v3(self,group=1,debug=False,initial_salary=None):
        '''
        Alussa ajettava funktio, joka tekee palkat yhtä episodia varten
        '''
        n_time = int(np.round((self.max_age-self.min_age)*self.inv_timestep))+2
        self.salary=np.zeros(n_time)

        if group>2: # naiset
            r=self.g_r[0,group-3]
            if initial_salary is not None:
                a0=initial_salary
            else:
                a0=self.palkat_ika_naiset[0]*r
            
            a1=self.palkat_ika_naiset[0]*r/5
            self.salary[0]=np.maximum(self.min_salary,np.random.normal(loc=a0,scale=a1,size=1)[0]) # e/y

            k=0
            r0=self.g_r[0,group-3]
            a0=self.palkat_ika_naiset[0]*r0
            s0=self.salary[0]
            for age in np.arange(self.min_age,self.max_age,self.timestep):
                r1=self.g_r[k,group-3]
                a1=self.palkat_ika_naiset[k]*r1
                self.salary[self.map_age(age)]=self.wage_process_TK_v3(s0,a0,a1,g=group-3)
                k=k+1
                s0=self.salary[self.map_age(age)]
                a0=a1
                r0=r1
        else: # miehet
            r=self.g_r[0,group]
            if initial_salary is not None:
                a0=initial_salary
            else:
                a0=self.palkat_ika_miehet[0]*r
                
            a1=self.palkat_ika_miehet[0]*r/5
            self.salary[0]=np.maximum(self.min_salary,np.random.normal(loc=a0,scale=a1,size=1)[0]) # e/y

            k=0
            r0=self.g_r[0,group]
            a0=self.palkat_ika_miehet[0]*r0
            s0=self.salary[0]
            for age in np.arange(self.min_age,self.max_age,self.timestep):
                r1=self.g_r[k,group]
                a1=self.palkat_ika_miehet[k]*r1
                self.salary[self.map_age(age)]=self.wage_process_TK_v3(s0,a0,a1,g=group)
                k=k+1
                s0=self.salary[self.map_age(age)]
                a0=a1
                r0=r1


    def state_encode(self,emp,g,pension,old_wage,age,time_in_state,paid_pension,pink,
                        toe,toekesto,tyohist,next_wage,used_unemp_benefit,wage_reduction,
                        unemp_after_ra,unempwage,unempwage_basis,
                        children_under3,children_under7,children_under18,
                        unemp_benefit_left,alkanut_ansiosidonnainen,toe58,ove_paid,kassanjasenyys,
                        prefnoise):
        '''
        Tilan koodaus neuroverkkoa varten. Arvot skaalataan ja tilat one-hot-enkoodataan

        Käytetään, jos kuolleisuus ei mukana
        '''
        #if self.include_children:
        if self.include_preferencenoise:
            d=np.zeros(self.n_empl+self.n_groups+27)
        else:
            d=np.zeros(self.n_empl+self.n_groups+26)
        #else:
        #    if self.include_preferencenoise:
        #        d=np.zeros(self.n_empl+self.n_groups+19)
        #    else:
        #        d=np.zeros(self.n_empl+self.n_groups+18)
                    
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
            d[states2+4]=np.log(paid_pension/20_000+self.eps) # alkanut eläke
            d[states2+10]=np.log(next_wage/40_000+self.eps)
            d[states2+14]=np.log(unempwage/40_000+self.eps)
            d[states2+15]=np.log(unempwage_basis/40_000+self.eps)
        else:
            d[states2]=(pension-40_000)/40_000 # vastainen eläke
            d[states2+1]=(old_wage-40_000)/40_000
            d[states2+4]=(paid_pension-40_000)/40_000 # alkanut eläke
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
        #if self.include_children:
        d[states2+18]=(children_under3-5)/10
        d[states2+19]=(children_under7-5)/10
        d[states2+20]=(children_under18-5)/10
        d[states2+21]=toe58
        d[states2+22]=ove_paid
        if age>=self.min_ove_age:
            d[states2+23]=1
        
        d[states2+24]=kassanjasenyys
        d[states2+25]=toekesto-14/12
        if self.include_preferencenoise:
            d[states2+26]=prefnoise
        #else:
        #    if self.include_preferencenoise:
        #        d[states2+18]=prefnoise
        
        return d

    def state_decode(self,vec,return_nextwage=False):
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

        if self.log_transform:
            pension=(np.exp(vec[pos])-self.eps)*20_000
            wage=(np.exp(vec[pos+1])-self.eps)*40_000
            next_wage=(np.exp(vec[pos+10])-self.eps)*40_000
            paid_pension=(np.exp(vec[pos+4])-self.eps)*20_000
            unempwage=(np.exp(vec[pos+14])-self.eps)*40_000
            unempwage_basis=(np.exp(vec[pos+15])-self.eps)*40_000
        else:
            pension=vec[pos]*40_000+40_000
            wage=vec[pos+1]*40_000+40_000 
            next_wage=vec[pos+10]*40_000+40_000 
            paid_pension=vec[pos+4]*40_000+40_000
            unempwage=vec[pos+14]*40_000+40_000 
            unempwage_basis=vec[pos+15]*40_000+40_000 

        age=vec[pos+2]*20+(self.max_age+self.min_age)/2
        time_in_state=vec[pos+3]*10+10
        #if self.include300:
        pink=vec[pos+5] # irtisanottu vai ei 
        toe=vec[pos+6]+14/12 # työssäoloehto, kesto
        tyohist=vec[pos+7]*20+10 # työhistoria
        used_unemp_benefit=vec[pos+11]+1 # käytetty työttömyyspäivärahapäivien määrä
        wage_reduction=vec[pos+12] # käytetty työttömyyspäivärahapäivien määrä
        unemp_after_ra=vec[pos+13]*2+1
        unemp_left=vec[pos+9]+1
        alkanut_ansiosidonnainen=int(vec[pos+17])
        #if self.include_children:
        children_under3=int(vec[pos+18]*10+5)
        children_under7=int(vec[pos+19]*10+5)
        children_under18=int(vec[pos+20]*10+5)
        toe58=int(vec[pos+21])
        ove_paid=vec[pos+22]
        kassanjasen=vec[pos+24]
        toekesto=vec[pos+25]+14/12
        if self.include_preferencenoise:
            prefnoise=vec[pos+26]
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

        if return_nextwage:
            return int(emp),int(g),pension,wage,age,time_in_state,paid_pension,int(pink),toe,toekesto,\
                   tyohist,used_unemp_benefit,wage_reduction,unemp_after_ra,\
                   unempwage,unempwage_basis,prefnoise,children_under3,children_under7,children_under18,\
                   unemp_left,alkanut_ansiosidonnainen,toe58,ove_paid,kassanjasen,next_wage
        else:
            return int(emp),int(g),pension,wage,age,time_in_state,paid_pension,int(pink),toe,toekesto,\
                   tyohist,used_unemp_benefit,wage_reduction,unemp_after_ra,\
                   unempwage,unempwage_basis,prefnoise,children_under3,children_under7,children_under18,\
                   unemp_left,alkanut_ansiosidonnainen,toe58,ove_paid,kassanjasen

    def unit_test_code_decode(self):
        for k in range(10):
            emp=random.randint(0,3)
            g=np.random.randint(0,6)
            pension=np.random.uniform(0,80_000)
            old_wage=np.random.uniform(0,80_000)
            age=np.random.randint(0,60)
            time_in_state=np.random.uniform(0,30)
            paid_pension=np.random.uniform(0,80_000)
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
        
            vec=self.state_encode(emp,g,pension,old_wage,age,time_in_state,paid_pension,pink,
                                toe,toekesto,tyohist,next_wage,used_unemp_benefit,wage_reduction,
                                unemp_after_ra,unempwage,unempwage_basis,children_under3,
                                children_under7,children_under18,unemp_benefit_left,alkanut_ansiosidonnainen,
                                toe58,ove_paid,kassanjasenyys,prefnoise)
                                
            emp2,g2,pension2,wage2,age2,time_in_state2,paid_pension2,pink2,toe2,toekesto2,\
            tyohist2,used_unemp_benefit2,wage_reduction2,unemp_after_ra2,\
            unempwage2,unempwage_basis2,prefnoise2, \
            children_under3_2,children_under7_2,children_under18_2,unemp_benefit_left2,\
            alkanut_ansiosidonnainen2,toe58_2,ove_paid_2,jasen_2,next_wage2\
                =self.state_decode(vec,return_nextwage=True)
                
            self.check_state(emp,g,pension,old_wage,age,time_in_state,paid_pension,pink,
                                toe,tyohist,next_wage,used_unemp_benefit,wage_reduction,
                                unemp_after_ra,unempwage,unempwage_basis,
                                prefnoise,children_under3,children_under7,children_under18,
                                unemp_benefit_left,alkanut_ansiosidonnainen,toe58,ove_paid,
                                emp2,g2,pension2,wage2,age2,time_in_state2,paid_pension2,pink2,toe2,
                                tyohist2,used_unemp_benefit2,wage_reduction2,unemp_after_ra2,
                                unempwage2,unempwage_basis2,prefnoise2,
                                children_under3_2,children_under7_2,children_under18_2,
                                unemp_benefit_left2,alkanut_ansiosidonnainen2,toe58_2,ove_paid_2,next_wage2)
        
    
    def check_state(self,emp,g,pension,old_wage,age,time_in_state,paid_pension,pink,
                                toe,tyohist,next_wage,used_unemp_benefit,wage_reduction,
                                unemp_after_ra,unempwage,unempwage_basis,
                                prefnoise,children_under3,children_under7,children_under18,
                                unemp_benefit_left,alkanut_ansiosidonnainen,toe58,ove_paid,
                                emp2,g2,pension2,old_wage2,age2,time_in_state2,paid_pension2,pink2,toe2,
                                tyohist2,used_unemp_benefit2,wage_reduction2,unemp_after_ra2,
                                unempwage2,unempwage_basis2,prefnoise2,
                                children_under3_2,children_under7_2,children_under18_2,
                                unemp_benefit_left2,alkanut_ansiosidonnainen2,toe58_2,ove_paid_2,next_wage2):
        if not emp==emp2:  
            print('emp: {} vs {}'.format(emp,emp2))
        if not g==g2:  
            print('g: {} vs {}'.format(g,g2))
        if not pension==pension2:  
            print('pension: {} vs {}'.format(pension,pension2))
        if not old_wage==old_wage2:  
            print('old_wage: {} vs {}'.format(old_wage,old_wage2))
        if not age==age2:  
            print('age: {} vs {}'.format(age,age2))
        if not time_in_state==time_in_state2:  
            print('time_in_state: {} vs {}'.format(time_in_state,time_in_state2))
        if not paid_pension==paid_pension2:  
            print('paid_pension: {} vs {}'.format(paid_pension,paid_pension2))
        if not pink==pink2:  
            print('pink: {} vs {}'.format(pink,pink2))
        if not tyohist==tyohist2:  
            print('tyohist: {} vs {}'.format(tyohist,tyohist2))
        if not next_wage==next_wage2:  
            print('next_wage: {} vs {}'.format(next_wage,next_wage2))
        if not used_unemp_benefit==used_unemp_benefit2:  
            print('used_unemp_benefit: {} vs {}'.format(used_unemp_benefit,used_unemp_benefit2))
        if not wage_reduction==wage_reduction2:  
            print('wage_reduction: {} vs {}'.format(wage_reduction,wage_reduction2))
        if not unemp_after_ra==unemp_after_ra2:  
            print('unemp_after_ra: {} vs {}'.format(unemp_after_ra,unemp_after_ra2))
        if not unempwage==unempwage2:  
            print('unempwage: {} vs {}'.format(unempwage,unempwage2))
        if not unempwage_basis==unempwage_basis2:  
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
        if not unemp_benefit_left==unemp_benefit_left2:  
            print('unemp_benefit_left2: {} vs {}'.format(unemp_benefit_left,unemp_benefit_left2))
        if not alkanut_ansiosidonnainen==alkanut_ansiosidonnainen2:  
            print('alkanut_ansiosidonnainen: {} vs {}'.format(alkanut_ansiosidonnainen,alkanut_ansiosidonnainen2))
        if not toe58==toe58_2:  
            print('toe58: {} vs {}'.format(toe58,toe58_2))
        if not ove_paid==ove_paid_2:  
            print('ove_paid: {} vs {}'.format(ove_paid,ove_paid_2))
    
    def reset(self,init=None):
        '''
        Open AI-interfacen mukainen reset-funktio, joka nollaa laskennan alkutilaan
        '''
        self.init_state()
        self.steps_beyond_done = None
        
        if self.plotdebug:
            self.render()

        return np.array(self.state)
        
    def init_state(self):
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
        
        # set up salary for the entire career
        g=random.choices(np.array([0,1,2],dtype=int),weights=[0.3,0.5,0.2])[0]
        gender=random.choices(np.array([0,1],dtype=int),weights=[0.5,0.5])[0]
        group=int(g+gender*3)
                
        employment_state=random.choices(np.array([13,0,1,10,3,11,12],dtype=int),
                weights=self.initial_weights[group,:])[0]

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
        
        self.compute_salary_TK_v3(group=group,initial_salary=initial_salary)
        old_wage=self.get_wage(self.min_age,wage_reduction)
        next_wage=old_wage
        if not reset_exp:
            if employment_state==0:
                tyohist=1.0
                toe=0.0
                toekesto=1.0
                wage_reduction=np.random.uniform(low=0.05,high=0.35)
                used_unemp_benefit=0.0
                unempwage_basis=old_wage
                alkanut_ansiosidonnainen=1
                unempwage=0
            elif employment_state==13:
                tyohist=0.0
                toe=0.0
                wage_reduction=np.random.uniform(low=0.10,high=0.50) # 20-70
                used_unemp_benefit=2.0
            elif employment_state==12:
                tyohist=0.0
                toe=0.0
                wage_reduction=np.random.uniform(low=0.10,high=0.30)
                used_unemp_benefit=0.0
            elif employment_state==11:
                tyohist=0.0
                toe=0.0
                wage_reduction=np.random.uniform(low=0.15,high=0.50) # 15-50
            elif employment_state==3:
                wage5y=next_wage
                paid_pension=pension
                # takuueläke voidaan huomioida jo tässä
                paid_pension=self.ben.laske_kokonaiselake(age,paid_pension/12,include_kansanelake=self.include_kansanelake,include_takuuelake=False,disability=True)*12
                pension=0
            elif employment_state==2:
                wage5y=next_wage
                paid_pension=pension
                # takuueläke voidaan huomioida jo tässä
                paid_pension=self.ben.laske_kokonaiselake(age,paid_pension/12,include_kansanelake=self.include_kansanelake,include_takuuelake=False,disability=True)*12
                pension=0
        else:        
            if employment_state==0:
                tyohist=np.random.uniform(low=0.0,high=age-18)
                toe=np.random.uniform(low=0.0,high=28/12)
                toekesto=toe
                wage_reduction=np.random.uniform(low=0.0,high=0.45)
                used_unemp_benefit=np.random.uniform(low=0.0,high=2.0)
                unempwage_basis=old_wage
                alkanut_ansiosidonnainen=1
                unempwage=np.random.uniform(low=0.0,high=90_000.0)
            elif employment_state==13:
                tyohist=np.random.uniform(low=0.0,high=age-18)
                toe=0.0
                toekesto=toe
                wage_reduction=np.random.uniform(low=0.0,high=0.60)
                used_unemp_benefit=2.0
            elif employment_state==10:
                tyohist=np.random.uniform(low=0.0,high=age-18)
                toe=0.0
                toekesto=toe
                wage_reduction=np.random.uniform(low=0.0,high=0.60)
                used_unemp_benefit=2.0
            elif employment_state==12:
                tyohist=0.0
                toe=0.0
                wage_reduction=np.random.uniform(low=0.0,high=0.50)
                used_unemp_benefit=0.0
            elif employment_state==11:
                tyohist=0.0
                toe=0.0
                wage_reduction=np.random.uniform(low=0.0,high=0.35)
            elif employment_state==3:
                wage5y=next_wage
                paid_pension=pension
                # takuueläke voidaan huomioida jo tässä
                paid_pension=self.ben.laske_kokonaiselake(age,paid_pension/12,include_kansanelake=self.include_kansanelake,include_takuuelake=False,disability=True)*12
                pension=0
            elif employment_state==2:
                wage5y=next_wage
                paid_pension=pension
                # takuueläke voidaan huomioida jo tässä
                paid_pension=self.ben.laske_kokonaiselake(age,paid_pension/12,include_kansanelake=self.include_kansanelake,include_takuuelake=False,disability=True)*12
                pension=0
        
        if employment_state in set([1,10]):
            unempwage=old_wage
            
        unemp_benefit_left=self.comp_unempdays_left(used_unemp_benefit,tyohist,age,toe,employment_state,alkanut_ansiosidonnainen,toe58,toe)
        
        self.init_infostate(age=age)
        kassanjasenyys=self.get_kassanjasenyys()
        if employment_state in set([0,4]):
            self.set_kassanjasenyys(1)

        # tarvitseeko alkutilassa laskea muita tietoja uusiksi? ei kait

        if self.plotdebug:
            print('emp {} gender {} g {} old_wage {} next_wage {} age {} kassanjäsen {}'.format(employment_state,gender,g,old_wage,next_wage,age,kassanjasenyys))

        if self.include_preferencenoise:
            # lognormaali
            #prefnoise=np.random.normal(loc=-0.5*self.preferencenoise_std*self.preferencenoise_std,scale=self.preferencenoise_std,size=1)[0]
            # normaali
            prefnoise=max(1e-6,np.random.normal(loc=1.0,scale=self.preferencenoise_std,size=1)[0])
        else:
            prefnoise=0
            
        self.state = self.state_encode(employment_state,group,pension,old_wage,age,
                                       time_in_state,0,pink,toe,toekesto,tyohist,next_wage,
                                       used_unemp_benefit,wage_reduction,unemp_after_ra,
                                       unempwage,unempwage_basis,
                                       children_under3,children_under7,children_under18,
                                       unemp_benefit_left,alkanut_ansiosidonnainen,toe58,
                                       ove_paid,kassanjasenyys,
                                       prefnoise)

    def render(self, mode='human', close=False, done=False, reward=None, netto=None):
        '''
        Tulostus-rutiini
        '''
        emp,g,pension,wage,age,time_in_state,paid_pension,pink,toe,toekesto,tyohist,used_unemp_benefit,\
            wage_red,unemp_after_ra,unempwage,unempwage_basis,prefnoise,c3,c7,c18,\
            unemp_left,oikeus,toe58,ove_paid,jasen,next_wage=self.state_decode(self.state,return_nextwage=True)
            
        #if emp in set([0,4]):
        #    old_toe=self.comp_oldtoe()
        #    print(old_toe)
        #else:
        #    old_toe=0

        if jasen:
            kassassa='+'
        else:
            kassassa='-'
            
            
        if reward is None:
            print('s {} g {} sal {:.0f} nw {:.0f} ikä {:.2f} tis {:.2f} tule {:.0f} alk e {:.0f} irti {} toe {:.2f}{} tk {:.2f} työuraura {:.2f} ueb {:.2f} wr {:.2f} uew {:.0f} (b {:.0f}) c3 {:.0f} c7 {:.0f} c18 {:.0f} uleft {:.2f} aa {:.0f} 58 {:.0f} ove {:.0f}'.format(\
                emp,g,wage,next_wage,age,time_in_state,pension,paid_pension,pink,toe,kassassa,toekesto,tyohist,used_unemp_benefit,wage_red,unempwage,unempwage_basis,c3,c7,c18,unemp_left,oikeus,toe58,ove_paid))
        elif netto is None:
            print('s {} g {} sal {:.0f} nw {:.0f} ikä {:.2f} tis {:.2f} tule {:.0f} alk e {:.0f} irti {} toe {:.2f}{} tk {:.2f} ura {:.2f} ueb {:.2f} wr {:.2f} uew {:.0f} (b {:.0f}) c3 {:.0f} c7 {:.0f} c18 {:.0f} uleft {:.2f} aa {:.0f} 58 {:.0f} ove {:.0f} r {:.4f}'.format(\
                emp,g,wage,next_wage,age,time_in_state,pension,paid_pension,pink,toe,kassassa,toekesto,tyohist,used_unemp_benefit,wage_red,unempwage,unempwage_basis,c3,c7,c18,unemp_left,oikeus,toe58,ove_paid,reward))
        else:
            print('s {} g {} sal {:.0f} nw {:.0f} ikä {:.2f} tis {:.2f} tule {:.0f} alk e {:.0f} irti {} toe {:.2f}{} tk {:.2f} ura {:.2f} ueb {:.2f} wr {:.2f} uew {:.0f} (b {:.0f}) c3 {:.0f} c7 {:.0f} c18 {:.0f} uleft {:.2f} aa {:.0f} 58 {:.0f} ove {:.0f} r {:.4f} n {:.0f}'.format(\
                emp,g,wage,next_wage,age,time_in_state,pension,paid_pension,pink,toe,kassassa,toekesto,tyohist,used_unemp_benefit,wage_red,unempwage,unempwage_basis,c3,c7,c18,unemp_left,oikeus,toe58,ove_paid,reward,netto))
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

        # korjaa
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
            state_min,
            state_min,
            state_min,
            toe_min,
            state_min]
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
            state_max,
            state_max,
            state_max,
            toe_max,
            state_max]
            
        #if self.include_mort: # if mortality is included, add one more state
        #      low.insert(0,state_min)
        #      high.insert(0,state_max)
              
        #if self.include_children:
        low.append(child_min)
        high.append(child_max)
        low.append(child_min)
        high.append(child_max)
        low.append(child_min)
        high.append(child_max)
                  
        if self.include_preferencenoise:
            low.append(pref_min)
            high.append(pref_max)
                
        self.low=np.array(low)
        self.high=np.array(high)

    def explain(self):
        '''
        Tulosta laskennan parametrit
        '''
        print('Parameters of lifecycle:\ntimestep {}\ngamma {} ({} per anno)\nmin_age {}\nmax_age {}\nmin_retirementage {}'.format(self.timestep,self.gamma,self.gamma**(1.0/self.timestep),self.min_age,self.max_age,self.min_retirementage))
        print('max_retirementage {}\nansiopvraha_kesto300 {}\nansiopvraha_kesto400 {}\nansiopvraha_kesto500 {}\nansiopvraha_toe {}'.format(self.max_retirementage,self.ansiopvraha_kesto300,self.ansiopvraha_kesto400,self.ansiopvraha_kesto500,self.ansiopvraha_toe))
        print('perustulo {}\nkarenssi_kesto {}\nmortality {}\nrandomness {}'.format(self.perustulo,self.karenssi_kesto,self.include_mort,self.randomness))
        print('include_putki {}\ninclude_pinkslip {}'.format(self.include_putki,self.include_pinkslip))
        print(f'perustulo {self.perustulo}\nsigma_reduction {self.use_sigma_reduction}\nplotdebug {self.plotdebug}')
        print('additional_tyel_premium {}\nscale_tyel_accrual {}\ninclude_ove {}'.format(self.additional_tyel_premium,self.scale_tyel_accrual,self.include_ove))
        print(f'unemp_limit_reemp {self.unemp_limit_reemp}\n')

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
        self.infostat_kassanjasenyys_rate()
            
    def init_infostate(self,lapsia=0,lasten_iat=np.zeros(15),lapsia_paivakodissa=0,age=18):
        '''
        Alustaa infostate-dictorionaryn
        Siihen talletetaan tieto aiemmista tiloista, joiden avulla lasketaan statistiikkoja
        '''
        self.infostate={}
        self.infostate['states']=np.zeros(self.n_time)-1
        self.infostate['wage']=np.zeros(self.n_time)-1
        self.infostate['unempbasis']=np.zeros(self.n_time)-1
        self.infostate['latest']=0
        self.infostate['children_n']=0
        self.infostate['children_date']=np.zeros(15)
        self.infostate['enimmaisaika_alkaa']=0
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
            
    def infostate_set_enimmaisaika(self,age):
        t=int((age-self.min_age)/self.timestep)
        self.infostate['enimmaisaika_alkaa']=t
        
    def update_infostate(self,t,state,wage,unempbasis):
        self.infostate['states'][t]=state
        self.infostate['latest']=int(t)
        self.infostate['unempbasis'][t]=unempbasis
        if state==1:
            self.infostate['wage'][t]=wage
        elif state==10:
            self.infostate['wage'][t]=wage*0.5
        elif state in set([5,6]):
            self.infostate['wage'][t]=wage
        else:
            self.infostate['wage'][t]=0
        
    def render_infostate(self):
        print('states {}'.format(self.infostate['states']))
        
    def get_kassanjasenyys(self):
        return self.infostate['kassanjasen']

    def set_kassanjasenyys(self,value):
        self.infostate['kassanjasen']=value
        
    def infostat_kassanjasenyys_rate(self):
        rate_age=np.array([18,30,40,50,60,74])
        #rate_obs=np.array([0.429,0.836,0.917,0.946,0.933,0.933])
        rate_obs=np.array([0.20,0.836,0.917,0.946,0.933,0.933])
        self.initial_kassanjasenia=rate_obs[0]
        
        x1=self.min_age
        x2=self.max_age+1
        f = interp1d(rate_age, rate_obs)
        n_time = int(np.round((x2-x1)*self.inv_timestep))+2
        jasenyys_x = np.linspace(x1, x2, num=n_time, endpoint=True)

        rate=f(jasenyys_x)
        self.kassanjasenyys_joinrate=rate*0
        for k in range(0,rate.shape[0]-1):
            self.kassanjasenyys_joinrate[k+1]=(rate[k+1]-rate[k])/(1-rate[k]) #*self.timestep
            
        self.kassanjasenyys_rate=rate
            
    def infostat_kassanjasenyys_update(self,age):
        if self.infostate['kassanjasen']<1:
            sattuma = np.random.uniform(size=1)
            intage=self.map_age(age)
            if sattuma<self.kassanjasenyys_joinrate[intage]:
                self.infostate['kassanjasen']=1
        
    def comp_toe_wage_nykytila(self):
        lstate=int(self.infostate['states'][self.infostate['latest']])
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
                    t2=self.infostate['latest']
                    nt=0
                    while nt<n_toe and t2>=0:
                        emps=self.infostate['states'][t2]
                        if emps in family_states:
                            pass
                        elif emps in emp_states:
                            w=self.infostate['wage'][t2]
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
                    t2=self.infostate['latest']
                    nt=0
                    t0=self.infostate['enimmaisaika_alkaa']
                    while nt<n_toe and t2>=t0:
                        emps=self.infostate['states'][t2]
                        if emps in family_states:
                            pass
                        elif emps in emp_states:
                            w=self.infostate['wage'][t2]
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
        
    def comp_toe_wage_porrastus(self):
        lstate=int(self.infostate['states'][self.infostate['latest']])
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
            t2=self.infostate['latest']
            nt=0
            t0=self.infostate['enimmaisaika_alkaa']
            while nt<n_toe and t2>=t0:
                emps=self.infostate['states'][t2]
                if emps in family_states:
                    pass
                elif emps in emp_states:
                    w=self.infostate['wage'][t2]
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
            t2=self.infostate['latest']
            nt=0
            while nt<n_toe and t2>=0:
                emps=self.infostate['states'][t2]
                if emps in family_states:
                    pass
                elif emps in emp_states:
                    w=self.infostate['wage'][t2]
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
        
    def comp_infostats(self,age):
        # laske työssäoloehto tarkasti
        # laske työttömyysturvaan vaikuttavat lasten määrät
        
        self.infostat_kassanjasenyys_update(age)
        
        toes,toekesto,wage=self.comp_toe_wage()
                
        #print('toes',toes)
        
        #start_t=max(self.infostate['latest']-n_toe,self.infostate['enimmaisaika_alkaa'])
        #print('-->',start_t,self.infostate['latest'],self.infostate['states'][start_t:self.infostate['latest']],
        #    self.infostate['wage'][start_t:self.infostate['latest']],toes)
        
        if self.include_children:
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
        else: # pidetään kirjaa vain yhdestä lapsesta
            children_under18=0
            children_under7=0
            children_under3=0
            for k in range(self.infostate['children_n']):
                c_age=age-self.infostate['children_date'][k]
                if c_age<3:
                    children_under3=1
                    break
                    
            children_under18=children_under3
            children_under7=children_under3

        return toes,toekesto,wage,children_under3,children_under7,children_under18

    def infostat_comp_5y_ave_wage(self):
        emp_states={1,10}
        unemp_states={0,4,13}
        family_states={5,6}
    
        lstate=int(self.infostate['latest'])
        n=int(np.ceil(5/self.timestep))
        wage=0
        for x in range(lstate-n,lstate):
            if x<0:
                pass
            else:
                if self.infostate['states'][x] in emp_states:
                    wage+=self.infostate['wage'][x]*self.timestep/5
                elif self.infostate['states'][x] in family_states:
                    wage+=self.infostate['wage'][x]*self.timestep/5
                elif self.infostate['states'][x] in unemp_states:
                    wage+=self.infostate['unempbasis'][x]*self.timestep/5
                elif self.infostate['states'][x]==13:
                    wage+=self.disabbasis_tmtuki*self.timestep/5
                elif self.infostate['states'][x]==12:
                    wage+=self.disabbasis_tmtuki*self.timestep/5
                    
                #wage+=self.infostate['wage'][x]*self.timestep/5

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

    def infostat_check_aareset(self,age):
        t=int((age-self.min_age)/self.timestep)
        ed_t=self.infostate['enimmaisaika_alkaa']
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
        
    def print_q(self,a):
        '''
        pretty printer for dict
        '''
        for x in a.keys():
            if a[x]>0 or a[x]<0:
                print('{}:{:.2f} '.format(x,a[x]),end='')
                
        print('')
        
    def comp_oldtoe(self,printti=False):
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
        
        lstate=int(self.infostate['states'][self.infostate['latest']])
        
        #if lstate!=0:
        #    return 0
        
        nt=0
        t2=max(0,self.infostate['enimmaisaika_alkaa']-1)
        emp_states={1,10}
        unemp_states={0,4}
        family_states={5,6,7}
        while nt<n_toe:
            emps=self.infostate['states'][t2]
            if printti:
                print('emps {} t2 {} toes {}'.format(emps,t2,toes))
            if emps in family_states:
                pass
            elif emps in emp_states:
                if self.infostate['wage'][t2]>self.min_toewage:
                    toes+=self.timestep
                elif self.include_halftoe and self.infostate['wage'][t2]>=self.min_halftoewage and emps==10:
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

    def check_toe58(self,age,toe,tyoura,toe58):
        '''
        laske työttömyysjaksoa edeltävä työssäoloehto tarkasti
        '''
        toes=0
        n_toe=int(np.floor(self.max_toe/self.timestep))
        emp_states={1,10}
        unemp_states={0,4}
        family_states={5,6,7}
        ret_states={2,3,8,9}
        lstate=int(self.infostate['states'][self.infostate['latest']])
        
        if age<self.minage_500 or lstate in ret_states:
            return 0

        t=self.map_age(age)
        t58=self.map_age(58)
        
        #if lstate!=0:
        #    return 0
        
        nt=0
        if lstate in unemp_states:
            t2=max(0,self.infostate['enimmaisaika_alkaa']-1)
        else:
            t2=max(0,self.infostate['latest'])
        
        while nt<n_toe and nt<t-t58:
            emps=self.infostate['states'][t2]
            if emps in family_states:
                pass
            elif emps in emp_states:
                if self.infostate['wage'][t2]>self.min_toewage:
                    toes+=self.timestep
                elif self.include_halftoe and self.infostate['wage'][t2]>=self.min_halftoewage and emps==10:
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
