"""

    unemployment_v2


    Gym module implementing the Finnish social security including earnings-related components,
    e.g., the unemployment benefit
    
    updated
    korjauksia julkaistuun versioon:
    - kansaeläkkeen yhteensovitus huomioitu väärin, joten kansaneläke huomioitiin liian pienenä, ei kuitenkaan vaikuttanut
      takuueläkkeeseen, joten tulokset eivät välttämättä paljon muutu (tarkasta)
    - 20-vuotiaan palkka korjattu pois nollasta

"""

import math
import gym
from gym import spaces, logger, utils, error
from gym.utils import seeding   
import numpy as np
import fin_benefits
import random

# class StayDict(dict):
#     '''
#     Apuluokka, jonka avulla tehdään virheenkorjausta 
#     '''
#     def __missing__(self, key):
#         return 'Unknown state '+key


class UnemploymentLargeEnv_v2(gym.Env):
    """
    Description:
        The Finnish Unemployment Pension Scheme 

    Source:
        This environment corresponds to the environment of the Finnish Social Security

    Observation: 
        Type: Box(17) 
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
       16    Preferenssikohina         

    Employment states:
        Type: Int
        Num     State
        0   Unemployed
        1   Employed
        2   Retired
        3   Disabled
        4   Työttömyysputki
        5   Äitiysvapaa
        6   Isyysvapaa
        7   Kotihoidontuki
        8   Vanhuuseläke+Osa-aikatyö   
        9   Vanhuuseläke+Kokoaikatyö   
        10  Osa-aikatyö
        11  Työvoiman ulkopuolella, ei tukia
        12  Opiskelija
        13  Työmarkkinatuki
        14  Armeijassa
        15  Osa-aikatyö + OVE
        16  Täysiaikatyö + OVE
        17  Työtön + OVE
        18  Osatk + työ
        19  Osatk + työtön
        20  Kuollut (jos kuolleisuus mukana)

    Actions:
        These really depend on the state (see function step)
    
        Type: Discrete(4)
        Num    Action
        0    Stay in the current state
        1    Switch to the other state (work -> unemployed; unemployed -> work)
        2    Retire if >=min_retirementage
        3    Some other transition

    Reward:
        Reward is the sum of wage and benefit for every step taken, including the termination step

    Starting State:
        Starting state in unemployed at age 20

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
        self.ansiopvraha_kesto500=500
        self.minage_500=58 # minimi-ikä 500 päivälle
        self.ansiopvraha_kesto400=400
        self.ansiopvraha_kesto300=300
        self.min_salary=1000 # julkaistut laskelmat olettavat tämän
        #self.min_salary=10000 # julkaistujen laskelmien jälkeen

        self.timestep=0.25
        self.max_age=71
        self.min_age=20
        self.min_retirementage=63.5 #65
        self.max_retirementage=68 # 70

        self.elinaikakerroin=0.925 # etk:n arvio 1962 syntyneille
        reaalinen_palkkojenkasvu=1.016
        
        self.reset_exploration_go=True
        self.reset_exploration_ratio=0.4
        self.train=False

        self.include_mort=False # onko kuolleisuus mukana laskelmissa
        self.include_preferencenoise=False # onko työllisyyspreferenssissä hajonta mukana 
        #self.include300=True # onko työuran kesto mukana laskelmissa
        self.perustulo=False # onko Kelan perustulo laskelmissa
        self.randomness=True # onko stokastiikka mukana
        self.mortstop=True # pysäytä kuolleisuuden jälkeen
        self.include_putki=True # työttömyysputki mukana
        self.include_pinkslip=True # irtisanomiset mukana
        self.use_sigma_reduction=True # kumpi palkkareduktio        
        self.include_children=True # yksinhuoltajat ja lapset mukana mallissa
        gamma=0.92
        
        # etuuksien laskentavuosi
        self.year=2018

        self.plotdebug=False # tulostetaanko rivi riviltä tiloja

        if 'kwargs' in kwargs:
            kwarg=kwargs['kwargs']
        else:
            kwarg={}

        for key, value in kwarg.items():
            if key=='step':
                if value is not None:
                    self.timestep=value
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
            elif key=='ansiopvraha_toe':
                if value is not None:
                    self.ansiopvraha_toe=value
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
 
        # ei skaalata!
        #self.ansiopvraha_kesto400=self.ansiopvraha_kesto400/(12*21.5)
        #self.ansiopvraha_kesto300=self.ansiopvraha_kesto300/(12*21.5)              

        self.gamma=gamma**self.timestep # discounting
        self.palkkakerroin=(0.8*1+0.2*1.0/reaalinen_palkkojenkasvu)**self.timestep
        self.elakeindeksi=(0.2*1+0.8*1.0/reaalinen_palkkojenkasvu)**self.timestep
        self.kelaindeksi=(1.0/reaalinen_palkkojenkasvu)**self.timestep
        self.n_age = self.max_age-self.min_age+1
        self.inv_timestep=int(np.round(1/self.timestep)) # pitäisi olla kokonaisluku
        self.n_time = int(np.round((self.n_age-1)*self.inv_timestep))+2

        # paljonko työstä poissaolo vaikuttaa palkkaan
        self.salary_const=0.05*self.timestep
        self.salary_const_up=0.015*self.timestep # työssäolo palauttaa ansioita tämän verran vuodessa
        self.salary_const_student=0.015*self.timestep # opiskelu nostaa leikkausta tämän verran vuodessa

        # karttumaprosentit
        self.acc=0.015*self.timestep
        self.acc_over_52=0.019*self.timestep
        self.acc_family=1.15*self.acc
        self.acc_family_over_52=1.15*self.acc_over_52
        self.acc_unemp=0.75*self.acc
        self.acc_unemp_over_52=0.75*self.acc_over_52

        # parametrejä
        self.max_toe=28/12
        self.accbasis_kht=719.0*12
        self.accbasis_tmtuki=1413.75*12

        self.n_age=self.max_age-self.min_age+1

        # male low income, male mid, male high, female low, female mid, female high income
        self.n_groups=6

        # käytetäänkö exp/log-muunnosta tiloissa vai ei?
        self.log_transform=False
        self.eps=1e-20

        self.salary=np.zeros(self.max_age+1)

        # ryhmäkohtaisia muuttujia
        #self.disability_intensity=self.get_disability_rate()*self.timestep # tn tulla työkyvyttömäksi
        self.disability_intensity=self.get_eff_disab_rate()*self.timestep # tn tulla työkyvyttömäksi
        
        if self.include_pinkslip:
            self.pinkslip_intensity=np.zeros(6)
            if True:
                self.pinkslip_intensity[0:3]=0.05*self.timestep # todennäköisyys tulla irtisanotuksi vuodessa, miehet
                self.pinkslip_intensity[3:6]=0.05*self.timestep # todennäköisyys tulla irtisanotuksi vuodessa, naiset
            else:
                self.pinkslip_intensity[0]=0.08*self.timestep # todennäköisyys tulla irtisanotuksi vuodessa, miehet
                self.pinkslip_intensity[1]=0.05*self.timestep # todennäköisyys tulla irtisanotuksi vuodessa, miehet
                self.pinkslip_intensity[2]=0.03*self.timestep # todennäköisyys tulla irtisanotuksi vuodessa, miehet
                self.pinkslip_intensity[3]=0.05*self.timestep # todennäköisyys tulla irtisanotuksi vuodessa, naiset
                self.pinkslip_intensity[4]=0.03*self.timestep # todennäköisyys tulla irtisanotuksi vuodessa, naiset
                self.pinkslip_intensity[5]=0.02*self.timestep # todennäköisyys tulla irtisanotuksi vuodessa, naiset
        else:
            self.pinkslip_intensity=0 # .05*self.timestep # todennäköisyys tulla irtisanotuksi vuodessa, skaalaa!
        
        self.birth_intensity=self.get_birth_rate()*self.timestep # todennäköisyys saada lapsi, skaalaa!
        self.mort_intensity=self.get_mort_rate()*self.timestep # todennäköisyys , skaalaa!
        self.student_inrate,self.student_outrate=self.get_student_rate()
        self.student_inrate=self.student_inrate*self.timestep
        self.student_outrate=self.student_outrate*self.timestep
        self.army_outrate=self.get_army_rate()*self.timestep
        self.outsider_inrate,self.outsider_outrate=self.get_outsider_rate()
        self.outsider_inrate=self.outsider_inrate*self.timestep
        self.outsider_outrate=self.outsider_outrate*self.timestep
        self.npv,self.npv0=self.comp_npv()

        self.set_state_limits()
        if self.include_mort: # and not self.mortstop:
            if self.include_mort and self.mortstop:
                print('Mortality included, stopped')
            else:
                print('Mortality included, not stopped')

            self.n_empl=16 # state of employment, 0,1,2,3,4
            self.state_encode=self.state_encode_mort
        else:
            print('No mortality included')
            self.n_empl=15 # state of employment, 0,1,2,3,4
            self.state_encode=self.state_encode_nomort

        self.n_actions=4 # valittavien toimenpiteiden määrä

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        #print(self.low.shape)
        
        #print(self.use_sigma_reduction)
        if self.use_sigma_reduction:
            self.update_wage_reduction=self.update_wage_reduction_sigma
        else:
            self.update_wage_reduction=self.update_wage_reduction_baseline

        #self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

        #self.log_utility=self.log_utility_randomness

        if self.perustulo:
            self.ben = fin_benefits.BasicIncomeBenefits(**kwargs)
        else:
            self.ben = fin_benefits.Benefits(**kwargs)
            
        self.ben.set_year(self.year)
        
        self.init_infostate()
        
        self.explain()
        
        #self.unit_test_code_decode()

    def get_n_states(self):
        '''
        Palauta parametrien arvoja
        '''
        return self.n_empl,self.n_actions

    def comp_npv(self):
        '''
        lasketaan montako timestep:iä (diskontattuna) max_age:n jälkeen henkilö on vanhuuseläkkeellä 
        hyvin yksinkertainen toteutus. Tulos on odotettu lukumäärä timestep:jä
        
        npv <- diskontattu
        npv0 <- ei ole diskontattu
        '''
        npv=np.zeros(self.n_groups)
        npv0=np.zeros(self.n_groups)

        for g in range(self.n_groups):
            cpsum=1
            cpsum0=1
            for x in np.arange(100,self.max_age,-self.timestep):
                intx=int(np.floor(x))
                m=self.mort_intensity[intx,g]
                cpsum=m*1+(1-m)*(1+self.gamma*cpsum)
                cpsum0=m*1+(1-m)*(1+cpsum0)
            npv[g]=cpsum
            npv0[g]=cpsum0
            
        if self.plotdebug:
            print('npv: {}',format(npv))

        return npv,npv0

    def comp_benefits(self,wage,old_wage,pension,employment_state,time_in_state,children_under7,children_under18,ika,
                      irtisanottu=0,tyohistoria=0,retq=True):
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

        p['perustulo']=self.perustulo
        p['opiskelija']=0
        p['toimeentulotuki_vahennys']=0
        p['ika']=ika
        if self.include_children:
            p['lapsia']=children_under18
            p['paivahoidossa']=children_under7
            p['alle_kouluikaisia']=children_under7
        else:
            p['lapsia']=0
            p['paivahoidossa']=0
            p['alle_kouluikaisia']=0
        p['aikuisia']=1
        p['veromalli']=0
        p['kuntaryhma']=3
        p['lapsia_kotihoidontuella']=0
        p['alle3v']=0
        p['tyottomyyden_kesto']=1
        p['puoliso_tyottomyyden_kesto']=10
        p['isyysvapaalla']=0
        p['aitiysvapaalla']=0
        p['kotihoidontuella']=0
        p['tyoelake']=0
        p['elakkeella']=0
        p['sairauspaivarahalla']=0
        if employment_state==1:
            p['tyoton']=0 # voisi olla työtön siinä mielessä, että oikeutettu soviteltuun päivärahaan
            p['t']=wage/12
            p['vakiintunutpalkka']=wage/12
            p['saa_ansiopaivarahaa']=0
        elif employment_state==0: # työtön, ansiopäivärahalla
            if ika<65:
                #self.render()
                p['tyoton']=1
                p['t']=0
                p['vakiintunutpalkka']=old_wage/12
                if irtisanottu>0:
                    p['tyottomyyden_kesto']=12*21.5*time_in_state
                else:
                    p['tyottomyyden_kesto']=12*21.5*time_in_state-self.karenssi_kesto # tämän voisi tehdä täsmällisemmin
                #if ((tyohistoria>=self.tyohistoria_vaatimus and p['tyottomyyden_kesto']<=self.ansiopvraha_kesto400) \
                #    or (p['tyottomyyden_kesto']<=self.ansiopvraha_kesto300) \
                #    or (ika>=self.minage_500 and tyohistoria>=self.tyohistoria_vaatimus500 and p['tyottomyyden_kesto']<=self.ansiopvraha_kesto500)) \
                if self.paivarahapaivia_jaljella(p['tyottomyyden_kesto'],tyohistoria,ika) and (irtisanottu>0 or time_in_state>=self.karenssi_kesto): # karenssi, jos ei irtisanottu
                    p['saa_ansiopaivarahaa']=1
                else:
                    p['saa_ansiopaivarahaa']=0
            else:
                p['tyoton']=0 # ei oikeutta työttömyysturvaan
                p['t']=0
                p['vakiintunutpalkka']=0
                p['saa_ansiopaivarahaa']=0
        elif employment_state==13: # työmarkkinatuki
            if ika<65:
                p['tyoton']=1
                p['t']=0
                p['vakiintunutpalkka']=0
                p['tyottomyyden_kesto']=12*21.5*time_in_state
                p['saa_ansiopaivarahaa']=0
            else:
                p['tyoton']=0 # ei oikeutta työttömyysturvaan
                p['t']=0
                p['vakiintunutpalkka']=0
                p['saa_ansiopaivarahaa']=0
        elif employment_state==3: # tk
            p['tyoton']=0
            p['saa_ansiopaivarahaa']=0
            p['t']=0
            p['vakiintunutpalkka']=0
            p['elakkeella']=1 
            #p['elake']=pension
        elif employment_state==4: # työttömyysputki
            if ika<65:
                p['tyoton']=1
                p['t']=0
                p['vakiintunutpalkka']=old_wage/12
                p['saa_ansiopaivarahaa']=1
                p['tyottomyyden_kesto']=12*21.5*time_in_state
            else:
                p['tyoton']=0 # ei oikeutta työttömyysturvaan
                p['t']=0
                p['vakiintunutpalkka']=0
                p['saa_ansiopaivarahaa']=0
        elif employment_state==5: # ansiosidonnainen vanhempainvapaa, äidit
            p['aitiysvapaalla']=1
            p['tyoton']=0
            p['aitiysvapaa_kesto']=0
            p['t']=0
            p['vakiintunutpalkka']=old_wage/12
            p['saa_ansiopaivarahaa']=1
        elif employment_state==6: # ansiosidonnainen vanhempainvapaa, isät
            p['isyysvapaalla']=1
            p['tyoton']=0
            p['t']=0
            p['vakiintunutpalkka']=old_wage/12
            p['saa_ansiopaivarahaa']=1
        elif employment_state==7: # hoitovapaa
            p['kotihoidontuella']=1
            if self.include_children:
                p['lapsia']=children_under18
                p['paivahoidossa']=0
                p['alle_3v']=children_under7
                p['alle_kouluikaisia']=children_under7
                p['lapsia_kotihoidontuella']=children_under7
            else:
                p['lapsia']=1
                p['paivahoidossa']=1
                p['alle_3v']=1
                p['lapsia_kotihoidontuella']=p['lapsia']
                p['alle_kouluikaisia']=1
            p['kotihoidontuki_kesto']=time_in_state
            p['t']=0
            p['vakiintunutpalkka']=old_wage/12
            p['saa_ansiopaivarahaa']=0
        elif employment_state==2: # vanhuuseläke
            if ika>self.min_retirementage:
                p['tyoton']=0
                p['saa_ansiopaivarahaa']=0
                p['t']=0
                p['vakiintunutpalkka']=0
                p['elakkeella']=1  
                p['tyoelake']=pension/12
            else:
                p['tyoton']=0
                p['saa_ansiopaivarahaa']=0
                p['t']=0
                p['vakiintunutpalkka']=0
                p['elakkeella']=0
                p['tyoelake']=0
        elif employment_state==8: # ve+työ
            p['tyoton']=0
            p['saa_ansiopaivarahaa']=0
            p['t']=wage/12
            p['vakiintunutpalkka']=0
            p['elakkeella']=1  
            p['tyoelake']=pension/12
        elif employment_state==9: # ve+osatyö
            p['tyoton']=0
            p['saa_ansiopaivarahaa']=0
            p['t']=wage/12
            p['vakiintunutpalkka']=0
            p['elakkeella']=1  
            p['tyoelake']=pension/12
        elif employment_state==10: # osa-aikatyö
            p['tyoton']=0
            p['saa_ansiopaivarahaa']=0
            p['t']=wage/12
            p['vakiintunutpalkka']=0
        elif employment_state==11: # työelämän ulkopuolella
            p['tyoton']=0
            p['toimeentulotuki_vahennys']=0 # oletetaan että ei kieltäytynyt työstä
            p['saa_ansiopaivarahaa']=0
            p['t']=0
            p['vakiintunutpalkka']=0
        elif employment_state==12: # opiskelija
            p['tyoton']=0
            p['opiskelija']=1
            p['saa_ansiopaivarahaa']=0
            p['t']=0
            p['vakiintunutpalkka']=0
        elif employment_state==14: # armeijassa, korjaa! ei tosin vaikuta tuloksiin.
            p['tyoton']=0
            p['opiskelija']=1
            p['saa_ansiopaivarahaa']=0
            p['t']=0
            p['vakiintunutpalkka']=0
        else:
            print('Unknown employment_state ',employment_state)

        # tarkastellaan yksinasuvia henkilöitä
        if employment_state==12: # opiskelija
            p['asumismenot_toimeentulo']=250
            p['asumismenot_asumistuki']=250
        else: # muu
            p['asumismenot_toimeentulo']=500
            p['asumismenot_asumistuki']=500

        p['ansiopvrahan_suojaosa']=1
        p['ansiopvraha_lapsikorotus']=1
        p['puoliso_tulot']=0
        p['puoliso_tyoton']=0  
        p['puoliso_vakiintunutpalkka']=0  
        p['puoliso_saa_ansiopaivarahaa']=0

        netto,benefitq=self.ben.laske_tulot(p)
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

    def get_mort_rate(self,debug=False):
        '''
        Kuolleisuus-intensiteetit eri ryhmille
        '''
        mort=np.zeros((101,self.n_groups))
        if debug:
            dfactor=np.array([1.0,1.0,1.0,1.0,1.0,1.0])
        else:
            dfactor=np.array([1.3,1.0,0.8,1.15,1.0,0.85])
        # tilastokeskuksen kuolleisuusdata 2017 sukupuolittain
        mort[:,1]=np.array([2.12,0.32,0.17,0.07,0.07,0.10,0.00,0.09,0.03,0.13,0.03,0.07,0.10,0.10,0.10,0.23,0.50,0.52,0.42,0.87,0.79,0.66,0.71,0.69,0.98,0.80,0.77,1.07,0.97,0.76,0.83,1.03,0.98,1.20,1.03,0.76,1.22,1.29,1.10,1.26,1.37,1.43,1.71,2.32,2.22,1.89,2.05,2.15,2.71,2.96,3.52,3.54,4.30,4.34,5.09,4.75,6.17,5.88,6.67,8.00,9.20,10.52,10.30,12.26,12.74,13.22,15.03,17.24,18.14,17.78,20.35,25.57,23.53,26.50,28.57,31.87,34.65,40.88,42.43,52.28,59.26,62.92,68.86,72.70,94.04,99.88,113.11,128.52,147.96,161.89,175.99,199.39,212.52,248.32,260.47,284.01,319.98,349.28,301.37,370.17,370.17])/1000.0
        mort[:,0]=dfactor[0]*mort[:,1]
        mort[:,2]=dfactor[2]*mort[:,1]
        mort[:,4]=np.array([1.89,0.30,0.11,0.03,0.14,0.03,0.16,0.07,0.13,0.03,0.00,0.07,0.07,0.07,0.18,0.14,0.07,0.31,0.31,0.30,0.33,0.26,0.18,0.33,0.56,0.17,0.32,0.29,0.35,0.24,0.55,0.35,0.23,0.39,0.48,0.38,0.35,0.80,0.42,0.65,0.50,0.68,0.80,1.12,0.99,0.88,1.13,1.01,1.07,1.68,1.79,2.16,1.87,2.32,2.67,2.69,2.88,2.86,3.73,4.19,3.66,4.97,5.20,5.52,6.05,7.17,7.48,7.32,8.88,10.33,10.72,12.77,12.13,13.30,16.18,18.30,17.50,24.63,26.53,29.88,32.65,38.88,46.95,51.30,60.00,64.73,79.35,90.94,105.11,118.46,141.44,155.07,163.11,198.45,207.92,237.21,254.75,311.31,299.59,356.64,356.64])/1000.0
        mort[:,3]=dfactor[3]*mort[:,4]
        mort[:,5]=dfactor[5]*mort[:,4]

        return mort

    def get_student_rate(self,debug=False):
        '''
        opiskelijoiden intensiteetit eri ryhmille
        '''
        inrate=np.zeros((101,self.n_groups))
        miehet_in=np.array([0.15202 ,0.09165 ,0.08517 ,0.07565 ,0.05787 ,0.04162 ,0.03061 ,0.02336 ,0.01803 ,0.01439 ,0.03214 ,0.02674 ,0.02122 ,0.02005 ,0.01776 ,0.01610 ,0.01490 ,0.01433 ,0.01307 ,0.01175 ,0.01081 ,0.01069 ,0.00921 ,0.00832 ,0.00808 ,0.00783 ,0.00738 ,0.00727 ,0.00712 ,0.00621 ,0.00578 ,0.00540 ,0.00505 ,0.00411 ,0.00434 ,0.00392 ,0.00415 ,0.00362 ,0.00279 ,0.00232 ,0.00184 ,0.00196 ,0.00126 ,0.00239 ,0.00402 ,0.00587 ,0.00587 ,0.00754 ,0 ,0 ])
        naiset_in=np.array([0.12538 ,0.09262 ,0.08467 ,0.06923 ,0.05144 ,0.03959 ,0.03101 ,0.02430 ,0.02103 ,0.01834 ,0.03984 ,0.03576 ,0.03300 ,0.03115 ,0.02934 ,0.02777 ,0.02454 ,0.02261 ,0.02127 ,0.01865 ,0.01711 ,0.01631 ,0.01496 ,0.01325 ,0.01251 ,0.01158 ,0.01148 ,0.01034 ,0.00935 ,0.00911 ,0.00848 ,0.00674 ,0.00636 ,0.00642 ,0.00605 ,0.00517 ,0.00501 ,0.00392 ,0.00330 ,0.00291 ,0.00202 ,0.00155 ,0.00118 ,0.00193 ,0.00376 ,0.00567 ,0.00779 ,0.00746 ,0 ,0 ])
        inrate[20:70,0] =miehet_in
        inrate[20:70,1] =miehet_in
        inrate[20:70,2] =miehet_in
        inrate[20:70,3] =naiset_in
        inrate[20:70,4] =naiset_in
        inrate[20:70,5] =naiset_in
        outrate=np.zeros((101,self.n_groups))
        miehet_ulos=np.array([0.20000,0.20000,0.27503,0.38096,0.43268,0.42941,0.41466,0.40854,0.38759,0.30057,0.66059,0.69549,0.55428,0.61274,0.58602,0.57329,0.53688,0.58737,0.59576,0.58190,0.50682,0.63749,0.59542,0.53201,0.53429,0.55827,0.51792,0.52038,0.63078,0.57287,0.57201,0.56673,0.69290,0.44986,0.60497,0.45890,0.64129,0.73762,0.68664,0.73908,0.47708,0.92437,0.27979,0.54998,0.60635,0.72281,0.45596,0.48120,0.41834,0.55567])
        naiset_ulos=np.array([0.2,0.226044511,0.34859165,0.404346193,0.378947854,0.379027678,0.393658729,0.312799282,0.312126148,0.325150199,0.5946454,0.564144808,0.555376244,0.556615568,0.545757439,0.61520002,0.577306728,0.558805476,0.618014582,0.584596312,0.542579298,0.581755996,0.612559266,0.559683811,0.577041852,0.51024909,0.602288269,0.594473782,0.529303275,0.573062208,0.709297989,0.559692954,0.499632245,0.560546551,0.654820741,0.547514252,0.728319756,0.668454496,0.637200351,0.832907039,0.763936815,0.823014939,0.439925972,0.400593267,0.57729364,0.432838681,0.720728303,0.45569566,0.756655823,0.210470698])
        outrate[20:70,0]=miehet_ulos
        outrate[20:70,1]=miehet_ulos
        outrate[20:70,2]=miehet_ulos
        outrate[20:70,3]=naiset_ulos
        outrate[20:70,4]=naiset_ulos
        outrate[20:70,5]=naiset_ulos

        return inrate,outrate

    def get_outsider_rate_old(self,debug=False):
        '''
        sairauspäivärahalle jäävien osuudet
        '''
        inrate=np.zeros((101,self.n_groups))
        max_spv=70
        
        miehet_in=np.array([0.00598,0.00236,0.00195,0.00179,0.00222,0.00150,0.00363,0.00142,0.00138,0.00149,0.00561,0.00140,0.00291,0.00390,0.00130,0.00548,0.00120,0.00476,0.00118,0.00315,0.00111,0.00346,0.00117,0.00203,0.00105,0.00189,0.00154,0.00104,0.00488,0.00103,0.00273,0.00104,0.00375,0.00108,0.00314,0.00256,0.00188,0.00115,0.00115,0.00112,0.00112,0.00106,0.00112,0.00000,0.00000,0.00000,0.00257,0.00359,0,0 ])
        naiset_in=np.array([0.00246,0.00210,0.00212,0.00211,0.00205,0.00217,0.00233,0.00355,0.00246,0.00247,0.00248,0.00239,0.00238,0.00225,0.00209,0.00194,0.00179,0.01151,0.00823,0.00802,0.00990,0.00515,0.00418,0.00644,0.00334,0.00101,0.00098,0.00256,0.00093,0.00092,0.00089,0.00172,0.00089,0.00248,0.00107,0.00170,0.00105,0.00143,0.00140,0.00233,0.00108,0.00104,0.00112,0.00000,0.00000,0.00000,0.00000,0.00000,0,0 ])
        inrate[20:max_spv,0] =miehet_in
        inrate[20:max_spv,1] =miehet_in
        inrate[20:max_spv,2] =miehet_in
        inrate[20:max_spv,3] =naiset_in
        inrate[20:max_spv,4] =naiset_in
        inrate[20:max_spv,5] =naiset_in
        outrate=np.zeros((101,self.n_groups))
        miehet_ulos=np.array([0.54528,0.21972,0.05225,0.08766,0.02000,0.07014,0.02000,0.07964,0.05357,0.02000,0.02000,0.12421,0.02000,0.02000,0.09464,0.02000,0.06655,0.02000,0.04816,0.02000,0.09763,0.02000,0.02000,0.02000,0.03777,0.02000,0.02000,0.10725,0.02000,0.05159,0.02000,0.04831,0.02000,0.08232,0.02000,0.02000,0.02000,0.02931,0.07298,0.05129,0.11783,0.07846,0.45489,0.58986,0.15937,0.43817,0.00000,0.00000,0.25798,0.00000])
        naiset_ulos=np.array([0.47839484,0.190435122,0.12086902,0.081182033,0.030748876,0.184119897,0.075833908,0.02,0.029741112,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.032506855,0.026333043,0.02,0.023692146,0.050057587,0.037561449,0.02,0.024524018,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.092785925,0.054435714,0.439187202,0.465046705,0.39008036,0.384356347,0.169971142,0.031645066,0,0])
        outrate[20:max_spv,0]=miehet_ulos
        outrate[20:max_spv,1]=miehet_ulos
        outrate[20:max_spv,2]=miehet_ulos
        outrate[20:max_spv,3]=naiset_ulos
        outrate[20:max_spv,4]=naiset_ulos
        outrate[20:max_spv,5]=naiset_ulos

        return inrate,outrate

    def get_outsider_rate(self,debug=False):
        '''
        sairauspäivärahalle jäävien osuudet
        '''
        inrate=np.zeros((101,self.n_groups))
        max_spv=70
        
        #miehet_in=np.array([0.00598,0.00236,0.00195,0.00179,0.00222,0.00150,0.00363,0.00142,0.00138,0.00149,0.00561,0.00140,0.00291,0.00390,0.00130,0.00548,0.00120,0.00476,0.00118,0.00315,0.00111,0.00346,0.00117,0.00203,0.00105,0.00189,0.00154,0.00104,0.00488,0.00103,0.00273,0.00104,0.00375,0.00108,0.00314,0.00256,0.00188,0.00115,0.00115,0.00112,0.00112,0.00106,0.00112,0.00000,0.00000,0.00000,0.00257,0.00359,0,0 ])
        #miehet_in=np.array([0.00578,0.00226,0.00187,0.00170,0.00248,0.00143,0.00230,0.00134,0.00130,0.00213,0.00450,0.00132,0.00300,0.00353,0.00123,0.00481,0.00112,0.00364,0.00109,0.00295,0.00103,0.00335,0.00095,0.00213,0.00095,0.00094,0.00148,0.00093,0.00400,0.00093,0.00342,0.00097,0.00370,0.00099,0.00259,0.00221,0.00244,0.00106,0.00102,0.00101,0.00099,0.00098,0.00095,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0 ])
        miehet_in=np.array([0.00578,0.00226,0.00187,0.00170,0.00153,0.00143,0.00137,0.00134,0.00130,0.00129,0.00210,0.00132,0.00348,0.00358,0.00123,0.00312,0.00112,0.00368,0.00109,0.00162,0.00103,0.00271,0.00095,0.00252,0.00095,0.00094,0.00093,0.00093,0.00400,0.00093,0.00342,0.00097,0.00370,0.00099,0.00259,0.00221,0.00244,0.00106,0.00102,0.00101,0.00099,0.00098,0.00095,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.0 ])
        #naiset_in=np.array([0.00246,0.00210,0.00212,0.00211,0.00205,0.00217,0.00233,0.00355,0.00246,0.00247,0.00248,0.00239,0.00238,0.00225,0.00209,0.00194,0.00179,0.01151,0.00823,0.00802,0.00990,0.00515,0.00418,0.00644,0.00334,0.00101,0.00098,0.00256,0.00093,0.00092,0.00089,0.00172,0.00089,0.00248,0.00107,0.00170,0.00105,0.00143,0.00140,0.00233,0.00108,0.00104,0.00112,0.00000,0.00000,0.00000,0.00000,0.00000,0,0 ])
        #naiset_in=np.array([0.00236,0.00203,0.00205,0.00206,0.00198,0.00210,0.00228,0.00227,0.00241,0.00241,0.00242,0.00234,0.00231,0.00217,0.00202,0.00187,0.00169,0.00817,0.00764,0.00667,0.00984,0.00532,0.00407,0.00593,0.00370,0.00092,0.00089,0.00326,0.00085,0.00084,0.00083,0.00142,0.00080,0.00295,0.00187,0.00086,0.00118,0.00089,0.00166,0.00100,0.00094,0.00092,0.00097,0.00000,0.00000,0.00000,0.00000,0.00000,0,0 ])
        naiset_in=np.array([0.00236,0.00203,0.00205,0.00206,0.00198,0.00210,0.00228,0.00227,0.00241,0.00241,0.00242,0.00234,0.00231,0.00217,0.00202,0.00187,0.01046,0.00997,0.00293,0.00918,0.00231,0.00401,0.00850,0.00266,0.00394,0.00172,0.00089,0.00262,0.00113,0.00084,0.00083,0.00142,0.00080,0.00295,0.00187,0.00086,0.00118,0.00089,0.00166,0.00100,0.00094,0.00092,0.00097,0.00000,0.00000,0.00000,0.00000,0.00000,0.0,0.00])
        inrate[20:max_spv,0] =miehet_in
        inrate[20:max_spv,1] =miehet_in
        inrate[20:max_spv,2] =miehet_in
        inrate[20:max_spv,3] =naiset_in
        inrate[20:max_spv,4] =naiset_in
        inrate[20:max_spv,5] =naiset_in
        outrate=np.zeros((101,self.n_groups))
        #miehet_ulos=np.array([0.54528,0.21972,0.05225,0.08766,0.02000,0.07014,0.02000,0.07964,0.05357,0.02000,0.02000,0.12421,0.02000,0.02000,0.09464,0.02000,0.06655,0.02000,0.04816,0.02000,0.09763,0.02000,0.02000,0.02000,0.03777,0.02000,0.02000,0.10725,0.02000,0.05159,0.02000,0.04831,0.02000,0.08232,0.02000,0.02000,0.02000,0.02931,0.07298,0.05129,0.11783,0.07846,0.45489,0.58986,0.15937,0.43817,0.00000,0.00000,0.25798,0.00000])
        #miehet_ulos=np.array([0.55164,0.21920,0.06385,0.08468,0.02000,0.07242,0.02000,0.08007,0.05701,0.02000,0.02000,0.11639,0.02000,0.02000,0.10085,0.02000,0.06289,0.02000,0.03766,0.02000,0.10860,0.02000,0.02765,0.02000,0.03089,0.02520,0.02000,0.08840,0.02000,0.04616,0.02000,0.06061,0.02000,0.08866,0.02000,0.02000,0.02000,0.07034,0.04439,0.08118,0.06923,0.16061,0.51689,0.55980,0.23310,0.25554,0.01519,0.12491,0.06625,0.00000])
        miehet_ulos=np.array([0.54333,0.18208,0.09452,0.06729,0.05128,0.03646,0.02952,0.04198,0.02505,0.02711,0.02000,0.07864,0.02000,0.02000,0.09971,0.02000,0.05071,0.02000,0.03236,0.02000,0.09185,0.02000,0.03203,0.02000,0.03167,0.03064,0.02161,0.08480,0.02000,0.04616,0.02000,0.06061,0.02000,0.08866,0.02000,0.02000,0.02000,0.07034,0.04439,0.08118,0.06923,0.16061,0.51689,0.55980,0.23310,0.25554,0.01519,0.12491,0.06625,0.00000]) 
        #naiset_ulos=np.array([0.47839484,0.190435122,0.12086902,0.081182033,0.030748876,0.184119897,0.075833908,0.02,0.029741112,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.032506855,0.026333043,0.02,0.023692146,0.050057587,0.037561449,0.02,0.024524018,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.092785925,0.054435714,0.439187202,0.465046705,0.39008036,0.384356347,0.169971142,0.031645066,0,0])
        #naiset_ulos=np.array([0.485631713,0.198175734,0.114871531,0.090719936,0.034279871,0.183220816,0.054071766,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.032778805,0.03154085,0.02,0.021885099,0.03347865,0.070837788,0.02,0.032940802,0.02,0.02,0.02027967,0.02,0.043477638,0.02,0.02,0.080038155,0.071876772,0.477291934,0.454819524,0.428913696,0.287380262,0.140803001,0.054164949,0,0])
        naiset_ulos=np.array([0.371419539,0.205661569,0.135265873,0.102702654,0.055240889,0.048992378,0.107111533,0.059592465,0.032056939,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.06991043,0.02,0.02,0.036157545,0.070163829,0.02,0.032241992,0.02,0.02,0.02027967,0.02,0.043477638,0.02,0.02,0.080038155,0.071876772,0.477291934,0.454819524,0.428913696,0.287380262,0.140803001,0.054164949,0,0])
        outrate[20:max_spv,0]=miehet_ulos
        outrate[20:max_spv,1]=miehet_ulos
        outrate[20:max_spv,2]=miehet_ulos
        outrate[20:max_spv,3]=naiset_ulos
        outrate[20:max_spv,4]=naiset_ulos
        outrate[20:max_spv,5]=naiset_ulos

        return inrate,outrate

    def get_army_rate(self,debug=False):
        '''
        armeija intensiteetit eri ryhmille
        '''
        outrate=np.zeros((101,self.n_groups))
        miehet_ulos=np.array([0.826082957,0.593698994,0.366283368,0.43758429,0.219910436,0.367689675,0.111588214,0.234498521,0.5,0.96438943,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
        naiset_ulos=np.array([0.506854911,0.619103706,0.181591468,0.518294319,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
        outrate[20:70,0]=miehet_ulos
        outrate[20:70,1]=miehet_ulos
        outrate[20:70,2]=miehet_ulos
        outrate[20:70,3]=naiset_ulos
        outrate[20:70,4]=naiset_ulos
        outrate[20:70,5]=naiset_ulos

        return outrate
        
    def get_disability_rate(self,debug=False):
        '''
        Työkyvyttömyys-alkavuudet eri ryhmille
        Data ETK:n tilastotietokannasta ja skaalattu ikäluokittaisillä miesten ja naisten määrillä
        '''
        disab=np.zeros((self.max_age+1,self.n_groups))
        # male low, male mid, male high, female low, female mid, female high
        if debug:
            dfactor=np.array([1.0,1.0,1.0,1.0,1.0,1.0])
        else:
             # uusitalon selvityksestä Työkyvyttömyyden vuoksi menetetty työura
             # skaalattu alaspäin, jotta tk:laisten kokonaismäärä menee paremmin oikein
            dfactor=np.array([1.2,0.8,0.4,1.1,0.8,0.5])*0.9
        
        dis_miehet=np.array([0.004697942,0.004435302,0.003631736,0.003141361,0.003457091,0.003005607,0.002905609,0.003029283,0.002289213,0.002137714,0.001854558,0.002813517,0.002607335,0.00292628,0.002937462,0.002784612,0.002846377,0.002776506,0.003017675,0.003129845,0.003349059,0.002991577,0.00305634,0.003446143,0.003633971,0.004045113,0.004002001,0.004517725,0.005527525,0.005565513,0.006319492,0.007399175,0.00731299,0.009142823,0.010254463,0.011784364,0.013783743,0.015299156,0.018282001,0.024051257,0.032338044,0.028290544,0.019444444,0.00454486,0.000330718,0,0,0,0,0,0])
        dis_naiset=np.array([0.00532654,0.004917401,0.00453191,0.003799551,0.003253733,0.003092307,0.002822592,0.003309772,0.002482279,0.002615887,0.002416545,0.003546203,0.002665276,0.003095104,0.003129633,0.003406418,0.003171677,0.003320357,0.003391292,0.004007371,0.004310094,0.00438571,0.004267343,0.004889399,0.005043702,0.005793425,0.005569451,0.006298434,0.006363081,0.007043361,0.009389811,0.007457667,0.009251373,0.011154836,0.009524088,0.013689796,0.014658423,0.017440417,0.022804727,0.02677838,0.037438459,0.034691279,0.022649573,0.004414073,0.000264568,0,0,0,0,0,0])
        # ei varhaiseläkkeitä mukana, joten oletetaan ettei tk-intensiteetti laske
        dis_miehet[41:51]=np.maximum(dis_miehet[41:51],0.02829054)
        dis_naiset[41:51]=np.maximum(dis_naiset[41:51],0.03469128)
            
        for g in range(3):
            disab[20:71,g]=dfactor[g]*dis_miehet
            disab[70:(self.max_age+1),g]=24.45*dfactor[g]/1000
        for g in range(3,6):
            disab[20:71,g]=dfactor[g]*dis_naiset
            disab[70:(self.max_age+1),g]=24.45*dfactor[g]/1000

        return disab        
        
    def get_eff_disab_rate(self,debug=False):
        '''
        Työkyvyttömyys-alkavuudet eri ryhmille
        Laskettu havaitusta työkyvyttömien lukumäärästä
        Siksi efektiivinen 
        '''
        disab=np.zeros((self.max_age+1,self.n_groups))
        # male low, male mid, male high, female low, female mid, female high
        if debug:
            dfactor=np.array([1.0,1.0,1.0,1.0,1.0,1.0])
        else:
             # uusitalon selvityksestä Työkyvyttömyyden vuoksi menetetty työura
             # skaalattu alaspäin, jotta tk:laisten kokonaismäärä menee paremmin oikein
            dfactor=np.array([1.3,0.95,0.6,1.2,1.0,0.9])
            
        dis_miehet=np.array([0.0068168,0.003341014,0,0.004279685,0.001118673,0.001802593,0.00217149,0,0,0.002157641,0,0.002545172,0,0.002960375,0.000767293,0,0.002265829,0.000286527,0,0.004899931,0,0.000677208,0.001155069,0.003796412,0.004896709,0.001921327,0.004668376,0.004630126,0.002478899,0.00642266,0.005795605,0.00558426,0.008096878,0.004548654,0.010179089,0.016100661,0.015144889,0.011688053,0.024563474,0.036719657,0.036573355,0.026898066,0.027508352,0.024176173,0.023621633,0.02058014,0.020290345,0.0202976,0.020304995,0.020282729,0.020282729])
        dis_naiset=np.array([0.004962318,0.002850008,0.004703008,0,0.001625749,0.000940874,0.001050232,0,0,4.34852E-05,0.003516261,0,8.21901E-05,0.002276047,0.000443789,0.002472653,0,0.001866348,0.002269429,0.001480588,0.00139571,0.002185668,0.002003531,0.003662852,0.003271301,0.003629155,0.002690071,0.003977974,0.005051223,0.00303663,0.008097507,0.004912787,0.005008356,0.007536173,0.007618452,0.017496524,0.012431715,0.020801345,0.025163258,0.027521298,0.039852895,0.023791604,0.025422742,0.02230225,0.021684456,0.01894045,0.018676988,0.018654938,0.01865384,0.018650795,0.018650795])
            
        for g in range(3):
            disab[20:71,g]=dfactor[g]*dis_miehet
            disab[70:(self.max_age+1),g]=24.45*dfactor[g]/1000
        for g in range(3,6):
            disab[20:71,g]=dfactor[g]*dis_naiset
            disab[70:(self.max_age+1),g]=24.45*dfactor[g]/1000

        return disab        

    def get_birth_rate(self,debug=False):
        '''
        Syntyvyysdata
        '''
        birth=np.zeros((self.max_age+1,self.n_groups))
        if debug:
            dfactor=np.array([1.0,1.0,1.0,1.0,1.0,1.0])
        else:
            dfactor=np.array([0.75,1.0,1.25,0.5,1.0,1.5])
        for g in range(self.n_groups):
            factor=dfactor[g] # tämä vaikeuttaa sovitetta
            birth[15,g]=0.000177167*factor
            birth[16,g]=0.001049171*factor
            birth[17,g]=0.002303504*factor
            birth[18,g]=0.00630474*factor
            birth[19,g]=0.014399394*factor
            birth[20,g]=0.023042239*factor
            birth[21,g]=0.03088312*factor
            birth[22,g]=0.039755923*factor
            birth[23,g]=0.047483352*factor
            birth[24,g]=0.055630287*factor
            birth[25,g]=0.067942889*factor
            birth[26,g]=0.077108925*factor
            birth[27,g]=0.085396679*factor
            birth[28,g]=0.096968809*factor
            birth[29,g]=0.10081728*factor
            birth[30,g]=0.105586061*factor
            birth[31,g]=0.1124004*factor
            birth[32,g]=0.102667839*factor
            birth[33,g]=0.098528489*factor
            birth[34,g]=0.084080311*factor
            birth[35,g]=0.072335459*factor
            birth[36,g]=0.065203338*factor
            birth[37,g]=0.053073374*factor
            birth[38,g]=0.044054569*factor
            birth[39,g]=0.032984136*factor
            birth[40,g]=0.024135797*factor
            birth[41,g]=0.0174215*factor
            birth[42,g]=0.011621238*factor
            birth[43,g]=0.006909705*factor
            birth[44,g]=0.003977037*factor
            birth[45,g]=0.002171444*factor
            birth[46,g]=0.00115119*factor
            birth[47,g]=0.000712692*factor
            birth[48,g]=9.16478E-05*factor
            birth[49,g]=0.000113167*factor

        # syntyvyys on lasten määrä suhteessa naisten määrään
        # ei siis tarvetta kertoa kahdella, vaikka isät pääsevät isyysvapaalle

        return birth

    def scale_pension(self,pension,age,scale=True,unemp_after_ra=0):
        '''
        Elinaikakertoimen ja lykkäyskorotuksen huomiointi
        '''
        if scale:
            return self.elinaikakerroin*pension*self.elakeindeksi*(1+0.048*(age-self.min_retirementage-unemp_after_ra)) 
        else:
            return self.elinaikakerroin*pension*self.elakeindeksi
        
    def move_to_parttime(self,pension,old_wage,age,tyoura,time_in_state,wage_reduction,children_under7,children_under18):
        '''
        Siirtymä osa-aikaiseen työskentelyyn
        '''
        employment_status = 10 # switch to part-time work
        intage=int(np.floor(age))
        wage=self.get_wage(intage,wage_reduction)
        parttimewage=0.5*self.get_wage(intage,wage_reduction)
        tyoura += self.timestep
        time_in_state=0
        old_wage=0
        pension=self.pension_accrual(age,parttimewage,pension,state=employment_status)
        netto,benq=self.comp_benefits(parttimewage,old_wage,0,employment_status,time_in_state,children_under7,children_under18,age,retq=True)
        pinkslip=0
        time_in_state=self.timestep
        wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)        

        return employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq

    def move_to_OVE_parttime(self,pension,old_wage,age,tyoura,time_in_state,wage_reduction,children_under7,children_under18):
        '''
        Siirtymä osa-aikaiseen työskentelyyn
        '''
        employment_status = 15 # switch to part-time work
        intage=int(np.floor(age))
        wage=self.get_wage(intage,wage_reduction)
        parttimewage=0.5*self.get_wage(intage,wage_reduction)
        tyoura += self.timestep
        time_in_state=0
        old_wage=0
        paid_pension = self.scale_pension(0.5*pension,age,scale=scale_acc)
        pension=self.pension_accrual(age,parttimewage,0.5*pension,state=employment_status)
        netto,benq=self.comp_benefits(parttimewage,old_wage,0,employment_status,time_in_state,children_under7,children_under18,age,retq=True)
        pinkslip=0
        time_in_state=self.timestep
        wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)        

        return employment_status,pension,paid_pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq

    def move_to_OVE_fulltime(self,pension,old_wage,age,tyoura,time_in_state,wage_reduction,children_under7,children_under18):
        '''
        Siirtymä osa-aikaiseen työskentelyyn
        '''
        employment_status = 16 # switch to part-time work
        intage=int(np.floor(age))
        wage=self.get_wage(intage,wage_reduction)
        tyoura += self.timestep
        time_in_state=0
        old_wage=0
        paid_pension = self.scale_pension(0.5*pension,age,scale=scale_acc)
        pension=self.pension_accrual(age,wage,0.5*pension,state=employment_status)
        netto,benq=self.comp_benefits(wage,old_wage,0,employment_status,time_in_state,children_under7,children_under18,age,retq=True)
        pinkslip=0
        time_in_state=self.timestep
        wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)        

        return employment_status,pension,paid_pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq

    def move_to_OVE_unemp(self,pension,old_wage,age,tyoura,time_in_state,wage_reduction,children_under7,children_under18):
        '''
        Siirtymä osa-aikaiseen työskentelyyn
        '''
        employment_status = 17 # switch to part-time work
        intage=int(np.floor(age))
        wage=self.get_wage(intage,wage_reduction)
        tyoura += self.timestep
        time_in_state=0
        old_wage=0
        paid_pension = self.scale_pension(0.5*pension,age,scale=scale_acc)
        pension=self.pension_accrual(age,0,0.5*pension,state=employment_status)
        netto,benq=self.comp_benefits(0,old_wage,0,employment_status,time_in_state,children_under7,children_under18,age,retq=True)
        pinkslip=0
        time_in_state=self.timestep
        wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)        

        return employment_status,pension,paid_pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq

    def move_to_work(self,pension,old_wage,age,time_in_state,tyoura,pinkslip,wage_reduction,children_under7,children_under18):
        '''
        Siirtymä täysiaikaiseen työskentelyyn
        '''
        employment_status = 1 # töihin
        intage=int(np.floor(age))
        wage=self.get_wage(intage,wage_reduction,pinkslip=pinkslip)
        time_in_state=0
        old_wage=0
        #toe=min(self.max_toe,toe+self.timestep)
        tyoura+=self.timestep
        #pinkslip=0
        pension=self.pension_accrual(age,wage,pension,state=employment_status) # poistettu *0.5
        netto,benq=self.comp_benefits(wage,old_wage,0,employment_status,time_in_state,children_under7,children_under18,age)
        time_in_state=self.timestep
        wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)        

        return employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq

    def move_to_retwork(self,pension,old_wage,age,time_in_state,paid_pension,wage_reduction,children_under7,children_under18):
        '''
        Siirtymä vanhuuseläkkeellä työskentelyyn
        '''
        employment_status = 8 # unchanged
        intage=int(np.floor(age))
        wage=self.get_wage(intage,wage_reduction)
        ptwage=wage*0.5
        paid_pension=paid_pension*self.elakeindeksi
        pension=self.pension_accrual(age,ptwage,pension,state=employment_status)
        time_in_state=0
        netto,benq=self.comp_benefits(ptwage,0,paid_pension,employment_status,time_in_state,children_under7,children_under18,age)
        time_in_state+=self.timestep
        wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)        
        
        # tyoura+= ??

        return employment_status,pension,wage,time_in_state,netto,wage_reduction,benq

    def move_to_student(self,pension,old_wage,age,time_in_state,tyoura,pinkslip,wage_reduction,children_under7,children_under18):
        '''
        Siirtymä opiskelijaksi
        Tässä ei muuttujien päivityksiä, koska se tehdään jo muualla!
        '''
        employment_status = 12
        intage=int(np.floor(age))
        time_in_state=0
        netto,benq=self.comp_benefits(0,0,0,employment_status,time_in_state,children_under7,children_under18,age)
        time_in_state+=self.timestep
        wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)        
        pinkslip=0
        wage=old_wage

        return employment_status,pension,wage,time_in_state,netto,pinkslip,wage_reduction,benq

    def move_to_retpartwork(self,pension,old_wage,age,time_in_state,paid_pension,wage_reduction,children_under7,children_under18):
        '''
        Siirtymä osa-aikaiseen vanhuuseläkkeellä työskentelyyn
        '''
        employment_status = 9 # unchanged
        intage=int(np.floor(age))
        wage=self.get_wage(intage,wage_reduction)
        ptwage=0.5*wage
        paid_pension=paid_pension*self.elakeindeksi
        pension=self.pension_accrual(age,ptwage,pension,state=employment_status)
        time_in_state=0
        netto,benq=self.comp_benefits(ptwage,0,paid_pension,employment_status,time_in_state,children_under7,children_under18,age)
        time_in_state+=self.timestep
        wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)        
        # tyoura+= ??

        return employment_status,pension,wage,time_in_state,netto,wage_reduction,benq

    def move_to_retirement(self,pension,old_wage,age,paid_pension,employment_status,
            wage_reduction,unemp_after_ra,children_under7,children_under18,all_acc=True,scale_acc=True):
        '''
        Siirtymä vanhuuseläkkeelle
        '''
        if age>=self.min_retirementage:
            if all_acc:
                if employment_status in set([2,8,9]): # ve, ve+työ, ve+osatyö
                    if age>=self.max_retirementage:
                        # ei lykkäyskorotusta
                        paid_pension = self.elinaikakerroin*self.elakeindeksi*pension+self.elakeindeksi*paid_pension
                        pension=0
                    else:
                        paid_pension = self.elakeindeksi*paid_pension
                elif employment_status==3: # tk
                    # do nothing
                    employment_status=3
                elif employment_status in set({15,16,17}): # OVE+oa tai OVE+työ tai OVE+työtön
                    paid_pension = self.scale_pension(pension,age,scale=scale_acc,unemp_after_ra=unemp_after_ra)+self.elakeindeksi*paid_pension
                    paid_pension += self.ben.laske_kansanelake(age,paid_pension/12,1)*12 # ben-modulissa palkat kk-tasolla
                    pension=0
                else:
                    # lykkäyskorotus
                    paid_pension = self.scale_pension(pension,age,scale=scale_acc,unemp_after_ra=unemp_after_ra)
                    paid_pension += self.ben.laske_kansanelake(age,paid_pension/12,1)*12 # ben-modulissa palkat kk-tasolla
                    pension=0

            time_in_state=self.timestep
            employment_status = 2 
            wage=old_wage
            netto,benq=self.comp_benefits(0,0,paid_pension,employment_status,0,children_under7,children_under18,age)
            wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)        
        else: # työvoiman ulkopuolella
            time_in_state=0
            employment_status = 2 
            wage=old_wage
            netto,benq=self.comp_benefits(0,0,0,employment_status,0,children_under7,children_under18,age)
            time_in_state+=self.timestep
            wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)        

        return employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,benq

    def move_to_retdisab(self,pension,old_wage,age,paid_pension,employment_status,wage_reduction,children_under7,children_under18):   
        '''
        Siirtymä vanhuuseläkkeelle, jossa ei voi tehdä työtä
        '''
        
        if age>=self.max_retirementage:
            paid_pension= self.elinaikakerroin*self.elakeindeksi*pension+self.elakeindeksi*paid_pension
            pension=0                        

        employment_status = 3
        wage=old_wage
        netto,benq=self.comp_benefits(0,0,paid_pension,employment_status,0,children_under7,children_under18,age)
        time_in_state=self.timestep
        wage_reduction=0.9

        return employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,benq
        
    def tyossaoloehto(self,toe,tyoura,age):
        if toe>=self.ansiopvraha_toe: # täyttyykö työssäoloehto
            return True
        else:
            return False
    
    def paivarahapaivia_jaljella(self,kesto,tyoura,age):
        if ((tyoura>=self.tyohistoria_vaatimus500 and kesto>=self.ansiopvraha_kesto500 and age>=self.minage_500) \
            or (tyoura>=self.tyohistoria_vaatimus and kesto>=self.ansiopvraha_kesto400 and (age<self.minage_500 or tyoura<self.tyohistoria_vaatimus500)) \
            or (tyoura<self.tyohistoria_vaatimus and kesto>=self.ansiopvraha_kesto300)):    
            return False
        else:
            return True
                #if ((tyohistoria>=self.tyohistoria_vaatimus and p['tyottomyyden_kesto']<=self.ansiopvraha_kesto400) \
                #    or (p['tyottomyyden_kesto']<=self.ansiopvraha_kesto300) \
                #    or (ika>=self.minage_500 and tyohistoria>=self.tyohistoria_vaatimus500 and p['tyottomyyden_kesto']<=self.ansiopvraha_kesto500)) \

    def move_to_unemp(self,pension,old_wage,age,paid_pension,toe,irtisanottu,tyoura,wage_reduction,
                    used_unemp_benefit,unemp_after_ra,unempwage,children_under7,children_under18):
        '''
        Siirtymä työttömyysturvalle
        '''
        if age>=self.min_retirementage: # ei uusia työttömiä enää alimman ve-iän jälkeen, vanhat jatkavat
            pinkslip=0
            employment_status=0
            unempwage_basis=0
            employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,benq=\
                self.move_to_retirement(pension,old_wage,age,paid_pension,employment_status,
                    wage_reduction,unemp_after_ra,children_under7,children_under18,all_acc=True)
                
            return employment_status,paid_pension,pension,wage,time_in_state,netto,\
                   wage_reduction,used_unemp_benefit,pinkslip,benq,unemp_after_ra,unempwage_basis
        else:
            #if toe>=self.ansiopvraha_toe: # täyttyykö työssäoloehto
            if self.tyossaoloehto(toe,tyoura,age):
                kesto=12*21.5*used_unemp_benefit
                #if ((tyoura>=self.tyohistoria_vaatimus500 and kesto>=self.ansiopvraha_kesto500 and age>=self.minage_500) \
                #    or (tyoura>=self.tyohistoria_vaatimus and kesto>=self.ansiopvraha_kesto400 and (age<self.minage_500 or tyoura<self.tyohistoria_vaatimus500)) \
                #    or (tyoura<self.tyohistoria_vaatimus and kesto>=self.ansiopvraha_kesto300)):
                if self.paivarahapaivia_jaljella(kesto,tyoura,age):
                    employment_status  = 0 # siirto ansiosidonnaiselle
                else:
                    if self.include_putki and age>=self.min_tyottputki_ika and tyoura>=self.tyohistoria_tyottputki: 
                        employment_status = 4 # siirto lisäpäiville
                    else:
                        employment_status = 13 # siirto työmarkkinatuelle
            else:
                employment_status  = 13 # siirto työmarkkinatuelle

            time_in_state=0                
            #toe=0 #max(0,toe-self.timestep) # nollataan työssäoloehto
            
            unempwage_basis=unempwage
            intage=int(np.floor(age))
            wage=self.get_wage(intage,wage_reduction)
            pension=self.pension_accrual(age,unempwage_basis,pension,state=employment_status)

            # hmm, omavastuupäivät puuttuvat!
            # omavastuupäiviä on 5/(21.5*12*self.timestep), kerroin tällöin
            # 1-5/(21.5*12*self.timestep)
            netto,benq=self.comp_benefits(0,unempwage_basis,0,employment_status,used_unemp_benefit,children_under7,children_under18,age,
                                     irtisanottu=irtisanottu,tyohistoria=tyoura)
            time_in_state=self.timestep
            #unemp_after_ra ei muutu
            wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)
            if irtisanottu: # muuten ei oikeutta ansiopäivärahaan karenssi vuoksi
                used_unemp_benefit+=self.timestep
                
            pinkslip=irtisanottu

        return employment_status,paid_pension,pension,wage,time_in_state,netto,\
               wage_reduction,used_unemp_benefit,pinkslip,benq,unemp_after_ra,unempwage_basis
               

    def move_to_outsider(self,pension,old_wage,age,irtisanottu,wage_reduction,children_under7,children_under18):
        '''
        Siirtymä työvoiman ulkopuolelle
        '''
        employment_status = 11 # switch
        time_in_state=0
        intage=int(np.floor(age))
        #old_wage=self.get_wage(intage-1,0)
        #toe=max(0,toe-self.timestep)
        wage=old_wage
        pension=pension*self.palkkakerroin

        netto,benq=self.comp_benefits(0,0,0,employment_status,time_in_state,children_under7,children_under18,age,irtisanottu=0)
        paid_pension=0
        time_in_state+=self.timestep
        wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)        
        pinkslip=0

        return employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,pinkslip,benq

    def move_to_disab(self,pension,old_wage,age,wage_reduction,children_under7,children_under18):
        '''
        Siirtymä työkyvyttömyyseläkkeelle
        '''
        employment_status = 3 # tk
        paid_pension=self.elinaikakerroin*pension*self.elakeindeksi + self.acc*old_wage*max(0,self.min_retirementage-age) # p.o. 5v keskiarvo
        paid_pension=self.ben.laske_kokonaiselake(65,paid_pension)
        pension=0
        #old_wage=0
        time_in_state=0
        wage=old_wage
        netto,benq=self.comp_benefits(0,0,paid_pension,employment_status,0,children_under7,children_under18,age)
        time_in_state+=self.timestep
        wage_reduction=0.60 # vastaa määritelmää

        return employment_status,pension,paid_pension,wage,time_in_state,netto,wage_reduction,benq

    def move_to_deceiced(self,pension,old_wage,age,children_under7,children_under18):
        '''
        Siirtymä tilaan kuollut
        '''
        employment_status = 15 # deceiced
        wage=old_wage
        pension=pension
        netto=0
        time_in_state=0

        return employment_status,pension,wage,time_in_state,netto

    def move_to_kht(self,pension,old_wage,age,wage_reduction,children_under7,children_under18):
        '''
        Siirtymä kotihoidontuelle
        '''
        employment_status = 7 # kotihoidontuelle
        wage=old_wage
        pension=self.pension_accrual(age,old_wage,pension,state=7)
        
        time_in_state=0
        netto,benq=self.comp_benefits(0,old_wage,0,employment_status,time_in_state,children_under7,children_under18,age)
        time_in_state+=self.timestep
        wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)        

        return employment_status,pension,wage,time_in_state,netto,wage_reduction,benq

    def move_to_fatherleave(self,pension,old_wage,age,wage_reduction,children_under7,children_under18):
        '''
        Siirtymä isyysvapaalle
        '''
        self.infostate_add_child(age)
        employment_status = 6 # isyysvapaa
        time_in_state=0
        wage=old_wage
        pension=self.pension_accrual(age,old_wage,pension,state=6)
        netto,benq=self.comp_benefits(0,old_wage,0,employment_status,0,children_under7,children_under18,age)
        time_in_state+=self.timestep        
        pinkslip=0
        wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)        
        
        return employment_status,pension,wage,time_in_state,netto,pinkslip,wage_reduction,benq

    def move_to_motherleave(self,pension,old_wage,age,wage_reduction,children_under7,children_under18):
        '''
        Siirtymä äitiysvapaalle
        '''
        self.infostate_add_child(age)
        employment_status = 5 # äitiysvapaa
        time_in_state=0
        wage=old_wage
        pension=self.pension_accrual(age,old_wage,pension,state=5)
        netto,benq=self.comp_benefits(0,old_wage,0,employment_status,0,children_under7,children_under18,age)
        time_in_state+=self.timestep
        pinkslip=0
        wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)        

        return employment_status,pension,wage,time_in_state,netto,pinkslip,wage_reduction,benq

    def stay_unemployed(self,employment_status,paid_pension,pension,time_in_state,toe,wage_reduction,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,children_under7,children_under18):
        '''
        Pysy tilassa työtön (0)
        '''
        time_in_state+=self.timestep
            
        if age>=65:
            employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,benq\
                =self.move_to_retirement(pension,0,age,paid_pension,employment_status,wage_reduction,unemp_after_ra,children_under7,children_under18,all_acc=True)
        elif action == 0 or (action == 2 and age < self.min_retirementage):
            employment_status = 0 # unchanged
            wage=self.get_wage(intage,wage_reduction)
            wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)

            netto,benq=self.comp_benefits(0,unempwage_basis,0,employment_status,used_unemp_benefit,children_under7,children_under18,age,tyohistoria=tyoura)
            if pinkslip or time_in_state>=self.karenssi_kesto: # muuten ei oikeutta ansiopäivärahaan karenssi vuoksi
                used_unemp_benefit+=self.timestep
            kesto=12*21.5*used_unemp_benefit
                
            if not self.paivarahapaivia_jaljella(kesto,tyoura,age):
            #if ((tyoura>=self.tyohistoria_vaatimus500 and kesto>=self.ansiopvraha_kesto500 and age>=self.minage_500) \
            #    or (tyoura>=self.tyohistoria_vaatimus and kesto>=self.ansiopvraha_kesto400 and (age<self.minage_500 or tyoura<self.tyohistoria_vaatimus500)) \
            #    or (tyoura<self.tyohistoria_vaatimus and kesto>=self.ansiopvraha_kesto300)):
                if self.include_putki and age>=self.min_tyottputki_ika and tyoura>=self.tyohistoria_tyottputki: 
                    employment_status = 4 # siirto lisäpäiville
                    pension=self.pension_accrual(age,unempwage_basis,pension,state=4)
                else:
                    employment_status = 13 # siirto työmarkkinatuelle
                    pension=self.pension_accrual(age,old_wage,pension,state=13)
            else:
                pension=self.pension_accrual(age,unempwage_basis,pension,state=0)
                
            if age >= self.min_retirementage:
                unemp_after_ra+=self.timestep

        elif action == 1: # 
            employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                self.move_to_work(pension,old_wage,age,time_in_state,tyoura,pinkslip,wage_reduction,children_under7,children_under18)
        elif action==2:
            if age >= self.min_retirementage: # ve
                employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,benq=\
                    self.move_to_retirement(pension,old_wage,age,paid_pension,employment_status,
                        wage_reduction,unemp_after_ra,children_under7,children_under18,scale_acc=False)
        elif action == 3: # osatyö 50%
            employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                self.move_to_parttime(pension,old_wage,age,tyoura,time_in_state,wage_reduction,children_under7,children_under18)
        elif action==11: # tk
            employment_status,pension,paid_pension,wage,time_in_state,netto,wage_reduction,benq=\
                self.move_to_disab(pension,old_wage,age,wage_reduction,children_under7,children_under18)
            pinkslip=0
        else:
            print('error 17')  
            
        return employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,\
            benq,pinkslip,unemp_after_ra,time_in_state,tyoura,used_unemp_benefit,unempwage_basis


    def stay_tyomarkkinatuki(self,employment_status,paid_pension,pension,time_in_state,toe,wage_reduction,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,
                        unempwage,unempwage_basis,action,age,sattuma,intage,g,children_under7,children_under18):
        '''
        Pysy tilassa työmarkkinatuki (13)
        '''
        time_in_state+=self.timestep
        if age>=65:
            employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,benq=\
                self.move_to_retirement(pension,0,age,paid_pension,employment_status,wage_reduction,
                        unemp_after_ra,children_under7,children_under18,all_acc=True)
        elif action == 0 or (action == 2 and age < self.min_retirementage):
            employment_status = 13 # unchanged
            wage=old_wage
            pension=self.pension_accrual(age,wage,pension,state=13)

            netto,benq=self.comp_benefits(0,old_wage,0,employment_status,used_unemp_benefit,children_under7,children_under18,age,tyohistoria=tyoura)
            used_unemp_benefit+=self.timestep
            wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)
            if age >= self.min_retirementage:
                unemp_after_ra+=self.timestep
        
        elif action == 1: # 
            employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                self.move_to_work(pension,old_wage,age,time_in_state,tyoura,pinkslip,wage_reduction,children_under7,children_under18)
        elif action==2:
            if age >= self.min_retirementage: # ve
                employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,benq=\
                    self.move_to_retirement(pension,old_wage,age,paid_pension,employment_status,
                        wage_reduction,unemp_after_ra,children_under7,children_under18,scale_acc=False)
        elif action == 3: # osatyö 50%
            employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                self.move_to_parttime(pension,old_wage,age,tyoura,time_in_state,wage_reduction,children_under7,children_under18)
        elif action==11: # tk
            employment_status,pension,paid_pension,wage,time_in_state,netto,wage_reduction,benq=\
                self.move_to_disab(pension,old_wage,age,wage_reduction,children_under7,children_under18)
        else:
            print('error 17')        
                
        return employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,\
               benq,pinkslip,unemp_after_ra,time_in_state,tyoura,used_unemp_benefit,unempwage_basis
                
    def stay_pipeline(self,employment_status,paid_pension,pension,time_in_state,toe,wage_reduction,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,children_under7,children_under18):
        '''
        Pysy tilassa työttömyysputki (4)
        '''
        time_in_state+=self.timestep
        if age>=65:
            employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,benq=\
                self.move_to_retirement(pension,0,age,paid_pension,employment_status,
                    wage_reduction,unemp_after_ra,children_under7,children_under18,all_acc=True)
        elif action == 0 or (action == 2 and age < self.min_retirementage):
            employment_status  = 4 # unchanged
            wage=self.get_wage(intage,wage_reduction)
            pension=self.pension_accrual(age,unempwage_basis,pension,state=4)
            wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)
                
            netto,benq=self.comp_benefits(0,unempwage_basis,0,employment_status,used_unemp_benefit,children_under7,children_under18,age,tyohistoria=tyoura)
            if pinkslip or time_in_state>=self.karenssi_kesto: # muuten ei oikeutta ansiopäivärahaan karenssi vuoksi
                used_unemp_benefit+=self.timestep
            if age >= self.min_retirementage:
                unemp_after_ra+=self.timestep
                
        elif action == 1: # 
            employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                self.move_to_work(pension,old_wage,age,time_in_state,tyoura,pinkslip,wage_reduction,children_under7,children_under18)
        elif action==2:
            if age >= self.min_retirementage: # ve
                employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,benq=\
                    self.move_to_retirement(pension,old_wage,age,paid_pension,employment_status,
                        wage_reduction,unemp_after_ra,children_under7,children_under18,scale_acc=False)
            pinkslip=0
        elif action == 3: # 
            employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                self.move_to_parttime(pension,old_wage,age,tyoura,time_in_state,wage_reduction,children_under7,children_under18)
        elif action==11: # tk
            employment_status,pension,paid_pension,wage,time_in_state,netto,wage_reduction,benq=\
                self.move_to_disab(pension,old_wage,age,wage_reduction,children_under7,children_under18)
        else:
            print('error 1: ',action)
            
        return employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,\
               benq,pinkslip,unemp_after_ra,time_in_state,tyoura,used_unemp_benefit,unempwage_basis
               
    def stay_employed(self,employment_status,paid_pension,pension,time_in_state,toe,wage_reduction,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,children_under7,children_under18):
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

        if action == 0 or (action == 2 and age < self.min_retirementage):
            employment_status = 1 # unchanged
            wage=self.get_wage(intage,wage_reduction)
            wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)
            tyoura+=self.timestep
            pension=self.pension_accrual(age,wage,pension,state=1)
            netto,benq=self.comp_benefits(wage,0,0,employment_status,time_in_state,children_under7,children_under18,age)
        elif action == 1: # työttömäksi
            employment_status,paid_pension,pension,wage,time_in_state,netto,\
                wage_reduction,used_unemp_benefit,pinkslip,benq,unemp_after_ra,unempwage_basis=\
                self.move_to_unemp(pension,old_wage,age,paid_pension,toe,pinkslip,tyoura,
                    wage_reduction,used_unemp_benefit,unemp_after_ra,unempwage,children_under7,children_under18)
        elif action==2:
            if age >= self.min_retirementage: # ve
                employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,benq=\
                    self.move_to_retirement(pension,old_wage,age,paid_pension,employment_status,wage_reduction,
                        unemp_after_ra,children_under7,children_under18) 
        elif action == 3: # osatyö 50%
            employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                self.move_to_parttime(pension,old_wage,age,tyoura,0,wage_reduction,children_under7,children_under18)
        elif action==11: # tk
            employment_status,pension,paid_pension,wage,time_in_state,netto,wage_reduction,benq=\
                self.move_to_disab(pension,old_wage,age,wage_reduction,children_under7,children_under18)
        else:
            print('error 12')    
            
        return employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,\
              benq,pinkslip,unemp_after_ra,time_in_state,tyoura,used_unemp_benefit,unempwage_basis
           
    def stay_disabled(self,employment_status,paid_pension,pension,time_in_state,toe,wage_reduction,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,children_under7,children_under18):
            
        '''
        Pysy tilassa työkyvytön (4)
        '''
        time_in_state+=self.timestep
        if age >= self.min_retirementage:
            employment_status = 3 # ve # miten kansaneläke menee?? takuueläke?
        else:
            employment_status = 3 # unchanged

        paid_pension=paid_pension*self.elakeindeksi
        wage=old_wage
        netto,benq=self.comp_benefits(0,0,paid_pension,employment_status,0,children_under7,children_under18,age)

        return employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,\
               benq,pinkslip,unemp_after_ra,time_in_state,tyoura,used_unemp_benefit,unempwage_basis           

    def stay_retired(self,employment_status,paid_pension,pension,time_in_state,toe,wage_reduction,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,children_under7,children_under18):
        '''
        Pysy tilassa vanhuuseläke (2)
        '''
        if age >= self.min_retirementage: # ve
            time_in_state+=self.timestep

            if age>=self.max_retirementage:
                paid_pension += self.elinaikakerroin*pension
                pension=0

            if action == 0 or action == 3 or ((action == 1 or action == 2) and age>=self.max_retirementage):
                employment_status = 2 # unchanged

                paid_pension=paid_pension*self.elakeindeksi
                pension=pension*self.palkkakerroin
                wage=self.get_wage(intage,wage_reduction)
                netto,benq=self.comp_benefits(0,0,paid_pension,employment_status,0,children_under7,children_under18,age)
                wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)
            elif action == 1 and age<self.max_retirementage:
                employment_status,pension,wage,time_in_state,netto,wage_reduction,benq=\
                    self.move_to_retwork(pension,old_wage,age,time_in_state,paid_pension,wage_reduction,children_under7,children_under18)
            elif action == 2 and age<self.max_retirementage:
                employment_status,pension,wage,time_in_state,netto,wage_reduction,benq=\
                    self.move_to_retpartwork(pension,old_wage,age,time_in_state,paid_pension,wage_reduction,children_under7,children_under18)
            elif action == 11:
                employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,benq=\
                    self.move_to_retdisab(pension,old_wage,age,time_in_state,paid_pension,wage_reduction,children_under7,children_under18)
            else:
                print('error 221, action {} age {}'.format(action,age))
        else:
            # työvoiman ulkopuolella
            time_in_state+=self.timestep
            if action == 0:
                employment_status = 2 # unchanged
                wage=old_wage
                #toe=max(0,toe-self.timestep)
                pension=pension*self.palkkakerroin
                netto,benq=self.comp_benefits(0,0,0,employment_status,time_in_state,children_under7,children_under18,age)
                wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)
            elif action == 1: # työttömäksi
                employment_status,paid_pension,pension,wage,time_in_state,netto,\
                    wage_reduction,used_unemp_benefit,pinkslip,benq,unemp_after_ra,unempwage_basis=\
                    self.move_to_unemp(pension,old_wage,age,paid_pension,toe,0,tyoura,
                        wage_reduction,used_unemp_benefit,unemp_after_ra,unempwage,children_under7,children_under18)
            elif action == 2: # töihin
                employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                    self.move_to_work(pension,old_wage,age,time_in_state,tyoura,pinkslip,wage_reduction,children_under7,children_under18)
            elif action == 3: # osatyö 50%
                employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                    self.move_to_parttime(pension,old_wage,age,tyoura,time_in_state,wage_reduction,children_under7,children_under18)
            elif action == 11: # tk
                employment_status,pension,paid_pension,wage,time_in_state,netto,wage_reduction,benq=\
                    self.move_to_disab(pension,old_wage,age,wage_reduction,children_under7,children_under18)
            else:
                print('error 12')
                
        return employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,\
               benq,pinkslip,unemp_after_ra,time_in_state,tyoura,used_unemp_benefit,unempwage_basis           

    def stay_motherleave(self,employment_status,paid_pension,pension,time_in_state,toe,wage_reduction,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,children_under7,children_under18):
        '''
        Pysy tilassa äitiysvapaa (5)
        '''
        if time_in_state>self.aitiysvapaa_kesto:
            pinkslip=0
            if action == 0:
                employment_status,paid_pension,pension,wage,time_in_state,netto,\
                    wage_reduction,used_unemp_benefit,pinkslip,benq,unemp_after_ra,unempwage_basis=\
                    self.move_to_unemp(pension,old_wage,age,paid_pension,toe,pinkslip,tyoura,
                        wage_reduction,used_unemp_benefit,unemp_after_ra,unempwage,children_under7,children_under18)
            elif action == 1: # 
                employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                    self.move_to_work(pension,old_wage,age,time_in_state,tyoura,pinkslip,wage_reduction,children_under7,children_under18)
            elif action == 2: # 
                employment_status,pension,wage,time_in_state,netto,wage_reduction,benq=\
                    self.move_to_kht(pension,old_wage,age,wage_reduction,children_under7,children_under18)
            elif action == 3: # osa-aikatyöhön
                employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                    self.move_to_parttime(pension,old_wage,age,tyoura,time_in_state,wage_reduction,children_under7,children_under18)
            elif action==11: # tk
                employment_status,pension,paid_pension,wage,time_in_state,netto,wage_reduction,benq=\
                    self.move_to_disab(pension,old_wage,age,wage_reduction,children_under7,children_under18)
            else:
                print('Error 21')
        else:
            pension=self.pension_accrual(age,old_wage,pension,state=5)
            wage=old_wage # self.get_wage(intage,wage_reduction)
            netto,benq=self.comp_benefits(0,old_wage,0,employment_status,0,children_under7,children_under18,age)
            time_in_state+=self.timestep
            wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)
                
        return employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,\
               benq,pinkslip,unemp_after_ra,time_in_state,tyoura,used_unemp_benefit,unempwage_basis           

    def stay_fatherleave(self,employment_status,paid_pension,pension,time_in_state,toe,wage_reduction,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,children_under7,children_under18):
        '''
        Pysy tilassa isyysvapaa (6)
        '''
        if time_in_state>=self.isyysvapaa_kesto:
            pinkslip=0
            if action == 0 or action==2:
                employment_status,paid_pension,pension,wage,time_in_state,netto,\
                    wage_reduction,used_unemp_benefit,pinkslip,benq,unemp_after_ra,unempwage_basis=\
                    self.move_to_unemp(pension,old_wage,age,paid_pension,toe,pinkslip,tyoura,
                        wage_reduction,used_unemp_benefit,unemp_after_ra,unempwage,children_under7,children_under18)
            elif action == 1: # 
                # ei vaikutusta palkkaan
                employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                    self.move_to_work(pension,old_wage,age,0,tyoura,pinkslip,wage_reduction,children_under7,children_under18)
            elif action == 2: # 
                employment_status,pension,wage,time_in_state,netto,wage_reduction,benq=\
                    self.move_to_kht(pension,old_wage,age,wage_reduction,children_under7,children_under18)
            elif action == 3: # osa-aikatöihin
                employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                    self.move_to_parttime(pension,old_wage,age,tyoura,0,wage_reduction,children_under7,children_under18)
            elif action==11: # tk
                employment_status,pension,paid_pension,wage,time_in_state,netto,wage_reduction,benq=\
                    self.move_to_disab(pension,old_wage,age,wage_reduction,children_under7,children_under18)
            else:
                print('Error 23')
        else:
            pension=self.pension_accrual(age,old_wage,pension,state=6)
            wage=old_wage
            netto,benq=self.comp_benefits(0,old_wage,0,employment_status,0,children_under7,children_under18,age)
            time_in_state+=self.timestep
            wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)

        return employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,\
               benq,pinkslip,unemp_after_ra,time_in_state,tyoura,used_unemp_benefit,unempwage_basis           

    def stay_khh(self,employment_status,paid_pension,pension,time_in_state,toe,wage_reduction,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,children_under7,children_under18):
        '''
        Pysy tilassa kotihoidontuki (0)
        '''
        time_in_state+=self.timestep

        if age >= self.min_retirementage: # ve
            employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,benq=\
                self.move_to_retirement(pension,old_wage,age,paid_pension,employment_status,
                    wage_reduction,unemp_after_ra,children_under7,children_under18,all_acc=True)
        elif action == 0 and (time_in_state<=self.kht_kesto or self.perustulo): # jos perustulo, ei aikarajoitetta
        #elif action == 0 and (time_in_state<=self.kht_kesto): # jos perustulo, ei aikarajoitetta
            employment_status  = 7 # stay
            wage=old_wage
            pension=self.pension_accrual(age,wage,pension,state=7)
            netto,benq=self.comp_benefits(0,old_wage,0,employment_status,time_in_state,children_under7,children_under18,age)
            wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)
        elif action == 2: # 
            pinkslip=0
            employment_status,paid_pension,pension,wage,time_in_state,netto,\
                wage_reduction,used_unemp_benefit,pinkslip,benq,unemp_after_ra,unempwage_basis=\
                self.move_to_unemp(pension,old_wage,age,paid_pension,toe,pinkslip,tyoura,
                    wage_reduction,used_unemp_benefit,unemp_after_ra,unempwage,children_under7,children_under18)
        elif action == 1: # 
            employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                self.move_to_work(pension,old_wage,age,time_in_state,tyoura,pinkslip,wage_reduction,children_under7,children_under18)
        elif action == 3: # 
            employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                self.move_to_parttime(pension,old_wage,age,tyoura,time_in_state,wage_reduction,children_under7,children_under18)
        elif action==11: # tk
            employment_status,pension,paid_pension,wage,time_in_state,netto,wage_reduction,benq=\
                self.move_to_disab(pension,old_wage,age,wage_reduction,children_under7,children_under18)
        elif time_in_state>self.kht_kesto: # 
            employment_status,paid_pension,pension,wage,time_in_state,netto,\
                wage_reduction,used_unemp_benefit,pinkslip,benq,unemp_after_ra,unempwage_basis=\
                self.move_to_unemp(pension,old_wage,age,paid_pension,toe,pinkslip,tyoura,
                    wage_reduction,used_unemp_benefit,unemp_after_ra,unempwage,children_under7,children_under18)
        else:
            print('Error 25')
            
        return employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,\
               benq,pinkslip,unemp_after_ra,time_in_state,tyoura,used_unemp_benefit,unempwage_basis           

    def stay_student(self,employment_status,paid_pension,pension,time_in_state,toe,wage_reduction,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,children_under7,children_under18):
        '''
        Pysy tilassa työtön (12)
        '''

        pinkslip=0
        tyoura=0
        if sattuma[5]>=self.student_outrate[intage,g]:
            employment_status = 12 # unchanged
            time_in_state+=self.timestep
            wage=old_wage
            pension=self.pension_accrual(age,0,pension,state=13)
            netto,benq=self.comp_benefits(0,0,0,employment_status,time_in_state,children_under7,children_under18,age,tyohistoria=tyoura)
            # opiskelu parantaa tuloja
            wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)
        elif action == 0: # 
            employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                self.move_to_work(pension,old_wage,age,0,tyoura,pinkslip,wage_reduction,children_under7,children_under18)
        elif action == 1 or action == 3:
            employment_status,paid_pension,pension,wage,time_in_state,netto,\
                wage_reduction,used_unemp_benefit,pinkslip,benq,unemp_after_ra,unempwage_basis=\
                self.move_to_unemp(pension,old_wage,age,paid_pension,toe,pinkslip,tyoura,
                    wage_reduction,used_unemp_benefit,unemp_after_ra,unempwage,children_under7,children_under18)
        elif action == 2:
            employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                self.move_to_parttime(pension,old_wage,age,tyoura,time_in_state,wage_reduction,children_under7,children_under18)
        elif action == 11: # tk
            employment_status,pension,paid_pension,wage,time_in_state,netto,wage_reduction,benq=\
                self.move_to_disab(pension,old_wage,age,wage_reduction,children_under7,children_under18)
        else:
            print('error 29: ',action)
            
        return employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,\
               benq,pinkslip,unemp_after_ra,time_in_state,tyoura,used_unemp_benefit,unempwage_basis

    def stay_oa_parttime(self,employment_status,paid_pension,pension,time_in_state,toe,wage_reduction,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,children_under7,children_under18):
        '''
        Pysy tilassa ve+(osa-aikatyö) (0)
        '''

        time_in_state+=self.timestep
        
        # irtisanominen
        if sattuma[1]<self.pinkslip_intensity[g]:
            action=2 # ve:lle

        if age>=self.max_retirementage:
            paid_pension += self.elinaikakerroin*pension
            pension=0

        if action == 0 or action == 3: # jatkaa osa-aikatöissä, ei voi saada työttömyyspäivärahaa
            employment_status = 9 # unchanged
            wage=self.get_wage(intage,wage_reduction)
            parttimewage=0.5*wage
            pension=self.pension_accrual(age,parttimewage,pension,state=9)

            paid_pension=paid_pension*self.elakeindeksi
            netto,benq=self.comp_benefits(parttimewage,0,paid_pension,employment_status,time_in_state,children_under7,children_under18,age)
            wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)
        elif action==1: # jatkaa täysin töissä, ei voi saada työttömyyspäivärahaa
            employment_status,pension,wage,time_in_state,netto,wage_reduction,benq=\
                self.move_to_retwork(pension,old_wage,age,0,paid_pension,wage_reduction,children_under7,children_under18)
        elif action==2: # eläkkeelle, eläkeaikana karttunutta eläkettä ei vielä maksuun
            employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,benq=\
                self.move_to_retirement(pension,old_wage,age,paid_pension,employment_status,
                    wage_reduction,0,children_under7,children_under18,all_acc=False)
        elif action == 11:
            # no more working, move to "disab" with no change in paid_pension
            employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,benq=\
                self.move_to_retdisab(pension,old_wage,age,time_in_state,paid_pension,wage_reduction,children_under7,children_under18)
        else:
            print('error 14, action {} age {}'.format(action,age))

        return employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,\
               benq,pinkslip,unemp_after_ra,time_in_state,tyoura,used_unemp_benefit,unempwage_basis           

    def stay_oa_emp(self,employment_status,paid_pension,pension,time_in_state,toe,wage_reduction,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,children_under7,children_under18):
        '''
        Pysy tilassa ve+työ (0)
        '''

        time_in_state+=self.timestep        
        # irtisanominen
        if sattuma[1]<self.pinkslip_intensity[g]:
            action=2 # ve:lle

        if age>=self.max_retirementage:
            paid_pension += self.elinaikakerroin*pension
            pension=0

        if action == 0 or action == 3: # jatkaa töissä, ei voi saada työttömyyspäivärahaa
            employment_status = 8 # unchanged
            wage=self.get_wage(intage,wage_reduction)
            pension=self.pension_accrual(age,wage,pension,state=8)
            
            paid_pension=paid_pension*self.elakeindeksi
            netto,benq=self.comp_benefits(wage,0,paid_pension,employment_status,time_in_state,children_under7,children_under18,age)
            wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)
        elif action == 1: # jatkaa osa-aikatöissä, ei voi saada työttömyyspäivärahaa
            employment_status,pension,wage,time_in_state,netto,wage_reduction,benq=\
                self.move_to_retpartwork(pension,old_wage,age,0,paid_pension,wage_reduction,children_under7,children_under18)
        elif action==2: # eläkkeelle, eläkeaikana karttunutta eläkettä ei vielä maksuun
            employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,benq=\
                self.move_to_retirement(pension,old_wage,age,paid_pension,employment_status,
                    wage_reduction,0,children_under7,children_under18,all_acc=False)
        elif action == 11:
            # no more working, move to "disab" with no change in paid_pension
            employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,benq=\
                self.move_to_retdisab(pension,old_wage,age,time_in_state,paid_pension,wage_reduction,children_under7,children_under18)
        else:
            print('error 14, action {} age {}'.format(action,age))
            
        return employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,\
               benq,pinkslip,unemp_after_ra,time_in_state,tyoura,used_unemp_benefit,unempwage_basis           

    def stay_parttime(self,employment_status,paid_pension,pension,time_in_state,toe,wage_reduction,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,children_under7,children_under18):
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

        if action == 0 or (action == 2 and age < self.min_retirementage):
            employment_status = 10 # unchanged
            #if time_in_state>1:
            #    prev_unempl=0 # nollataan työttömyyden vaikutus palkkaan vuoden jälkeen

            wage=self.get_wage(intage,wage_reduction)
            parttimewage=0.5*wage
            tyoura+=self.timestep
            
            pension=self.pension_accrual(age,parttimewage,pension,state=10)
            netto,benq=self.comp_benefits(parttimewage,0,0,employment_status,time_in_state,children_under7,children_under18,age)
            wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)
        elif action == 1: # työttömäksi
            employment_status,paid_pension,pension,wage,time_in_state,netto,\
                wage_reduction,used_unemp_benefit,pinkslip,benq,unemp_after_ra,unempwage_basis=\
                self.move_to_unemp(pension,old_wage,age,paid_pension,toe,pinkslip,tyoura,
                    wage_reduction,used_unemp_benefit,unemp_after_ra,unempwage,children_under7,children_under18)
        elif action==2:
            if age >= self.min_retirementage: # ve
                employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,benq=\
                    self.move_to_retirement(pension,old_wage,age,paid_pension,employment_status,
                        wage_reduction,unemp_after_ra,children_under7,children_under18,all_acc=True)
        elif action==3:
            employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                self.move_to_work(pension,old_wage,age,0,tyoura,pinkslip,wage_reduction,children_under7,children_under18)
        elif action==11: # tk
            employment_status,pension,paid_pension,wage,time_in_state,netto,wage_reduction,benq=\
                self.move_to_disab(pension,old_wage,age,wage_reduction,children_under7,children_under18)
        else:
            print('error 12')
        return employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,\
               benq,pinkslip,unemp_after_ra,time_in_state,tyoura,used_unemp_benefit,unempwage_basis           


    def stay_army(self,employment_status,paid_pension,pension,time_in_state,toe,wage_reduction,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,children_under7,children_under18):
        '''
        Pysy tilassa armeija/siviilipalvelus (14)
        '''

        pinkslip=0
        tyoura=0
        if sattuma[6]>=self.army_outrate[intage,g]: # vain ulos
            employment_status = 14 # unchanged
            time_in_state+=self.timestep
            wage=old_wage
            netto,benq=self.comp_benefits(0,0,0,employment_status,time_in_state,children_under7,children_under18,age,tyohistoria=tyoura)
            # opiskelu parantaa tuloja
            wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)
        elif action == 0: # 
            employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                self.move_to_work(pension,old_wage,age,0,tyoura,pinkslip,wage_reduction,children_under7,children_under18)
        elif action == 1 or action == 3:
            employment_status,paid_pension,pension,wage,time_in_state,netto,\
                wage_reduction,used_unemp_benefit,pinkslip,benq,unemp_after_ra,unempwage_basis=\
                self.move_to_unemp(pension,old_wage,age,paid_pension,toe,pinkslip,tyoura,
                    wage_reduction,used_unemp_benefit,unemp_after_ra,unempwage,children_under7,children_under18)
        elif action == 2:
            employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                self.move_to_parttime(pension,old_wage,age,tyoura,time_in_state,wage_reduction,children_under7,children_under18)
        elif action == 11: # tk
            employment_status,pension,paid_pension,wage,time_in_state,netto,wage_reduction,benq=\
                self.move_to_disab(pension,old_wage,age,wage_reduction,children_under7,children_under18)
        else:
            print('error 39: ',action) 
                
        return employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,\
               benq,pinkslip,unemp_after_ra,time_in_state,tyoura,used_unemp_benefit,unempwage_basis           

    def stay_outsider(self,employment_status,paid_pension,pension,time_in_state,toe,wage_reduction,
                        tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,
                        unempwage_basis,action,age,sattuma,intage,g,children_under7,children_under18):
        '''
        Pysy tilassa työvoiman ulkopuolella (11)
        '''

        if age>=self.min_retirementage:
            employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,benq=\
                self.move_to_retirement(pension,old_wage,age,paid_pension,employment_status,
                wage_reduction,unemp_after_ra,children_under7,children_under18,all_acc=True)
        elif sattuma[5]>=self.outsider_outrate[intage,g]:
            time_in_state+=self.timestep
            employment_status = 11 # unchanged
            wage=old_wage
            pension=self.pension_accrual(age,wage,pension,state=11)
            netto,benq=self.comp_benefits(0,old_wage,0,employment_status,time_in_state,children_under7,children_under18,age,tyohistoria=tyoura)
            wage_reduction=self.update_wage_reduction(employment_status,wage_reduction)
        elif action == 1: # 
            employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                self.move_to_work(pension,old_wage,age,time_in_state,tyoura,pinkslip,wage_reduction,children_under7,children_under18)
        elif action == 2 or action == 0: # 
            pinkslip=0
            employment_status,paid_pension,pension,wage,time_in_state,netto,\
                wage_reduction,used_unemp_benefit,pinkslip,benq,unemp_after_ra,unempwage_basis=\
                self.move_to_unemp(pension,old_wage,age,paid_pension,toe,pinkslip,tyoura,wage_reduction,
                    used_unemp_benefit,unemp_after_ra,unempwage,children_under7,children_under18)
        elif action == 3: # 
            employment_status,pension,wage,time_in_state,netto,tyoura,pinkslip,wage_reduction,benq=\
                self.move_to_parttime(pension,old_wage,age,tyoura,time_in_state,wage_reduction,children_under7,children_under18)
        elif action == 11: # tk
            employment_status,pension,paid_pension,wage,time_in_state,netto,wage_reduction,benq=\
                self.move_to_disab(pension,old_wage,age,wage_reduction,children_under7,children_under18)
            pinkslip=0
        else:
            print('error 19: ',action)

        return employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,\
               benq,pinkslip,unemp_after_ra,time_in_state,tyoura,used_unemp_benefit,unempwage_basis
        
    def pension_accrual(self,age,wage,pension,state=1):
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
                pension=pension*self.palkkakerroin+acc*wage
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
            if age<self.min_retirementage:
                pension=pension*self.palkkakerroin+self.accbasis_tmtuki*acc
            else:
                pension=pension*self.palkkakerroin
        else: # 2,3,11,12,14 # ei karttumaa
            pension=pension*self.palkkakerroin # vastainen eläke, ei alkanut, ei karttumaa
            
        return pension

    def update_wage_reduction_baseline(self,state,wage_reduction):
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
            wage_reduction+=self.salary_const
        elif state in set([5,6]): # isyys tai vanhempainvapaa
            wage_reduction+=self.salary_const
        elif state in set([7,2]): # kotihoidontuki tai ve tai tk
            wage_reduction+=self.salary_const
        elif state in set([3,14,15]): # ei muutosta
            wage_reduction=wage_reduction
        else: # ylivuoto, ei tiloja
            wage_reduction=wage_reduction
        
        return wage_reduction

    def update_wage_reduction_sigma(self,state,wage_reduction):
        '''
        Pidetään kirjaa siitä, kuinka paljon palkkaa alennetaan työttömyyden keston suhteen,
        ja miten siitä palaudutaan
        
        Tämä malli ei mene koskaan nollaan.
        '''
        if state in set([1,10]): # töissä
            wage_reduction=max(0,wage_reduction-self.salary_const_up)
        if state in set([8,9]): # ve+töissä
            wage_reduction=max(0,wage_reduction-self.salary_const_up)
        elif state==12: # opiskelee
            wage_reduction=max(0,wage_reduction-self.salary_const_student)
        elif state in set([0,4,13,11]): # työtön tai työelämän ulkopuolella, tuleeko skaalaus kahteen kertaan?
            #wage_reduction=max(0,1.0-((1-self.salary_const)**self.timestep)*(1-wage_reduction))
            wage_reduction=max(0,1.0-(1-self.salary_const)*(1-wage_reduction))
        elif state in set([5,6]): # isyys tai vanhempainvapaa, ei vaikutusta
            wage_reduction=wage_reduction
        elif state in set([7,2]): # kotihoidontuki tai ve
            #wage_reduction=max(0,1.0-((1-self.salary_const)**self.timestep)*(1-wage_reduction))
            wage_reduction=max(0,1.0-(1-self.salary_const)*(1-wage_reduction))
        elif state in set([3,14,15]): # ei muutosta
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

        employment_status,g,pension,old_wage,age,time_in_state,paid_pension,pinkslip,toe,\
            tyoura,used_unemp_benefit,wage_reduction,unemp_after_ra,\
            unempwage,unempwage_basis,prefnoise,children_under7,children_under18\
                =self.state_decode(self.state)
            
        # simulointiin vaikuttavia ulkoisia tilamuuttujia, ei toteutettu
        #tyoura,lapsia,lapsen1_ika,lapsen2_ika,lapsen3_ika,lapsia_paivakodissa=self.decode_info_state()

        #toe,unempwage=self.comp_infostats()
        #print('toe {} vs toe2 {}'.format(toe,toe2))
        #print('old_wage {} vs unempwage {}'.format(old_wage,unempwage))

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
                    if self.infostat_can_have_children(age): # lasten väli vähintään vuosi
                        if g>2: # naiset
                            employment_status,pension,wage,time_in_state,netto,pinkslip,wage_reduction,benq=\
                                self.move_to_motherleave(pension,old_wage,age,wage_reduction,children_under7,children_under18)
                            pinkslip=0
                            moved=True
                        else: # miehet
                            # ikä valittu äidin iän mukaan. oikeastaan tämä ei mene ihan oikein miehille
                            if sattuma[4]<0.5:
                                employment_status,pension,wage,time_in_state,netto,pinkslip,wage_reduction,benq=\
                                    self.move_to_fatherleave(pension,old_wage,age,wage_reduction,children_under7,children_under18)
                                moved=True
                elif sattuma[2]<s3/move_prob:
                    if employment_status not in set([2,3,8,9,11,12,14]): # and False:
                        employment_status,pension,wage,time_in_state,netto,pinkslip,wage_reduction,benq=\
                            self.move_to_student(pension,old_wage,age,time_in_state,tyoura,pinkslip,wage_reduction,children_under7,children_under18)
                        moved=True
                #elif sattuma[2]<s4/move_prob: # and False:
                else:
                    if employment_status not in set([2,3,8,9,11,12,14]):
                        employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,pinkslip,benq=\
                            self.move_to_outsider(pension,old_wage,age,pinkslip,wage_reduction,children_under7,children_under18)
                        moved=True

            # voi aiheuttaa epästabiilisuutta
            if sattuma[3]<self.mort_intensity[intage,g] and self.include_mort: # and False:
                employment_status,pension,wage,time_in_state,netto=self.move_to_deceiced(pension,old_wage,age)
        else:
            # tn ei ole koskaan alle rajan, jos tämä on 1
            sattuma = np.ones(7)
            
        if employment_status==15: # deceiced
            #time_in_state+=self.timestep
            if not self.include_mort:
                print('emp state 15')
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
                            used_unemp_benefit,wage_reduction,unemp_after_ra,unemp_after_ra,unempwage,unempwage_basis,
                            children_under7=children_under7,children_under18=children_under18)
            reward=0
            return np.array(self.state), reward, done, {}
        elif age>=self.max_retirementage:
            employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,benq\
                =self.move_to_retirement(pension,0,age,paid_pension,employment_status,wage_reduction,unemp_after_ra,children_under7,children_under18,all_acc=True)
        else:
             if not moved:
                # hoidetaan tilasiirtymät ja -pysymiset alirutiineilla, joita kutsutaan mäppäämällä tila funktioksi,
                # jota sitten kutsutaan
                map_stays={0: self.stay_unemployed,  1: self.stay_employed,         2: self.stay_retired,     3: self.stay_disabled,
                           4: self.stay_pipeline,    5: self.stay_motherleave,      6: self.stay_fatherleave, 7: self.stay_khh,
                           8: self.stay_oa_emp,      9: self.stay_oa_parttime,     10: self.stay_parttime,   11: self.stay_outsider,
                           12: self.stay_student,   13: self.stay_tyomarkkinatuki, 14: self.stay_army}
                employment_status,paid_pension,pension,wage,time_in_state,netto,wage_reduction,benq,pinkslip,\
                unemp_after_ra,time_in_state,tyoura,used_unemp_benefit,unempwage_basis\
                    = map_stays[employment_status](employment_status,paid_pension,pension,time_in_state,toe,wage_reduction,
                                                   tyoura,used_unemp_benefit,pinkslip,unemp_after_ra,old_wage,unempwage,unempwage_basis,
                                                   action,age,sattuma,intage,g,children_under7,children_under18)

        done = age >= self.max_age
        done = bool(done)
        
        self.update_infostate(t,int(employment_status),wage)
        toe,unempwage,children_under7,children_under18=self.comp_infostats(age+self.timestep)
        work={1,10}
        if employment_status in work and self.tyossaoloehto(toe,tyoura,age):            
            used_unemp_benefit=0
        
        
        #self.render_infostate()

        if not done:
            reward = self.log_utility(netto,int(employment_status),age,g=g,pinkslip=pinkslip)
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            
            paid_pension += self.elinaikakerroin*pension
            pension=0
            
            netto,benq=self.comp_benefits(0,old_wage,paid_pension,employment_status,time_in_state,children_under7,children_under18,age)
            if employment_status in set([2,3,8,9]):
                reward = self.npv[g]*self.log_utility(netto,employment_status,age,pinkslip=0)
                
                # npv0 is undiscounted
                benq=self.scale_q(self.npv0[g],benq)                
            else:
                # giving up the pension
                reward = 0.0 #-self.npv[g]*self.log_utility(netto,employment_status,age)
                
            pinkslip=0
            #time_in_state+=self.timestep
        else:
            #if not dynprog: # tätä mallia on vaikea ajaa dynaamisella ohjelmoinnilla
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        # seuraava palkka tiedoksi valuaatioapproksimaattorille
        next_wage=self.get_wage(int(np.floor(age+self.timestep)),wage_reduction)
        
        if self.include_preferencenoise:
            self.state = self.state_encode(employment_status,g,pension,wage,age+self.timestep,time_in_state,
                                    paid_pension,pinkslip,toe,tyoura,next_wage,used_unemp_benefit,
                                    wage_reduction,unemp_after_ra,unempwage,unempwage_basis,
                                    prefnoise=prefnoise,children_under7=children_under7,children_under18=children_under18)
        else:
            self.state = self.state_encode(employment_status,g,pension,wage,age+self.timestep,time_in_state,
                                    paid_pension,pinkslip,toe,tyoura,next_wage,used_unemp_benefit,
                                    wage_reduction,unemp_after_ra,unempwage,unempwage_basis,
                                    children_under7=children_under7,children_under18=children_under18)

        if self.plotdebug:
            self.render(done=done,reward=reward, netto=netto)
            #self.render_infostate()

        return np.array(self.state), reward, done, benq
        
    def scale_q(self,npv,benq):
        benq['verot']*=npv
        benq['etuustulo_brutto']*=npv
        benq['valtionvero']*=npv
        benq['kunnallisvero']*=npv
        benq['asumistuki']*=npv
        benq['elake_maksussa']*=npv
        benq['kokoelake']*=npv
        benq['perustulo']*=npv
        benq['palkkatulot']*=npv
        benq['kateen']*=npv

        return benq        

    # WITH RANDOMNESS
    def log_utility(self,income,employment_state,age,g=0,pinkslip=0,prefnoise=0):
        '''
        Log-utiliteettifunktio muokattuna lähteestä Määttänen, 2013 & Hakola & Määttänen, 2005

        Tulot _income_ ovat vuositasolla, jotta askelpituuden muutos ei vaikuta vapaa-aika-vakioihin
        '''

        # kappa tells how much person values free-time
        if g<3: # miehet
            kappa_kokoaika=0.607 # 0.635 # 0.665
            mu_scale=0.203 # 0.14 # 0.30 # 0.16 # how much penalty is associated with work increase with age after mu_age
            mu_age=59.25 # P.O. 60??
            kappa_osaaika=0.65*kappa_kokoaika
            kappa_hoitovapaa=0.04
            kappa_ve=0.00 # ehkä 0.10?
        else: # naiset
            kappa_kokoaika=0.534 # 0.605 # 0.58
            mu_scale=0.151 # 0.25 # 0.25 # 0.17 # how much penalty is associated with work increase with age after mu_age
            mu_age=59.25 # 61 # P.O. 60??
            kappa_osaaika=0.44*kappa_kokoaika # 0.42*kappa_kokoaika
            kappa_hoitovapaa=0.08
            kappa_ve=0.17 # ehkä 0.10?
                
        if self.include_preferencenoise:
            kappa_kokoaika += prefnoise
        
        #if age<25:
        # alle 25-vuotiaalla eri säännöt, vanhempien tulot huomioidaan jne
        #    kappa_pinkslip = 0.25
        #else:
        if pinkslip>0: # irtisanottu
            kappa_pinkslip = 0 # irtisanotuille ei vaikutuksia
        else:
            kappa_pinkslip = 0.26 # irtisanoutumisesta seuraava alennus
        
        if age>mu_age:
            kappa_kokoaika *= (1+mu_scale*max(0,age-mu_age))
            kappa_osaaika *= (1+mu_scale*max(0,age-mu_age))

        if employment_state in set([1,8]):
            kappa= -kappa_kokoaika
        elif employment_state in set([9,10]):
            kappa= -kappa_osaaika
        elif employment_state in set([0,4,13]):
            kappa= -kappa_pinkslip
        elif employment_state == 2:
            kappa=kappa_ve
        elif employment_state == 7:
            kappa=kappa_hoitovapaa
        elif employment_state == 11:
            kappa=0 #kappa_outsider
        elif employment_state == 12:
            kappa=0 #kappa_opiskelija
        else: # states 3, 5, 6, 7, 14, 15
            kappa=0
        
        # hyöty/score
        #print(type(np),income,type(income))
        u=np.log(income)+kappa

        if u is np.inf:
            print('inf: state ',employment_state)

        if income<1:
            print('inf: state ',employment_state)

        return u/10 # tulot ovat vuositasolla, mutta skaalataan hyöty

    # From Määttänen, 2013
    def wage_process(self,w,age,ave=3300*12):
        '''
        Palkkaprosessi lähteestä Määttänen, 2013 
        '''
        eps=np.random.normal(loc=0,scale=0.02,size=1)[0]
        a0=ave
        a1=0.89
        if w>0:
            wt=a0*np.exp(a1*np.log(w/a0)+eps)
        else:
            wt=a0*np.exp(eps)

        return wt

    def wage_process_simple(self,w,age,ave=3300*12):
        '''
        debug-versio palkkaprosessista
        '''
        return w

    def compute_salary(self,group=1,debug=True):
        '''
        Alussa ajettava funktio, joka tekee palkat yhtä episodia varten
        '''
        group_ave=np.array([2000,3300,5000,0.85*2000,0.85*3300,0.85*5000])*12

        a0=group_ave[group]

        self.salary[self.min_age]=np.maximum(self.min_salary,np.random.normal(loc=a0,scale=12*1000,size=1)[0]) # e/y

        if debug:
            self.salary[self.min_age+1:self.max_age+1]=self.salary[self.min_age]
        else:
            for age in range(self.min_age+1,self.max_age+1):
                self.salary[age]=self.wage_process(self.salary[age-1],age,ave=a0)

    def get_wage(self,age,reduction,pinkslip=0):
        '''
        palkka age-ikäiselle time_in_state-vähennyksellä työllistymispalkkaan
        '''
        if age<self.max_age and age>=self.min_age-1:
            return np.maximum(self.min_salary,self.salary[int(np.floor(age))]*max(0,(1-reduction)))
        else:
            return 0

    # wage process reparametrized
    def wage_process_TK(self,w,age,a0=3300*12,a1=3300*12,g=1):
        '''
        Palkkaprosessi muokattu lähteestä Määttänen, 2013 
        '''
        #group_sigmas=[0.08,0.10,0.15]
        #group_sigmas=[0.09,0.10,0.13]
        group_sigmas=[0.05,0.05,0.05]
        sigma=group_sigmas[g]
        eps=np.random.normal(loc=0,scale=sigma,size=1)[0]
        c1=0.89
        if w>0:
             # pidetään keskiarvo/a1 samana kuin w/a0
            wt=a1*np.exp(c1*np.log(w/a0)+eps-0.5*sigma*sigma)
        else:
            wt=a1*np.exp(eps)

        # täysiaikainen vuositulo vähintään self.min_salary
        wt=np.maximum(self.min_salary,wt)

        return wt

    def compute_salary_TK(self,group=1,debug=False,initial_salary=None):
        '''
        Alussa ajettava funktio, joka tekee palkat yhtä episodia varten
        '''
        # TK:n aineisto vuodelta 2018
        # iät 20-70
        palkat_ika_miehet=12.5*np.array([2339.01,2489.09,2571.40,2632.58,2718.03,2774.21,2884.89,2987.55,3072.40,3198.48,3283.81,3336.51,3437.30,3483.45,3576.67,3623.00,3731.27,3809.58,3853.66,3995.90,4006.16,4028.60,4104.72,4181.51,4134.13,4157.54,4217.15,4165.21,4141.23,4172.14,4121.26,4127.43,4134.00,4093.10,4065.53,4063.17,4085.31,4071.25,4026.50,4031.17,4047.32,4026.96,4028.39,4163.14,4266.42,4488.40,4201.40,4252.15,4443.96,3316.92,3536.03,3536.03])
        palkat_ika_naiset=12.5*np.array([2223.96,2257.10,2284.57,2365.57,2443.64,2548.35,2648.06,2712.89,2768.83,2831.99,2896.76,2946.37,2963.84,2993.79,3040.83,3090.43,3142.91,3159.91,3226.95,3272.29,3270.97,3297.32,3333.42,3362.99,3381.84,3342.78,3345.25,3360.21,3324.67,3322.28,3326.72,3326.06,3314.82,3303.73,3302.65,3246.03,3244.65,3248.04,3223.94,3211.96,3167.00,3156.29,3175.23,3228.67,3388.39,3457.17,3400.23,3293.52,2967.68,2702.05,2528.84,2528.84])
        g_r=[0.77,1.0,1.23]
        #group_ave=np.array([2000,3300,5000,0.85*2000,0.85*3300,0.85*5000])*12

        if debug: # flat wages, no change in time, all randomness at initialization
            a0=3465.0*12.5 # keskiarvo TK:n aineistossa
            self.salary[self.min_age-1]=np.maximum(self.min_salary,np.random.normal(loc=a0,scale=12*1000,size=1)[0]) # e/y
            self.salary[self.min_age:self.max_age+1]=self.salary[self.min_age-1]
        else: # randomness and time-development included
            if group>2: # naiset
                r=g_r[group-3]
                if initial_salary is not None:
                    a0=initial_salary
                else:
                    a0=palkat_ika_naiset[0]*r
                
                a1=palkat_ika_naiset[0]*r/5
                self.salary[self.min_age-1]=np.maximum(self.min_salary,np.random.normal(loc=a0,scale=a1,size=1)[0]) # e/y

                for age in range(self.min_age,self.max_age+1):
                    a0=palkat_ika_naiset[age-1-self.min_age]*r
                    a1=palkat_ika_naiset[age-self.min_age]*r
                    self.salary[age]=self.wage_process_TK(self.salary[age-1],age,a0,a1)
            else: # miehet
                r=g_r[group]
                if initial_salary is not None:
                    a0=initial_salary
                else:
                    a0=palkat_ika_miehet[0]*r
                a1=palkat_ika_miehet[0]*r/5
                self.salary[self.min_age-1]=np.maximum(self.min_salary,np.random.normal(loc=a0,scale=a1,size=1)[0]) # e/y

                for age in range(self.min_age,self.max_age+1):
                    a0=palkat_ika_miehet[age-1-self.min_age]*r
                    a1=palkat_ika_miehet[age-self.min_age]*r
                    self.salary[age]=self.wage_process_TK(self.salary[age-1],age,a0,a1)


    def state_encode_mort(self,emp,g,pension,old_wage,age,time_in_state,paid_pension,pink,
                          toe,tyohist,next_wage,used_unemp_benefit,wage_reduction,
                          unemp_after_ra,unempwage,unempwage_basis,
                          prefnoise=0,children_under7=0,children_under18=0):
        '''
        Tilan koodaus neuroverkkoa varten. Arvot skaalataan ja tilat one-hot-enkoodataan

        Käytetään, jos kuolleisuus mukana
        '''
        if self.include_children:
            if self.include_preferencenoise:
                d=np.zeros(self.n_empl+self.n_groups+19)
            else:
                d=np.zeros(self.n_empl+self.n_groups+18)
        else:
            if self.include_preferencenoise:
                d=np.zeros(self.n_empl+self.n_groups+17)
            else:
                d=np.zeros(self.n_empl+self.n_groups+16)
        
        states=self.n_empl
        if emp==1:
            d[0:states]=np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        elif emp==0:
            d[0:states]=np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        elif emp==2:
            d[0:states]=np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
        elif emp==3:
            d[0:states]=np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0])
        elif emp==4:
            d[0:states]=np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])
        elif emp==5:
            d[0:states]=np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])
        elif emp==6:
            d[0:states]=np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0])
        elif emp==7:
            d[0:states]=np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0])
        elif emp==8:
            d[0:states]=np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])
        elif emp==9:
            d[0:states]=np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0])
        elif emp==10:
            d[0:states]=np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0])
        elif emp==11:
            d[0:states]=np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0])
        elif emp==12:
            d[0:states]=np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0])
        elif emp==13:
            d[0:states]=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0])
        elif emp==14:
            d[0:states]=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0])
        elif emp==15:
            d[0:states]=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])
        else:
            print('state_encode error '+str(emp))

        states2=states+self.n_groups
        if g==1:
            d[states:states2]=np.array([0,1,0,0,0,0])
        elif g==0:
            d[states:states2]=np.array([1,0,0,0,0,0])
        elif g==2:
            d[states:states2]=np.array([0,0,1,0,0,0])
        elif g==3:
            d[states:states2]=np.array([0,0,0,1,0,0])
        elif g==4:
            d[states:states2]=np.array([0,0,0,0,1,0])
        elif g==5:
            d[states:states2]=np.array([0,0,0,0,0,1])
        else:
            print('state_encode g-error '+str(g))

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

        if tyohist>self.tyohistoria_vaatimus:
            hist400=1
        else:
            hist400=0

        d[states2+2]=(age-(self.max_age+self.min_age)/2)/20
        d[states2+3]=(time_in_state-3)/10
        #if self.include300:
        d[states2+5]=pink # irtisanottu vai ei 
        d[states2+6]=toe-14/12 # työssäoloehto
        d[states2+7]=(tyohist-3)/20 # tyohistoria: 300/400 pv
        d[states2+8]=hist400
        if age>=self.min_retirementage:
            retaged=1
        else:
            retaged=0
        d[states2+9]=retaged
        d[states2+11]=used_unemp_benefit
        d[states2+12]=wage_reduction
        d[states2+13]=(unemp_after_ra-1)/2
        if self.include_preferencenoise:
            d[states2+15]=prefnoise
            if self.include_children:
                d[states2+17]=(children_under7-5)/10
                d[states2+18]=(children_under18-5)/10
        else:
            if self.include_children:
                d[states2+16]=(children_under7-5)/10
                d[states2+17]=(children_under18-5)/10
        
        return d


    def state_encode_nomort(self,emp,g,pension,old_wage,age,time_in_state,paid_pension,pink,
                            toe,tyohist,next_wage,used_unemp_benefit,wage_reduction,
                            unemp_after_ra,unempwage,unempwage_basis,
                            prefnoise=0,children_under7=0,children_under18=0):
        '''
        Tilan koodaus neuroverkkoa varten. Arvot skaalataan ja tilat one-hot-enkoodataan

        Käytetään, jos kuolleisuus ei mukana
        '''
        if self.include_children:
            if self.include_preferencenoise:
                d=np.zeros(self.n_empl+self.n_groups+19)
            else:
                d=np.zeros(self.n_empl+self.n_groups+18)
        else:
            if self.include_preferencenoise:
                d=np.zeros(self.n_empl+self.n_groups+17)
            else:
                d=np.zeros(self.n_empl+self.n_groups+16)
                    
        states=self.n_empl
        # d2=np.zeros(n_empl,1)
        # d2[emp]=1
        if emp==1:
            d[0:states]=np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
        elif emp==0:
            d[0:states]=np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        elif emp==2:
            d[0:states]=np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0])
        elif emp==3:
            d[0:states]=np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])
        elif emp==4:
            d[0:states]=np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])
        elif emp==5:
            d[0:states]=np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0])
        elif emp==6:
            d[0:states]=np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0])
        elif emp==7:
            d[0:states]=np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])
        elif emp==8:
            d[0:states]=np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0])
        elif emp==9:
            d[0:states]=np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0])
        elif emp==10:
            d[0:states]=np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0])
        elif emp==11:
            d[0:states]=np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0])
        elif emp==12:
            d[0:states]=np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0])
        elif emp==13:
            d[0:states]=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0])
        elif emp==14:
            d[0:states]=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])
        elif emp==15:
            print('no state 15 in state_encode_nomort')
        else:
            print('state_encode error '+str(emp))

        states2=states+self.n_groups
        if g==1:
            d[states:states2]=np.array([0,1,0,0,0,0])
        elif g==0:
            d[states:states2]=np.array([1,0,0,0,0,0])
        elif g==2:
            d[states:states2]=np.array([0,0,1,0,0,0])
        elif g==3:
            d[states:states2]=np.array([0,0,0,1,0,0])
        elif g==4:
            d[states:states2]=np.array([0,0,0,0,1,0])
        elif g==5:
            d[states:states2]=np.array([0,0,0,0,0,1])
        else:
            print('state_encode g-error '+str(g))

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
        d[states2+3]=(time_in_state-3)/10
        if age>=self.min_retirementage:
            retaged=1
        else:
            retaged=0

        #if self.include300:
        d[states2+5]=pink # irtisanottu vai ei 
        d[states2+6]=toe-14/12 # työssäoloehto
        d[states2+7]=(tyohist-3)/20 # tyohistoria: 300/400 pv
        if tyohist>self.tyohistoria_vaatimus:
            hist400=1
        else:
            hist400=0

        d[states2+8]=hist400
        d[states2+9]=retaged
        d[states2+11]=used_unemp_benefit-1
        d[states2+12]=wage_reduction
        d[states2+13]=(unemp_after_ra-1)/2
        if self.include_preferencenoise:
            d[states2+16]=prefnoise
            if self.include_children:
                d[states2+17]=(children_under7-5)/10
                d[states2+18]=(children_under18-5)/10
        else:
            if self.include_children:
                d[states2+16]=(children_under7-5)/10
                d[states2+17]=(children_under18-5)/10
        
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
        time_in_state=vec[pos+3]*10+3
        #if self.include300:
        pink=vec[pos+5] # irtisanottu vai ei 
        toe=vec[pos+6]+14/12 # työssäoloehto, kesto
        tyohist=vec[pos+7]*20+3 # työhistoria
        used_unemp_benefit=vec[pos+11]+1 # käytetty työttömyyspäivärahapäivien määrä
        wage_reduction=vec[pos+12] # käytetty työttömyyspäivärahapäivien määrä
        unemp_after_ra=vec[pos+13]*2+1
        
        if self.include_preferencenoise:
            prefnoise=vec[pos+16]
            if self.include_children:
                children_under7=int(vec[pos+17])
                children_under18=int(vec[pos+18])
            else:
                children_under7=0
                children_under18=0
        else:
            prefnoise=0
            if self.include_children:
                children_under7=int(vec[pos+16]*10+5)
                children_under18=int(vec[pos+17]*10+5)
            else:
                children_under7=0
                children_under18=0

        if return_nextwage:
            return int(emp),int(g),pension,wage,age,time_in_state,paid_pension,int(pink),toe,\
                   tyohist,used_unemp_benefit,wage_reduction,unemp_after_ra,\
                   unempwage,unempwage_basis,prefnoise,children_under7,children_under18,next_wage
        else:
            return int(emp),int(g),pension,wage,age,time_in_state,paid_pension,int(pink),toe,\
                   tyohist,used_unemp_benefit,wage_reduction,unemp_after_ra,\
                   unempwage,unempwage_basis,prefnoise,children_under7,children_under18

    def unit_test_code_decode(self):
        for k in range(10):
            emp=random.randint(0,3)
            g=np.random.randint(0,6)
            pension=random.uniform(0,80_000)
            old_wage=random.uniform(0,80_000)
            age=np.random.randint(0,60)
            time_in_state=random.uniform(0,30)
            paid_pension=random.uniform(0,80_000)
            pink=np.random.randint(2)
            toe=np.random.uniform(0,3)
            tyohist=np.random.uniform(0,20)
            next_wage=random.uniform(0,80_000)
            used_unemp_benefit=np.random.uniform(0,20)
            wage_reduction=random.uniform(0,1.0)
            unemp_after_ra=random.uniform(0,10.0)
            unempwage=random.uniform(0,80_000)
            unempwage_basis=random.uniform(0,80_000)
            prefnoise=random.uniform(-1,1)
            children_under7=np.random.randint(0,10)
            children_under18=np.random.randint(0,10)
        
            vec=self.state_encode(emp,g,pension,old_wage,age,time_in_state,paid_pension,pink,
                                toe,tyohist,next_wage,used_unemp_benefit,wage_reduction,
                                unemp_after_ra,unempwage,unempwage_basis,
                                prefnoise=prefnoise,children_under7=children_under7,children_under18=children_under18)
            emp2,g2,pension2,wage2,age2,time_in_state2,paid_pension2,pink2,toe2,\
            tyohist2,used_unemp_benefit2,wage_reduction2,unemp_after_ra2,\
            unempwage2,unempwage_basis2,prefnoise2,children_under7_2,children_under18_2,next_wage2\
                =self.state_decode(vec,return_nextwage=True)
                
            self.check_state(emp,g,pension,old_wage,age,time_in_state,paid_pension,pink,
                                toe,tyohist,next_wage,used_unemp_benefit,wage_reduction,
                                unemp_after_ra,unempwage,unempwage_basis,
                                prefnoise,children_under7,children_under18,
                                emp2,g2,pension2,wage2,age2,time_in_state2,paid_pension2,pink2,toe2,\
                                tyohist2,used_unemp_benefit2,wage_reduction2,unemp_after_ra2,\
                                unempwage2,unempwage_basis2,prefnoise2,children_under7_2,children_under18_2,next_wage2)
        
    
    def check_state(self,emp,g,pension,old_wage,age,time_in_state,paid_pension,pink,
                                toe,tyohist,next_wage,used_unemp_benefit,wage_reduction,
                                unemp_after_ra,unempwage,unempwage_basis,
                                prefnoise,children_under7,children_under18,
                                emp2,g2,pension2,old_wage2,age2,time_in_state2,paid_pension2,pink2,toe2,\
                                tyohist2,used_unemp_benefit2,wage_reduction2,unemp_after_ra2,\
                                unempwage2,unempwage_basis2,prefnoise2,children_under7_2,children_under18_2,next_wage2):
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
        if not children_under7==children_under7_2:  
            print('children_under7: {} vs {}'.format(children_under7,children_under7_2))
        if not children_under18==children_under18_2:  
            print('children_under18: {} vs {}'.format(children_under18,children_under18_2))
    
    def reset(self,init=None):
        '''
        Open AI-interfacen mukainen reset-funktio, joka nollaa laskennan alkutilaan
        '''
        age=int(self.min_age)
        pension=0
        time_in_state=0
        pink=0
        toe=0
        tyohist=0
        
        # set up salary for the entire career
        g=random.choices(np.array([0,1,2],dtype=int),weights=[0.3,0.5,0.2])[0]
        gender=random.choices(np.array([0,1],dtype=int),weights=[0.5,0.5])[0]
        group=int(g+gender*3)
        initial_salary=None
        if self.reset_exploration_go and self.train:
            if self.reset_exploration_ratio>np.random.uniform():
                #print('exploration')
                initial_salary=np.random.uniform(low=1_000,high=100_000)
                pension=random.uniform(0,80_000)
                toe=random.choices(np.array([0,0.25,0.5,0.75,1.0,1.5,2.0,2.5],dtype=int),
                    weights=[0.3,0.1,0.1,0.1,0.1,0.1,0.1,0.1])[0]
                toe=random.choices(np.array([0,0.25,0.5,0.75,1.0,1.5,2.0,2.5],dtype=int),
                    weights=[0.3,0.1,0.1,0.1,0.1,0.1,0.1,0.1])[0]
        
        self.compute_salary_TK(group=group,initial_salary=initial_salary)
        old_wage=self.salary[self.min_age]
        next_wage=old_wage
        used_unemp_benefit=0
        wage_reduction=0
        unemp_after_ra=0
        unempwage=0
        unempwage_basis=old_wage
        children_under7=0
        children_under18=0
        
        
        self.init_infostate()
        
        if gender==0: # miehet
            employment_state=random.choices(np.array([13,0,1,10,3,11,12,14],dtype=int),
                weights=[0.133*3/5,0.133*2/5,0.68*0.374,0.32*0.374,0.014412417,0.151,0.240,0.089])[0]
        else: # naiset
            employment_state=random.choices(np.array([13,0,1,10,3,11,12,14],dtype=int),
                weights=[0.073*3/5,0.073*2/5,0.44*0.550,0.56*0.550,0.0121151,0.077,0.283,0.00362])[0] 

        if employment_state==0:
            tyohist=1.0
            toe=1.0
            wage_reduction=0.05
        elif employment_state==13:
            tyohist=0.0
            toe=0.0
            wage_reduction=0.05
        elif employment_state==11:
            tyohist=0.0
            toe=0.0
            wage_reduction=0.10
        #elif employment_state==12:
        #    wage_reduction=0.25
            
        # tarvitseeko alkutilassa laskea muita tietoja uusiksi? ei kait

        if self.plotdebug:
            print('emp {} gender {} g {} old_wage {} next_wage {}'.format(employment_state,gender,g,old_wage,next_wage))

        if self.include_preferencenoise:
            prefnoise=np.random.normal(loc=0,scale=0.1,size=1)[0]
            self.state = self.state_encode(employment_state,group,pension,old_wage,self.min_age,
                                            time_in_state,0,pink,toe,tyohist,next_wage,
                                            used_unemp_benefit,wage_reduction,unemp_after_ra,
                                            unempwage,unempwage_basis,
                                            prefnoise=prefnoise,children_under7=children_under7,children_under18=children_under18)
        else:
            self.state = self.state_encode(employment_state,group,pension,old_wage,self.min_age,
                                            time_in_state,0,pink,toe,tyohist,next_wage,
                                            used_unemp_benefit,wage_reduction,unemp_after_ra,
                                            unempwage,unempwage_basis,
                                            children_under7=children_under7,children_under18=children_under18)
        #print('shape=',self.state.shape)
        self.steps_beyond_done = None

        return np.array(self.state)

    def render(self, mode='human', close=False, done=False, reward=None, netto=None):
        '''
        Tulostus-rutiini
        '''
        emp,g,pension,wage,age,time_in_state,paid_pension,pink,toe,tyohist,used_unemp_benefit,\
        wage_red,unemp_after_ra,unempwage,unempwage_basis,prefnoise,c7,c18,next_wage=self.state_decode(self.state,return_nextwage=True)
        if reward is None:
            print('s {} g {} palkka {:.2f} nw {:.2f} ikä {:.2f} t-i-s {:.2f} tul.eläke {:.2f} alk.eläke {:.2f} irti {} toe {:.2f} työhist {:.2f} ueb {:.2f} wr {:.4f} uew {:.2f} (b {:.2f}) c7 {:.0f} c18 {:.0f}'.format(\
                emp,g,wage,next_wage,age,time_in_state,pension,paid_pension,pink,toe,tyohist,used_unemp_benefit,wage_red,unempwage,unempwage_basis,c7,c18))
        elif netto is None:
            print('s {} g {} palkka {:.2f} nw {:.2f} ikä {:.2f} t-i-s {:.2f} tul.eläke {:.2f} alk.eläke {:.2f} irti {} toe {:.2f} työhist {:.2f} ueb {:.2f} wr {:.4f} uew {:.2f} (b {:.2f}) c7 {:.0f} c18 {:.0f} r {:.4f}'.format(\
                emp,g,wage,next_wage,age,time_in_state,pension,paid_pension,pink,toe,tyohist,used_unemp_benefit,wage_red,unempwage,unempwage_basis,c7,c18,reward))
        else:
            print('s {} g {} palkka {:.2f} nw {:.2f} ikä {:.2f} t-i-s {:.2f} tul.eläke {:.2f} alk.eläke {:.2f} irti {} toe {:.2f} työhist {:.2f} ueb {:.2f} wr {:.4f} uew {:.2f} (b {:.2f}) c7 {:.0f} c18 {:.0f} r {:.4f} n {:.2f}'.format(\
                emp,g,wage,next_wage,age,time_in_state,pension,paid_pension,pink,toe,tyohist,used_unemp_benefit,wage_red,unempwage,unempwage_basis,c7,c18,reward,netto))
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
        tis_max=(self.max_age-self.min_age-3)/10
        tis_min=-3/10
        pink_min=0 # irtisanottu vai ei 
        pink_max=1 # irtisanottu vai ei 
        toe_min=0-self.max_toe*0.5 # työssäoloehto
        toe_max=self.max_toe-self.max_toe*0.5 # työssäoloehto
        thist_min=-3/20 # tyohistoria: 300/400 pv
        thist_max=(self.max_age-self.min_age-3)/20 # tyohistoria: 300/400 pv
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
            state_min,
            state_min,
            wage_min,
            #out_min,
            ben_min,
            wr_min,
            unra_min,
            wage_min,
            wage_min]
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
            state_max,
            state_max,
            wage_max,
            #out_max,
            ben_max,
            wr_max,
            unra_max,
            wage_max,
            wage_max]
            
        if self.include_mort: # if mortality is included, add one more state
              low.prepend(state_min)
              high.prepend(state_max)
              
        if self.include_preferencenoise:
            low.append(pref_min)
            high.append(pref_max)
                
        if self.include_children:
              low.append(child_min)
              high.append(child_max)
              low.append(child_min)
              high.append(child_max)
                  
        self.low=np.array(low)
        self.high=np.array(high)
                    
    def explain(self):
        '''
        Tulosta laskennan parametrit
        '''
        print('Parameters of lifecycle:\ntimestep {}\ngamma {} ({} per anno)\nmin_age {}\nmax_age {}\nmin_retirementage {}'.format(self.timestep,self.gamma,self.gamma**(1.0/self.timestep),self.min_age,self.max_age,self.min_retirementage))
        print('max_retirementage {}\nansiopvraha_kesto300 {}\nansiopvraha_kesto400 {}\nansiopvraha_toe {}'.format(self.max_retirementage,self.ansiopvraha_kesto300,self.ansiopvraha_kesto400,self.ansiopvraha_toe))
        print('perustulo {}\nkarenssi_kesto {}\nmortality {}\nrandomness {}'.format(self.perustulo,self.karenssi_kesto,self.include_mort,self.randomness))
        print('include_putki {}\ninclude_pinkslip {}'.format(self.include_putki,self.include_pinkslip))
        print('sigma_reduction {}\nplotdebug {}\n'.format(self.use_sigma_reduction,self.plotdebug))
        

    def unempright_left(self,emp,tis,bu,ika,tyohistoria):
        '''
        Tilastointia varten lasketaan jäljellä olevat ansiosidonnaiset työttömyysturvapäivät
        '''
        if ika>=self.minage_500 and tyohistoria>=self.tyohistoria_vaatimus500:
            kesto=self.ansiopvraha_kesto500
        elif tyohistoria>=self.tyohistoria_vaatimus:
            kesto=self.ansiopvraha_kesto400
        else:
            kesto=self.ansiopvraha_kesto300
        
        kesto=kesto/(12*21.5)
        #if irtisanottu<1 and time_in_state<self.karenssi_kesto: # karenssi, jos ei irtisanottu
        
        if emp==13:
            return tis
        else:
            return kesto-bu
            
    def init_infostate(self,lapsia=0,lasten_iat=np.zeros(15),lapsia_paivakodissa=0):
        '''
        Alustaa infostate-dictorionaryn
        Siihen talletetaan tieto aiemmista tiloista, joiden avulla lasketaan statistiikkoja
        '''
        self.infostate={}
        self.infostate['states']=np.zeros(self.n_time)-1
        self.infostate['wage']=np.zeros(self.n_time)-1
        self.infostate['latest']=0
        self.infostate['children_n']=0
        self.infostate['children_date']=np.zeros(15)
        #self.infostate['unemp_after_retage']=0 # kauanko ollut työtön alimman vanhuuseläkeiän jälkeen
        
    def infostate_add_child(self,age):
        if self.infostate['children_n']<14:
            self.infostate['children_date'][self.infostate['children_n']]=age
            self.infostate['children_n']=self.infostate['children_n']+1
        
    def update_infostate(self,t,state,wage):
        self.infostate['states'][t]=state
        self.infostate['latest']=int(t)
        if state==1:
            self.infostate['wage'][t]=wage
        elif state==10:
            self.infostate['wage'][t]=wage*0.5
        else:
            self.infostate['wage'][t]=0
        #if age>=self.min_retirementage:
        #    self.infostate['unemp_after_retage']=self.infostate['unemp_after_retage']+self.timestep
        
    def render_infostate(self):
        print('states {}'.format(self.infostate['states']))
        
    def render_stats(self):
        print('states {}'.format(self.infostate['states']))
        
    def comp_infostats(self,age):
        # laske työssäoloehto tarkasti
        # laske työttömyysturvaan vaikuttavat lasten määrät
        toes=0
        n_toe=int(np.floor(self.max_toe/self.timestep))
        emp_states={1,10}
        unemp_states={0,4}
        family_states={5,6,7}
        ret_states={2,3,8,9}
        start_t=max(self.infostate['latest']-n_toe,0)
        wage=0
        #print(start_t,self.infostate['latest'],self.infostate['states'][start_t:self.infostate['latest']],
        #    self.infostate['wage'][start_t:self.infostate['latest']])
        
        if self.infostate['latest'] not in ret_states:
            if self.infostate['latest'] in family_states:
                # laskee, onko ollut riittävä toe ansiosidonnaiseen, ei onko päiviä jäljellä
                t2=self.infostate['latest']
                nt=0
                while nt<n_toe and t2>=0:
                    if self.infostate['states'][t2] in family_states:
                        pass
                    elif self.infostate['states'][t2] in emp_states:
                        toes+=self.timestep
                        wage+=self.infostate['wage'][t2]*self.timestep
                        nt=nt+1
                    elif self.infostate['states'][t2] in unemp_states:
                        nt=nt+1
                    else:
                        nt=nt+1
                    t2=t2-1
            else:
                # laskee, onko toe täyttynyt viimeisimmän ansiosidonnaisen jakson jälkeen
                t2=self.infostate['latest']
                nt=0
                while nt<n_toe and t2>=0:
                    if self.infostate['states'][t2] in family_states:
                        pass
                    elif self.infostate['states'][t2] in emp_states:
                        toes+=self.timestep
                        wage+=self.infostate['wage'][t2]*self.timestep
                        nt=nt+1
                    elif self.infostate['states'][t2] in unemp_states:
                        nt=n_toe
                    else:
                        nt=nt+1
                    t2=t2-1
            if toes>=self.ansiopvraha_toe and toes>0:
                wage=wage/toes
            else:
                wage=0
            
        children_under18=0
        children_under7=0
        for k in range(self.infostate['children_n']):
            c_age=age-self.infostate['children_date'][k]
            if c_age<18:
                children_under18=children_under18+1
                if c_age<7:
                    children_under7=children_under7+1

        return toes,wage,children_under7,children_under18

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
            
    def infostat_check_80percent(self,t):
        pass