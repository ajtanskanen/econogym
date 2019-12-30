"""

    unemloyment_v1


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

class UnemploymentLargeEnv(gym.Env):
    """
    Description:
        The Finnish Unemployment Pension Scheme 

    Source:
        This environment corresponds to the environment of the Finnish Social Security

    Observation: 
        Type: Box(10) 
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
        9    Työstä pois (aika)            0            100
     ( 10    Irtisanottu (jos valittu)     0              1 )

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
        13  Kuollut (jos kuolleisuus mukana)

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

        self.toe_vaatimus=0.5 # = 6kk
        self.karenssi_kesto=0.25 #0.25 # = 3kk
        self.isyysvapaa_kesto=0.25 # = 3kk
        self.aitiysvapaa_kesto=0.75 # = 9kk ml vanhempainvapaa
        self.min_tyottputki_ika=58.5 # vuotta. Ikä, jolloin työttömyyden pitää alkaa, jotta pääsee putkeen
        self.kht_kesto=2.0 # kotihoidontuen kesto 2 v
        self.tyohistoria_vaatimus=3.0 # 3 vuotta
        self.ansiopvraha_kesto400=400
        self.ansiopvraha_kesto300=300

        self.timestep=0.25
        self.gamma=0.92**self.timestep # discounting
        self.max_age=71
        self.min_age=20
        self.min_retirementage=63.5 #65
        self.max_retirementage=68.5 # 70

        self.elinaikakerroin=0.925 # etk:n arvio 1962 syntyneille
        reaalinen_palkkojenkasvu=1.016
        self.palkkakerroin=(0.8*1+0.2*1.0/reaalinen_palkkojenkasvu)**self.timestep
        self.elakeindeksi=(0.2*1+0.8*1.0/reaalinen_palkkojenkasvu)**self.timestep
        self.kelaindeksi=(1.0/reaalinen_palkkojenkasvu)**self.timestep

        self.include_mort=False # onko kuolleisuus mukana laskelmissa
        #self.include300=True # onko työuran kesto mukana laskelmissa
        self.perustulo=False # onko Kelan perustulo laskelmissa
        self.randomness=True # onko stokastiikka mukana
        self.mortstop=True # pysäytä kuolleisuuden jälkeen
        self.include_putki=True # työttömyysputki mukana
        self.include_pinkslip=True # irtisanomiset mukana

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
                    self.mortstop=value**self.timestep
            elif key=='gamma':
                if value is not None:
                    self.gamma=value**self.timestep
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
            elif key=='plotdebug':
                if value is not None:
                    self.plotdebug=value  
 
        # ei skaalata!
        #self.ansiopvraha_kesto400=self.ansiopvraha_kesto400/(12*21.5)
        #self.ansiopvraha_kesto300=self.ansiopvraha_kesto300/(12*21.5)              

        # paljonko työstä poissaolo vaikuttaa palkkaan
        self.salary_const=0.05*self.timestep

        # karttumaprosentit
        self.acc=0.015*self.timestep
        self.acc_family=1.15*self.acc
        self.acc_unemp=0.75*self.acc

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

        #print('n_empl',self.n_empl)

        self.salary=np.zeros(self.max_age+1)

        # ryhmäkohtaisia muuttujia
        self.disability_intensity=self.get_disability_rate()*self.timestep # tn tulla työkyvyttömäksi
        if self.include_pinkslip:
            self.pinkslip_intensity=0.05*self.timestep # todennäköisyys tulla irtisanotuksi vuodessa, skaalaa!
        else:
            self.pinkslip_intensity=0 # .05*self.timestep # todennäköisyys tulla irtisanotuksi vuodessa, skaalaa!
        
        self.birth_intensity=self.get_birth_rate()*self.timestep # todennäköisyys saada lapsi, skaalaa!
        self.mort_intensity=self.get_mort_rate()*self.timestep # todennäköisyys , skaalaa!
        self.npv=self.comp_npv()

        self.set_state_limits()
        if self.include_mort: # and not self.mortstop:
            if self.include_mort and self.mortstop:
                print('Mortality included, stopped')
            else:
                print('Mortality included, not stopped')

            self.n_empl=14 # state of employment, 0,1,2,3,4
            self.state_encode=self.state_encode_mort
        else:
            print('No mortality included')
            self.n_empl=13 # state of employment, 0,1,2,3,4
            self.state_encode=self.state_encode_nomort

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        #self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

        if self.randomness:
            self.log_utility=self.log_utility_randomness
        else:
            self.log_utility=self.log_utility_norandomness

        if self.perustulo:
            self.log_utility=self.log_utility_perustulo

        self.ben = fin_benefits.Benefits()

    def comp_npv(self):
        '''
        lasketaan montako timestep:iä (diskontattuna) max_age:n jälkeen henkilö on vanhuuseläkkeellä 
        hyvin yksinkertainen toteutus. Tulos on odotettu lukumäärä timestep:jä
        '''
        npv=np.zeros(self.n_groups)

        for g in range(self.n_groups):
            cpsum=1
            for x in np.arange(100,self.max_age,-self.timestep):
                intx=int(np.floor(x))
                m=self.mort_intensity[intx,g]
                cpsum=m*1+(1-m)*(1+self.gamma*cpsum)
            npv[g]=cpsum

        return npv

    def comp_benefits(self,wage,old_wage,pension,employment_status,time_in_state,ika=25,
                      irtisanottu=0,tyossaoloehto=0,tyohistoria=0):
        '''
        Kutsuu fin_benefits-modulia, jonka avulla lasketaan etuudet ja huomioidaan verotus
        Laske etuuksien arvo, kun 
            wage on palkka
            old_wage on vanha palkka
            pension on eläkkeen määrä
            employment_status on töissä olo (0)/työttömyys (1)/eläkkeellä olo (2)
            prev_empl on työttömyyden kesto (0/1/2)
            ika on henkilön ikä
        '''
        p={}

        p['perustulo']=self.perustulo
        p['opiskelija']=self.perustulo
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
        p['elakkeella']=0
        if employment_status==1:
            p['tyoton']=0 # voisi olla työtön siinä mielessä, että oikeutettu soviteltuun päivärahaan
            p['t']=wage/12
            p['vakiintunutpalkka']=wage/12
            p['saa_ansiopaivarahaa']=0
        elif employment_status==0: # työtön, ansiopäiväraha alle 60 ja työmarkkinatuki
            if ika<65:
                p['tyoton']=1
                p['t']=0
                p['vakiintunutpalkka']=old_wage/12
                p['tyottomyyden_kesto']=12*21.5*time_in_state
                if ((tyohistoria>=self.tyohistoria_vaatimus and p['tyottomyyden_kesto']<=self.ansiopvraha_kesto400) or (p['tyottomyyden_kesto']<=self.ansiopvraha_kesto300)) \
                    and (tyossaoloehto>=self.toe_vaatimus) \
                    and (irtisanottu>0 or (time_in_state>=self.karenssi_kesto)): # karenssi, jos ei irtisanottu
                    p['saa_ansiopaivarahaa']=1
                else:
                    p['saa_ansiopaivarahaa']=0
            else:
                p['tyoton']=0 # ei oikeutta työttömyysturvaan
                p['t']=0
                p['vakiintunutpalkka']=0
                p['saa_ansiopaivarahaa']=0
        elif employment_status==3: # tk
            p['tyoton']=0
            p['saa_ansiopaivarahaa']=0
            p['t']=0
            p['vakiintunutpalkka']=0
            p['elakkeella']=1 
            #p['elake']=pension
        elif employment_status==4: # työttömyysputki
            if ika<65:
                p['tyoton']=1
                p['t']=0
                p['vakiintunutpalkka']=old_wage/12
                if (tyossaoloehto>=self.toe_vaatimus) and (irtisanottu>0 or (time_in_state>=self.karenssi_kesto)): # karenssi, jos ei irtisanottu
                    p['saa_ansiopaivarahaa']=1
                else:
                    p['saa_ansiopaivarahaa']=0
            else:
                p['tyoton']=0 # ei oikeutta työttömyysturvaan
                p['t']=0
                p['vakiintunutpalkka']=0
                p['saa_ansiopaivarahaa']=0
        elif employment_status==5: # ansiosidonnainen vanhempainvapaa, äidit
            p['aitiysvapaalla']=1
            p['tyoton']=0
            p['aitiysvapaa_kesto']=0
            p['t']=0
            p['vakiintunutpalkka']=old_wage/12
            p['saa_ansiopaivarahaa']=1
        elif employment_status==6: # ansiosidonnainen vanhempainvapaa, isät
            p['isyysvapaalla']=1
            p['tyoton']=0
            p['t']=0
            p['vakiintunutpalkka']=old_wage/12
            p['saa_ansiopaivarahaa']=1
        elif employment_status==7: # hoitovapaa
            p['kotihoidontuella']=1
            p['lapsia']=1
            p['tyoton']=0
            p['alle3v']=1
            p['kotihoidontuki_kesto']=time_in_state
            p['lapsia_kotihoidontuella']=p['lapsia']
            p['t']=0
            p['vakiintunutpalkka']=old_wage/12
            p['saa_ansiopaivarahaa']=0
        elif employment_status==2: # vanhuuseläke
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
        elif employment_status==8: # ve+työ
            p['tyoton']=0
            p['saa_ansiopaivarahaa']=0
            p['t']=wage/12
            p['vakiintunutpalkka']=0
            p['elakkeella']=1  
            p['tyoelake']=pension/12
        elif employment_status==9: # ve+osatyö
            p['tyoton']=0
            p['saa_ansiopaivarahaa']=0
            p['t']=wage/12
            p['vakiintunutpalkka']=0
            p['elakkeella']=1  
            p['tyoelake']=pension/12
        elif employment_status==10: # osa-aikatyö
            p['tyoton']=0
            p['saa_ansiopaivarahaa']=0
            p['t']=wage/12
            p['vakiintunutpalkka']=0
        elif employment_status==11: # työelämän ulkopuolella
            p['tyoton']=0
            p['toimeentulotuki_vahennys']=1 # oletetaan että kieltäytynyt työstä
            p['saa_ansiopaivarahaa']=0
            p['t']=0
            p['vakiintunutpalkka']=0
        elif employment_status==12: # opiskelija
            p['tyoton']=0
            p['opiskelija']=1
            p['saa_ansiopaivarahaa']=0
            p['t']=0
            p['vakiintunutpalkka']=0
        else:
            print('Unknown employment_status ',employment_status)

        # tarkastellaan yksinasuvia henkilöitä
        if employment_status==12: # opiskelija
            p['asumismenot_toimeentulo']=250
            p['asumismenot_asumistuki']=250
        else: # muu
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

        #print(p,benefitq)
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

    def get_wage(self,age,out_of_work):
        '''
        palkka age-ikäiselle time_in_state-vähennyksellä työllistymispalkkaan
        '''
        if age<self.max_age and age>=self.min_age-1:
            return self.salary[int(np.floor(age))]*(1-min(out_of_work,5)*self.salary_const)
        else:
            return 0

    def get_mort_rate(self,debug=False):
        '''
        Kuolleisuus-intensiteetit eri ryhmille
        '''
        mort=np.zeros((101,self.n_groups))
        if debug:
            dfactor=np.array([1.0,1.0,1.0,1.0,1.0,1.0])
        else:
            dfactor=np.array([1.3,1.0,0.7,1.2,1.0,0.8])
        # tilastokeskuksen kuolleisuusdata 2017 sukupuolittain
        mort[:,1]=np.array([2.12,0.32,0.17,0.07,0.07,0.10,0.00,0.09,0.03,0.13,0.03,0.07,0.10,0.10,0.10,0.23,0.50,0.52,0.42,0.87,0.79,0.66,0.71,0.69,0.98,0.80,0.77,1.07,0.97,0.76,0.83,1.03,0.98,1.20,1.03,0.76,1.22,1.29,1.10,1.26,1.37,1.43,1.71,2.32,2.22,1.89,2.05,2.15,2.71,2.96,3.52,3.54,4.30,4.34,5.09,4.75,6.17,5.88,6.67,8.00,9.20,10.52,10.30,12.26,12.74,13.22,15.03,17.24,18.14,17.78,20.35,25.57,23.53,26.50,28.57,31.87,34.65,40.88,42.43,52.28,59.26,62.92,68.86,72.70,94.04,99.88,113.11,128.52,147.96,161.89,175.99,199.39,212.52,248.32,260.47,284.01,319.98,349.28,301.37,370.17,370.17])/1000.0
        mort[:,0]=dfactor[0]*mort[:,1]
        mort[:,2]=dfactor[2]*mort[:,1]
        mort[:,4]=np.array([1.89,0.30,0.11,0.03,0.14,0.03,0.16,0.07,0.13,0.03,0.00,0.07,0.07,0.07,0.18,0.14,0.07,0.31,0.31,0.30,0.33,0.26,0.18,0.33,0.56,0.17,0.32,0.29,0.35,0.24,0.55,0.35,0.23,0.39,0.48,0.38,0.35,0.80,0.42,0.65,0.50,0.68,0.80,1.12,0.99,0.88,1.13,1.01,1.07,1.68,1.79,2.16,1.87,2.32,2.67,2.69,2.88,2.86,3.73,4.19,3.66,4.97,5.20,5.52,6.05,7.17,7.48,7.32,8.88,10.33,10.72,12.77,12.13,13.30,16.18,18.30,17.50,24.63,26.53,29.88,32.65,38.88,46.95,51.30,60.00,64.73,79.35,90.94,105.11,118.46,141.44,155.07,163.11,198.45,207.92,237.21,254.75,311.31,299.59,356.64,356.64])/1000.0
        mort[:,3]=dfactor[3]*mort[:,4]
        mort[:,5]=dfactor[5]*mort[:,4]

        return mort

    def get_disability_rate_unisex(self,debug=False):
        '''
        Työkyvyttömyys-alkavuudet eri ryhmille
        '''
        disab=np.zeros((self.max_age+1,self.n_groups))
        # male low, male mid, male high, female low, female mid, female high
        if debug:
            dfactor=np.array([1.0,1.0,1.0,1.0,1.0,1.0])
        else:
            dfactor=np.array([1.5,1.0,0.5,1.2,1.0,0.8])
            
        for g in range(self.n_groups):
            factor=dfactor[g]
            disab[20,g]=1.99*factor
            disab[21,g]=1.99*factor
            disab[22,g]=1.99*factor
            disab[23,g]=1.99*factor
            disab[24,g]=1.99*factor
            disab[25,g]=1.99*factor
            disab[26,g]=1.84*factor
            disab[27,g]=2.45*factor
            disab[28,g]=1.95*factor
            disab[29,g]=2.06*factor
            disab[30,g]=1.74*factor
            disab[31,g]=2.20*factor
            disab[32,g]=2.37*factor
            disab[33,g]=2.35*factor
            disab[34,g]=2.52*factor
            disab[35,g]=2.33*factor
            disab[36,g]=2.83*factor
            disab[37,g]=2.50*factor
            disab[38,g]=2.77*factor
            disab[39,g]=2.91*factor
            disab[40,g]=3.47*factor
            disab[41,g]=3.17*factor
            disab[42,g]=3.16*factor
            disab[43,g]=3.48*factor
            disab[44,g]=4.21*factor
            disab[45,g]=4.16*factor
            disab[46,g]=4.13*factor
            disab[47,g]=4.43*factor
            disab[48,g]=5.08*factor
            disab[49,g]=5.70*factor
            disab[50,g]=5.89*factor
            disab[51,g]=6.76*factor
            disab[52,g]=7.43*factor
            disab[53,g]=8.43*factor
            disab[54,g]=8.79*factor
            disab[55,g]=10.40*factor
            disab[56,g]=12.41*factor
            disab[57,g]=14.54*factor
            disab[58,g]=17.12*factor
            disab[59,g]=21.69*factor
            disab[60,g]=28.88*factor
            disab[61,g]=30.86*factor
            disab[62:(self.max_age+1),g]=24.45*factor

        disab=disab/1000
        return disab
        
    def get_disability_rate(self,debug=False):
        '''
        Työkyvyttömyys-alkavuudet eri ryhmille
        '''
        disab=np.zeros((self.max_age+1,self.n_groups))
        # male low, male mid, male high, female low, female mid, female high
        if debug:
            dfactor=np.array([1.0,1.0,1.0,1.0,1.0,1.0])
        else:
            dfactor=np.array([1.5,1.0,0.5,1.2,1.0,0.8])
            
        dis_miehet=np.array([ 0.004630,0.004356,0.003559,0.003081,0.003381,0.002937,0.002835,0.002951,0.002232,0.002088,0.001808,0.002747,0.002540,0.002851,0.002854,0.002704,0.002764,0.002691,0.002923,0.003033,0.003231,0.002886,0.002947,0.003319,0.003487,0.003864,0.003816,0.004289,0.005225,0.005248,0.005923,0.006897,0.006780,0.008412,0.009394,0.010691,0.012313,0.013471,0.015919,0.020455,0.026545,0.022420,0.015017,0.003898,0.000286,0.015017,0.015017,0.015017,0.015017,0.015017,0.015017 ])
        dis_naiset=np.array([ 0.005557,0.005064,0.004733,0.003931,0.003340,0.003180,0.002937,0.003415,0.002572,0.002710,0.002500,0.003701,0.002778,0.003217,0.003242,0.003512,0.003326,0.003402,0.003504,0.004077,0.004424,0.004442,0.004355,0.005005,0.005032,0.005847,0.005579,0.006115,0.006144,0.006849,0.008927,0.007070,0.008676,0.010337,0.008784,0.012450,0.013127,0.015199,0.019386,0.022249,0.029828,0.026609,0.016825,0.003603,0.000211,0.016825,0.016825,0.016825,0.016825,0.016825,0.016825 ])
            
        for g in range(3):
            disab[20:70,g]=dfactor[g]*dis_miehet
            disab[70:(self.max_age+1),g]=24.45*dfactor[g]/1000
        for g in range(3,6):
            disab[20:70,g]=dfactor[g]*dis_naiset
            disab[70:(self.max_age+1),g]=24.45*dfactor[g]/1000

        return disab        

    def get_birth_rate(self,debug=False):
        birth=np.zeros((69,self.n_groups)) # 
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

    def scale_pension(self,pension,age):
        '''
        Elinaikakertoimen ja lykkäyskorotuksen huomiointi
        '''
        return self.elinaikakerroin*pension*self.elakeindeksi*(1+0.048*(age-self.min_retirementage)) 

    def move_to_parttime(self,pension,old_wage,age,toe,tyoura,time_in_state,out_of_work):
        '''
        Siirtymä osa-aikaiseen työskentelyyn
        '''
        employment_status = 10 # switch to part-time work
        intage=int(np.floor(age))
        wage=self.get_wage(intage,out_of_work)
        parttimewage=0.5*wage
        toe=min(self.max_toe,toe+self.timestep)
        tyoura += self.timestep
        time_in_state=0
        out_of_work=0
        old_wage=0
        pension=pension*self.palkkakerroin+self.acc*parttimewage
        netto=self.comp_benefits(parttimewage,old_wage,0,employment_status,time_in_state,age)

        return employment_status,pension,wage,time_in_state,netto,toe,tyoura,out_of_work

    def move_to_work(self,pension,old_wage,age,time_in_state,toe,tyoura,out_of_work):
        '''
        Siirtymä täysiaikaiseen työskentelyyn
        '''
        employment_status = 1 # töihin
        intage=int(np.floor(age))
        wage=self.get_wage(intage,out_of_work)
        time_in_state=0
        old_wage=0
        toe=min(self.max_toe,toe+self.timestep)
        tyoura+=self.timestep
        out_of_work=0

        pension=pension*self.palkkakerroin+self.acc*wage
        netto=self.comp_benefits(wage,old_wage,0,employment_status,time_in_state,age)

        return employment_status,pension,wage,time_in_state,netto,toe,tyoura,out_of_work

    def move_to_retwork(self,pension,old_wage,age,time_in_state,paid_pension,out_of_work):
        '''
        Siirtymä vanhuuseläkkeellä työskentelyyn
        '''
        employment_status = 8 # unchanged
        intage=int(np.floor(age))
        wage=self.get_wage(intage,out_of_work)
        paid_pension=paid_pension*self.elakeindeksi
        pension=pension*self.palkkakerroin+self.acc*wage
        netto=self.comp_benefits(wage,0,paid_pension,employment_status,time_in_state,age)
        time_in_state=0
        out_of_work=0
        # tyoura+= ??

        return employment_status,pension,wage,time_in_state,netto,out_of_work

    def move_to_retpartwork(self,pension,old_wage,age,time_in_state,paid_pension,out_of_work):
        '''
        Siirtymä osa-aikaiseen vanhuuseläkkeellä työskentelyyn
        '''
        employment_status = 9 # unchanged
        intage=int(np.floor(age))
        wage=self.get_wage(intage,out_of_work)
        paid_pension=paid_pension*self.elakeindeksi
        pension=pension*self.palkkakerroin+self.acc*wage*0.5
        netto=self.comp_benefits(wage*0.5,0,paid_pension,employment_status,time_in_state,age)
        time_in_state=0
        out_of_work=0
        # tyoura+= ??

        return employment_status,pension,wage,time_in_state,netto,out_of_work

    def move_to_retirement(self,pension,old_wage,age,paid_pension,employment_status,out_of_work,all_acc=True):   
        '''
        Siirtymä vanhuuseläkkeelle
        '''
        if age>=self.min_retirementage:
            if all_acc:
                if employment_status in set([2,8,9]): # ve, ve+työ, ve+osatyö
                    if age>=self.max_retirementage:
                        paid_pension= self.elinaikakerroin*self.elakeindeksi*pension+paid_pension
                        pension=0
                    else:
                        paid_pension= self.elinaikakerroin*paid_pension
                elif employment_status==3: # tk
                    # do nothing
                    employment_status=3
                else:
                    paid_pension=self.scale_pension(pension,age)
                    paid_pension += self.ben.laske_kansanelake(age,paid_pension,1)
                    pension=0

            time_in_state=0
            employment_status = 2 
            wage=old_wage
            out_of_work+=self.timestep
            netto=self.comp_benefits(0,0,paid_pension,employment_status,0,age)
        else: # työvoiman ulkopuolella
            time_in_state=0
            employment_status = 2 
            wage=old_wage
            netto=self.comp_benefits(0,0,0,employment_status,0,age)

        return employment_status,paid_pension,pension,wage,time_in_state,netto,out_of_work

    def move_to_retdisab(self,pension,old_wage,age,paid_pension,employment_status,out_of_work):   
        '''
        Siirtymä vanhuuseläkkeelle, jossa ei voi tehdä työtä
        '''
        
        if age>=self.max_retirementage:
            paid_pension= self.elinaikakerroin*self.elakeindeksi*pension+paid_pension
            pension=0                        

        employment_status = 3
        out_of_work+=self.timestep
        wage=old_wage
        netto=self.comp_benefits(0,0,paid_pension,employment_status,0,age)
        time_in_state=0

        return employment_status,paid_pension,pension,wage,time_in_state,netto,out_of_work

    def move_to_unemp(self,pension,old_wage,age,toe,irtisanottu,out_of_work,tyoura):
        '''
        Siirtymä työttömyysturvalle
        '''
        if age>=self.min_tyottputki_ika and self.include_putki:
            employment_status  = 4 # switch
        else:
            employment_status  = 0 # switch
        time_in_state=0
        out_of_work+=self.timestep        
        intage=int(np.floor(age))
        wage=self.get_wage(intage,0)
        if age<65:
            pension=pension*self.palkkakerroin+self.acc_unemp*old_wage

        # hmm, omavastuupäivät puuttuvat!
        # omavastuupäiviä on 5/(21.5*12*self.timestep), kerroin tällöin
        # 1-5/(21.5*12*self.timestep)
        netto=self.comp_benefits(0,old_wage,0,employment_status,time_in_state,age,
                                 irtisanottu=irtisanottu,tyossaoloehto=toe,tyohistoria=tyoura)

        return employment_status,pension,wage,time_in_state,netto,toe,out_of_work

    def move_to_outsider(self,pension,old_wage,age,toe,irtisanottu,out_of_work):
        '''
        Siirtymä työvoiman ulkopuolelle
        '''
        employment_status = 11 # switch
        time_in_state=0
        out_of_work+=self.timestep        
        intage=int(np.floor(age))
        old_wage=self.get_wage(intage-1,0)
        toe=max(0,toe-self.timestep)
        wage=old_wage
        pension=pension*self.palkkakerroin

        # hmm, omavastuupäivät puuttuvat!
        # omavastuupäiviä on 5/(21.5*12*self.timestep), kerroin tällöin
        # 1-5/(21.5*12*self.timestep)
        netto=self.comp_benefits(0,0,0,employment_status,time_in_state,age,irtisanottu=0)
        paid_pension=0

        return employment_status,paid_pension,pension,wage,time_in_state,toe,netto,out_of_work

    def move_to_disab(self,pension,old_wage,age,out_of_work):
        '''
        Siirtymä työkyvyttömyyseläkkeelle
        '''
        employment_status = 3 # tk
        paid_pension=self.elinaikakerroin*pension*self.elakeindeksi + self.acc*old_wage*max(0,self.min_retirementage-age) # p.o. 5v keskiarvo
        paid_pension=self.ben.laske_kokonaiselake(65,paid_pension)
        pension=0
        #old_wage=0
        time_in_state=0
        out_of_work+=self.timestep        
        wage=old_wage
        netto=self.comp_benefits(0,0,paid_pension,employment_status,0,age)

        return employment_status,pension,paid_pension,wage,time_in_state,netto,out_of_work

    def move_to_deceiced(self,pension,old_wage,age):
        '''
        Siirtymä tilaan kuollut
        '''
        employment_status = 13 # deceiced
        wage=old_wage
        pension=pension
        netto=0
        time_in_state=0

        return employment_status,pension,wage,time_in_state,netto

    def move_to_kht(self,pension,old_wage,age,out_of_work):
        '''
        Siirtymä kotihoidontuelle
        '''
        employment_status = 7 # kotihoidontuelle
        wage=old_wage
        pension=pension*self.palkkakerroin+self.acc*self.accbasis_kht
        time_in_state=0
        out_of_work+=self.timestep        
        netto=self.comp_benefits(0,old_wage,0,employment_status,time_in_state,age)

        return employment_status,pension,wage,time_in_state,netto,out_of_work

    def move_to_fatherleave(self,pension,old_wage,age,out_of_work):
        '''
        Siirtymä isyysvapaalle
        '''
        employment_status = 6 # isyysvapaa
        time_in_state=0
        wage=old_wage
        out_of_work+=self.timestep        
        pension=pension*self.palkkakerroin+self.acc_family*wage
        netto=self.comp_benefits(0,old_wage,0,employment_status,0,age)

        return employment_status,pension,wage,time_in_state,netto,out_of_work

    def move_to_motherleave(self,pension,old_wage,age,out_of_work):
        '''
        Siirtymä äitiysvapaalle
        '''
        employment_status = 5 # äitiysvapaa
        time_in_state=0
        wage=old_wage
        out_of_work+=self.timestep        
        pension=pension*self.palkkakerroin+self.acc_family*wage
        netto=self.comp_benefits(0,old_wage,0,employment_status,0,age)

        return employment_status,pension,wage,time_in_state,netto,out_of_work

    def reset_info_state(self):
        self.infostate=(0,0,-1,-1,-1,-1)

    def decode_info_state(self):
        tyoura,lapsia,lapsen_ika,lapsia_paivakodissa,n_lapsilisa,n_tyotturva=self.infostate
        return tyoura,lapsia,lapsen_ika,lapsia_paivakodissa,n_lapsilisa,n_tyotturva

    def encode_info_state(self,tyoura,lapsia,lapsen_ika,lapsia_paivakodissa,n_lapsilisa,n_tyotturva):
        self.infostate=(tyoura,lapsia,lapsen_ika,lapsia_paivakodissa,n_lapsilisa,n_tyotturva)

    def update_info_state(self):
        tyoura,lapsia,lapsen_ika,lapsia_paivakodissa,n_lapsilisa,n_tyotturva=self.decode_info_state()

        lapsia_paivakodissa=0
        n_lapsilisa=0
        n_tyotturva=0
        if lapsia<1:
            return
        for l in range(lapsia):
            lapsen_ika[l]+=self.timestep
            if lapsen_ika[l]>=1.0 and lapsen_ika[l]<7:
                lapsia_paivakodissa += 1
            if lapsen_ika[l]<17:
                n_lapsilisa += 1
            if lapsen_ika[l]<18:
                n_tyotturva += 1

        self.encode_info_state(tyoura,lapsia,lapsen_ika,lapsia_paivakodissa,n_lapsilisa,n_tyotturva)

    def step(self, action, dynprog=False, debug=False):
        '''
        Open AI interfacen mukainen step-funktio, joka tekee askeleen eteenpäin
        toiminnon action mukaan 

        Keskeinen funktio simuloinnissa
        '''
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        employment_status,g,pension,old_wage,age,time_in_state,paid_pension,pinkslip,toe,tyoura,out_of_work\
            =self.state_decode(self.state)
            
        # simulointiin vaikuttavia ulkoisia tilamuuttujia, ei toteutettu
        #tyoura,lapsia,lapsen1_ika,lapsen2_ika,lapsen3_ika,lapsia_paivakodissa=self.decode_info_state()
        #tyoura,lapsia,lapsen1_ika,lapsen2_ika,lapsen3_ika,lapsia_paivakodissa=self.decode_info_state()

        intage=int(np.floor(age))

        if self.randomness:
            # kaikki satunnaisuus kerralla
            sattuma = np.random.uniform(size=5)
            # tk-alkavuus
            if sattuma[0]<self.disability_intensity[intage,g]: # age<self.min_retirementage and 
                action=11 # disability

            if age<50 and sattuma[2]<self.birth_intensity[intage,g]:
                if g>2: # naiset
                    action=5 # äitiysvapaa, kaikki äidit pitävät
                else: # miehet
                    # ikä valittu äidin iän mukaan. oikeastaan tämä ei mene ihan oikein miehille
                    if sattuma[4]<0.5:
                        action=6 # isyysvapaa, 50% isistä pitää
        
            # aiheuttaa epästabiilisuutta
            if sattuma[3]<self.mort_intensity[intage,g] and self.include_mort: # and False:
                employment_status,pension,wage,time_in_state,netto=self.move_to_deceiced(pension,old_wage,age)
        else:
            # tn ei ole koskaan alle rajan, jos tämä on 1
            sattuma = np.ones(5)

        if employment_status==13: # deceiced
            #time_in_state+=self.timestep
            if not self.include_mort:
                print('emp state 13')
            wage=old_wage
            nextwage=wage
            toe=0
            if self.mortstop:
                done=True
            else:
                done = age >= self.max_age
                done = bool(done)

            self.state = self.state_encode(employment_status,g,pension,wage,age+self.timestep,
                                            time_in_state,paid_pension,pinkslip,toe,tyoura,nextwage,out_of_work)
            reward=0
            return np.array(self.state), reward, done, {}
        elif age>=self.max_retirementage:
            employment_status,paid_pension,pension,wage,time_in_state,netto,out_of_work\
                =self.move_to_retirement(pension,0,age,paid_pension,employment_status,out_of_work,all_acc=True)
        elif employment_status == 0:
            time_in_state+=self.timestep
            out_of_work+=self.timestep
            if age>65:
                employment_status,paid_pension,pension,wage,time_in_state,netto,out_of_work\
                    =self.move_to_retirement(pension,0,age,paid_pension,employment_status,out_of_work,all_acc=True)
            elif action == 0: # or (action == 2 and age < self.min_retirementage):
                employment_status = 0 # unchanged
                wage=old_wage # self.get_wage(intage,time_in_state)
                toe=max(0.0,toe-self.timestep)
                if age<65:                
                    if time_in_state<self.ansiopvraha_kesto400: # 1.5 years
                        pension=pension*self.palkkakerroin+self.acc_unemp*old_wage
                    else:
                        pension=pension*self.palkkakerroin+self.acc*self.accbasis_tmtuki
                else:
                    pension=pension*self.palkkakerroin

                netto=self.comp_benefits(0,old_wage,0,employment_status,time_in_state,age,tyossaoloehto=toe,tyohistoria=tyoura)
            elif action == 1: # 
                employment_status,pension,wage,time_in_state,netto,toe,tyoura,out_of_work=\
                    self.move_to_work(pension,old_wage,age,time_in_state,toe,tyoura,out_of_work)
                pinkslip=0
            elif action==2:
                if age >= self.min_retirementage: # ve
                    employment_status,paid_pension,pension,wage,time_in_state,netto,out_of_work=\
                        self.move_to_retirement(pension,old_wage,age,paid_pension,employment_status,out_of_work)
                else:
                    employment_status,paid_pension,pension,wage,time_in_state,toe,netto,out_of_work=\
                        self.move_to_outsider(pension,old_wage,age,toe,0,out_of_work)
                    #employment_status,paid_pension,pension,wage,time_in_state,netto=\
                        #self.move_to_retirement(pension,old_wage,age,paid_pension,employment_status)
                pinkslip=0
            elif action == 3: # osatyö 50%
                employment_status,pension,wage,time_in_state,netto,toe,tyoura,out_of_work=\
                    self.move_to_parttime(pension,old_wage,age,toe,tyoura,time_in_state,out_of_work)
                pinkslip=0
            elif action==11: # tk
                employment_status,pension,paid_pension,wage,time_in_state,netto,out_of_work=\
                    self.move_to_disab(pension,old_wage,age,out_of_work)
                pinkslip=0
            elif action==5:
                employment_status,pension,wage,time_in_state,netto,out_of_work=\
                    self.move_to_motherleave(pension,old_wage,age,out_of_work)
                pinkslip=0
            elif action==6: 
                employment_status,pension,wage,time_in_state,netto,out_of_work=\
                    self.move_to_fatherleave(pension,old_wage,age,out_of_work)
                pinkslip=0
            else:
                print('error 17')
        elif employment_status == 4: # työttömyysputki
            time_in_state+=self.timestep
            out_of_work+=self.timestep
            if age>self.min_retirementage:
                employment_status,paid_pension,pension,wage,time_in_state,netto,out_of_work\
                    =self.move_to_retirement(pension,0,age,paid_pension,employment_status,out_of_work,all_acc=True)
            elif action == 0: # or (action == 2 and age < self.min_retirementage):
                employment_status  = 4 # unchanged
                wage=old_wage # self.get_wage(intage,time_in_state)
                toe=max(0,toe-self.timestep)
                if age<65:
                    pension=pension*self.palkkakerroin+self.acc_unemp*old_wage
                else:
                    pension=pension*self.palkkakerroin
    
                netto=self.comp_benefits(0,old_wage,0,employment_status,time_in_state,age,tyossaoloehto=toe,tyohistoria=tyoura)
                #time_in_state=time_in_state+self.timestep
            elif action == 1: # 
                employment_status,pension,wage,time_in_state,netto,toe,tyoura,out_of_work=\
                    self.move_to_work(pension,old_wage,age,time_in_state,toe,tyoura,out_of_work)
                pinkslip=0
            elif action==2:
                if age >= self.min_retirementage: # ve
                    employment_status,paid_pension,pension,wage,time_in_state,netto,out_of_work=\
                        self.move_to_retirement(pension,old_wage,age,paid_pension,employment_status,out_of_work)
                else:
                    employment_status,paid_pension,pension,wage,time_in_state,toe,netto,out_of_work=\
                        self.move_to_outsider(pension,old_wage,age,toe,0,out_of_work)
                    #employment_status,paid_pension,pension,wage,time_in_state,netto=\
                    #    self.move_to_retirement(pension,old_wage,age,paid_pension,employment_status)
                pinkslip=0
            elif action == 3: # 
                employment_status,pension,wage,time_in_state,netto,toe,tyoura,out_of_work=\
                    self.move_to_parttime(pension,old_wage,age,toe,tyoura,time_in_state,out_of_work)
                pinkslip=0
            elif action==11: # tk
                employment_status,pension,paid_pension,wage,time_in_state,netto,out_of_work=\
                    self.move_to_disab(pension,old_wage,age,out_of_work)
                pinkslip=0
            else:
                print('error 1: ',action)
        elif employment_status == 1:
            # irtisanominen
            time_in_state+=self.timestep
            out_of_work=0
            if sattuma[1]<self.pinkslip_intensity:
                if age<self.min_retirementage:
                    pinkslip=1
                    action=1 # unemp
                else:
                    pinkslip=0
                    action=2 # ve
            else:
                pinkslip=0

            if action == 0 or (action == 2 and age < self.min_retirementage):
                employment_status  = 1 # unchanged
    
                wage=self.get_wage(intage,0)
                toe=min(self.max_toe,toe+self.timestep)
                tyoura+=self.timestep
                out_of_work=0
                
                if age<self.max_retirementage:
                    pension=pension*self.palkkakerroin+self.acc*wage
                else:
                    pension=pension*self.palkkakerroin
                    
                netto=self.comp_benefits(wage,0,0,employment_status,time_in_state,age)
            elif action == 1: # työttömäksi
                employment_status,pension,wage,time_in_state,netto,toe,out_of_work=\
                    self.move_to_unemp(pension,old_wage,age,toe,pinkslip,out_of_work,tyoura)
            elif action==2:
                if age >= self.min_retirementage: # ve
                    employment_status,paid_pension,pension,wage,time_in_state,netto,out_of_work=\
                        self.move_to_retirement(pension,old_wage,age,paid_pension,employment_status,out_of_work)                        
                else: # työttömäksi
                    employment_status,paid_pension,pension,wage,time_in_state,toe,netto,out_of_work=\
                        self.move_to_outsider(pension,old_wage,age,toe,pinkslip,out_of_work)
                    #employment_status,paid_pension,pension,wage,time_in_state,netto=self.move_to_retirement(pension,old_wage,age,paid_pension,employment_status)
                    #employment_status,pension,wage,time_in_state,netto,toe=self.move_to_unemp(pension,old_wage,age,toe,pinkslip)
            elif action == 3: # osatyö 50%
                employment_status,pension,wage,time_in_state,netto,toe,tyoura,out_of_work=\
                    self.move_to_parttime(pension,old_wage,age,toe,tyoura,0,out_of_work)
            elif action==5: 
                employment_status,pension,wage,time_in_state,netto,out_of_work=\
                    self.move_to_motherleave(pension,old_wage,age,out_of_work)
            elif action==6: 
                employment_status,pension,wage,time_in_state,netto,out_of_work=\
                    self.move_to_fatherleave(pension,old_wage,age,out_of_work)
            elif action==11: # tk
                employment_status,pension,paid_pension,wage,time_in_state,netto,out_of_work=\
                    self.move_to_disab(pension,old_wage,age,out_of_work)
            else:
                print('error 12')
        elif employment_status == 3: # tk, ei voi siirtyä ve:lle
            time_in_state+=self.timestep
            out_of_work+=self.timestep
            if age >= self.min_retirementage:
                employment_status = 3 # ve # miten kansaneläke menee?? takuueläke?
            else:
                employment_status = 3 # unchanged

            toe=max(0,toe-self.timestep)
            paid_pension=paid_pension*self.elakeindeksi
            wage=old_wage
            netto=self.comp_benefits(0,0,paid_pension,employment_status,0,age)
        elif employment_status == 2: # eläkkeellä, voi palata töihin
            if age >= self.min_retirementage: # ve
                time_in_state+=self.timestep
                out_of_work+=self.timestep

                if age>=self.max_retirementage:
                    paid_pension += self.elinaikakerroin*pension
                    pension=0

                if action == 0 or action == 3 or ((action == 1 or action == 2) and age>=self.max_retirementage):
                    employment_status = 2 # unchanged
                    #old_wage=0

                    paid_pension=paid_pension*self.elakeindeksi
                    pension=pension*self.palkkakerroin
    
                    wage=self.get_wage(intage,time_in_state)
                    netto=self.comp_benefits(0,0,paid_pension,employment_status,0,age)
                elif action == 1 and age<self.max_retirementage:
                    employment_status,pension,wage,time_in_state,netto,out_of_work=\
                        self.move_to_retwork(pension,old_wage,age,time_in_state,paid_pension,out_of_work)
                elif action == 2 and age<self.max_retirementage:
                    employment_status,pension,wage,time_in_state,netto,out_of_work=\
                        self.move_to_retpartwork(pension,old_wage,age,time_in_state,paid_pension,out_of_work)
                elif action == 11:
                    employment_status,paid_pension,pension,wage,time_in_state,netto,out_of_work=\
                        self.move_to_retdisab(pension,old_wage,age,time_in_state,paid_pension,out_of_work)
                else:
                    print('error 221, action {} age {}'.format(action,age))
            else:
                # työvoiman ulkopuolella
                time_in_state+=self.timestep
                out_of_work+=self.timestep
                if action == 0:
                    employment_status = 2 # unchanged
                    wage=old_wage
                    toe=max(0,toe-self.timestep)
                    pension=pension*self.palkkakerroin
                    netto=1 #self.comp_benefits(0,0,0,employment_status,time_in_state,age)
                elif action == 1: # työttömäksi
                    employment_status,pension,wage,time_in_state,netto,toe,out_of_work=self.move_to_unemp(pension,old_wage,age,toe,0,out_of_work,tyoura)
                elif action == 2: # töihin
                    employment_status,pension,wage,time_in_state,netto,toe,tyoura,out_of_work=self.move_to_work(pension,old_wage,age,time_in_state,toe,tyoura,out_of_work)
                elif action == 3: # osatyö 50%
                    employment_status,pension,wage,time_in_state,netto,toe,tyoura,out_of_work=self.move_to_parttime(pension,old_wage,age,toe,tyoura,time_in_state,out_of_work)
                elif action == 5: 
                    employment_status,pension,wage,time_in_state,netto,out_of_work=self.move_to_motherleave(pension,old_wage,age,out_of_work)
                elif action == 6: 
                    employment_status,pension,wage,time_in_state,netto,out_of_work=self.move_to_fatherleave(pension,old_wage,age,out_of_work)
                elif action == 11: # tk
                    employment_status,pension,paid_pension,wage,time_in_state,netto,out_of_work=self.move_to_disab(pension,old_wage,age,out_of_work)
                else:
                    print('error 12')
        elif employment_status == 5: # äitiysvapaa
            time_in_state+=self.timestep
            out_of_work+=self.timestep
            if time_in_state>self.aitiysvapaa_kesto:
                pinkslip=0
                if action == 0:
                    employment_status,pension,wage,time_in_state,netto,toe,out_of_work=self.move_to_unemp(pension,old_wage,age,toe,pinkslip,out_of_work,tyoura)
                elif action == 1: # 
                    employment_status,pension,wage,time_in_state,netto,toe,tyoura,out_of_work=self.move_to_work(pension,old_wage,age,time_in_state,toe,tyoura,out_of_work)
                elif action == 2: # 
                    employment_status,pension,wage,time_in_state,netto,out_of_work=self.move_to_kht(pension,old_wage,age,out_of_work)
                elif action == 3: # osa-aikatyöhön
                    employment_status,pension,wage,time_in_state,netto,toe,tyoura,out_of_work=self.move_to_parttime(pension,old_wage,age,toe,tyoura,time_in_state,out_of_work)
                elif action==5: # uudelleen äitiysvapaalle
                    employment_status,pension,wage,time_in_state,netto,out_of_work=self.move_to_motherleave(pension,old_wage,age,out_of_work)
                elif action==6:  # isyysvapaalle
                    employment_status,pension,wage,time_in_state,netto,out_of_work=self.move_to_fatherleave(pension,old_wage,age,out_of_work)
                elif action==11: # tk
                    employment_status,pension,paid_pension,wage,time_in_state,netto,out_of_work=self.move_to_disab(pension,old_wage,age,out_of_work)
                else:
                    print('Error 21')
            else:
                pension=pension*self.palkkakerroin+self.acc_family*old_wage
                wage=old_wage #self.get_wage(intage,time_in_state)
                netto=self.comp_benefits(0,old_wage,0,employment_status,0,age)
        elif employment_status == 6: # isyysvapaa
            time_in_state+=self.timestep
            out_of_work+=self.timestep
            if time_in_state>=self.isyysvapaa_kesto:
                pinkslip=0
                if action == 0 or action==2:
                    employment_status,pension,wage,time_in_state,netto,toe,out_of_work=self.move_to_unemp(pension,old_wage,age,toe,pinkslip,out_of_work,tyoura)
                elif action == 1: # 
                    # ei vaikutusta palkkaan
                    employment_status,pension,wage,time_in_state,netto,toe,tyoura,out_of_work=self.move_to_work(pension,old_wage,age,0,toe,tyoura,out_of_work)
                elif action == 2: # 
                    employment_status,pension,wage,time_in_state,netto,out_of_work=self.move_to_kht(pension,old_wage,age,out_of_work)
                elif action == 3: # osa-aikatöihin
                    employment_status,pension,wage,time_in_state,netto,toe,tyoura,out_of_work=self.move_to_parttime(pension,old_wage,age,toe,tyoura,0,out_of_work)
                elif action==5: 
                    employment_status,pension,wage,time_in_state,netto,out_of_work=self.move_to_motherleave(pension,old_wage,age,out_of_work)
                elif action==6: 
                    employment_status,pension,wage,time_in_state,netto,out_of_work=self.move_to_fatherleave(pension,old_wage,age,out_of_work)
                elif action==11: # tk
                    employment_status,pension,paid_pension,wage,time_in_state,netto,out_of_work=self.move_to_disab(pension,old_wage,age,out_of_work)
                else:
                    print('Error 23')
            else:
                pension=pension*self.palkkakerroin+self.acc_family*old_wage
                wage=old_wage #self.get_wage(intage,time_in_state)
                netto=self.comp_benefits(0,old_wage,0,employment_status,0,age)
        elif employment_status == 7: # kotihoidontuki
            time_in_state+=self.timestep
            out_of_work+=self.timestep
            
            #self.render()

            if action == 0 and time_in_state<=self.kht_kesto:
                employment_status  = 7 # stay
                wage=old_wage #self.get_wage(intage,time_in_state)
                toe=max(0,toe-self.timestep)
                pension=pension*self.palkkakerroin+self.acc*self.accbasis_kht
                netto=self.comp_benefits(0,old_wage,0,employment_status,time_in_state,age)
            elif action == 2: # 
                pinkslip=0
                employment_status,pension,wage,time_in_state,netto,toe,out_of_work=self.move_to_unemp(pension,old_wage,age,toe,pinkslip,out_of_work,tyoura)
            elif action == 1: # 
                employment_status,pension,wage,time_in_state,netto,toe,tyoura,out_of_work=self.move_to_work(pension,old_wage,age,time_in_state,toe,tyoura,out_of_work)
            elif action == 3: # 
                employment_status,pension,wage,time_in_state,netto,toe,tyoura,out_of_work=self.move_to_parttime(pension,old_wage,age,toe,tyoura,time_in_state,out_of_work)
            elif action==5: 
                employment_status,pension,wage,time_in_state,netto,out_of_work=self.move_to_motherleave(pension,old_wage,age,out_of_work)
            elif action==6: 
                employment_status,pension,wage,time_in_state,netto,out_of_work=self.move_to_fatherleave(pension,old_wage,age,out_of_work)
            elif action==11: # tk
                employment_status,pension,paid_pension,wage,time_in_state,netto,out_of_work=self.move_to_disab(pension,old_wage,age,out_of_work)
            elif time_in_state>self.kht_kesto: # 
                pinkslip=0
                employment_status,pension,wage,time_in_state,netto,toe,out_of_work=self.move_to_unemp(pension,old_wage,age,toe,pinkslip,out_of_work,tyoura)
            else:
                print('Error 25')
        elif employment_status == 8: # töissä ja ve:llä
            time_in_state+=self.timestep
            out_of_work=0
        
            # irtisanominen
            if sattuma[1]<self.pinkslip_intensity:
                action=2 # ve:lle

            if age>=self.max_retirementage:
                paid_pension += self.elinaikakerroin*pension
                pension=0

            if action == 0 or action == 3: # jatkaa töissä, ei voi saada työttömyyspäivärahaa
                employment_status = 8 # unchanged
                wage=self.get_wage(intage,0)
                if age<self.max_retirementage:
                    pension=pension*self.palkkakerroin+self.acc*wage
                
                paid_pension=paid_pension*self.elakeindeksi
                netto=self.comp_benefits(wage,0,paid_pension,employment_status,time_in_state,age)
            elif action == 1: # jatkaa osa-aikatöissä, ei voi saada työttömyyspäivärahaa
                employment_status,pension,wage,time_in_state,netto,out_of_work=\
                    self.move_to_retpartwork(pension,old_wage,age,0,paid_pension,out_of_work)
            elif action==2: # eläkkeelle, eläkeaikana karttunutta eläkettä ei vielä maksuun
                employment_status,paid_pension,pension,wage,time_in_state,netto,out_of_work=\
                    self.move_to_retirement(pension,old_wage,age,paid_pension,employment_status,out_of_work,all_acc=False)
            elif action == 11:
                # no more working, move to "disab" with no change in paid_pension
                employment_status,paid_pension,pension,wage,time_in_state,netto,out_of_work=\
                    self.move_to_retdisab(pension,old_wage,age,time_in_state,paid_pension,out_of_work)
            else:
                print('error 14, action {} age {}'.format(action,age))
        elif employment_status == 9: # osatöissä ja ve:llä
            time_in_state+=self.timestep
            out_of_work=0
            # irtisanominen
            if sattuma[1]<self.pinkslip_intensity:
                if self.plotdebug:
                    print('pinkslip')
                action=2 # ve:lle

            if age>=self.max_retirementage:
                paid_pension += self.elinaikakerroin*pension
                pension=0

            if action == 0 or action == 3: # jatkaa osa-aikatöissä, ei voi saada työttömyyspäivärahaa
                employment_status = 9 # unchanged
                wage=self.get_wage(intage,0)
                parttimewage=0.5*wage
                if age<self.max_retirementage:
                    pension=pension*self.palkkakerroin+self.acc*parttimewage

                paid_pension=paid_pension*self.elakeindeksi
                netto=self.comp_benefits(parttimewage,0,paid_pension,employment_status,time_in_state,age)
            elif action==1: # jatkaa täysin töissä, ei voi saada työttömyyspäivärahaa
                employment_status,pension,wage,time_in_state,netto,out_of_work=\
                    self.move_to_retwork(pension,old_wage,age,0,paid_pension,out_of_work)
            elif action==2: # eläkkeelle, eläkeaikana karttunutta eläkettä ei vielä maksuun
                employment_status,paid_pension,pension,wage,time_in_state,netto,out_of_work=\
                    self.move_to_retirement(pension,old_wage,age,paid_pension,employment_status,out_of_work,all_acc=False)
            elif action == 11:
                # no more working, move to "disab" with no change in paid_pension
                employment_status,paid_pension,pension,wage,time_in_state,netto,out_of_work=\
                    self.move_to_retdisab(pension,old_wage,age,time_in_state,paid_pension,out_of_work)
            else:
                print('error 14, action {} age {}'.format(action,age))
        elif employment_status == 10: # osatöissä, ei ve:llä
            time_in_state+=self.timestep
            out_of_work=0
            # irtisanominen
            if sattuma[1]<self.pinkslip_intensity:
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
    
                wage=self.get_wage(intage,0)
                parttimewage=0.5*wage
                toe=min(28/12,toe+self.timestep)
                if age<self.max_retirementage:
                    pension=pension*self.palkkakerroin+self.acc*parttimewage
                else:
                    pension=pension*self.palkkakerroin
                netto=self.comp_benefits(parttimewage,0,0,employment_status,time_in_state,age)
            elif action == 1: # työttömäksi
                employment_status,pension,wage,time_in_state,netto,toe,out_of_work=\
                    self.move_to_unemp(pension,old_wage,age,toe,pinkslip,out_of_work,tyoura)
            elif action==2:
                if age >= self.min_retirementage: # ve
                    employment_status,paid_pension,pension,wage,time_in_state,netto,out_of_work=\
                        self.move_to_retirement(pension,old_wage,age,paid_pension,employment_status,out_of_work)
                else:
                    #employment_status,paid_pension,pension,wage,time_in_state,netto=self.move_to_retirement(pension,old_wage,age,paid_pension,employment_status)
                    employment_status,paid_pension,pension,wage,time_in_state,toe,netto=\
                        self.move_to_outsider(pension,old_wage,age,toe,pinkslip,out_of_work)
            elif action==3:
                employment_status,pension,wage,time_in_state,netto,toe,tyoura,out_of_work=\
                    self.move_to_work(pension,old_wage,age,0,toe,tyoura,out_of_work)
            elif action==5: 
                employment_status,pension,wage,time_in_state,netto,out_of_work=\
                    self.move_to_motherleave(pension,old_wage,age,out_of_work)
            elif action==6: 
                employment_status,pension,wage,time_in_state,netto,out_of_work=\
                    self.move_to_fatherleave(pension,old_wage,age,out_of_work)
            elif action==11: # tk
                employment_status,pension,paid_pension,wage,time_in_state,netto,out_of_work=\
                    self.move_to_disab(pension,old_wage,age,out_of_work)
            else:
                print('error 12')
        elif employment_status == 11: # työvoiman ulkopuolella, ei töissä, ei hae töitä
            out_of_work+=self.timestep
            time_in_state+=self.timestep
            if age>=self.min_retirementage:
                employment_status,paid_pension,pension,wage,time_in_state,netto,out_of_work=\
                    self.move_to_retirement(pension,old_wage,age,paid_pension,employment_status,out_of_work)
            elif action == 0:
                employment_status = 11 # unchanged
                wage=old_wage
                toe=max(0,toe-self.timestep)
                pension=pension*self.palkkakerroin
                netto=self.comp_benefits(0,old_wage,0,employment_status,time_in_state,age,tyossaoloehto=toe,tyohistoria=tyoura)
            elif action == 1: # 
                employment_status,pension,wage,time_in_state,netto,toe,tyoura,out_of_work=self.move_to_work(pension,old_wage,age,time_in_state,toe,tyoura,out_of_work)
                pinkslip=0
            elif action == 2: # 
                employment_status,pension,wage,time_in_state,netto,toe,out_of_work=self.move_to_unemp(pension,old_wage,age,toe,pinkslip,out_of_work,tyoura)
                pinkslip=0
            elif action==5:
                employment_status,pension,wage,time_in_state,netto,out_of_work=self.move_to_motherleave(pension,old_wage,age,out_of_work)
                pinkslip=0
            elif action==6: 
                employment_status,pension,wage,time_in_state,netto,out_of_work=self.move_to_fatherleave(pension,old_wage,age,out_of_work)
                pinkslip=0
            elif action == 3: # 
                employment_status,pension,wage,time_in_state,netto,toe,tyoura,out_of_work=self.move_to_parttime(pension,old_wage,age,toe,tyoura,time_in_state,out_of_work)
                pinkslip=0
            elif action==11: # tk
                employment_status,pension,paid_pension,wage,time_in_state,netto,out_of_work=self.move_to_disab(pension,old_wage,age,out_of_work)
                pinkslip=0
            else:
                print('error 19: ',action)
        elif employment_status == 12: # opiskelija
            out_of_work=0 #self.timestep
            pinkslip=0
            tyoura=0
            toe=0
            #if time_in_state>7:
            #    action=2

            if action == 0 or (action==2 and age<25):
                employment_status = 12 # unchanged
                time_in_state+=self.timestep
                wage=old_wage
                toe=max(0,toe-self.timestep)
                pension=pension*self.palkkakerroin
                netto=self.comp_benefits(0,0,0,employment_status,time_in_state,age,tyossaoloehto=toe,tyohistoria=tyoura)
            elif action == 1: # 
                employment_status,pension,wage,time_in_state,netto,toe,tyoura,out_of_work=self.move_to_work(pension,old_wage,age,0,toe,tyoura,out_of_work)
                pinkslip=0
            elif action == 2: # 
                employment_status,pension,wage,time_in_state,netto,toe,out_of_work=self.move_to_unemp(pension,old_wage,age,toe,pinkslip,out_of_work,tyoura)
                pinkslip=0
            elif action == 3: # 
                employment_status,pension,wage,time_in_state,netto,toe,tyoura,out_of_work=self.move_to_parttime(pension,old_wage,age,toe,tyoura,0,out_of_work)
                pinkslip=0
            elif action==5:
                employment_status,pension,wage,time_in_state,netto,out_of_work=self.move_to_motherleave(pension,old_wage,age,out_of_work)
                pinkslip=0
            elif action==6: 
                employment_status,pension,wage,time_in_state,netto,out_of_work=self.move_to_fatherleave(pension,old_wage,age,out_of_work)
                pinkslip=0
            elif action==11: # tk
                employment_status,pension,paid_pension,wage,time_in_state,netto,out_of_work=self.move_to_disab(pension,old_wage,age,out_of_work)
                pinkslip=0
            else:
                print('error 19: ',action)
        else:
            print('Unknown employment_status {s} of type {t}'.format(s=employment_status,t=type(employment_status)))

        done = age >= self.max_age
        done = bool(done)

        if not done:
            reward = self.log_utility(netto,int(employment_status),age,g=g)
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            
            paid_pension += self.elinaikakerroin*pension
            pension=0
            
            netto=self.comp_benefits(0,old_wage,paid_pension,employment_status,time_in_state,age)
            if employment_status in set([2,8,9]):
                reward = self.npv[g]*self.log_utility(netto,employment_status,age)
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

        next_wage=self.get_wage(int(np.floor(age+self.timestep)),0)

        self.state = self.state_encode(employment_status,g,pension,wage,age+self.timestep,time_in_state,
                                        paid_pension,pinkslip,toe,tyoura,next_wage,out_of_work)

        if self.plotdebug:
            self.render(done=done,reward=reward, netto=netto)

        return np.array(self.state), reward, done, {}

    # NO RANDOMNESS
    def log_utility_norandomness(self,income,employment_state,age,g=0):
        '''
        Log-utiliteettifunktio hieman muokattuna lähteestä Määttänen, 2013 & Hakola & Määttänen, 2005

        Käytetään, jos laskelmissa ei satunnaisuutta
        '''
        # kappa tells how much person values free-time
        # kappa_kokoaika=0.58 # vuositasolla sopiva
        # kappa_osaaika=0.29 # vuositasolla sopiva

        kappa_kokoaika=0.70
        kappa_osaaika=0.66*kappa_kokoaika
        kappa_ve=0.20
        kappa_opiskelija=2.00
        mu=0.20 # how much penalty increase (with age) is associated with work
        min_student_age=20
        max_student_age=27

        if age>=58:
            kappa_kokoaika *= (1+mu*max(0,age-58))
            kappa_osaaika *= (1+mu*max(0,age-58))

        if employment_state == 1 or employment_state == 8:
            u=np.log(income)-kappa_kokoaika
        elif employment_state == 10 or employment_state == 9:
            u=np.log(income)-kappa_osaaika
        elif employment_state == 2 and age>=self.min_retirementage:
            u=np.log(income)+kappa_ve
        elif employment_state == 11:
            u=np.log(income)+kappa_ve
        elif employment_state == 12:
            if age<max_student_age:
                kappa=max(0,(max_student_age-age)/(max_student_age-min_student_age)*kappa_opiskelija)
                u=np.log(income)+kappa
                print(u)
            else:
                u=np.log(income)+kappa_ve
        else:
            u=np.log(income)

        if u is np.inf:
            print('inf: state ',employment_state)

        if income<1:
            print('inf: state ',employment_state)

        return u/10 # skaalataan

    # WITH RANDOMNESS
    def log_utility_randomness(self,income,employment_state,age,g=0):
        '''
        Log-utiliteettifunktio hieman muokattuna lähteestä Määttänen, 2013 & Hakola & Määttänen, 2005

        Käytetään, jos laskelmissa on mukana satunnaisuutta
        '''

        # kappa tells how much person values free-time
        kappa_kokoaika=0.78
        kappa_osaaika=2/3*kappa_kokoaika
        kappa_ve=0.55
        
        if g==0 or g==3: # pienituloinen ryhmä
            kappa_opiskelija=2.0
        elif g==1 or g==4: # keskituloinen ryhmä
            kappa_opiskelija=2.25
        elif g==2 or g==5:  # suurituloisin ryhmä
            kappa_opiskelija=2.5
        else: # muissa ryhmissä ei opiskelupreferenssiä
            kappa_opiskelija=0
            
        mu=0.10 # how much penalty is associated with work increase with age
        min_student_age=20
        max_student_age=25

        if age>58:
            kappa_kokoaika *= (1+mu*max(0,age-58))
            kappa_osaaika *= (1+mu*max(0,age-58))

        if employment_state == 1 or employment_state == 8:
            u=np.log(income)-kappa_kokoaika
        elif employment_state == 10 or employment_state == 9:
            u=np.log(income)-kappa_osaaika
        elif employment_state == 2:
            u=np.log(income)+kappa_ve
        elif employment_state == 11:
            u=np.log(income)
        elif employment_state == 12:
            if age<max_student_age:
                kappa=max(0,(max_student_age-max(min_student_age,age))/(max_student_age-min_student_age)*kappa_opiskelija)
                u=np.log(income)+kappa
            else:
                u=np.log(income)
        else:
            u=np.log(income)

        if u is np.inf:
            print('inf: state ',employment_state)

        if income<1:
            print('inf: state ',employment_state)

        return u/10 # skaalataan
   
    def log_utility_perustulo(self,income,employment_state,age,g=0):
        '''
        Log-utiliteettifunktio hieman muokattuna lähteestä Määttänen, 2013 & Hakola & Määttänen, 2005

        Käytetään, jos perustulolaskelmissa
        '''
        # kappa tells how much person values free-time
        # kappa_kokoaika=0.58 # vuositasolla sopiva
        # kappa_osaaika=0.29 # vuositasolla sopiva

        kappa_kokoaika=0.60
        kappa_osaaika=0.40
        kappa_ve=0.18
        kappa_opiskelija=1.00
        mu=0.20 # how much penalty is associated with work increase with age

        if age>=58:
            kappa_kokoaika *= (1+mu*max(0,age-58))
            kappa_osaaika *= (1+mu*max(0,age-58))

        if employment_state == 1 or employment_state == 8:
            u=np.log(income)-kappa_kokoaika
        elif employment_state == 10 or employment_state == 9:
            u=np.log(income)-kappa_osaaika
        elif employment_state == 2:
            u=np.log(income)+kappa_ve
        elif employment_state == 11:
            u=np.log(income)+kappa_ve
        elif employment_state == 12:
            if age<25:
                kappa=max(0,(25-age)/5*kappa_opiskelija)
                u=np.log(income)+kappa
            else:
                u=np.log(income)+kappa_ve
        else:
            u=np.log(income)

        if u is np.inf:
            print('inf: state ',employment_state)

        if income<1:
            print('inf: state ',employment_state)

        return u/10 # skaalataan
   
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

        self.salary[self.min_age]=np.maximum(8000,np.random.normal(loc=a0,scale=12*1000,size=1)[0]) # e/y

        if debug:
            self.salary[self.min_age+1:self.max_age+1]=self.salary[self.min_age]
        else:
            for age in range(self.min_age+1,self.max_age+1):
                self.salary[age]=self.wage_process(self.salary[age-1],age,ave=a0)


    # From Määttänen, 2013
    def wage_process_TK(self,w,age,a0=3300*12,a1=3300*12,g=1):
        '''
        Palkkaprosessi lähteestä Määttänen, 2013 
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

        # täysiaikainen vuositulo vähintään 8000e
        wt=np.maximum(8000,wt)

        return wt

    def compute_salary_TK(self,group=1,debug=False):
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
            self.salary[self.min_age]=np.maximum(8000,np.random.normal(loc=a0,scale=12*1000,size=1)[0]) # e/y
            self.salary[self.min_age+1:self.max_age+1]=self.salary[self.min_age]
        else: # randomness and time-development included
            if group>2: # naiset
                r=g_r[group-3]
                a0=palkat_ika_naiset[0]*r
                a1=palkat_ika_naiset[0]*r/5
                self.salary[self.min_age]=np.maximum(8000,np.random.normal(loc=a0,scale=a1,size=1)[0]) # e/y

                for age in range(self.min_age+1,self.max_age+1):
                    a0=palkat_ika_naiset[age-1-self.min_age]*r
                    a1=palkat_ika_naiset[age-self.min_age]*r
                    self.salary[age]=self.wage_process_TK(self.salary[age-1],age,a0,a1)
            else: # miehet
                r=g_r[group]
                a0=palkat_ika_miehet[0]*r
                a1=palkat_ika_miehet[0]*r/5
                self.salary[self.min_age]=np.maximum(8000,np.random.normal(loc=a0,scale=a1,size=1)[0]) # e/y

                for age in range(self.min_age+1,self.max_age+1):
                    a0=palkat_ika_miehet[age-1-self.min_age]*r
                    a1=palkat_ika_miehet[age-self.min_age]*r
                    self.salary[age]=self.wage_process_TK(self.salary[age-1],age,a0,a1)


    def state_encode_mort(self,emp,g,pension,old_wage,age,time_in_state,paid_pension,pink,
                          toe,tyohist,next_wage,out_of_work):
        '''
        Tilan koodaus neuroverkkoa varten. Arvot skaalataan ja tilat one-hot-enkoodataan

        Käytetään, jos kuolleisuus mukana
        '''
        #if self.include300:
        d=np.zeros(self.n_empl+self.n_groups+11)
        #else:
        #    d=np.zeros(self.n_empl+self.n_groups+7)

        states=self.n_empl
        if emp==1:
            d[0:states]=np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0])
        elif emp==0:
            d[0:states]=np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0])
        elif emp==2:
            d[0:states]=np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0])
        elif emp==3:
            d[0:states]=np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0])
        elif emp==4:
            d[0:states]=np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0])
        elif emp==5:
            d[0:states]=np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0])
        elif emp==6:
            d[0:states]=np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0])
        elif emp==7:
            d[0:states]=np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0])
        elif emp==8:
            d[0:states]=np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0])
        elif emp==9:
            d[0:states]=np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0])
        elif emp==10:
            d[0:states]=np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0])
        elif emp==11:
            d[0:states]=np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0])
        elif emp==12:
            d[0:states]=np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0])
        elif emp==13:
            d[0:states]=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1])
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
        else:
            d[states2]=(pension-20_000)/10_000 # vastainen eläke
            d[states2+1]=(old_wage-40_000)/15_000
            d[states2+4]=(paid_pension-20_000)/10_000 # alkanut eläke
        if tyohist>self.tyohistoria_vaatimus:
            hist400=1
        else:
            hist400=0

        d[states2+2]=(age-(self.max_age+self.min_age)/2)/20
        d[states2+3]=(time_in_state-3)/10
        #if self.include300:
        #d[states2+5]=pink # irtisanottu vai ei 
        d[states2+5]=toe-14/12 # työssäoloehto
        d[states2+6]=(tyohist-3)/10 # tyohistoria: 300/400 pv
        d[states2+7]=hist400
        if age>=self.min_retirementage:
            retaged=1
        else:
            retaged=0
        d[states2+8]=retaged
        d[states2+9]=(next_wage-40_000)/15_000
        d[states2+10]=(out_of_work-3)/10        
        #d[states2+10]=lapsia # lapsien lkm
        #d[states2+11]=lapsia_paivakodissa # nuorimman lapsen ika

        return d


    def state_encode_nomort(self,emp,g,pension,old_wage,age,time_in_state,paid_pension,pink,
                            toe,tyohist,next_wage,out_of_work):
        '''
        Tilan koodaus neuroverkkoa varten. Arvot skaalataan ja tilat one-hot-enkoodataan

        Käytetään, jos kuolleisuus ei mukana
        '''
        #if self.include300:
        d=np.zeros(self.n_empl+self.n_groups+11)
        #else:
        #    d=np.zeros(self.n_empl+self.n_groups+8)
        states=self.n_empl
        if emp==1:
            d[0:states]=np.array([0,1,0,0,0,0,0,0,0,0,0,0,0])
        elif emp==0:
            d[0:states]=np.array([1,0,0,0,0,0,0,0,0,0,0,0,0])
        elif emp==2:
            d[0:states]=np.array([0,0,1,0,0,0,0,0,0,0,0,0,0])
        elif emp==3:
            d[0:states]=np.array([0,0,0,1,0,0,0,0,0,0,0,0,0])
        elif emp==4:
            d[0:states]=np.array([0,0,0,0,1,0,0,0,0,0,0,0,0])
        elif emp==5:
            d[0:states]=np.array([0,0,0,0,0,1,0,0,0,0,0,0,0])
        elif emp==6:
            d[0:states]=np.array([0,0,0,0,0,0,1,0,0,0,0,0,0])
        elif emp==7:
            d[0:states]=np.array([0,0,0,0,0,0,0,1,0,0,0,0,0])
        elif emp==8:
            d[0:states]=np.array([0,0,0,0,0,0,0,0,1,0,0,0,0])
        elif emp==9:
            d[0:states]=np.array([0,0,0,0,0,0,0,0,0,1,0,0,0])
        elif emp==10:
            d[0:states]=np.array([0,0,0,0,0,0,0,0,0,0,1,0,0])
        elif emp==11:
            d[0:states]=np.array([0,0,0,0,0,0,0,0,0,0,0,1,0])
        elif emp==12:
            d[0:states]=np.array([0,0,0,0,0,0,0,0,0,0,0,0,1])
        elif emp==13:
            print('state 13 in state_encode_nomort!')
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
        else:
            d[states2]=(pension-20_000)/10_000 # vastainen eläke
            d[states2+1]=(old_wage-40_000)/15_000
            d[states2+4]=(paid_pension-20_000)/10_000 # alkanut eläke

        d[states2+2]=(age-(self.max_age+self.min_age)/2)/20
        d[states2+3]=(time_in_state-3)/10
        if age>=self.min_retirementage:
            retaged=1
        else:
            retaged=0

        #if self.include300:
        #d[states2+5]=pink # irtisanottu vai ei 
        d[states2+5]=toe-14/12 # työssäoloehto
        d[states2+6]=(tyohist-3)/10 # tyohistoria: 300/400 pv
        if tyohist>self.tyohistoria_vaatimus:
            hist400=1
        else:
            hist400=0

        d[states2+7]=hist400
        d[states2+8]=retaged
        d[states2+9]=(next_wage-40_000)/15_000
        d[states2+10]=(out_of_work-3)/10        
        #else:
        #    d[states2+5]=toe-14/12 # työssäoloehto
        #    d[states2+6]=retaged
        #    d[states2+7]=(next_wage-40_000)/15_000

        return d

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

        if self.log_transform:
            pension=(np.exp(vec[pos])-self.eps)*20_000
            wage=(np.exp(vec[pos+1])-self.eps)*40_000
            paid_pension=(np.exp(vec[pos+4])-self.eps)*20_000
        else:
            pension=vec[pos]*10_000+20_000
            wage=vec[pos+1]*15_000+40_000 
            paid_pension=vec[pos+4]*10_000+20_000

        age=vec[pos+2]*20+(self.max_age+self.min_age)/2
        time_in_state=vec[pos+3]*10+3
        #if self.include300:
        pink=0 #vec[pos+5] # irtisanottu vai ei 
        toe=vec[pos+5]+14/12 # työssäoloehto, kesto
        tyohist=vec[pos+6]*10+3 # työhistoria
        out_of_work=vec[pos+10]*10+3 # kesto poissa työelämästä
        #else:
        #    toe=vec[pos+5]+14/12 # työssäoloehto, kesto
        #    pink=0
        #    tyohist=10 # fake
        #ave5y=vec[pos+6] # 5v palkkojen ka

        return int(emp),int(g),pension,wage,age,time_in_state,paid_pension,int(pink),toe,tyohist,out_of_work

    def reset(self,init=None):
        '''
        Open AI-interfacen mukainen reset-funktio, joka nollaa laskennan alkutilaan
        '''

        #initial=(0,0,0,0,self.min_age)

        #if init is None:
        #    employment_status, prev_unempl, pension, wage, age = self.np_random.uniform(low=self.low, high=self.high, size=(5,))
        #else:
        #    employment_status, prev_unempl, pension, wage, age = init

        #employment_status, prev_unempl, pension, wage, age = initial

        employment_status=12 # opiskelija
        age=int(self.min_age)
        pension=0
        time_in_state=0
        pink=0
        toe=0
        tyohist=0

        # set up salary for the entire career
        #group=np.random.randint(self.n_groups)
        g=random.choices(np.array([0,1,2],dtype=int),weights=[0.3,0.5,0.2])[0]
        gender=random.choices(np.array([0,1],dtype=int),weights=[0.5,0.5])[0]
        group=int(g+gender*3)
        self.compute_salary_TK(group=group)
        old_wage=self.salary[self.min_age]
        next_wage=old_wage
        out_of_w=0
        
        if gender==0:
            employment_status=random.choices(np.array([0,1,3,11,12],dtype=int),weights=[0.133,0.374,0.012,0.151,0.240])[0]
        else:
            employment_status=random.choices(np.array([0,1,3,11,12],dtype=int),weights=[0.073,0.550,0.010,0.034,0.283])[0]

        # tarvitseeko alkutilassa laskea muita tietoja uusiksi? ei kait

        if self.plotdebug:
            print('emp {} gender {} g {} old_wage {} next_wage {}'.format(employment_status,gender,g,old_wage,next_wage))

        self.state = self.state_encode(employment_status,group,pension,old_wage,self.min_age,
                                        time_in_state,0,pink,toe,tyohist,next_wage,out_of_w)
        self.steps_beyond_done = None

        return np.array(self.state)

    def render(self, mode='human', close=False, done=False, reward=None, netto=None):
        '''
        Tulostus-rutiini
        '''
        emp,g,pension,wage,age,time_in_state,paid_pension,pink,toe,tyohist,out_of_work=self.state_decode(self.state)
        if reward is None:
            print('Tila {} ryhmä {} palkka {:.2f} ikä {} t-i-s {} tul.eläke {:.2f} alk.eläke {:.2f} irtisanottu {} toe {:.2f} työhist {:.2f} o-o-w {:.2f}'.format(\
                emp,g,wage,age,time_in_state,pension,paid_pension,pink,toe,tyohist,out_of_work))
        elif netto is None:
            print('Tila {} ryhmä {} palkka {:.2f} ikä {} t-i-s {} tul.eläke {:.2f} alk.eläke {:.2f} irtisanottu {} toe {:.2f} työhist {:.2f} o-o-w {:.2f} r {:.4f}'.format(\
                emp,g,wage,age,time_in_state,pension,paid_pension,pink,toe,tyohist,out_of_work,reward))
        else:
            print('Tila {} ryhmä {} palkka {:.2f} ikä {} t-i-s {} tul.eläke {:.2f} alk.eläke {:.2f} irtisanottu {} toe {:.2f} työhist {:.2f} o-o-w {:.2f} r {:.4f} n {:.2f}'.format(\
                emp,g,wage,age,time_in_state,pension,paid_pension,pink,toe,tyohist,out_of_work,reward,netto))
        if done:
            print('-------------------------------------------------------------------------------------------------------')

    def close(self):
        '''
        Ei käytössä
        '''
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def set_state_limits(self,debug=False):
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
            pension_max=(200_000-20_000)/10_000 # vastainen eläke
            pension_min=(0-20_000)/10_000 # vastainen eläke
            wage_max=(500_000-40_000)/15_000
            wage_min=(0-40_000)/15_000
            paid_pension_min=(0-20_000)/10_000 # alkanut eläke
            paid_pension_max=(200_000-20_000)/10_000 # alkanut eläke

        age_max=(self.max_age-(self.max_age+self.min_age)/2)/20
        age_min=(self.min_age-(self.max_age+self.min_age)/2)/20
        tis_max=(self.max_age-self.min_age-3)/10
        tis_min=-3/10
        pink_min=0 # irtisanottu vai ei 
        pink_max=1 # irtisanottu vai ei 
        toe_min=0-14/12 # työssäoloehto
        toe_max=28/12-14/12 # työssäoloehto
        thist_min=-3/10 # tyohistoria: 300/400 pv
        thist_max=(self.max_age-self.min_age-3)/10 # tyohistoria: 300/400 pv
        out_max=100
        out_min=0

        group_min=0
        group_max=1
        state_min=0
        state_max=1

        # korjaa
        if self.include_mort: # and not self.mortstop:
            #if self.include300:
            # Limits on states
            self.low = np.array([
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
                #pink_min,
                toe_min,
                thist_min,
                state_min,
                state_min,
                wage_min,
                out_min])
            self.high = np.array([
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
                #pink_max,
                toe_max,
                thist_max,
                state_max,
                state_max,
                wage_max,
                out_max])  
        else:
            self.low = np.array([
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
                #pink_min,
                toe_min,
                thist_min,
                state_min,
                state_min,
                wage_min,
                out_min])
            self.high = np.array([
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
                #pink_max,
                toe_max,
                thist_max,
                state_max,
                state_max,
                wage_max,
                out_max])
        if debug:  
            print(self.low.shape,self.high.shape)