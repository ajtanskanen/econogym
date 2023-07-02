import math
import gym
import numpy as np
import random
from . util import compare_q_print,crosscheck_print
        
class Statevector_v6():
    '''
    Implements class for handling of state vector for UnemploymentLargeEnv_v5
    '''
    def __init__(self,n_empl: int,n_groups: int,n_parttime_action: int,include_mort: bool,min_age: float,max_age: float,include_preferencenoise: bool,
                min_retirementage: float,min_ove_age: float,get_paid_wage: float,timestep: float):
        self.n_empl=n_empl
        self.n_groups=n_groups
        self.n_empl=n_empl
        self.n_parttime_action=n_parttime_action
        self.include_mort=include_mort
        self.setup_state_encoding()
        self.state = None
        self.log_transform=False
        self.pensionscale=25_000 # neuroverkon eläkeskaalaus
        self.wagescale=50_000 # neuroverkon palkkaskaalaus
        self.min_age=min_age
        self.max_age=max_age
        self.include_preferencenoise=include_preferencenoise
        self.min_retirementage=min_retirementage
        self.min_ove_age=min_ove_age
        self.get_paid_wage=get_paid_wage
        self.timestep=timestep
        self.n_states=self.n_empl+self.n_groups+self.n_empl+self.n_parttime_action+self.n_parttime_action+56
        
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
                        main_paid_wage : float,spouse_paid_wage : float,
                        main_pt_action : int, spouse_pt_action : int,
                        main_wage_basis : float,spouse_wage_basis : float,
                        main_life_left: float,spouse_life_left: float,
                        prefnoise : float):     
        '''
        Tilan koodaus neuroverkkoa varten. Arvot skaalataan ja tilat one-hot-enkoodataan

        Käytetään, jos kuolleisuus ei mukana
        '''
        n_states=self.n_empl+self.n_groups+self.n_empl+self.n_parttime_action+self.n_parttime_action+56
        if self.include_preferencenoise:
            d=np.zeros(n_states)
        else:
            d=np.zeros(n_states+1)
                    
        states=self.n_empl
        d[0:states]=self.state_encoding[emp,:]
        if emp==15 and not self.include_mort:
            print('no state 15 in state_encode_nomort')
        elif emp>15:
            print('state_encode error '+str(emp))

        states2=states+self.n_groups
        d[states:states2]=self.group_encoding[g,:]
        
        states3=states2+self.n_empl
        d[states2:states3]=self.spousestate_encoding[puoliso_tila,:]        
        
        if emp in set([1,8,9,10]):
            d[states3]=1
            if emp in set([8,10]):
                d[states3+1]=1
            else:
                d[states3+1]=0
        else:
            d[states3+0]=0
            d[states3+1]=0
        states4=states3+2+self.n_parttime_action
        d[(states3+2):states4]=self.ptstate_encoding[main_pt_action]
        if puoliso_tila in set([1,8,9,10]):
            d[states4]=1
            if puoliso_tila in set([8,10]):
                d[states4+1]=1
            else:
                d[states4+1]=0
        else:
            d[states4]=0
            d[states4+1]=0
            
        states5=states4+2+self.n_parttime_action
        d[(states4+2):states5]=self.ptstate_encoding[spouse_pt_action]
        
        if self.log_transform:
            d[states5]=math.log(pension/self.pensionscale+self.eps) # vastainen eläke
            d[states5+1]=math.log(old_wage/self.wagescale+self.eps)
            d[states5+4]=math.log(tyoelake_maksussa/self.pensionscale+self.eps) # alkanut eläke
            d[states5+10]=math.log(next_wage/self.wagescale+self.eps)
            d[states5+14]=math.log(unempwage/self.wagescale+self.eps)
            d[states5+15]=math.log(unempwage_basis/self.wagescale+self.eps)
        else:
            d[states5]=(pension-self.pensionscale)/self.pensionscale # vastainen eläke
            d[states5+1]=(old_wage-self.wagescale)/self.wagescale
            d[states5+4]=(tyoelake_maksussa-self.pensionscale)/self.pensionscale # alkanut eläke
            d[states5+10]=(next_wage-self.wagescale)/self.wagescale
            d[states5+14]=(unempwage-self.wagescale)/self.wagescale
            d[states5+15]=(unempwage_basis-self.wagescale)/self.wagescale

        age_scale = (self.max_age-self.min_age)/2
        age_mid = (self.max_age+self.min_age)/2
        age_dur = 2*age_mid
        d[states5+2]=(age-age_mid)/age_scale
        d[states5+3]=(time_in_state-10)/10
        if age>=self.min_retirementage:
            retaged=1
        else:
            retaged=0

        d[states5+5]=pink # irtisanottu vai ei 
        d[states5+6]=toe-14/12 # työssäoloehto **
        d[states5+7]=(tyohist-10)/20 # tyohistoria: 300/400 pv
        d[states5+8]=(self.min_retirementage-age)/age_dur
        d[states5+9]=unemp_benefit_left-1
        d[states5+11]=used_unemp_benefit-1
        d[states5+12]=wage_reduction
        d[states5+13]=(unemp_after_ra-1)/2
        d[states5+16]=retaged
        d[states5+17]=alkanut_ansiosidonnainen
        d[states5+18]=(children_under3-5)/10
        d[states5+19]=(children_under7-5)/10
        d[states5+20]=(children_under18-5)/10
        d[states5+21]=toe58
        d[states5+22]=ove_paid
        if age>=self.min_ove_age:
            d[states5+23]=1
        else:
            d[states5+23]=0
        
        d[states5+24]=kassanjasenyys
        d[states5+25]=toekesto-14/12
        d[states5+26]=puoliso
        
        d[states5+27]=(puoliso_old_wage-self.wagescale)/self.wagescale
        d[states5+28]=(puoliso_pension-self.pensionscale)/self.pensionscale
        d[states5+29]=puoliso_wage_reduction
        d[states5+30]=(puoliso_tyoelake_maksussa-self.pensionscale)/self.pensionscale # alkanut eläke
        d[states5+31]=(puoliso_next_wage-self.wagescale)/self.wagescale
        d[states5+32]=puoliso_used_unemp_benefit-1
        d[states5+33]=puoliso_unemp_benefit_left-1
        d[states5+34]=(puoliso_unemp_after_ra-1)/2
        d[states5+35]=(puoliso_unempwage-self.wagescale)/self.wagescale
        d[states5+36]=(puoliso_unempwage_basis-self.wagescale)/self.wagescale
        d[states5+37]=puoliso_alkanut_ansiosidonnainen
        d[states5+38]=puoliso_toe58
        d[states5+39]=puoliso_toe-14/12
        d[states5+40]=puoliso_toekesto-14/12
        d[states5+41]=(puoliso_tyoura-10)/20
        d[states5+42]=(puoliso_time_in_state-10)/10
        d[states5+43]=puoliso_pinkslip
        d[states5+44]=puoliso_ove_paid
        d[states5+45]=(kansanelake-self.pensionscale)/self.pensionscale
        d[states5+46]=(puoliso_kansanelake-self.pensionscale)/self.pensionscale
        d[states5+47]=(main_paid_wage-self.wagescale)/self.wagescale # vastainen eläke
        d[states5+48]=(spouse_paid_wage-self.wagescale)/self.wagescale # vastainen eläke
        d[states5+49]=(main_wage_basis-self.wagescale)/self.wagescale # vastainen eläke
        d[states5+50]=(spouse_wage_basis-self.wagescale)/self.wagescale # vastainen eläke

        age_maxdur = (100-self.min_age)/2
        d[states5+51]=(main_life_left-age_maxdur)/age_maxdur # elinaikaa jäljellä
        d[states5+52]=(spouse_life_left-age_maxdur)/age_maxdur # 

        if self.include_preferencenoise:
            d[states5+53]=prefnoise
            
        return d

    def get_state_name(self):
        if self.include_preferencenoise:
            d=np.empty(self.n_states, dtype='U256')
        else:
            d=np.empty(self.n_states+1, dtype='U256')
                    
        states=self.n_empl
        states2=states+self.n_groups
        states3=states2+self.n_empl
        states4=states3+2+self.n_parttime_action
        states5=states4+2+self.n_parttime_action
        d[0]='emp'
        d[0:states]='emp'
        d[states:states2]='group'
        d[states2:states3]='puoliso_tila'
        d[(states3+2):states4]='main_pt_action'
            
        d[(states4+2):states5]='spouse_pt_action'
        d[states5]='pension'
        d[states5+1]='old_wage'
        d[states5+4]='tyoelake_maksussa'
        d[states5+10]='next_wage'
        d[states5+14]='unempwage'
        d[states5+15]='unempwage_basis'
        d[states5+2]='age'
        d[states5+3]='time_in_state'

        d[states5+5]='pink'
        d[states5+6]='toe'
        d[states5+7]='tyohist'
        d[states5+8]='self.min_retirementage'
        d[states5+9]='unemp_benefit_left'
        d[states5+11]='used_unemp_benefit'
        d[states5+12]='wage_reduction'
        d[states5+13]='unemp_after_ra'
        d[states5+16]='retaged'
        d[states5+17]='alkanut_ansiosidonnainen'
        d[states5+18]='children_under3'
        d[states5+19]='children_under7'
        d[states5+20]='children_under18'
        d[states5+21]='toe58'
        d[states5+22]='ove_paid'
        d[states5+23]='over_ove_age'
        
        d[states5+24]='kassanjasenyys'
        d[states5+25]='toekesto'
        d[states5+26]='puoliso'
        
        d[states5+27]='puoliso_old_wage'
        d[states5+28]='puoliso_pension'
        d[states5+29]='puoliso_wage_reduction'
        d[states5+30]='puoliso_tyoelake_maksussa'
        d[states5+31]='puoliso_next_wage'
        d[states5+32]='puoliso_used_unemp_benefit'
        d[states5+33]='puoliso_unemp_benefit_left'
        d[states5+34]='puoliso_unemp_after_ra'
        d[states5+35]='puoliso_unempwage'
        d[states5+36]='puoliso_unempwage_basis'
        d[states5+37]='puoliso_alkanut_ansiosidonnainen'
        d[states5+38]='puoliso_toe58'
        d[states5+39]='puoliso_toe'
        d[states5+40]='puoliso_toekesto'
        d[states5+41]='puoliso_tyoura'
        d[states5+42]='puoliso_time_in_state'
        d[states5+43]='puoliso_pinkslip'
        d[states5+44]='puoliso_ove_paid'
        d[states5+45]='kansanelake'
        d[states5+46]='puoliso_kansanelake'
        d[states5+47]='main_paid_wage'
        d[states5+48]='spouse_paid_wage'
        d[states5+49]='main_wage_basis'
        d[states5+50]='spouse_wage_basis'
        d[states5+51]='main_life_left'
        d[states5+52]='spouse_life_left'

        if self.include_preferencenoise:
            d[states5+53]='prefnoise'
            
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

    def get_onehot(self,vec,a,n,desc):
        g=-1
        pos=a
        for k in range(n):
            if vec[pos+k]>0:
                g=k
                break

        g=int(g)
        if g<0:
            print(f'{desc}:state error at {a} len {n}:'+str(vec[a:a+n]))
        
        return g

    def state_decode(self,vec):
        '''
        Tilan dekoodaus laskentaa varten

        Käytetään, jos aina
        '''

        pos=0
        emp=self.get_onehot(vec,pos,self.n_empl,'emp')
        pos+=self.n_empl

        g=self.get_onehot(vec,pos,self.n_groups,'g')
        spouse_g=self.get_spouse_g(g)
        pos+=self.n_groups

        puoliso_tila=self.get_onehot(vec,pos,self.n_empl,'s_emp')
        pos+=self.n_empl+2
                
        pt_act=self.get_onehot(vec,pos,self.n_parttime_action,'pt1')
        pos+=self.n_parttime_action+2

        sp_pt_act=self.get_onehot(vec,pos,self.n_parttime_action,'pt2')
        pos+=self.n_parttime_action

        if self.log_transform:
            pension=(math.exp(vec[pos])-self.eps)*self.pensionscale
            wage=(math.exp(vec[pos+1])-self.eps)*self.wagescale
            next_wage=(math.exp(vec[pos+10])-self.eps)*self.wagescale
            tyoelake_maksussa=(math.exp(vec[pos+4])-self.eps)*self.pensionscale
            unempwage=(math.exp(vec[pos+14])-self.eps)*self.wagescale
            unempwage_basis=(math.exp(vec[pos+15])-self.eps)*self.wagescale
        else:
            pension=vec[pos]*self.pensionscale+self.pensionscale
            wage=vec[pos+1]*self.wagescale+self.wagescale 
            next_wage=vec[pos+10]*self.wagescale+self.wagescale 
            tyoelake_maksussa=vec[pos+4]*self.pensionscale+self.pensionscale
            unempwage=vec[pos+14]*self.wagescale+self.wagescale 
            unempwage_basis=vec[pos+15]*self.wagescale+self.wagescale 

        age_scale = (self.max_age-self.min_age)/2
        age_mid = (self.max_age+self.min_age)/2
        age_dur = 2*age_mid

        age=vec[pos+2]*age_scale+age_mid
        time_in_state=vec[pos+3]*10+10
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
        puoliso_old_wage=vec[pos+27]*self.wagescale+self.wagescale
        puoliso_pension=vec[pos+28]*self.pensionscale+self.pensionscale
        puoliso_wage_reduction=vec[pos+29]
        puoliso_tyoelake_maksussa=vec[pos+30]*self.pensionscale+self.pensionscale
        puoliso_next_wage=vec[pos+31]*self.wagescale+self.wagescale
        puoliso_used_unemp_benefit=vec[pos+32]+1
        puoliso_unemp_benefit_left=vec[pos+33]+1
        puoliso_unemp_after_ra=2*vec[pos+34]+1
        puoliso_unempwage=vec[pos+35]*self.wagescale+self.wagescale
        puoliso_unempwage_basis=vec[pos+36]*self.wagescale+self.wagescale
        puoliso_alkanut_ansiosidonnainen=int(vec[pos+37])
        puoliso_toe58=int(vec[pos+38])
        puoliso_toe=vec[pos+39]+14/12
        puoliso_toekesto=vec[pos+40]+14/12
        puoliso_tyoura=vec[pos+41]*20+10
        puoliso_time_in_state=vec[pos+42]*10+10
        puoliso_pinkslip=int(vec[pos+43])
        puoliso_ove_paid=int(vec[pos+44])
        
        kansanelake=vec[pos+45]*self.pensionscale+self.pensionscale
        puoliso_kansanelake=vec[pos+46]*self.pensionscale+self.pensionscale
        paid_pension=tyoelake_maksussa+kansanelake
        puoliso_paid_pension=puoliso_tyoelake_maksussa+puoliso_kansanelake

        main_paid_wage=vec[pos+47]*self.wagescale+self.wagescale
        spouse_paid_wage=vec[pos+48]*self.wagescale+self.wagescale
        main_basis_wage=vec[pos+49]*self.wagescale+self.wagescale
        spouse_basis_wage=vec[pos+50]*self.wagescale+self.wagescale

        age_maxdur = (100-self.min_age)/2
        main_life_left=vec[pos+51]*age_maxdur+age_maxdur
        spouse_life_left=vec[pos+52]*age_maxdur+age_maxdur

        if self.include_preferencenoise:
            prefnoise=vec[pos+53]
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
               main_paid_wage,spouse_paid_wage,pt_act,sp_pt_act,\
               main_basis_wage,spouse_basis_wage,\
               main_life_left,spouse_life_left
                
                              
    def random_init_state(self,minage: float=18,maxage: float=70):
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
        main_wage_basis=np.random.uniform(0,50000)
        spouse_wage_basis=np.random.uniform(0,50000)
        paid_pension=kansanelake+tyoelake_maksussa
        puoliso_paid_pension=puoliso_kansanelake+puoliso_tyoelake_maksussa
        main_pt_action=np.random.randint(0,2)
        spouse_pt_action=np.random.randint(0,2)
        main_paid_wage,main_pt_factor=self.get_paid_wage(old_wage,emp,main_pt_action)
        spouse_paid_wage,spouse_pt_factor=self.get_paid_wage(puoliso_old_wage,puoliso_tila,spouse_pt_action)
        main_life_left,spouse_life_left=np.random.uniform(0,100.0),np.random.uniform(0,100.0)
        
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
                main_paid_wage,spouse_paid_wage,
                main_pt_action,spouse_pt_action,
                main_wage_basis,spouse_wage_basis,
                main_life_left,spouse_life_left,
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
            paid_pension=kansanelake+tyoelake_maksussa
            puoliso_paid_pension=puoliso_kansanelake+puoliso_tyoelake_maksussa
            main_pt_action=np.random.randint(0,3)
            spouse_pt_action=np.random.randint(0,3)
            old_paid=np.random.uniform(0,50000)
            spouse_old_paid=np.random.uniform(0,50000)
            main_paid_wage,main_pt_factor=self.get_paid_wage(old_wage,emp,main_pt_action)
            spouse_paid_wage,spouse_pt_factor=self.get_paid_wage(puoliso_old_wage,puoliso_tila,spouse_pt_action)
            main_wage_basis=np.random.uniform(0,50000)
            spouse_wage_basis=np.random.uniform(0,50000)
            main_life_left=np.random.uniform(0,100)
            spouse_life_left=np.random.uniform(0,100)
        
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
                                main_paid_wage,spouse_paid_wage,
                                main_pt_action,spouse_pt_action,
                                main_wage_basis,spouse_wage_basis,
                                main_life_left,spouse_life_left,
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
                main_paid_wage2,spouse_paid_wage2,pt_act2,s_pt_act2,\
                main_wage_basis2,spouse_wage_basis2,\
                main_life_left2,spouse_life_left2\
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
                                main_paid_wage,spouse_paid_wage,main_pt_action,spouse_pt_action,main_wage_basis,spouse_wage_basis,
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
                                next_wage2,main_paid_wage2,spouse_paid_wage2,pt_act2,s_pt_act2,
                                main_wage_basis2,spouse_wage_basis2,main_life_left,spouse_life_left,main_life_left2,spouse_life_left2)
        
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
                    old_paid,spouse_old_paid,main_pt_action,spouse_pt_action,
                    main_wage_basis,spouse_wage_basis,
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
                    next_wage2,old_paid2,spouse_old_paid2,pt_act2,s_pt_act2,main_wage_basis2,spouse_wage_basis2,
                    main_life_left,spouse_life_left,main_life_left2,spouse_life_left2):
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
        if not math.isclose(old_paid,old_paid2):  
            print('old_paid: {} vs {}'.format(old_paid,old_paid2))
        if not math.isclose(spouse_old_paid,spouse_old_paid2):  
            print('spouse_old_paid: {} vs {}'.format(spouse_old_paid,spouse_old_paid2))
        if not math.isclose(spouse_old_paid,spouse_old_paid2):  
            print('spouse_old_paid: {} vs {}'.format(spouse_old_paid,spouse_old_paid2))
        if not math.isclose(main_pt_action,pt_act2):  
            print('main_pt_action: {} vs {}'.format(main_pt_action,pt_act2))
        if not math.isclose(spouse_pt_action,s_pt_act2):  
            print('spouse_pt_action: {} vs {}'.format(spouse_pt_action,s_pt_act2))
        if not math.isclose(main_wage_basis,main_wage_basis2):  
            print('main_wage_basis: {} vs {}'.format(main_wage_basis,main_wage_basis2))
        if not math.isclose(spouse_wage_basis,spouse_wage_basis2):  
            print('spouse_wage_basis: {} vs {}'.format(spouse_wage_basis,spouse_wage_basis2))
        if not math.isclose(main_life_left,main_life_left2):  
            print('main_life_left: {} vs {}'.format(main_life_left,main_life_left2))
        if not math.isclose(spouse_life_left,spouse_life_left2):  
            print('spouse_life_left: {} vs {}'.format(spouse_life_left,spouse_life_left2))
    
    def check_state_vec(self,vec1,vec2):
        emp2,g2,pension2,old_wage2,age2,time_in_state2,paid_pension2,pink2,toe2,toekesto2,\
            tyohist2,used_unemp_benefit2,wage_reduction2,unemp_after_ra2,\
            unempwage2,unempwage_basis2,prefnoise2,\
            children_under3_2,children_under7_2,children_under18_2,unemp_benefit_left2,\
            alkanut_ansiosidonnainen2,toe58_2,ove_paid_2,jasen_2,p2,p2_tila,p2_g,p2_old_wage,p2_pension,\
            p2_wage_reduction,p2_paid_pension,p2_next_wage,p2_used_unemp_benefit,p2_unemp_benefit_left,\
            p2_unemp_after_ra,p2_unempwage,p2_unempwage_basis,p2_alkanut_ansiosidonnainen,p2_toe58,p2_toe,\
            p2_toekesto,p2_tyoura,p2_time_in_state,p2_pinkslip,p2_ove_paid,\
            kansanelake2,p2_kansanelake,tyoelake_maksussa2,p2_tyoelake_maksussa,next_wage2,\
            main_paid_wage2,spouse_paid_wage2,\
            pt_act2,s_pt_act2,\
            main_wage_basis2,spouse_wage_basis2,main_life_left2,spouse_life_left2\
            =self.state_decode(vec2)

        emp,g,pension,old_wage,age,time_in_state,paid_pension,pink,toe,toekesto,\
            tyohist,used_unemp_benefit,wage_reduction,unemp_after_ra,\
            unempwage,unempwage_basis,prefnoise,\
            children_under3,children_under7,children_under18,unemp_benefit_left,\
            alkanut_ansiosidonnainen,toe58,ove_paid,jasen,puoliso,p_tila,p_g,p_old_wage,p_pension,\
            p_wage_reduction,p_paid_pension,p_next_wage,p_used_unemp_benefit,p_unemp_benefit_left,\
            p_unemp_after_ra,p_unempwage,p_unempwage_basis,p_alkanut_ansiosidonnainen,p_toe58,p_toe,\
            p_toekesto,p_tyoura,p_time_in_state,p_pinkslip,p_ove_paid,\
            kansanelake,p_kansanelake,tyoelake_maksussa,p_tyoelake_maksussa,next_wage,\
            main_paid_wage,spouse_paid_wage,\
            pt_act,s_pt_act,\
            main_wage_basis,spouse_wage_basis,main_life_left,spouse_life_left\
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
        if not math.isclose(main_paid_wage,main_paid_wage2):  
            print('main_paid_wage: {} vs {}'.format(main_paid_wage,main_paid_wage2))    
        if not math.isclose(spouse_paid_wage,spouse_paid_wage2):  
            print('spouse_paid_wage: {} vs {}'.format(spouse_paid_wage,spouse_paid_wage2))    
        if not math.isclose(pt_act,pt_act2):  
            print('main_pt_action: {} vs {}'.format(pt_act,pt_act2))
        if not math.isclose(s_pt_act,s_pt_act2):  
            print('spouse_pt_action: {} vs {}'.format(s_pt_act,s_pt_act2))
        if not math.isclose(main_wage_basis,main_wage_basis2):  
            print('main_wage_basis: {} vs {}'.format(main_wage_basis,main_wage_basis2))
        if not math.isclose(spouse_wage_basis,spouse_wage_basis2):  
            print('spouse_wage_basis: {} vs {}'.format(spouse_wage_basis,spouse_wage_basis2))

    def setup_state_encoding(self):
        self.state_encoding=np.zeros((self.n_empl,self.n_empl))
        for s in range(self.n_empl):
            self.state_encoding[s,s]=1
        self.group_encoding=np.zeros((self.n_groups,self.n_groups))
        for s in range(self.n_groups):
            self.group_encoding[s,s]=1
        self.spousestate_encoding=np.zeros((self.n_empl,self.n_empl))
        for s in range(self.n_empl):
            self.spousestate_encoding[s,s]=1
        self.ptstate_encoding=np.zeros((self.n_parttime_action,self.n_parttime_action))
        for s in range(self.n_parttime_action):
            self.ptstate_encoding[s,s]=1

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
            main_paid_wage,spouse_paid_wage,main_pt_action,spouse_pt_action,\
            main_wage_basis,spouse_wage_basis,main_life_left,spouse_life_left\
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
            spouse_paid_wage,main_paid_wage,
            spouse_pt_action,main_pt_action,
            spouse_wage_basis,main_wage_basis,
            spouse_life_left,main_life_left,
            prefnoise)
            
    def set_state_limits(self,debug=True):
        '''
        Rajat tiloille
        '''
        if self.log_transform:
            pension_min=math.log(0/self.pensionscale+self.eps) # vastainen eläke
            pension_max=math.log(200_000/self.pensionscale+self.eps) # vastainen eläke
            wage_max=math.log(500_000/self.wagescale+self.eps)
            wage_min=math.log(0/self.wagescale+self.eps)
            paid_pension_max=math.log(200_00/self.pensionscale+self.eps) # alkanut eläke
            paid_pension_min=math.log(0/self.pensionscale+self.eps) # alkanut eläke
        else:
            pension_max=(200_000-self.pensionscale)/self.pensionscale # vastainen eläke
            pension_min=(0-self.pensionscale)/self.pensionscale # vastainen eläke
            wage_max=(300_000-self.wagescale)/self.wagescale
            wage_min=(0-self.wagescale)/self.wagescale
            paid_pension_min=(0-self.pensionscale)/self.pensionscale # alkanut eläke
            paid_pension_max=(200_000-self.pensionscale)/self.pensionscale # alkanut eläke

        age_max=(self.max_age+1-(self.max_age+self.min_age)/2)/20
        age_min=(self.min_age-1-(self.max_age+self.min_age)/2)/20
        tis_max=(self.max_age-self.min_age-10)/10
        tis_min=-10/10
        pink_min=0 # irtisanottu vai ei 
        pink_max=1 # irtisanottu vai ei 
        toe_min=-15/12 # työssäoloehto
        toe_max=15/12 # työssäoloehto
        thist_min=-10/20 # tyohistoria: 300/400 pv
        thist_max=(self.max_age-self.min_age-10)/20 # tyohistoria: 300/400 pv

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
        tr_min=(63-self.max_age)/(self.max_age-self.min_age)
        tr_max=(68-self.min_age)/(self.max_age-self.min_age)
        left_min=-1
        left_max=1
        lleftmin=-2
        lleftmax=2
        
        low=[]
        high=[]
        low_mid = [
            pension_min, # vastainen eläke
            wage_min, # old_wage
            age_min, # age
            tis_min, # time_in_state
            paid_pension_min, # maksussa oleva eläke
            state_min, # pink
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
            state_min, # on puolisoa tai ei
            ]
        
        low_end=[wage_min, # puoliso old wage
            paid_pension_min, # puoliso pension
            wr_min, # puoliso wr
            paid_pension_min, # työeläke maksussa
            wage_min, # puoliso next wage
            ben_min, # puoliso_used_unemp_benefit
            ben_min, # puoliso_unemp_benefit_left
            unra_min, # puoliso_unemp_after_ra
            wage_min, # puoliso_unempwage
            wage_min, # puoliso_unempwage_basis
            state_min, # puoliso alkanut ansiosidonnainen
            toe_min, # puoliso toe58
            toe_min, # puoliso_toe
            toe_min, # puoliso_toekesto
            thist_min, # puoliso_tyoura
            tis_min, # puoliso_time_in_state
            state_min, # puoliso_pinkslip
            state_min, # puoliso_ove_paid
            paid_pension_min, # kansanelake
            paid_pension_min, # puoliso_kansanelake        
            wage_min, # main_paid_wage
            wage_min, # spouse_paid_wage
            wage_min, # main_wage_basis
            wage_min, # spouse_wage_basis
            lleftmin,
            lleftmin
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
            toe_max, # puoliso toe58
            toe_max,
            toe_max,
            thist_max,
            tis_max,
            state_max,
            state_max,
            paid_pension_max, #kansaneläke
            paid_pension_max, #kansaneläke
            wage_max, # main_paid_wage
            wage_max, # spouse_paid_wage
            wage_max, # main_wage_basis
            wage_max, # spouse_wage_basis
            lleftmax, # left
            lleftmax  # puoliso left
            ]
                    
        for k in range(self.n_empl):
            low.append(state_min)
            high.append(state_max)
        for k in range(self.n_groups):
            low.append(group_min)
            high.append(group_max)
        for k in range(self.n_empl):
            low.append(state_min)
            high.append(state_max)
        for k in range(2*(self.n_parttime_action+2)):
            low.append(state_min)
            high.append(state_max)
            
        low.extend(low_mid)
        high.extend(high_mid)
            
        low.extend(low_end)
        high.extend(high_end)
              
        if self.include_preferencenoise:
            low.append(pref_min)
            high.append(pref_max)
                
        low=np.array(low,dtype=np.float32)
        high=np.array(high,dtype=np.float32)            
        
        return low,high