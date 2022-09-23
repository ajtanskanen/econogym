"""

    infostate_v5

    - toimii unemployment_v5:n muistina
    - tämän avulla lasketaan mm. tulevan ajan palkka työkyvyttömyyteen
    
    
"""

import math
import numpy as np
import random

class Infostate():
    def __init__(self,n_time,timestep,min_age,lapsia=0,lasten_iat=np.zeros(15),
        lapsia_paivakodissa=0,age=18,kassanjasenyys_joinrate=None,kassanjasenyys_rate=None,include_halftoe=None,min_toewage=None):
        '''
        Alustaa infostate-dictorionaryn
        Siihen talletetaan tieto aiemmista tiloista, joiden avulla lasketaan statistiikkoja
        '''
        self.n_time=n_time
        self.timestep=timestep
        self.min_age=min_age
        self.include_halftoe=include_halftoe
        self.min_toewage=min_toewage
        self.kassanjasenyys_joinrate=kassanjasenyys_joinrate
        self.kassanjasenyys_rate=kassanjasenyys_rate
        self.reset(lapsia=lapsia,lasten_iat=lasten_iat,lapsia_paivakodissa=lapsia_paivakodissa,age=age)
        
    def reset(self,lapsia=0,lasten_iat=np.zeros(15),lapsia_paivakodissa=0,age=18,spouse=False):
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
        
        #print('age {} sattuma {} rate {}'.format(age,sattuma,self.kassanjasenyys_rate[t]))

    #def init_inforate(self):
    #    self.kassanjasenyys_joinrate,self.kassanjasenyys_rate=self.rates.get_kassanjasenyys_rate()
        
    def infostate_add_child(self,age : float):
        if self.infostate['children_n']<14:
            self.infostate['children_date'][self.infostate['children_n']]=age
            self.infostate['children_n']=self.infostate['children_n']+1
            
    def infostate_set_enimmaisaika(self,age : float,is_spouse=False):
        states,latest,enimaika,palkka,voc_unempbasis,member=self.infostate_vocabulary(is_spouse=is_spouse)
        t=int((age-self.min_age)/self.timestep)
        self.infostate[enimaika]=t
        
    def update_infostate(self,t : int,state : int,paid_wage : float,basiswage : float,unempbasis : float,is_spouse=False):
        states,latest,enimaika,palkka,voc_unempbasis,member=self.infostate_vocabulary(is_spouse=is_spouse)
    
        self.infostate[states][t]=state
        self.infostate[latest]=int(t)
        self.infostate[voc_unempbasis][t]=unempbasis
        self.infostate[member][t]=self.infostate['kassanjasen']
        if state==1:
            self.infostate[palkka][t]=paid_wage
        elif state==10:
            self.infostate[palkka][t]=paid_wage
        elif state in set([5,6,14]):
            self.infostate[palkka][t]=basiswage
        else:
            self.infostate[palkka][t]=0
        
    def render(self):
        print('states {}'.format(self.infostate['states']))
        
    def get_kassanjasenyys(self):
        return self.infostate['kassanjasen']

    def set_kassanjasenyys(self,value : int):
        self.infostate['kassanjasen']=value

    def infostate_kassanjasenyys_update(self,age : float):
        if self.infostate['kassanjasen']<1:
            sattuma = np.random.uniform(size=1)
            if sattuma[0]<self.kassanjasenyys_joinrate[self.map_age(age)] and self.randomness:
                self.set_kassanjasenyys(1)
        
    def comp_toe_wage_nykytila(self,is_spouse=False):
        states,latest,enimaika,palkka,voc_unempbasis,member=self.infostate_vocabulary(is_spouse=is_spouse)
        lstate=int(self.infostate[states][self.infostate[latest]])
        toes=0
        wage=0
        n_toe=int(np.floor(self.max_toe/self.timestep))
        emp_states={1,10}
        unemp_states={0,4}
        family_states={5,6,7,14}
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
        family_states={5,6,7,14}
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
                    elif self.include_halftoe and w>self.min_halftoewage: # and emps==10:
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
                    elif self.include_halftoe and w>self.min_halftoewage: # and emps==10:
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
        family_states={5,6,14}
        
        states,latest,enimaika,voc_wage,voc_unempbasis,member=self.infostate_vocabulary(is_spouse=is_spouse)
    
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
                    value=self.infostate[voc_wage][x] # FIXME!
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
        
    def infostate_comp_svpaivaraha_1v(self,is_spouse=False):
        emp_states={1,10}
        unemp_states={0,4}
        family_states={5,6,14}
        
        states,latest,enimaika,voc_wage,voc_unempbasis,member=self.infostate_vocabulary(is_spouse=is_spouse)
    
        lstate=int(self.infostate[latest])
        n=int(np.ceil(1/self.timestep))
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
                    value=self.infostate[voc_wage][x] # FIXME!
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
                    
                wage += value*self.timestep
                #truewage += w*self.timestep

        return wage#,truewage
        
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
        if (t-ed_t)/self.timestep:
            return True
        else:
            return False
        
    def comp_oldtoe(self,printti=False,is_spouse=False):
        '''
        laske työttämyysjaksoa edeltävä työssäoloehto tarkasti
        '''
        toes=0
        n_toe=int(np.floor(self.max_toe/self.timestep))
        emp_states={1,10}
        unemp_states={0,4,13}
        family_states={5,6,7,14}
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
        family_states={5,6,7,14}
        while nt<n_toe:
            emps=self.infostate[states][t2]
            if printti:
                print('emps {} t2 {} toes {}'.format(emps,t2,toes))
            if emps in family_states:
                pass
            elif emps in emp_states:
                if self.infostate[palkka][t2]>self.min_toewage:
                    toes+=self.timestep
                elif self.include_halftoe and self.infostate[palkka][t2]>=self.min_halftoewage: # and emps==10:
                    toes+=0.5*self.timestep
                nt=nt+1
            elif emps in unemp_states:
                nt=nt+1
            else:
                nt=nt+1
            t2=t2-1

        return toes
        
    def comp_svperuste(self,printti=False,is_spouse=False):
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
        
        states,latest,enimaika,palkka,voc_unempbasis,member=self.infostate_vocabulary(is_spouse=is_spouse)
        
        lstate=int(self.infostate[states][self.infostate[latest]])
        
        nt=0
        t2=max(0,self.infostate[enimaika]-1)
        emp_states={1,10}
        unemp_states={0,4}
        family_states={5,6,7,14}
        while nt<n_svp:
            emps=self.infostate[states][t2]
            if printti:
                print('emps {} t2 {} toes {}'.format(emps,t2,toes))
            if emps in family_states:
                pass
            elif emps in emp_states:
                if self.infostate[palkka][t2]:
                    toes+=self.infostate[palkka][t2]
                nt=nt+1
            elif emps in unemp_states:
                if self.infostate[palkka][t2]:
                    toes+=self.infostate[palkka][t2]
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
        family_states={5,6,7,14}
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
                elif self.include_halftoe and self.infostate[palkka][t2]>=self.min_halftoewage: # and emps==10:
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