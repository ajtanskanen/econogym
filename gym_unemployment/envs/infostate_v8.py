
###############################################
###
###    INFOSTATE
###
###############################################

from . rates import Rates
import random

class Infostate_v8(gym.Env):
    def __init__(self,lapsia: int=0,lasten_iat=np.zeros(15),lapsia_paivakodissa: int=0,age: int=18,spouse:bool =False): -> None:
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

    def init_inforate(self):
        self.kassanjasenyys_joinrate,self.kassanjasenyys_rate = self.rates.get_kassanjasenyys_rate()

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