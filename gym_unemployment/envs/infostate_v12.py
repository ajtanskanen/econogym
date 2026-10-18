"""

    infostate_v12

    Kotitalouden (päähenkilö + puoliso) tilahistorian hallinta unemployment_v12:lle.

    Historian avulla lasketaan suureet, joita ei ole tilavektorissa:
    - työssäoloehto (toe) ja sen palkka
    - 58-vuotiaiden lisäpäiväsääntö (toe58)
    - sairauspäivärahan ja vanhempainetuuksien perustepalkka (1 v)
    - työkyvyttömyyseläkkeen tulevan ajan 5 v keskipalkka
    - lasten lukumäärät ikäryhmittäin
    - työttömyyskassan jäsenyys

    Erotettu UnemploymentEnv_v12-luokasta. Logiikka on siirretty muuttamatta,
    ks. test_infostate_v12.py, joka vertaa uutta toteutusta vanhaan.

    Huom: kassanjäsenyys on kotitaloustasoinen lippu (yksi yhteinen päähenkilölle ja puolisolle),
    kuten alkuperäisessä toteutuksessa. Henkilökohtaiseksi muuttaminen onnistuu siirtämällä
    `kassanjasen` PersonHistory-luokkaan, mutta se muuttaa mallin käyttäytymistä ja vaatii
    tilavektoriin toisen jäsenyyskentän.

"""

import random
import numpy as np # type: ignore
from typing import Optional, Tuple

EMP_STATES = frozenset({1, 10})
UNEMP_STATES = frozenset({0, 4})
RET_STATES = frozenset({2, 3, 8, 9})
TOE_ACCEPTED_STATES = frozenset({5, 6, 7, 12, 14}) # pidentävät jaksot: eivät kuluta tarkastelujaksoa


class PersonHistory():
    '''
    Yhden henkilön tilahistoria. Indeksi t = askel iästä min_age alkaen.
    '''
    __slots__ = ('states', 'wage', 'unempbasis', 'wagebasis', 'member', 'latest', 'enimmaisaika')

    def __init__(self, n_time: int) -> None:
        self.states = np.zeros(n_time)-1         # työllisyystila, -1 = ei vielä täytetty
        self.wage = np.zeros(n_time)-1           # maksettu palkka (1,10) tai etuuden perustepalkka (5,6,14), muuten 0
        self.unempbasis = np.zeros(n_time)-1     # ansiopäivärahan perustepalkka
        self.wagebasis = np.zeros(n_time)-1      # sv-/vanhempainpäivärahan perustepalkka (5,6,14)
        self.member = np.zeros(n_time, dtype=np.int8) # kassanjäsenyys kullakin askeleella
        self.latest = 0                          # viimeisin täytetty indeksi
        self.enimmaisaika = 0                    # indeksi, jolloin enimmäisaika viimeksi alkoi

    def latest_state(self) -> int:
        return round(self.states[self.latest])


class Infostate_v12():
    '''
    Kotitalouden historia: kaksi PersonHistory-oliota, yhteiset lapset ja kassanjäsenyys.

    Parametrit (toe-rajat yms.) asetetaan set_params-metodilla. UnemploymentEnv_v12
    kutsuu sitä set_year-, init_inforate- ja reset-metodeissa, joten ympäristön
    attribuuttien muutokset siirtyvät tänne viimeistään seuraavan episodin alussa.
    '''

    MAX_CHILDREN = 14

    def __init__(self, n_time: int, min_age: float, timestep: float) -> None:
        self.n_time = n_time
        self.min_age = min_age
        self.timestep = timestep
        self.inv_timestep = int(np.round(1/timestep))

        # oletusparametrit; ympäristö ylikirjoittaa nämä set_params-kutsulla
        self.max_toe: float = 28/12
        self.min_toewage: float = 0.0
        self.min_halftoewage: float = 0.0
        self.include_halftoe: bool = True
        self.ansiopvraha_toe: float = 0.5
        self.disabbasis_tmtuki: float = 0.0
        self.minage_500: float = 58
        self.tyohistoria_vaatimus500: float = 10.0
        self.porrasta_toe: bool = False
        self.randomness: bool = True
        self.kassanjasenyys_joinrate: Optional[np.ndarray] = None

        self.reset()

    #############################
    # parametrit ja alustus
    #############################

    def set_params(self, max_toe: float, min_toewage: float, min_halftoewage: float, include_halftoe: bool,
                   ansiopvraha_toe: float, disabbasis_tmtuki: float, minage_500: float,
                   tyohistoria_vaatimus500: float, porrasta_toe: bool, randomness: bool,
                   kassanjasenyys_joinrate: Optional[np.ndarray] = None) -> None:
        self.max_toe = max_toe
        self.min_toewage = min_toewage
        self.min_halftoewage = min_halftoewage
        self.include_halftoe = include_halftoe
        self.ansiopvraha_toe = ansiopvraha_toe
        self.disabbasis_tmtuki = disabbasis_tmtuki
        self.minage_500 = minage_500
        self.tyohistoria_vaatimus500 = tyohistoria_vaatimus500
        self.porrasta_toe = porrasta_toe
        self.randomness = randomness
        if kassanjasenyys_joinrate is not None:
            self.kassanjasenyys_joinrate = kassanjasenyys_joinrate

    def reset(self) -> None:
        '''
        Tyhjentää koko kotitalouden historian (molemmat henkilöt, lapset, kassanjäsenyys)
        '''
        self.main = PersonHistory(self.n_time)
        self.spouse = PersonHistory(self.n_time)
        self.children_n = 0
        self.children_date = np.zeros(15)
        self.kassanjasen = 0

    def person(self, is_spouse: bool = False) -> PersonHistory:
        return self.spouse if is_spouse else self.main

    def map_age(self, age: float) -> int:
        return round((age-self.min_age)*self.inv_timestep)

    def age_to_t(self, age: float) -> int:
        return round((age-self.min_age)/self.timestep)

    #############################
    # päivitys
    #############################

    def update(self, t: int, state: int, paid_wage: float, basiswage: float, unempbasis: float, is_spouse: bool = False) -> None:
        '''
        Tallentaa askeleen t tiedot
        '''
        p = self.person(is_spouse)
        p.states[t] = state
        p.latest = int(t)
        p.unempbasis[t] = unempbasis
        p.member[t] = self.kassanjasen
        if state in EMP_STATES:
            p.wage[t] = paid_wage
        elif state in {5, 6, 14}:
            p.wage[t] = basiswage
            p.wagebasis[t] = basiswage
        else:
            p.wage[t] = 0

    def set_enimmaisaika(self, age: float, is_spouse: bool = False) -> None:
        self.person(is_spouse).enimmaisaika = self.age_to_t(age)

    def check_aareset(self, age: float, is_spouse: bool = False) -> bool:
        '''
        Onko edellisestä uudelleenmäärittelystä alle vuosi aikaa
        '''
        t = self.age_to_t(age)
        return (t-self.person(is_spouse).enimmaisaika)*self.timestep < 1.0

    #############################
    # kassanjäsenyys (kotitaloustasoinen)
    #############################

    def get_kassanjasenyys(self) -> bool:
        return bool(self.kassanjasen)

    def set_kassanjasenyys(self, value: int) -> None:
        self.kassanjasen = value

    def kassanjasenyys_update(self, age: float, empstate: int) -> None:
        '''
        Satunnainen liittyminen kassaan alle 45-vuotiaana.
        random.uniform kutsutaan aina (myös kun randomness=False), jotta satunnaislukujono pysyy ennallaan.
        '''
        if self.kassanjasen < 1 and age < 45:
            if empstate in {0, 1, 10}:
                kerroin = 1.5
            else:
                kerroin = 0.1
            if random.uniform(0, 1) < kerroin*self.kassanjasenyys_joinrate[self.map_age(age)] and self.randomness:
                self.set_kassanjasenyys(1)

    #############################
    # lapset (kotitaloustasoinen)
    #############################

    def add_child(self, age: float) -> None:
        if self.children_n < self.MAX_CHILDREN:
            self.children_date[self.children_n] = age
            self.children_n += 1

    def comp_children(self, age: float) -> Tuple[int, int, int]:
        '''
        Lasten määrät ikäryhmittäin. <=, jotta koko ikä 0-3, 0-7 ja 0-18 huomioidaan
        '''
        c3, c7, c18 = 0, 0, 0
        for k in range(self.children_n):
            c_age = age-self.children_date[k]
            if c_age <= 18:
                c18 += 1
                if c_age <= 7:
                    c7 += 1
                    if c_age <= 3:
                        c3 += 1
        return c3, c7, c18

    def can_have_children(self, age: float) -> bool:
        for k in range(self.children_n):
            if age-self.children_date[k] < 1.0:
                return False
        return True

    #############################
    # työssäoloehto
    #############################

    def tyossaoloehto(self, toe: float) -> bool:
        return toe >= self.ansiopvraha_toe

    def comp_toe_wage(self, is_spouse: bool = False) -> Tuple[float, float, float]:
        if self.porrasta_toe:
            return self.comp_toe_wage_porrastus(is_spouse=is_spouse)
        else:
            return self.comp_toe_wage_nykytila(is_spouse=is_spouse)

    def _toe_backwards(self, p: PersonHistory, t_stop: int, half: bool) -> Tuple[float, float]:
        '''
        Kulkee historiaa taaksepäin viimeisimmästä askeleesta indeksiin t_stop asti, enintään max_toe:n verran
        tarkastelujaksoa. Pidentävät jaksot (TOE_ACCEPTED_STATES) eivät kuluta tarkastelujaksoa,
        ellei henkilö ollut silloin kassan ulkopuolella.

        Palauttaa (toe, palkkasumma)
        '''
        n_toe = int(np.floor(self.max_toe/self.timestep))
        toes, wage = 0.0, 0.0
        t2 = p.latest
        nt = 0
        while nt < n_toe and t2 >= t_stop:
            emps = p.states[t2]
            if p.member[t2] < 1:
                nt += 1
            elif emps in TOE_ACCEPTED_STATES:
                pass
            elif emps in EMP_STATES:
                w = p.wage[t2]
                if w > self.min_toewage:
                    toes += self.timestep
                    wage += w*self.timestep
                elif half and self.include_halftoe and w > self.min_halftoewage:
                    toes += 0.5*self.timestep
                    wage += w*self.timestep
                nt += 1
            else:
                nt += 1
            t2 -= 1
        return toes, wage

    def comp_toe_wage_nykytila(self, is_spouse: bool = False) -> Tuple[float, float, float]:
        '''
        Nykytilan mukainen toe ja toe-ajan palkka
        '''
        p = self.person(is_spouse)
        toes, wage = 0.0, 0.0

        if self.kassanjasen > 0:
            lstate = p.latest_state()
            if lstate not in RET_STATES:
                if lstate in TOE_ACCEPTED_STATES:
                    # onko ollut riittävä toe ansiosidonnaiseen (ei: onko päiviä jäljellä)
                    toes, wage = self._toe_backwards(p, 0, half=False)
                else:
                    # onko toe täyttynyt viimeisimmän ansiosidonnaisen alkamisen jälkeen
                    toes, wage = self._toe_backwards(p, p.enimmaisaika, half=False)
                if toes >= self.ansiopvraha_toe and toes > 0:
                    wage = wage/toes
                else:
                    wage = 0

        return toes, toes, wage

    def comp_toe_wage_porrastus(self, is_spouse: bool = False) -> Tuple[float, float, float]:
        '''
        Porrastetun toen mukainen toe, toekesto ja palkka
        '''
        p = self.person(is_spouse)
        lstate = p.latest_state()

        if self.kassanjasen > 0 and lstate not in RET_STATES:
            # toe viimeisimmän ansiosidonnaisen alkamisen jälkeen (palkkasummaa ei käytetä)
            toes, _ = self._toe_backwards(p, p.enimmaisaika, half=True)
            # toe ansiosidonnaiseen ilman alkuhetken rajausta
            toekesto, wage = self._toe_backwards(p, 0, half=True)

            if toekesto >= self.ansiopvraha_toe and toekesto > 0:
                wage = wage/toekesto
            else:
                wage = 0

            if lstate in TOE_ACCEPTED_STATES:
                toes = toekesto
        else:
            toes, toekesto, wage = 0, 0, 0

        return toes, toekesto, wage

    def comp_oldtoe(self, is_spouse: bool = False) -> float:
        '''
        Työttömyysjaksoa edeltävä työssäoloehto
        '''
        p = self.person(is_spouse)
        n_toe = int(np.floor(self.max_toe/self.timestep))
        toes = 0.0
        nt = 0
        t2 = max(0, p.enimmaisaika-1)
        while nt < n_toe and t2 >= 0:
            emps = p.states[t2]
            if emps in TOE_ACCEPTED_STATES:
                pass
            elif emps in EMP_STATES:
                if p.wage[t2] > self.min_toewage:
                    toes += self.timestep
                elif self.include_halftoe and p.wage[t2] >= self.min_halftoewage:
                    toes += 0.5*self.timestep
                nt += 1
            else:
                nt += 1
            t2 -= 1

        return toes

    def check_toe58(self, age: float, tyoura: float, is_spouse: bool = False) -> bool:
        '''
        58 vuotta täyttäneiden lisäpäiväsääntö: toe täyttynyt 58 v täyttämisen jälkeen
        '''
        p = self.person(is_spouse)
        lstate = p.latest_state()

        if age < self.minage_500 or lstate in RET_STATES or self.kassanjasen < 1:
            return False

        n_toe = int(np.floor(self.max_toe/self.timestep))
        t = self.map_age(age)
        t58 = self.map_age(58)

        if lstate in UNEMP_STATES:
            t2 = max(0, p.enimmaisaika-1)
        else:
            t2 = max(0, p.latest)

        toes = 0.0
        nt = 0
        while nt < n_toe and nt < t-t58 and t2 >= 0:
            emps = p.states[t2]
            if emps in TOE_ACCEPTED_STATES:
                pass
            elif emps in EMP_STATES:
                if p.wage[t2] > self.min_toewage:
                    toes += self.timestep
                elif self.include_halftoe and p.wage[t2] >= self.min_halftoewage:
                    toes += 0.5*self.timestep
                nt += 1
            else:
                nt += 1
            t2 -= 1

        return self.tyossaoloehto(toes) and tyoura >= self.tyohistoria_vaatimus500

    #############################
    # muut historiasta lasketut suureet
    #############################

    def comp_studenttime(self, is_spouse: bool = False) -> float:
        '''
        Opiskeluaika neljän viimeisen opiskelun ulkopuolisen vuoden aikana
        '''
        p = self.person(is_spouse)
        studenttime = 0.0
        n_student = int(np.floor(4.0/self.timestep))
        t2 = p.latest
        nt = 0
        while nt < n_student and t2 >= 0:
            if p.states[t2] == 12:
                studenttime += self.timestep
            else:
                nt += 1
            t2 -= 1

        return studenttime

    def comp_infostats(self, age: float, is_spouse: bool = False) -> Tuple[float, float, float, int, int, int]:
        '''
        Työssäoloehto ja työttömyysturvaan vaikuttavat lasten määrät
        '''
        toes, toekesto, wage = self.comp_toe_wage(is_spouse=is_spouse)
        c3, c7, c18 = self.comp_children(age)
        return toes, toekesto, wage, c3, c7, c18

    def comp_5y_ave_wage(self, is_spouse: bool = False, render: bool = False) -> Tuple[float, float]:
        '''
        Työkyvyttömyyseläkkeen tulevan ajan perusteena oleva 5 v keskiansio.
        Palauttaa (keskiansio etuudet mukaan lukien, toteutunut palkkasumma 5 v ajalta)
        '''
        p = self.person(is_spouse)
        lstate = round(p.latest)+1
        n = int(np.ceil(5/self.timestep))
        wage_ave = 0.0
        sum_of_wage = 0.0
        for x in range(max(0, lstate-n), lstate):
            empstate = p.states[x]
            w = 0
            if empstate in EMP_STATES:
                value = p.wage[x]
                w = value
            elif empstate in {5, 6}:
                value = 1.17*p.wagebasis[x] # FIXME!
            elif empstate == 14:
                value = 0.62*p.wagebasis[x] # FIXME!
            elif empstate in UNEMP_STATES:
                value = 0.75*p.unempbasis[x]
            elif empstate in {7, 12, 13}:
                value = self.disabbasis_tmtuki
            else:
                value = 0

            if render:
                print(f'{empstate}: {value:.2f}')

            wage_ave += value*self.timestep/5
            sum_of_wage += w*self.timestep

        return wage_ave, sum_of_wage

    def comp_svpaivaraha_1v(self, is_spouse: bool = False, render: bool = False) -> float:
        '''
        Sairaus- ja vanhempainpäivärahan perustepalkka edeltävän vuoden ajalta (vuositasolla)
        '''
        p = self.person(is_spouse)
        lstate = round(p.latest)+1
        n = int(np.ceil(1/self.timestep))
        wage = 0.0
        for x in range(max(0, lstate-n), lstate):
            empstate = p.states[x]
            if empstate in EMP_STATES:
                value = p.wage[x]
            elif empstate in {5, 6, 14}:
                value = p.wagebasis[x] # FIXME!
            elif empstate in {7, 12, 13}:
                value = self.disabbasis_tmtuki
            elif empstate in UNEMP_STATES:
                value = p.unempbasis[x]
            else:
                value = 0

            if render:
                print(f'{empstate}: {value:.2f}')

            wage += value*self.timestep

        return wage

    def comp_svperuste(self, printti: bool = False, is_spouse: bool = False) -> float:
        '''
        Sairauspäivärahan perustepalkka (ei käytössä)
        '''
        p = self.person(is_spouse)
        n_svp = int(np.floor(1/self.timestep))
        toes = 0.0
        nt = 0
        t2 = max(0, p.enimmaisaika-1)
        while nt < n_svp and t2 >= 0:
            emps = p.states[t2]
            if printti:
                print(f'emps {emps} t2 {t2} toes {toes}')
            if emps in {5, 6, 7, 14}:
                pass
            elif emps in EMP_STATES or emps in UNEMP_STATES:
                if p.wage[t2]:
                    toes += p.wage[t2]
                nt += 1
            else:
                nt += 1
            t2 -= 1

        return toes

    #############################
    # tulostus
    #############################

    def render(self) -> None:
        print(f'main states {self.main.states}')
        print(f'spouse states {self.spouse.states}')

    def print_ages(self, age: float) -> None:
        c3, c7, c18 = self.comp_children(age)
        first = True
        for k in range(self.children_n):
            c_age = age-self.children_date[k]
            if c_age <= 20:
                if first:
                    print(f'age {age:.2f} child {k}/{self.children_n}: age {c_age:.2f} c3: {c3} c7: {c7} c18: {c18}')
                    first = False
                else:
                    print(f'age {age:.2f} child {k}/{self.children_n}: age {c_age:.2f}')