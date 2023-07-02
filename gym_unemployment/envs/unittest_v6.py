class test():

    def __init__()

    def test_life_left(self,cc,n: int=100000):
        npv0=np.zeros(self.n_groups)

        startage=cc.min_age
        for g in range(cc.n_groups):
            cpsum0=1.0
            for x in np.arange(100,startage,-cc.timestep):
                m=cc.mort_intensity[int(np.floor(x)),g]
                cpsum0=m*1+(1-m)*(1+cpsum0) # no discount
            npv0[g]=cpsum0

        medlifeleft=np.zeros(cc.n_groups)
        for g in range(cc.n_groups):
            age=startage
            for k in range(n):
                #age=np.randint(100)
                lifeleft=cc.comp_life_left(g,age)
                medlifeleft[g] += lifeleft
            medlifeleft[g] /= n
            print(f'll {medlifeleft[g]} vs med {npv0[g]}')
