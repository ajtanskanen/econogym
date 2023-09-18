'''

    wages.py

    rates for moving agents from state to another

'''

import numpy as np
import math
from scipy.interpolate import interp1d

class Wages_v1():
    def __init__(self,year: int=2018,silent: bool=True,max_age: float=70,n_groups: int=6,timestep: float=0.25,min_age: float=18,
                 inv_timestep: float=0,min_retirementage: float=65,min_salary: float=2_000):
        self.year=year
        self.max_age=max_age
        self.min_age=min_age
        self.n_groups=n_groups
        self.timestep=timestep
        self.inv_timestep=inv_timestep
        self.silent=silent
        self.min_retirementage=min_retirementage
        self.min_salary=min_salary
        
        self.setup_salaries()

    def map_age(self,age,start_zero: bool=False):
        if start_zero:
            return int((age)*self.inv_timestep)
        else:
            return int((age-self.min_age)*self.inv_timestep)
    
    def setup_salaries(self):
        # TK:n aineisto vuodelta 2018
        # iät 18-70
        # päivitetty 2022
        
        if self.year==2018:
            palkat_ika_miehet=12.5*np.array([2039.15,2256.56,2339.01,2489.09,2571.40,2632.58,2718.03,2774.21,2884.89,2987.55,3072.40,3198.48,3283.81,3336.51,3437.30,3483.45,3576.67,3623.00,3731.27,3809.58,3853.66,3995.90,4006.16,4028.60,4104.72,4181.51,4134.13,4157.54,4217.15,4165.21,4141.23,4172.14,4121.26,4127.43,4134.00,4093.10,4065.53,4063.17,4085.31,4071.25,4026.50,4031.17,4047.32,4026.96,4028.39,4163.14,4266.42,4488.40,4201.40,4252.15,4443.96,3316.92,3536.03,3536.03])
            palkat_ika_naiset=12.5*np.array([2058.55,2166.68,2223.96,2257.10,2284.57,2365.57,2443.64,2548.35,2648.06,2712.89,2768.83,2831.99,2896.76,2946.37,2963.84,2993.79,3040.83,3090.43,3142.91,3159.91,3226.95,3272.29,3270.97,3297.32,3333.42,3362.99,3381.84,3342.78,3345.25,3360.21,3324.67,3322.28,3326.72,3326.06,3314.82,3303.73,3302.65,3246.03,3244.65,3248.04,3223.94,3211.96,3167.00,3156.29,3175.23,3228.67,3388.39,3457.17,3400.23,3293.52,2967.68,2702.05,2528.84,2528.84])
        elif self.year==2019:
            ind=1.018
            palkat_ika_miehet=ind*12.5*np.array([2339.01,2339.01,2339.01,2489.09,2571.40,2632.58,2718.03,2774.21,2884.89,2987.55,3072.40,3198.48,3283.81,3336.51,3437.30,3483.45,3576.67,3623.00,3731.27,3809.58,3853.66,3995.90,4006.16,4028.60,4104.72,4181.51,4134.13,4157.54,4217.15,4165.21,4141.23,4172.14,4121.26,4127.43,4134.00,4093.10,4065.53,4063.17,4085.31,4071.25,4026.50,4031.17,4047.32,4026.96,4028.39,4163.14,4266.42,4488.40,4201.40,4252.15,4443.96,3316.92,3536.03,3536.03])
            palkat_ika_naiset=ind*12.5*np.array([2223.96,2223.96,2223.96,2257.10,2284.57,2365.57,2443.64,2548.35,2648.06,2712.89,2768.83,2831.99,2896.76,2946.37,2963.84,2993.79,3040.83,3090.43,3142.91,3159.91,3226.95,3272.29,3270.97,3297.32,3333.42,3362.99,3381.84,3342.78,3345.25,3360.21,3324.67,3322.28,3326.72,3326.06,3314.82,3303.73,3302.65,3246.03,3244.65,3248.04,3223.94,3211.96,3167.00,3156.29,3175.23,3228.67,3388.39,3457.17,3400.23,3293.52,2967.68,2702.05,2528.84,2528.84])
        elif self.year==2020:
            ind=1.018*1.0199
            palkat_ika_miehet=ind*12.5*np.array([2339.01,2339.01,2339.01,2489.09,2571.40,2632.58,2718.03,2774.21,2884.89,2987.55,3072.40,3198.48,3283.81,3336.51,3437.30,3483.45,3576.67,3623.00,3731.27,3809.58,3853.66,3995.90,4006.16,4028.60,4104.72,4181.51,4134.13,4157.54,4217.15,4165.21,4141.23,4172.14,4121.26,4127.43,4134.00,4093.10,4065.53,4063.17,4085.31,4071.25,4026.50,4031.17,4047.32,4026.96,4028.39,4163.14,4266.42,4488.40,4201.40,4252.15,4443.96,3316.92,3536.03,3536.03])
            palkat_ika_naiset=ind*12.5*np.array([2223.96,2223.96,2223.96,2257.10,2284.57,2365.57,2443.64,2548.35,2648.06,2712.89,2768.83,2831.99,2896.76,2946.37,2963.84,2993.79,3040.83,3090.43,3142.91,3159.91,3226.95,3272.29,3270.97,3297.32,3333.42,3362.99,3381.84,3342.78,3345.25,3360.21,3324.67,3322.28,3326.72,3326.06,3314.82,3303.73,3302.65,3246.03,3244.65,3248.04,3223.94,3211.96,3167.00,3156.29,3175.23,3228.67,3388.39,3457.17,3400.23,3293.52,2967.68,2702.05,2528.84,2528.84])
        elif self.year==2021:
            ind=1.018*1.0199*1.0216
            palkat_ika_miehet=ind*12.5*np.array([2339.01,2339.01,2339.01,2489.09,2571.40,2632.58,2718.03,2774.21,2884.89,2987.55,3072.40,3198.48,3283.81,3336.51,3437.30,3483.45,3576.67,3623.00,3731.27,3809.58,3853.66,3995.90,4006.16,4028.60,4104.72,4181.51,4134.13,4157.54,4217.15,4165.21,4141.23,4172.14,4121.26,4127.43,4134.00,4093.10,4065.53,4063.17,4085.31,4071.25,4026.50,4031.17,4047.32,4026.96,4028.39,4163.14,4266.42,4488.40,4201.40,4252.15,4443.96,3316.92,3536.03,3536.03])
            palkat_ika_naiset=ind*12.5*np.array([2223.96,2223.96,2223.96,2257.10,2284.57,2365.57,2443.64,2548.35,2648.06,2712.89,2768.83,2831.99,2896.76,2946.37,2963.84,2993.79,3040.83,3090.43,3142.91,3159.91,3226.95,3272.29,3270.97,3297.32,3333.42,3362.99,3381.84,3342.78,3345.25,3360.21,3324.67,3322.28,3326.72,3326.06,3314.82,3303.73,3302.65,3246.03,3244.65,3248.04,3223.94,3211.96,3167.00,3156.29,3175.23,3228.67,3388.39,3457.17,3400.23,3293.52,2967.68,2702.05,2528.84,2528.84])
        elif self.year==2022:
            ind=1.018*1.0199*1.0216*1.0232
            palkat_ika_miehet=ind*12.5*np.array([2339.01,2339.01,2339.01,2489.09,2571.40,2632.58,2718.03,2774.21,2884.89,2987.55,3072.40,3198.48,3283.81,3336.51,3437.30,3483.45,3576.67,3623.00,3731.27,3809.58,3853.66,3995.90,4006.16,4028.60,4104.72,4181.51,4134.13,4157.54,4217.15,4165.21,4141.23,4172.14,4121.26,4127.43,4134.00,4093.10,4065.53,4063.17,4085.31,4071.25,4026.50,4031.17,4047.32,4026.96,4028.39,4163.14,4266.42,4488.40,4201.40,4252.15,4443.96,3316.92,3536.03,3536.03])
            palkat_ika_naiset=ind*12.5*np.array([2223.96,2223.96,2223.96,2257.10,2284.57,2365.57,2443.64,2548.35,2648.06,2712.89,2768.83,2831.99,2896.76,2946.37,2963.84,2993.79,3040.83,3090.43,3142.91,3159.91,3226.95,3272.29,3270.97,3297.32,3333.42,3362.99,3381.84,3342.78,3345.25,3360.21,3324.67,3322.28,3326.72,3326.06,3314.82,3303.73,3302.65,3246.03,3244.65,3248.04,3223.94,3211.96,3167.00,3156.29,3175.23,3228.67,3388.39,3457.17,3400.23,3293.52,2967.68,2702.05,2528.84,2528.84])
        elif self.year==2023:
            ind=1.018*1.0199*1.0216*1.0232*1.0315
            palkat_ika_miehet=ind*12.5*np.array([2339.01,2339.01,2339.01,2489.09,2571.40,2632.58,2718.03,2774.21,2884.89,2987.55,3072.40,3198.48,3283.81,3336.51,3437.30,3483.45,3576.67,3623.00,3731.27,3809.58,3853.66,3995.90,4006.16,4028.60,4104.72,4181.51,4134.13,4157.54,4217.15,4165.21,4141.23,4172.14,4121.26,4127.43,4134.00,4093.10,4065.53,4063.17,4085.31,4071.25,4026.50,4031.17,4047.32,4026.96,4028.39,4163.14,4266.42,4488.40,4201.40,4252.15,4443.96,3316.92,3536.03,3536.03])
            palkat_ika_naiset=ind*12.5*np.array([2223.96,2223.96,2223.96,2257.10,2284.57,2365.57,2443.64,2548.35,2648.06,2712.89,2768.83,2831.99,2896.76,2946.37,2963.84,2993.79,3040.83,3090.43,3142.91,3159.91,3226.95,3272.29,3270.97,3297.32,3333.42,3362.99,3381.84,3342.78,3345.25,3360.21,3324.67,3322.28,3326.72,3326.06,3314.82,3303.73,3302.65,3246.03,3244.65,3248.04,3223.94,3211.96,3167.00,3156.29,3175.23,3228.67,3388.39,3457.17,3400.23,3293.52,2967.68,2702.05,2528.84,2528.84])
        elif self.year==2024:
            ind=1.018*1.0199*1.0216*1.0232*1.0315*1.024
            palkat_ika_miehet=ind*12.5*np.array([2339.01,2339.01,2339.01,2489.09,2571.40,2632.58,2718.03,2774.21,2884.89,2987.55,3072.40,3198.48,3283.81,3336.51,3437.30,3483.45,3576.67,3623.00,3731.27,3809.58,3853.66,3995.90,4006.16,4028.60,4104.72,4181.51,4134.13,4157.54,4217.15,4165.21,4141.23,4172.14,4121.26,4127.43,4134.00,4093.10,4065.53,4063.17,4085.31,4071.25,4026.50,4031.17,4047.32,4026.96,4028.39,4163.14,4266.42,4488.40,4201.40,4252.15,4443.96,3316.92,3536.03,3536.03])
            palkat_ika_naiset=ind*12.5*np.array([2223.96,2223.96,2223.96,2257.10,2284.57,2365.57,2443.64,2548.35,2648.06,2712.89,2768.83,2831.99,2896.76,2946.37,2963.84,2993.79,3040.83,3090.43,3142.91,3159.91,3226.95,3272.29,3270.97,3297.32,3333.42,3362.99,3381.84,3342.78,3345.25,3360.21,3324.67,3322.28,3326.72,3326.06,3314.82,3303.73,3302.65,3246.03,3244.65,3248.04,3223.94,3211.96,3167.00,3156.29,3175.23,3228.67,3388.39,3457.17,3400.23,3293.52,2967.68,2702.05,2528.84,2528.84])
        else:
            print('Unknown year ',self.year)
            error(1001)
            
        def map_age18(x):
            return int(x-18)
            
        def ifunc(palkat,x1,x2):
            x = np.linspace(18, 72, num=54, endpoint=True)
            f = interp1d(x, palkat, fill_value='extrapolate')
            n_time = int(np.round((x2-x1)*self.inv_timestep))+2
            palkat_x = np.linspace(x1, x2, num=n_time, endpoint=True)
        
            return f(palkat_x)        

        def gfunc(palkat,x1,x2):
            x = np.linspace(18, 73, num=55, endpoint=True)
            n_time = int(np.round((x2-x1)*self.inv_timestep))+2
            palkat_x = np.linspace(x1, x2, num=n_time, endpoint=True)
            g=np.zeros((n_time,3))
            for k in range(3):
                f = interp1d(x, palkat[:,k],fill_value='extrapolate')
                g[:,k]=f(palkat_x)
        
            return g
        
        # filtteri
        m_age=int(self.min_retirementage-1)
        palkat_miehet_ve=palkat_ika_miehet[map_age18(m_age)]
        palkat_ika_miehet[map_age18(m_age):]=np.minimum(palkat_ika_miehet[map_age18(m_age):],palkat_miehet_ve)
        palkat_naiset_ve=palkat_ika_naiset[map_age18(m_age)]
        palkat_ika_naiset[map_age18(m_age):]=np.minimum(palkat_ika_naiset[map_age18(m_age):],palkat_naiset_ve)

        self.palkat_ika_miehet=ifunc(palkat_ika_miehet,self.min_age,self.max_age)
        self.palkat_ika_naiset=ifunc(palkat_ika_naiset,self.min_age,self.max_age)
        
        g_r=np.array([[0.81436,1.00992,1.2469],[0.81436,1.00992,1.2469],[0.81436,1.00992,1.2469],[0.8144725,1.011105,1.2677425],[0.814585,1.01229,1.288585],[0.8146975,1.013475,1.3094275],[0.81481,1.01466,1.33027],[0.81481,1.01466,1.33027],[0.798635,1.0116225,1.34756],[0.78246,1.008585,1.36485],[0.766285,1.0055475,1.38214],[0.75011,1.00251,1.39943],[0.75011,1.00251,1.39943],[0.74420478,0.99822499,1.40395702],[0.73829957,0.99393997,1.40848403],[0.73239435,0.98965496,1.41301105],[0.72648913,0.98536995,1.41753806],[0.72058391,0.98108493,1.42206508],[0.7146787,0.97679992,1.4265921],[0.70877348,0.9725149,1.43111911],[0.70286826,0.96822989,1.43564613],[0.69696304,0.96394488,1.44017314],[0.69105783,0.95965986,1.44470016],[0.68515261,0.95537485,1.44922718],[0.67924739,0.95108984,1.45375419],[0.67334218,0.94680482,1.45828121],[0.66743696,0.94251981,1.46280822],[0.66153174,0.93823479,1.46733524],[0.65562652,0.93394978,1.47186226],[0.64972131,0.92966477,1.47638927],[0.64381609,0.92537975,1.48091629],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433],[0.63791087,0.92109474,1.4854433]])
        self.g_r=gfunc(g_r,self.min_age,self.max_age)
        
        return palkat_ika_miehet,palkat_ika_naiset,g_r
        
    
    def wage_process_TK_v4(self,w: float,a0: float=3300*12,a1: float=3300*12,g: int=1):
        '''
        Palkkaprosessi muokattu lähteestä Määttänen, 2013 
        '''
        group_sigmas=np.array([0.05,0.07,0.10])*math.sqrt(self.timestep)
        sigma=group_sigmas[g]
        eps=np.random.normal(loc=0,scale=sigma,size=1)[0]
        c1=0.89**self.timestep
        if w>0:
             # pidetään keskiarvo/a1 samana kuin w/a0
            wt=a1*math.exp(c1*np.log(w/a0)+eps-0.5*sigma*sigma)
        else:
            wt=a1*math.exp(eps)

        # täysiaikainen vuositulo vähintään self.min_salary
        wt=max(self.min_salary,wt)

        return wt
                
    def compute_salary(self,group: int=1,debug: bool=False,initial_salary: float=None):
        '''
        Alussa ajettava funktio, joka tekee palkat yhtä episodia varten
        '''
        n_time = int(np.round((self.max_age-self.min_age)*self.inv_timestep))+2
        salary=np.zeros(n_time)

        if group>2: # naiset
            r=self.g_r[0,group-3]
            if initial_salary is not None:
                a0=initial_salary
            else:
                a0=self.palkat_ika_naiset[0]*r
            
            a1=self.palkat_ika_naiset[0]*r/5
            salary[0]=np.maximum(self.min_salary,np.random.normal(loc=a0,scale=a1,size=1)[0]) # e/y

            k=0
            r0=self.g_r[0,group-3]
            a0=self.palkat_ika_naiset[0]*r0
            s0=salary[0]
            for age in np.arange(self.min_age,self.max_age,self.timestep):
                r1=self.g_r[k,group-3]
                a1=self.palkat_ika_naiset[k]*r1
                salary[self.map_age(age)]=self.wage_process_TK_v4(s0,a0,a1,g=group-3)
                k=k+1
                s0=salary[self.map_age(age)]
                a0=a1
                r0=r1
        else: # miehet
            r=self.g_r[0,group]
            if initial_salary is not None:
                a0=initial_salary
            else:
                a0=self.palkat_ika_miehet[0]*r
                
            a1=self.palkat_ika_miehet[0]*r/5
            salary[0]=np.maximum(self.min_salary,np.random.normal(loc=a0,scale=a1,size=1)[0]) # e/y

            k=0
            r0=self.g_r[0,group]
            a0=self.palkat_ika_miehet[0]*r0
            s0=salary[0]
            for age in np.arange(self.min_age,self.max_age,self.timestep):
                r1=self.g_r[k,group]
                a1=self.palkat_ika_miehet[k]*r1
                salary[self.map_age(age)]=self.wage_process_TK_v4(s0,a0,a1,g=group)
                k=k+1
                s0=salary[self.map_age(age)]
                a0=a1
                r0=r1
                
        self.salary=salary

    def get_wage(self,age: float,reduction: float):
        '''
        palkka age-ikäiselle time_in_state-vähennyksellä työllistymispalkkaan, step-kohtaisesti, ei vuosikohtaisesti
        '''
        intage=self.map_age(age)
        if age<self.max_age and age>=self.min_age-1:
            return np.maximum(self.min_salary,self.salary[intage])*max(0,(1-reduction))
        else:
            return 0
            
    def get_potential_wage(self,age: float):
        '''
        palkka age-ikäiselle ILMAN time_in_state-vähennystä työllistymispalkkaan, step-kohtaisesti, ei vuosikohtaisesti
        '''
        intage=self.map_age(age)
        if age<self.max_age and age>=self.min_age-1:
            return np.maximum(self.min_salary,self.salary[intage])
        else:
            return 0
            
            