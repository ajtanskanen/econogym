'''

Util functions

'''


def pretty_print(q,q2):
    '''
    Helper function that prettyprints arrays
    '''
    import copy
    for key in q:
        if (q[key]>0 or q[key]<0) and not ('puoliso_' in key):
            key2=copy.copy(key)
            if 'omat_' in key:
                key2=key.replace('omat_','puoliso_')
            if 'puoliso_' in key:
                key2=key.replace('puoliso_','omat_')
            if key in q and key in q2:
                if not math.isclose(q[key],q2[key2]):
                    print(key,': {:.2f} vs {:.2f}, {} {} ({:.2f} vs {:.2f})'.format(q[key],q2[key2],key,key2,q[key2],q2[key],))
            else:
                if key in q:
                    print(key2,' not in q2')
                else:
                    print(key,' not in q')
                    
                    
def check_q(q,num=-1):
    for person in set(['omat_','puoliso_']):
        d1=q[person+'verot']
        d2=q[person+'valtionvero']+q[person+'kunnallisvero']+q[person+'ptel']+q[person+'tyotvakmaksu']+\
            q[person+'ylevero']+q[person+'sairausvakuutusmaksu']
        
        if np.abs(d2-d1)>1e-6:
            print(f'check_q {num}: {person} {d2-d1}')
    
def print_q(a):
    '''
    pretty printer for dict
    '''
    for x in a.keys():
        if a[x]>0 or a[x]<0:
            print('{}:{:.2f} '.format(x,a[x]),end='')
            
    print('')
        