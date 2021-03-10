import numpy as np

######################################################################
# Non-commercial
# project link: git@github.com:VeritasXu/DDEA-python.git
# official link: https://github.com/HandingWang/DDEA-SE.git
######################################################################

def mutation(POP, pm, lb, ub, eta_m =15):
    """
    Routine for real polynomial mutation of an individual
    :param POP: Input Population
    :param pm:  Mutation Probability
    :param lb:  Lower Bound
    :param ub:  Upper Bound
    :param eta_m:
    DATE:       Feb 2020
    """
    N = POP.shape[0]
    C = POP.shape[1]
    NPOP = POP
    N_L = [i for i in range(N)]
    C_L = [i for i in range(C)]

    for i in N_L:
        k = i
        NPOP[i,:] = POP[k, :]

        for j in C_L:
            r1 = np.random.rand()
            if r1 <= pm:
                y = POP[k, j]
                yd, yu = lb[j], ub[j]
                if y > yd:
                    if (y-yd)<(yu-y):
                        delta = (y-yd)/(yu-yd)
                    else:
                        delta = (yu-y)/(yu-yd)

                    r2 = np.random.rand()
                    indi = 1/(eta_m+1)
                    if r2 <= 0.5:
                        xy = 1-delta
                        val = 2 * r2 + (1 - 2 * r2) * np.power(xy, (eta_m + 1))
                        deltaq = np.power(val, indi) -1
                    else:
                        xy = 1- delta
                        val = 2 * (1 - r2) + 2 * (r2 - 0.5) * np.power(xy, (eta_m + 1))
                        deltaq = 1 - np.power(val, indi)

                    y = y + deltaq*(yu-yd)

                    y = min(y,yu)
                    NPOP[i,j] = max(y,yd)
                else:
                    NPOP[i,j] = np.random.rand()*(yu-yd)+yd

    return NPOP
