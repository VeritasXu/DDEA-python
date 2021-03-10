import numpy as np

######################################################################
# Non-commercial
# project link: git@github.com:VeritasXu/DDEA-python.git
# official link: https://github.com/HandingWang/DDEA-SE.git
######################################################################

def SBX(POP, pc, lb, ub, eta_c =15):
    """
    Simulated Binary Crossover
    :param POP: Input Population
    :param pc:  Crossover Probability
    :param lb:  Lower Bound
    :param ub:  Upper Bound
    :param eta_c:
    """
    N = POP.shape[0]
    C = POP.shape[1]
    NPOP = np.zeros((2*N, C))
    N_L = [i for i in range(N)]
    C_L = [i for i in range(C)]

    for i in N_L:
        r1 = np.random.rand()
        if r1 <= pc:
            A = np.random.permutation(N_L)
            k = i
            if A[1] < A[0]:
                y = A[1]
            else:
                y = A[0]
            if k == y:
                k = A[2]

            # d = np.sqrt(np.sum(np.linalg.norm(POP[y,:] - POP[k,:])**2))

            if k != y:
                for j in C_L:
                    par1, par2 = POP[y,j], POP[k,j]
                    yd, yu = lb[j], ub[j]
                    r2 = np.random.rand()
                    if r2 <= 0.5:
                        y1 = min(par1,par2)
                        y2 = max(par1,par2)
                        ## y1 和 y2 可能相同
                        if y1 == y2:
                            tmp_diff = 0.0000000001
                        else:
                            tmp_diff = y2 - y1
                        if (y1 - yd) > (yu - y2):
                            beta = 1 + 2 * (yu - y2) / tmp_diff
                        else:
                            beta = 1 + 2 * (y1 - yd) / tmp_diff
                        expp = eta_c+1
                        beta = 1/beta
                        alpha = 2.0 - np.power(beta, expp)

                        r3 = np.random.rand()
                        if r3 <= 1/alpha:
                            alpha = alpha * r3
                            expp = 1/(eta_c + 1.0)
                            betaq = np.power(alpha, expp)
                        else:
                            alpha = 1 / (2.0 - alpha * r3)
                            expp = 1 / (eta_c + 1.0)
                            betaq = np.power(alpha, expp)

                        child1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                        child2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

                        aa = max(child1,yd)
                        bb = max(child2,yd)
                        if np.random.rand() > 0.5:
                            NPOP[2 * i, j] = min(aa, yu)
                            NPOP[2 * i + 1, j] = min(bb, yu)
                        else:
                            NPOP[2 * i + 1, j] = min(aa, yu)
                            NPOP[2 * i, j] = min(bb, yu)
                    else:
                        NPOP[2 * i, j] = par1
                        NPOP[2 * i + 1, j] = par2
    return NPOP
