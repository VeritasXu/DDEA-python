from pyDOE import lhs

######################################################################
# Non-commercial
# project link: git@github.com:VeritasXu/DDEA-python.git
# official link: https://github.com/HandingWang/DDEA-SE.git
######################################################################

def initialize_pop(n, d, lb, ub):
    """
    :param n: number of samples
    :param d: number of the decision variable
    :param lb: lower bound
    :param ub: upper bound
    :return:sample
    """

    result = lhs(d, samples = n)

    POP = result * (ub-lb) + lb

    return POP
