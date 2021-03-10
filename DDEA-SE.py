import copy
import time
import numpy as np
from datetime import datetime
from Operation.SBX import SBX
from utils.config import args_parser
from Operation.mutation import mutation
from utils.bootstrap import index_bootstrap
from utils.data_generate import init_samples
from Surrogate.RBF import RBFNet, rbf_predict
from Operation.model_select import Model_Selection
from Operation.initialize_pop import initialize_pop


######################################################################
# Non-commercial
# project link: git@github.com:VeritasXu/DDEA-python.git
# official link: https://github.com/HandingWang/DDEA-SE.git
######################################################################

if __name__ == '__main__':
    # parse args
    args = args_parser()

    now = datetime.now()
    clock = 100 * (now.year + now.month + now.day + now.hour + now.minute + now.second)
    np.random.seed(clock)

    if args.func_name == 'Ellipsoid':
        from OF.Ellipsoid import Ellipsoid as Function
    elif args.func_name == 'Ackley':
        from OF.Ackley import Rastrigin as Function
    else:
        Function = None
        print('Error in Test Function')

    num_T = args.num_users
    num_Q = args.num_model
    num_data = args.num_b4_d*args.d
    boot_prob = args.boot_prob
    # dimension of decision variable
    d = args.d
    best_pop = []
    ub = args.ub
    multi_ub = ub * np.ones(d)
    lb = args.lb
    multi_lb = lb * np.ones(d)
    num_pop = args.num_pop
    # initial pop, sbx pop, mutation pop
    t_num_pop = num_pop * 4
    # generate first generations
    pop = initialize_pop(num_pop, d, lb, ub)
    pops_fitness = np.zeros((t_num_pop, num_T))

    train_x, train_y = init_samples(func=Function,
                                    num_b4_d=args.num_b4_d,
                                    d=d, xlb=lb, xub=ub)

    weight_locals, center_locals, bias_locals, spread_locals = [], [], [], []

    record_best = []

    t0 = time.time()

    for e in range(args.epochs):
        print('Round #', e)
        m = max(num_T, 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        t1 = time.time()
        if e == 0:
            for idx in idxs_users:
                data_index = index_bootstrap(num_data, boot_prob)
                RBF = RBFNet(k = d)
                local_c, local_w, local_b, local_s = RBF.local_update(train_x[data_index], train_y[data_index])
                center_locals.append(local_c)
                weight_locals.append(local_w)
                bias_locals.append(local_b)
                spread_locals.append(local_s)

            t2 = time.time()
            print('train time: ', t2-t1)
        tmp_fitness = np.zeros((num_pop, num_Q))

        # model management
        if e != 0:
            #model selection
            model_index = Model_Selection(center_locals, weight_locals, bias_locals, spread_locals, best_pop,
                                          num_set=num_Q)
            print('selected models: ', model_index)
            for _i in range(num_Q):
                _ = model_index[_i]
                tmp_fitness[:, _i] = rbf_predict(center_locals[_], weight_locals[_], bias_locals[_], spread_locals[_],
                                                 pop).flatten()
            t5 = time.time()
        # SBX
        sbx_pops = SBX(copy.deepcopy(pop), pc = 1, lb=multi_lb, ub=multi_ub, eta_c=15)

        # mutations
        mutated_pops = mutation(copy.deepcopy(pop), pm = 1/d, lb = multi_ub, ub = multi_ub, eta_m=15)

        new_pop = np.vstack((pop, sbx_pops, mutated_pops))

        t7 = time.time()
        # step 1: find best solution x_b
        for i in range(0, num_T):
            pops_fitness[:,i] = rbf_predict(center_locals[i], weight_locals[i], bias_locals[i],spread_locals[i], new_pop).flatten()
        t8 = time.time()
        print('T predict time: ', t8-t7)
        # calculate the mean fitness for each row
        YAVE = np.mean(pops_fitness, axis=1)
        sorted_fit_index = np.argsort(YAVE)
        pop = new_pop[sorted_fit_index[0:100], :]
        best_pop = pop[0, :].reshape(-1, d)
        best_fit = Function(best_pop.flatten())
        # print('best',best_pop.flatten())
        print('Best fitness: ',best_fit)
        record_best.append(np.hstack((best_pop.flatten(),best_fit)))

    t_final = time.time()
    print('Elapsed time: ', t_final-t0,' seconds')