import random
import numpy as np
from Surrogate.RBF import rbf_predict

######################################################################
# Non-commercial
# project link: git@github.com:VeritasXu/DDEA-python.git
# official link: https://github.com/HandingWang/DDEA-SE.git
######################################################################



def Model_Selection(centers, weights, biases, spreads, best_x, num_set):

    num_models = len(centers)

    results = np.zeros((num_models, 1))

    selected_index = []

    for i in range(0, num_models):
        c, w, b, s= centers[i], weights[i], biases[i], spreads[i]
        results[i] = rbf_predict(c, w, b, s, best_x)

    gap = int(num_models/num_set)


    #sort the results list and get index
    # please mind that:
    # x = numpy.array([1.48,1.31,0.0,0.8])
    # print(x.argsort())
    # [2 3 1 0]
    # means that: the 0th element of the sorted array is the 2nd element of the unsorted array. Not error!
    results = np.asarray(results).flatten()
    sorted_index = np.argsort(results)

    #random select num_set models
    for j in range(0, num_set):
        selected_index.append(random.sample([k for k in sorted_index[j*gap:(j+1)*gap]],1)[0])
    # print(selected_index)
    return selected_index





