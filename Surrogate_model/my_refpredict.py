import numpy as np


def my_rbfpredict(model, Xtr, Xq):
    if len(Xq) < 3:
        raise ValueError('Too few input arguments.')

    if model['n'] != Xtr.shape[0]:
        raise ValueError('The matrix Xtr should be the same matrix with which the model was built.')

    nq = Xq.shape[0]
    Yq = np.zeros(nq)

    for t in range(nq):
        # dist = np.zeros(model['n'])
        if model['bf_type'].upper() == 'BH':
            dist = np.sqrt(np.sum((np.tile(Xq[t, :], (model['n'], 1)) - Xtr) ** 2, axis=1))
        elif model['bf_type'].upper() == 'IMQ':
            dist = 1 / np.sqrt(np.sum((np.tile(Xq[t, :], (model['n'], 1)) - Xtr) ** 2, axis=1) + model['bf_c'] ** 2)
        elif model['bf_type'].upper() == 'CUB':
            # bf_c = 0
            dist = np.sqrt(np.sum((np.tile(Xq[t, :], (model['n'], 1)) - Xtr) ** 2, axis=1) + model['bf_c'] ** 2) ** 3
        elif model['bf_type'].upper() == 'TPS':
            dist = np.sum((np.tile(Xq[t, :], (model['n'], 1)) - Xtr) ** 2, axis=1)
            dist = (dist + model['bf_c'] ** 2) * np.log(np.sqrt(dist + model['bf_c'] ** 2))
        elif model['bf_type'].upper() == 'G':
            dist = np.exp(-np.sum((np.tile(Xq[t, :], (model['n'], 1)) - Xtr) ** 2, axis=1) / (2 * model['bf_c'] ** 2))
        else:  # MQ
            dist = np.sqrt(np.sum((np.tile(Xq[t, :], (model['n'], 1)) - Xtr) ** 2, axis=1) + model['bf_c'] ** 2)

        if model['poly'] == 0:
            Yq[t] = model['meanY'] + np.dot(model['coefs'].T, dist)
        elif model['poly'] == 1:
            Yq[t] = np.dot(model['coefs'].T, np.concatenate((dist, [1], Xq[t, :])))
        # elif model['poly'] == 2:
        #     XQ = dace_regpoly2(Xq[t, :])
        #     Yq[t] = np.dot(model['coefs'].T, np.concatenate((dist, XQ)))

    return Yq
