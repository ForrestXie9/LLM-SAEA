import numpy as np

def my_rbfbuild(Xtr, Ytr, bf_type='MQ', bf_c=1, usePolyPart=0, verbose=1):
    Ytr = Ytr.reshape(-1, 1)
    n, d = Xtr.shape
    ny, dy = Ytr.shape

    if n < 2 or d < 1 or ny != n or dy != 1:
        raise ValueError('Wrong training data sizes.')

    if usePolyPart not in [0, 1, 2]:
        raise ValueError('Invalid value for usePolyPart. It must be 0, 1, or 2.')

    model = {}
    model['n'] = n
    model['meanY'] = np.mean(Ytr)
    model['bf_type'] = bf_type
    model['bf_c'] = bf_c
    model['poly'] = usePolyPart

    dist = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            dist[i, j] = np.linalg.norm(Xtr[i, :] - Xtr[j, :])
            dist[j, i] = dist[i, j]

    if bf_type.upper() == 'IMQ':
        dist += bf_c ** 2
        dist = 1 / np.sqrt(dist)
    elif bf_type.upper() == 'CUB':
        dist += bf_c ** 2
        dist **= 3
    elif bf_type.upper() == 'TPS':
        dist **= 2
        dist += bf_c ** 2
        dist = dist * np.log(np.sqrt(dist))
    elif bf_type.upper() == 'G':
        dist = np.exp(-dist ** 2 / (2 * bf_c ** 2))
    else:
        dist += bf_c ** 2
        dist = np.sqrt(dist)

    model['dist'] = dist

    if usePolyPart == 0:
        model['coefs'] = np.linalg.lstsq(dist, Ytr - model['meanY'], rcond=None)[0]
    else:
        Xt = Xtr
        ones_n_1 = np.ones((n, 1))

        # 创建全0矩阵
        zeros_d1_d1 = np.zeros((d + 1, d + 1))

        # 构建 A 矩阵的左上部分
        A_top_left = dist
        A_top_middle = ones_n_1
        A_top_right = Xt

        # 构建 A 矩阵的右上部分
        A_top = np.concatenate((A_top_left, A_top_middle, A_top_right), axis=1)

        # 构建 A 矩阵的左下部分
        A_bottom_left = np.concatenate((ones_n_1.T, Xt.T), axis=0)
        A_bottom_right = zeros_d1_d1
        A_bottom = np.concatenate((A_bottom_left, A_bottom_right), axis=1)
        A = np.concatenate((A_top, A_bottom), axis=0)
        # A = np.block([[dist, np.ones((n, 1)), Xt], [np.ones((n, 1)), Xt.T, np.zeros(((d + 1) * (d + 2) // 2, (d + 1) * (d + 2) // 2))]])
        # 拟合线性模型
        Ytr_zeros = np.zeros((d + 1, 1))
        Y_combined = np.concatenate((Ytr, Ytr_zeros), axis=0)
        coefs, _, _, _ = np.linalg.lstsq(A, Y_combined, rcond=None)

        model['coefs'] = coefs.reshape(-1, 1)

    # if verbose:
    #     print('Execution time: {:.2f} seconds'.format(time))

    return model
