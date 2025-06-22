from sklearn.neighbors import KNeighborsClassifier
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from llm.LLM_score import LLM_score

from sklearn.preprocessing import MinMaxScaler

def KNN_eor_arm(ghx, ghf, offspring, hx, hf, FUN, NFEs, level, CE, gfs, num_arm, paras):
    # Knn, L1-exploitation

    sidx = np.arange(1, len(ghf) + 1)  # ghx and ghf have been sorted
    train_label1 = np.ceil(sidx * level / len(ghf)).astype(int)  # Obtain the label for classifier
    Parents_L1 = ghx[np.where(train_label1 == 1)[0], :]

    mdl = KNeighborsClassifier(n_neighbors=1, metric='minkowski', p=2)  # knn classifier

    mdl.fit(ghx, train_label1)
    # mdl = GradientBoostingClassifier().fit(ghx, train_label1)
    # the prediction of knn
    label = mdl.predict(offspring.reshape(-1, ghx.shape[1]))  # rank 1
    select_pp = np.where(label == label.min())[0]
    dist = np.zeros((len(select_pp), Parents_L1.shape[0]))
    for ii3 in range(len(select_pp)):
        for j in range(Parents_L1.shape[0]):
            dist[ii3, j] = np.sqrt(np.sum((offspring[select_pp[ii3], :].reshape(1,-1) - Parents_L1[j, :].reshape(1,-1)) ** 2, axis=1))
    min_dist = np.min(dist, axis=1)
    in_ = np.argmax(min_dist)
    candidate_position = offspring[select_pp[in_], :]
    ih = np.where(np.all(hx == candidate_position, axis=1))[0]
    if len(ih) == 0:
        candidate_fit = FUN(candidate_position)
        NFEs += 1


        elite_fit = ghf
        elite_x = ghx
        min_value = np.min(elite_fit)
        mean_value = np.mean(elite_fit)
        max_value = np.max(elite_fit)
        hf_sum = np.append(elite_fit, candidate_fit)
        index = np.argsort(hf_sum)
        candidate_rank = (np.where(index == (len(hf_sum) - 1))[0][0]) + 1
        dim = elite_x.shape[1]
        num = elite_x.shape[0]
        ##################
        so = LLM_score(min_value, mean_value, max_value, candidate_fit, candidate_rank, num, dim, num_arm,
                       paras)
        scoring = so.Score()
        reward = scoring[0]
        # print(scoring)
        ######################
        # save candidate into dataset, and sort dataset
        hx = np.vstack((hx, candidate_position))
        hf = np.append(hf, candidate_fit)

        # Update CE for plotting
        CE[NFEs - 1, :] = [NFEs - 1, candidate_fit]
        gfs[NFEs-1] = np.min(CE[0:NFEs, 1])

        # update the low level arm reward
        Arm = 'KNN_L1-exploration '
        # reward = reward_function(ghf, hf, candidate_fit, NFEs, Arm)
        if candidate_fit == np.min(hf):
            print(f"Current optimal obtained by {Arm} arm is: {candidate_fit} NFE={NFEs}")

    else:
        reward = 0  # current database has not update

    return hx, hf, reward, NFEs, CE, gfs