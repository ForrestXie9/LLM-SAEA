import numpy as np

from Surrogate_model.srgtsPRSEvaluate import srgtsPRSEvaluate
from Surrogate_model.srgtsPRSFit import srgtsPRSFit
from Surrogate_model.srgtsPRSSetOptions import srgtsPRSSetOptions
from llm.LLM_score import LLM_score
from sklearn.preprocessing import MinMaxScaler
def PRS_pre_arm(ghx, ghf, offspring, hx, hf, FUN, NFEs, CE, gfs, num_arm, paras):
    # {RBF, prescreening}

    srgtOPT = srgtsPRSSetOptions(ghx, ghf)
    # Fit the polynomial response surface
    srgtSRGT = srgtsPRSFit(srgtOPT)

    fitnessModel = srgtsPRSEvaluate(offspring, srgtSRGT)

    sidx = np.argmin(fitnessModel)  # Get the best point indexs
    candidate_position = offspring[sidx, :]

    ih = np.where((hx == candidate_position).all(axis=1))[0]

    if len(ih) == 0:
        candidate_fit = FUN(candidate_position)  # Evaluation
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
        # Save candidate into dataset, and sort dataset
        hx = np.vstack((hx, candidate_position))
        hf = np.append(hf, candidate_fit)

        # Update CE for plotting
        CE[NFEs-1, :] = [NFEs-1, candidate_fit]
        gfs[NFEs-1] = np.min(CE[0:NFEs, 1])

        # Update the low level arm reward
        Arm = 'PRS_prescreening '
        # reward = reward_function(ghf, hf, candidate_fit, NFEs, Arm)
        if candidate_fit == np.min(hf):
            print(f"Current optimal obtained by {Arm} arm is: {candidate_fit} NFE={NFEs}")
    else:
        reward = 0  # current database has not updated

    return hx, hf, reward, NFEs, CE, gfs
