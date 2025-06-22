import numpy as np
from pyDOE import lhs
from Configurations.DE_operator import DEoperator
from Configurations.RBF_pre_arm import RBF_pre_arm
from Configurations.GP_lcb_arm import GP_lcb_arm
from Configurations.RBF_ls_arm import RBF_ls_arm
from Configurations.GP_EI_arm import GP_EI_arm
from Configurations.PRS_pre_arm import PRS_pre_arm
from Configurations.PRS_ls_arm import PRS_ls_arm
from Configurations.GBC_eoi_arm import KNN_eoi_arm
from Configurations.GBC_eor_arm import KNN_eor_arm
from llm.LLM_DAC import LLM_DAC
from llm.LLM_selection import LLM_selection


class LLM_SAEA_Code(object):
    def __init__(self, maxFEs, FUN, dim, LB, UB, paras):
        self.paras = paras
        self.maxFEs = maxFEs
        self.NFEs = 0
        self.popsize = 100
        self.initial_sample_size = 100

        self.F = 0.5
        self.CR = 0.9
        self.apha = 2.5
        self.level = 5


        self.dim = dim
        self.cxmin = np.array(LB)
        self.cxmax = np.array(UB)
        self.bound = self.cxmax - self.cxmin
        self.FUN = FUN

        self.database = None
        self.hx = None
        self.hf = None
        self.gen = None

        self.CE = np.zeros((maxFEs, 2))
        self.gfs = np.zeros(maxFEs)
        self.VRmin = np.tile(LB, (self.popsize, 1))
        self.VRmax = np.tile(UB, (self.popsize, 1))

        self.id = 0
        self.rbf_model = []
        self.gp_model = []
        self.knn_model = []
        self.prs_model = []

        self.Save_rp = []
        self.Save_rl = []
        self.Save_gl = []
        self.Save_ge = []
        self.Save_pp = []
        self.Save_pl = []
        self.Save_Gi = []
        self.Save_Go = []

        self.Succ_rp = 0
        self.Succ_rl = 0
        self.Succ_gl = 0
        self.Succ_ge = 0
        self.Succ_pp = 0
        self.Succ_pl = 0
        self.Succ_Gi = 0
        self.Succ_Go = 0



        self.num_arm = 8
        self.Succ_rate = np.zeros(self.num_arm)
        in1 = np.random.permutation(self.num_arm)

        self.q_value_m = np.zeros(self.num_arm).tolist()
        self.Num = np.zeros(self.num_arm).tolist()

    def initPop(self):
        sam = np.tile(self.cxmin, (self.initial_sample_size, 1)) + (np.tile(self.cxmax, (self.initial_sample_size, 1)) - np.tile(self.cxmin, (self.initial_sample_size, 1))) * lhs(self.dim, samples=self.initial_sample_size, criterion='center')
        fitness = np.zeros((self.initial_sample_size))
        for i in range(self.initial_sample_size):
            fitness[i] = self.FUN(sam[i, :])
            self.CE[self.NFEs, :] = [self.NFEs , self.FUN(sam[i, :])]
            self.NFEs += 1
            self.gfs[i] = np.min(self.CE[0:self.NFEs, 1])


        self.database = [sam,  fitness]
        self.hx = sam
        self.hf = fitness

    def update(self, index, reward):
        index2 = index - 1
        average_re = self.q_value_m[index2]
        reward = reward
        new_re = (average_re * self.Num[index2] + reward)/(self.Num[index2] + 1)
        self.q_value_m[index2] = new_re
        self.Num[index2] = self.Num[index2] + 1


    def run(self):
        if self.database is None:
            self.initPop()

        while self.NFEs < self.maxFEs:
            self.id += 1
            sort_index = np.argsort(self.hf)
            ghf = self.hf[sort_index[:self.initial_sample_size]]
            ghx = self.hx[sort_index[:self.initial_sample_size]]
            se = LLM_selection(self.q_value_m, self.Num, self.id, self.maxFEs-self.popsize, self.paras)

            index1 = se.selection()
            print(index1)
            num_index = len(index1)
            np.random.shuffle(index1)

            for ii in range(num_index):
                index = index1[ii]
                if index == 1:
                    offspring = DEoperator(ghx, self.initial_sample_size, self.dim, ghx, self.F, self.CR, self.VRmax,self. VRmin)
                    self.hx, self.hf, reward_rp, self.NFEs, self.CE, self.gfs = RBF_pre_arm(ghx, ghf, offspring, self.hx, self.hf, self.FUN, self.NFEs, self.CE, self.gfs, self.num_arm,   self.paras)

                    self.update(index, reward_rp)

                elif index == 2:
                    offspring = DEoperator(ghx, self.initial_sample_size, self.dim, ghx, self.F, self.CR, self.VRmax,self. VRmin)
                    self.hx, self.hf, reward_gl, self.NFEs, self.CE, self.gfs = GP_lcb_arm(ghx, ghf, offspring, self.hx, self.hf, self.FUN, self.NFEs, self.CE, self.gfs, self.num_arm,  self.paras)

                    self.update(index, reward_gl)

                elif index == 3:
                    self.hx, self.hf, reward_rl, self.NFEs, self.CE, self.gfs  = RBF_ls_arm(ghx, ghf, self.hx, self.hf, self.FUN, self.NFEs, self.CE, self.gfs, self.cxmin, self.cxmax, self.num_arm,  self.paras)

                    self.update(index, reward_rl)

                elif index == 4:
                    offspring = DEoperator(ghx, self.initial_sample_size, self.dim, ghx, self.F, self.CR, self.VRmax,self. VRmin)
                    self.hx, self.hf, reward_ge, self.NFEs, self.CE, self.gfs = GP_EI_arm(ghx, ghf, offspring, self.hx, self.hf, self.FUN, self.NFEs, self.CE, self.gfs, self.num_arm,   self.paras)

                    self.update(index, reward_ge)

                elif index == 5:
                    offspring = DEoperator(ghx, self.initial_sample_size, self.dim, ghx, self.F, self.CR, self.VRmax, self.VRmin)
                    self.hx, self.hf, reward_pp, self.NFEs, self.CE, self.gfs  = PRS_pre_arm(ghx, ghf, offspring,  self.hx, self.hf, self.FUN, self.NFEs, self.CE, self.gfs, self.num_arm,   self.paras)

                    self.update(index, reward_pp)

                elif index == 6:
                    self.hx, self.hf, reward_pl, self.NFEs, self.CE, self.gfs = PRS_ls_arm(ghx, ghf, self.hx, self.hf, self.FUN, self.NFEs, self.CE, self.gfs, self.cxmin, self.cxmax, self.num_arm,   self.paras)

                    self.update(index, reward_pl)

                elif index == 7:
                    offspring = DEoperator(ghx, self.initial_sample_size, self.dim, ghx, self.F, self.CR, self.VRmax, self.VRmin)
                    self.hx, self.hf, reward_Gi, self.NFEs, self.CE, self.gfs = KNN_eoi_arm(ghx, ghf, offspring, self.hx, self.hf, self.FUN, self.NFEs, self.level, self.CE, self.gfs, self.num_arm,   self.paras)

                    self.update(index, reward_Gi)

                elif index == 8:
                    offspring = DEoperator(ghx, self.initial_sample_size, self.dim, ghx, self.F, self.CR, self.VRmax, self.VRmin)
                    self.hx, self.hf, reward_Go, self.NFEs, self.CE, self.gfs = KNN_eor_arm(ghx, ghf, offspring, self.hx, self.hf, self.FUN, self.NFEs, self.level, self.CE, self.gfs, self.num_arm,   self.paras)

                    self.update(index, reward_Go)

                if self.hf[-1] == np.min(self.hf):

                    break


                if self.NFEs >= 1000:
                    break


        return self.hf, self.maxFEs, self.gfs

