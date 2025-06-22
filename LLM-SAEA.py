import numpy as np
import time

from llm.getParas  import Paras
from Benchmarkss.variable_domain import variable_domain
import os
import opfunu_v3

from LLM_SAEA_Code import LLM_SAEA_Code

def RUN_LLM_SAEA(MaxFEs, runs, D, FUN, fun_name, LB, UB, f_bias, paras):
    time_begin = time.time()
    np.seterr(all='ignore')  # 忽略警告


    gsamp1 = np.zeros((runs, MaxFEs))  # 初始化存储每次运行结果的数组

    for r in range(runs):
        # 主循环
        print('\n')
        print('FUNCTION:', fun_name, 'RUN:', r + 1)
        print('\n')
        opt = LLM_SAEA_Code(MaxFEs, FUN, D, LB, UB, paras)
        hisf, mf, gfs = opt.run()
        print('Best fitness:', min(hisf))
        gsamp1[r, :] = gfs[:mf]

    # 计算输出结果
    samp_mean = np.mean(gsamp1[:, -1])
    samp_mean_error = samp_mean - f_bias
    samp_median = np.median(gsamp1[:, -1])
    std_samp = np.std(gsamp1[:, -1])
    gsamp1_ave = np.mean(gsamp1, axis=0)

    # 时间复杂度
    os.makedirs('./result', exist_ok=True)
    time_cost = time.time() - time_begin
    last_value = [samp_mean, samp_mean_error, std_samp, time_cost]
    # np.save(f"result/NFE{mf}_{fun_name}_runs={runs}_Dim={D}", gsamp1)
    np.savetxt(f"result/NFE{mf}_{fun_name}_runs={runs}_Dim={D}.txt", gsamp1)
    # np.savetxt('./result/%s.txt' % fun_name, last_value)  # 均值与方差
    np.savetxt(f"./result/{fun_name}_runs={runs}_Dim={D}.txt", last_value)


if __name__ == "__main__":

    TestFuns = ['F12005', 'F22005', 'F32005', 'F42005', 'F52005', 'F62005', 'F72005',
                'F82005', 'F92005', 'F102005', 'F112005', 'F122005', 'F132005',
                'F142005', 'F152005']

    TestFuns2 = ['ELLIPSOID', 'ROSENBROCK', 'ACKLEY', 'GRIEWANK', 'RASTRIGIN']


    def Ackley(p):  # 32.768 f789  f12
        p = p[np.newaxis, :]
        N, D = p.shape
        tmp1 = np.sqrt(np.sum(np.square(p), axis=1) / D)
        tmp2 = np.sum(np.cos(2 * np.pi * p), axis=1)
        y = -20 * np.exp(-0.2 * tmp1) - np.exp(tmp2 / D) + 20 + np.exp(1)
        return y.item()


    def Griewank(p):
        p = p[np.newaxis, :]  # 600 f101112
        N, D = p.shape
        tmp1 = np.sum(np.square(p), axis=1) / 4000
        tmp2 = np.cos(p / np.sqrt(np.arange(0, D) + 1))
        y = 1 + tmp1 - np.prod(tmp2, axis=1)
        return y.item()


    def Ellipsoid(p):
        p = p[np.newaxis, :]
        N, D = p.shape
        for i in range(D):
            y = np.sum((i + 1) * p ** 2, axis=1)

        return y.item()


    def Rosenbrock(p):
        p = p[np.newaxis, :]
        N, D = p.shape  # 2.048 f456
        y = np.sum(100 * np.square(p[:, 1:] - np.square(p[:, :-1])) + np.square(p[:, :-1] - 1), axis=1)
        return y.item()


    def rastrigin(p):
        p = p[np.newaxis, :]
        N, D = p.shape
        f = np.zeros(N)
        for j in range(N):
            y = p[j, :]
            s = 0
            for i in range(D):
                s += (y[i] ** 2 - 10 * np.cos(2 * np.pi * y[i]) + 10)
            f[j] = s
        return f.item()


    functions1 = [Ellipsoid, Rosenbrock, Ackley, Griewank, rastrigin]




    ####llm
    paras = Paras()

    # Set parameters #
    paras.set_paras(llm_api_endpoint="********",  # Please add your llm_api_endpoint
                    llm_api_key="********",   # Please add your key
                    llm_model="********",
                    exp_debug_mode=False)


    #####
    dims = [10, 30]  # Dimensions
    Runs = 20  # Number of runs
    MaxFEs = 1000

    d = len(dims)
    o = len(TestFuns) + len(TestFuns2)  # Two benchmark sets

    f_bias_set = [-450, -450, -450, -450, -310, 390, -180, -140, -330, -330, 90, -460, -130, -300, 120]
    f_bias_set2 = [0, 0, 0, 0, 0]

    for i in range(0, d ):
        for j in range(0, o):
            if j < len(TestFuns):
                f_bias = f_bias_set[j]
                fun_name = TestFuns[j]

                funcs = opfunu_v3.get_functions_by_classname(fun_name)
                FUN = funcs[0](ndim=dims[i])
                F_FUN =  FUN.evaluate

            else:

                problem_index = j - len(TestFuns)
                f_bias = f_bias_set2[problem_index]
                F_FUN = functions1[problem_index]
                fun_name = TestFuns2[problem_index]

            Xmin, Xmax = variable_domain(fun_name)
            LB = [Xmin] * dims[i]
            UB = [Xmax] * dims[i]



            RUN_LLM_SAEA(MaxFEs, Runs, dims[i], F_FUN, fun_name, LB, UB, f_bias, paras)


