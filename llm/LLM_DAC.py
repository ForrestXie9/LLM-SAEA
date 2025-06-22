
import re
import time
from llm.ael_prompts import GetPrompts
from llm.interface_LLM import InterfaceLLM
import numpy as np

class LLM_DAC():

    def __init__(self, Save_rp, Save_gl, Save_rl, Save_ge, Save_pp, Save_pl, Save_Gi, Save_Go, paras):
        self.llm_api_endpoint  =  paras.llm_api_endpoint
        self.llm_api_key = paras.llm_api_key
        self.llm_model = paras.llm_model
        self.exp_debug_mode= paras.exp_debug_mode
        self.Save_rp = Save_rp
        self.Save_rl = Save_gl
        self.Save_gl = Save_rl
        self.Save_ge = Save_ge
        self.Save_pp = Save_pp
        self.Save_pl = Save_pl
        self.Save_Gi = Save_Gi
        self.Save_Go = Save_Go
        self.debug_mode = False

        # 构建算法部件与历史性能的映射
        self.component_performance = {
            1: self.Save_rl,
            2: self.Save_rl,
            3: self.Save_gl,
            4: self.Save_ge,
            5: self.Save_pp,
            6: self.Save_pl,
            7: self.Save_Gi,
            8: self.Save_Go,
            # 继续添加其它算法部件的映射
        }

        self.interface_llm = InterfaceLLM(self.llm_api_endpoint, self.llm_api_key, self.llm_model, self.debug_mode)
    def get_prompt_selection(self):
        description1 = ""
        for component, performance in self.component_performance.items():
            description1 += "The historical performance of the algorithm component {} is: {}\n".format(component, performance)

        prompt_content = "Now you will help me select the configuration of algorithm. There are a total of " + str(len(self.component_performance)) + \
                         " algorithm configurations available for selection. I recorded the historical performance of these " + str(len(self.component_performance)) + " algorithm configurations:\n" + description1 + \
                         " You need to provide a recommendation for an algorithm configuration from the existing ones to achieve better algorithm performance. " \
                         " You just need to output the index of your recommended algorithm configuration, and this index should start with <start> and end with <end>." \
                         " Do not write code. Do not give any explanation."

        return prompt_content


    def Selection(self):
        prompt_content = self.get_prompt_selection()
        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m2 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        response = self.interface_llm.get_response(prompt_content)
        # algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
        index_str = re.findall(r"<start>(.*?)<end>", response)
        index = np.array([int(num) for num in index_str])

        if self.debug_mode:
            # print("\n >>> check designed algorithm: \n", algorithm)
            # print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()


        return index