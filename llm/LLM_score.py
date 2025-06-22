
import re
import time
from llm.ael_prompts import GetPrompts
from llm.interface_LLM import InterfaceLLM
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class LLM_score():

    def __init__(self, min_value,  mean_value, max_value, candidate_fit, candidate_rank, num, dim, num_arm, paras):


        self.llm_api_endpoint  =  paras.llm_api_endpoint
        self.llm_api_key = paras.llm_api_key
        self.llm_model = paras.llm_model
        self.exp_debug_mode= paras.exp_debug_mode
        self.min_value = min_value
        self.mean_value = mean_value
        self.max_value = max_value
        self.candidate_fit = candidate_fit
        self.candidate_rank = candidate_rank
        self.dim = dim
        self.num = num
        self.num_arm = num_arm

        self.debug_mode = False


        self.interface_llm = InterfaceLLM(self.llm_api_endpoint, self.llm_api_key, self.llm_model, self.debug_mode)


    def get_prompt_score(self):

        Role_description = "As a scoring expert, you need to score point x in set P based on rigorous derivation ranging from 0 to 1, where 1 is the highest quality and 0 is the lowest. "

        Context = "Consider the following information: \n1) Point x ranks " + str(self.candidate_rank) + " out of " + str(self.num + 1) + " points in set P, ordered from best to worst based on the objective value. \n" \
                                                            "2) The objective value of point x is " + str(self.candidate_fit) + ".\n" \
                                                            "3) Excluding point x, the best, average, and worst objective values in set P are respectively  " + str(self.min_value) + ", " + str(self.mean_value) + ", and " + str(self.max_value) + ".\n "
                                            # "\n "
        Output_c = "Output only the score with two decimal places in the format <start>value<end>. Do not give explanations."

        prompt_content = Role_description + Context + Output_c

        return prompt_content
    def Score(self):
        # if self.candidate_rank < self.min_value:
        #     converted_numbers = [1]
        # else:
        prompt_content = self.get_prompt_score()
        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m2 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        response = self.interface_llm.get_response(prompt_content)
        # print(response)
        # algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
        # index_str = re.findall(r'\d+\.\d+', response)
        index_str = re.findall(r'\d+\.\d+|\d+', response)
        converted_numbers = []
        for number in index_str:
            if '.' in number:
                converted_numbers.append(float(number))
            else:
                converted_numbers.append(int(number))

        while len(converted_numbers) > 1 or len(converted_numbers) < 1:
            response = self.interface_llm.get_response(prompt_content)
            index_str = re.findall(r'\d+\.\d+|\d+', response)
            converted_numbers = []
            for number in index_str:
                if '.' in number:
                    converted_numbers.append(float(number))
                else:
                    converted_numbers.append(int(number))
     #   print(converted_numbers)
        if self.debug_mode:
            # print("\n >>> check designed algorithm: \n", algorithm)
            # print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()


        return converted_numbers