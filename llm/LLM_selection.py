
import re
import time
from llm.ael_prompts import GetPrompts
from llm.interface_LLM import InterfaceLLM
import numpy as np

class LLM_selection():

    def __init__(self, q_value_m, Num,  current_iter, total_iter, paras):
        # LLM_selection(q_value_m, v_vaule_m, Succ, Num)
        self.llm_api_endpoint  =  paras.llm_api_endpoint
        self.llm_api_key = paras.llm_api_key
        self.llm_model = paras.llm_model
        self.exp_debug_mode= paras.exp_debug_mode

        self.debug_mode = False
        self.num_arm = len(q_value_m)
        self.q_value_m = q_value_m
        total_num = np.sum(Num)
        self.current_iter = current_iter
        self.total_iter = total_iter
        # 构建算法部件与历史性能的映射
        if total_num == 0:
            self.component_performance = {
                1: [q_value_m[0], 0],
                2: [q_value_m[1], 0],
                3: [q_value_m[2], 0],
                4: [q_value_m[3], 0],
                5: [q_value_m[4], 0],
                6: [q_value_m[5], 0],
                7: [q_value_m[6], 0],
                8: [q_value_m[7], 0],
                # 继续添加其它算法部件的映射
            }
        else:
            self.component_performance = {
                1: [q_value_m[0],  Num[0] / total_num],
                2: [q_value_m[1],  Num[1] / total_num],
                3: [q_value_m[2],  Num[2] / total_num],
                4: [q_value_m[3],  Num[3] / total_num],
                5: [q_value_m[4],  Num[4] / total_num],
                6: [q_value_m[5],  Num[5] / total_num],
                7: [q_value_m[6],  Num[6] / total_num],
                8: [q_value_m[7], Num[7] / total_num],
                # 继续添加其它算法部件的映射
            }

        for key, value_list in self.component_performance.items():
            rounded_value_list = [round(float_value, 2) for float_value in value_list]
            self.component_performance[key] = rounded_value_list

        self.interface_llm = InterfaceLLM(self.llm_api_endpoint, self.llm_api_key, self.llm_model, self.debug_mode)



    def get_prompt_selection(self):

        component_performance_str = ""

        for key, value in self.component_performance.items():
            # component_str = f"<Configuration {key}: {value}>\n"
            component_str = f"Action {key}: average score is {value[0]} and selection frequency is {value[1]}\n"
            component_performance_str += component_str

        Role_description = "Your task is to select the best actions from the provided action set by considering the context information of time slot and action. " \
                          "Time slot context: \n\nTotal time slots: " + str(self.total_iter) + "\nCurrent time slot: " + str(self.current_iter) + "\nRemaining time slots: " + str(self.total_iter - self.current_iter) + ""

        Context = "\n\nAction set context:" \
                  "\n\n" + component_performance_str + "\n"

        Hint = "Firstly, you must ensure each action is explored (i.e., has a non-zero selection probability). " \
               "Moreover, use your strong reasoning abilities to strike a balance between exploration (choosing actions with lower selection frequency) and exploitation (choosing actions with higher average score). "

        Output = "Output only the index(es) of your selected action(s) and its confidence label in the format <start>index(confidence label) <end>, e.g., <start>1(certainty), 2(uncertainty)<end>. " \
                 "If you are confident in your selected action, use 'certainty' as its confidence label; otherwise, use 'uncertainty'. Do not give explanations."

        prompt_content = Role_description + Context + Hint + Output

        return prompt_content

    def roulette_wheel_selection(self):
        # e_values = np.exp(self.q_value_m - np.max(self.q_value_m))
        e_values = np.exp(self.q_value_m)
        probabilities = e_values / e_values.sum()
        cumulative_sum = np.cumsum(probabilities)
        r = np.random.rand()
        for index, value in enumerate(cumulative_sum):
            if r < value:
                return index+1

        return len(probabilities)



    def selection(self):
        # if self.candidate_rank < self.min_value:
        #     converted_numbers = [1]
        # else:
        prompt_content = self.get_prompt_selection()
        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m2 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()
        attempts = 0
        max_attempts = 3
        total_con = []
        [total_con.append(i) for i in range(1, 9)]
        response = self.interface_llm.get_response(prompt_content)
        print(response)
        attempts += 1
        matches = re.findall(r'(\d+)\((certainty|uncertainty)\)', response)
        # print(matches)
        converted_numbers = []
        numbers = []
        for index, status in matches:
            total_con.append(int(index))
            numbers.append(int(index))
            if status == 'certainty':
                converted_numbers.append(int(index))
            # else:
            #     total_con.append(int(index))

        while len(numbers) < 1:
            response = self.interface_llm.get_response(prompt_content)
            matches = re.findall(r'(\d+)\((certainty|uncertainty)\)', response)
            converted_numbers = []
            for index, status in matches:
                numbers.append(int(index))
                total_con.append(int(index))
                if status == 'certainty':
                    converted_numbers.append(int(index))
                # else:
                #     total_con.append(int(index))

        remaining_num = len(total_con) - self.num_arm - len(converted_numbers)
        for jj in range(remaining_num):
            selected_index = self.roulette_wheel_selection()
            # random_choice = np.random.choice(total_con)
            # converted_numbers.append(selected_index)
            if selected_index not in converted_numbers:
                converted_numbers.append(selected_index)


        if self.debug_mode:

            print(">>> Press 'Enter' to continue")
            input()

        return converted_numbers