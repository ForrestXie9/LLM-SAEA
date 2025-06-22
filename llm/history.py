def get_prompt_selection(self):
    component_performance_str = ""

    for key, value in self.component_performance.items():
        component_str = f"<Configuration {key}: {value}>\n"
        component_performance_str += component_str

    # Role_description = "The team aims to achieve automated algorithm configuration selection for solving a single-objective minimization optimization problem. As a configuration selection expert within a team of optimization experts, " \
    #                    "your role is to select the optimal configuration from among " + str(self.num_arm) + " options. \n"

    # problem_background = "The team aims to achieve automated algorithm configuration selection for solving a single-objective minimization optimization problem.  "

    Role_description = "As a configuration select expert, " \
                       "your task is to select one configuration from " + str(
        self.num_arm) + " configuration options in each iteration. "

    Context = "Each configuration's context is formatted as: <Configuration index: [average reward, selection frequency]>, listed below:" \
              "\n\n " + component_performance_str + "\n "

    # Output = "Given the context, your selection strategy should strike a balance between exploring unknown options (i.e., trying to choose options with low selection frequency) and utilizing known options (i.e., selecting the option that is currently performing the best). " \
    #          "Your response only includes the index of the configuration you selected, formatted as <start>index<end>. "

    Output = "Given the information, your selection strategy should strike a balance between exploring options with lower selection frequency and leveraging options with higher average reward. " \
             "Output only the index of the configuration you selected, formatted as <start>index<end>. Do not give explanation."

    prompt_content = Role_description + Context + Output

    return prompt_content


def get_prompt_selection(self):
    component_performance_str = ""

    for key, value in self.component_performance.items():
        # component_str = f"<Configuration {key}: {value}>\n"
        component_str = f"Configuration {key}: average reward is {value[0]} and selection frequency is {value[1]}\n"
        component_performance_str += component_str

    Role_description = "As a configuration select expert, " \
                       "your task is to select one configuration from " + str(self.num_arm) + " configuration options. "

    Context = "Each configuration's context is listed below:" \
              "\n\n " + component_performance_str + "\n "

    Output = "Given the information, your selection strategy is based on rigorous reasoning, striking a balance between exploring options with lower selection frequency and leveraging options with higher average rewards. " \
             "Output only the index of the configuration you selected, formatted as <start>index<end>. Do not give some explanation."

    prompt_content = Role_description + Context + Output

    return prompt_content