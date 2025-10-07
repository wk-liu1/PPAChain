# import torch
from ast import literal_eval
import xml.etree.ElementTree as ET
from xml.dom import minidom
from copy import deepcopy
from openai import OpenAI
import time
def generate(fixed_prompt, user_input, start_flag, end_flag):
#     """调用宿主机上的大模型api"""
    # model_choice可选：qwen2.5-coder:32b-instruct-fp16、qwen2.5-coder:32b-instruct-fp16等
    
    # print(0)
    # 调用deepseek api
    # 设置API密钥和请求地址（请根据实际API文档调整URL）
    # DEEPSEEK_API_KEY = "sk-28e4fe4d36cb47c6b08ed27b1dd33cfd"  # 替换为你的API密钥
    # API_URL = "https://api.deepseek.com"  
    
    # headers = {
    #     "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    #     "Content-Type": "application/json"
    # }
    
    # # 组装提示词
    # #full_prompt = f"{fixed_prompt}{user_input}"
    # # print(full_prompt)

    # print(1)
    # # 传输给api的数据
    # data = {
    #     "model":"deepseek-reasoner",
    #     "messages":[
    #         {"role": "system", "content": fixed_prompt},
    #         {"role": "user", "content": user_input},
    #     ],

    #     "temperature":0.0,
    #     "stream":False
    # }
    
    # try:
    #     print(2)
    #     response = requests.post(API_URL, json=data, headers=headers)
    #     print(response)
    #     response.raise_for_status() # 检查http错误
    #     result = response.json()
    #     print(result)
        

    #     # 提取回复内容
    #     if result.get("choices"):
    #         print(result["choices"][0]["message"]["content"])
    #         return result["choices"][0]["message"]["content"]
    #     return "Error: No response generated"
    
    # except requests.exceptions.RequestException as e:
    #     return f"Request error: {str(e)}"
    # except Exception as e:
    #     return f"Error: {str(e)}"
        
        
        
    # 以下是openai调用，但是openai import OpenAI报错
    client = OpenAI(api_key="sk-qkszhuzkeczxxdxrohzerrbhygsyfmmalapurdmxtbiahvwk", base_url="https://api.siliconflow.cn")

    while True:
        try:
            response = client.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3",
                messages=[
                    {"role": "system", "content": fixed_prompt},
                    {"role": "user", "content": user_input},
                ],
    
                temperature = 0.3,
                stream=False
            )
            
            output = response.choices[0].message.content
    
            #检查有没有按格式输出
            start = output.find(start_flag)
            if start==-1:
                continue
                # print(think_and_output)
                # return -1
            else:
                start+=len(start_flag)
                end = output.find(end_flag, start)
                if end==-1:
                    continue
                else:
                    output = output[start:end].strip()
                    print(output)
                # return 1
                    return output
            
        except Exception as e:
            return f"API Error: {str(e)}"


    # while True:
    #     try:
    #         response = requests.post(api_url, json=data)
    #         # print(response)
    #         # response.raise_for_status() # 检查http错误
    #         result = response.json()
    #         # print("result:", result)
    #         #print("result类型:", type(result)) 字典类型
    #         # print("response:", result["response"])
    #         # print("response类型:", type(result["response"]))
            
    #         #截取response，去除think的内容，仅保留输出
    #         think_and_output = result["response"]
    #         # print(think_and_output)


    #         if model_choice=="qwq:32b-fp16" or model_choice=="qwq:32b-q8_0" or model_choice=="deepseek-r1:70b" or model_choice=="deepseek-r1:qwen-32b-distill-fp16":      #如果是深度思考模型
    #             #检查有没有输出</think>
    #             think_end = think_and_output.find("</think>")
    #             if think_end==-1:
    #                 continue
                
    #             #检查有没有按格式输出
    #             start = think_and_output.find(start_flag, think_end)
    #             if start==-1:
    #                 continue
    #                 # print(think_and_output)
    #                 # return -1
    #             else:
    #                 start+=len(start_flag)
    #                 end = think_and_output.find(end_flag, start)
    #                 if end==-1:
    #                     continue
    #                 else:
    #                     output = think_and_output[start:end].strip()
    #                     print(output)
    #                 # return 1
    #                     return output
    #         else:
    #             #检查有没有按格式输出
    #             start = think_and_output.find(start_flag)
    #             if start==-1:
    #                 continue
    #                 # print(think_and_output)
    #                 # return -1
    #             else:
    #                 start+=len(start_flag)
    #                 end = think_and_output.find(end_flag, start)
    #                 if end==-1:
    #                     continue
    #                 else:
    #                     output = think_and_output[start:end].strip()
    #                     print(output)
    #                 # return 1
    #                     return output

    #     except Exception as e:
    #         print(f"Error calling model API: {e}")
    #         return None

# 装备解析Agent
def equipment_component_parse(equipment_description):

    print("-------------------------装备解析-------------------------")

    s_time = time.time()
    #装备能力解析的提示词模板
    prompt_template_for_component_generate =    """
请先自我强调：决不能过度、反复思考!!!

你是一个具有武器装备基本常识和严谨逻辑思维的建模师，现在有一段自然语言形式的[装备描述]，你能按照[工作流程]从中解析和推理出装备具有哪些动作。但是在你开始工作之前，必须自我强调一遍[注意事项]。

[工作流程]：
第一步：完整地找出装备描述中所有明确提及的与动作相关的表述，如果有某些表述不能确定是否与动作相关也要保留。比如装备描述是“一辆坦克最快能以40km/h的速度行驶，它的125mm口径火炮射程为1000m，该辆坦克可以在坡度小于10的地形上行驶。该坦克上有一名指挥员可以目视2000m以内的情况。”，其中明确提及了坦克具有行驶能力，火炮可以发射炮弹，还提到了指挥员能目视。
第二步：通过简单推理将第一步中找出的与动作相关的表述转换为主谓或主谓宾结构，这里的主语优先选择装备组件（装备中的人员也可视为组件），比如装备描述是“潜艇有一套主动声呐系统能探测2000m范围内的物体”，这里应转换为“声呐探测”，主语也可以是装备本身，比如装备描述是“一架旋翼无人机携带了4枚射程为2km的空对地导弹”，就应该转换为“旋翼无人机发射空对地导弹。
第三步：将第二步中的动词或动宾短语合并为一个字符串元素构成的动作列表。

[注意事项]：
1.在你思考的时候，不允许自我修正。
2.必须严格按照[输出格式]输出最终结果，除此之外不允许输出任何其他内容。


[输出格式]：
△这里是字符串列表△

[示例]：
输入：一艘潜艇，最大航速32节，有一台水下蜂巢可以在距海平面100m以内的水位发射50架无人机，还有一套主动声呐系统能探测2000m范围内的物体和一套被动声呐系统能探测1000m范围内的物体，一套浮标通信系统，通信范围为10000m，该潜艇还具有水下定位与导航系统。
输出：△["潜艇航行", "水下蜂巢发射无人机", "主动声呐探测", "被动声呐探测", "浮标通信", "水下定位与导航"]△

输入：一辆坦克最快能以40km/h的速度行驶，它的125mm口径火炮射程为1000m，该辆坦克可以在坡度小于10的地形上行驶。该坦克上有一名指挥员可以目视2000m以内的情况。
输出：△["坦克行驶", "火炮发射炮弹", "指挥员目视"]△

[装备描述]：
    """
    prompt_variables_for_component_generate = equipment_description

    #生成输出
    llm_paerse_component_list_str = generate(prompt_template_for_component_generate, prompt_variables_for_component_generate,  "△", "△")
    # print(type(llm_paerse_component_list_str))

    #检查如果有中文引号，将其替换为英文引号
    quotes = {'“': '"', '”': '"', '，': ','}
    for chinese, english in quotes.items():
        llm_paerse_component_list_str = llm_paerse_component_list_str.replace(chinese, english)

    #将字符串转为列表
    llm_paerse_component_list = literal_eval(llm_paerse_component_list_str)

    # 后剪枝，到原子动作库中找相似
    # init_milvus_connection()
    retrieve_components_list = []
    # for component in llm_paerse_component_list:
    #     milvus_retrieve_component_list = retrieve_from_milvus("hisim_atom_action_better", component, 5)  #milvus_retrieve_component_list是一个列表，其中的每个元素是字典类型
    #     for i in range(0, len(milvus_retrieve_component_list)):
    #         print("大模型生成组件：", component, "检索结果：", milvus_retrieve_component_list[i].get("动作名称"),milvus_retrieve_component_list[0].get("相似度"))
    #     retrieve_components_list.append({"动作名称": milvus_retrieve_component_list[0].get("动作名称"), "动作ID": milvus_retrieve_component_list[0].get("动作ID")})

    e_time = time.time()
    run_time = e_time-s_time
    print(f"-------------------推理时间：{run_time}-------------------")
    
    return llm_paerse_component_list, retrieve_components_list
#任务解析Agent
def equipment_task_parse(task_description):

    print("-------------------------任务推理-------------------------")

    s_time = time.time()
    
    prompt_template_for_task_parse = '''
你是一个具有武器装备基本常识和严谨逻辑思维的规划师，现在有一段自然语言形式的[任务描述]说明了某个装备所要完成的任务，你要按照[工作流程]从中解析和推理出装备的最终目标。但是在你开始工作之前，必须自我强调一遍[注意事项]。

[工作流程]：
第一步：找出[任务描述]中体现装备任务目标的表述。比如任务描述是“该无人机的任务是按照航线搜索指定区域，打击被发现的地面敌方目标”，其中体现装备目标的表述包括“搜索指定区域”、“发现地面敌方目标”和“打击地面敌方目标”。
第二步：分析第一步中找出的多个体现装备目标的表述之间的依赖关系，形成一条目标依赖链路。比如“搜索指定区域”是为了“发现地面敌方目标”，“发现地面敌方目标”才能“打击地面敌方目标”，所以目标依赖链路是“搜索指定区域”→“发现地面敌方目标”→“打击地面敌方目标”。
第三步：第二步生成的目标链路的最终节点就是装备的最终目标，将其转换为布尔表达式的形式。如“打击地面敌方目标”就转换为已毁伤(地面敌方目标)==True。

[注意事项]：
1.在你思考的时候，不允许自我修正。
2.必须严格按照[输出格式]输出最终结果，除此之外不允许输出任何其他内容。
3.装备的最终目标有且只有一个，只能用单一布尔表达式表示，不能用复合布尔表达式。

[输出格式]：
△这里是装备最终目标的单一布尔表达式△

[示例]：
输入：该潜艇的任务是按照给定航线机动至A军港附近，行进过程中避开敌方舰艇侦察，机动完成后发射30架无人机到预定点位，并成功获得所有无人机回传的A军港敌方海军图像。
输出：△数量(无人机回传图像)==30△

输入：该坦克的作战任务是掩护步兵班推进至A高地，推进过程中需向敌方火力点实施火力压制。
输出：△位置(步兵班)==A高地△

[任务描述]：
    '''
    prompt_variables_for_task_parse = task_description

    llm_paerse_task_str = generate(prompt_template_for_task_parse, prompt_variables_for_task_parse, "△", "△")
    # print(llm_paerse_task_str)

    e_time = time.time()
    run_time = e_time-s_time
    print(f"-------------------推理时间：{run_time}-------------------")
    
    return llm_paerse_task_str
# 动作生成
def equipment_action_choose(task_description, task, equipment_components_list):
    print(f"-------------------------动作选取 for {task}-------------------------")

    s_time = time.time()
    prompt_template_for_action_choose = f'''
你是一个具有武器装备基本常识和严谨逻辑思维的规划师，现在有一段[任务描述]说明了某个装备所要完成的任务，[目标]则是从任务描述中提取的任务目标，你要按照[工作流程]从该装备的[动作列表]中找出一个直接实现装备目标的动作，并推理出该动作的作用对象。但是在你开始工作之前，必须自我强调一遍[注意事项]。

[工作流程]：
第一步：结合[任务描述]分析[目标]的含义。如“位置(步兵班)==A高地”表示“步兵班抵达A高地”。
第二步：从[动作列表]中选出能直接实现目标的动作。如坦克行驶仅改变坦克位置，不能使步兵班抵达A高地；火炮发射炮弹可以摧毁或压制阻碍步兵抵达A高地的敌方火力点；指挥员目视可以观察步兵班和敌方火力点的位置，但是不能使步兵班抵达A高地。因此能直接实现目标的动作是"火炮发射炮弹"。如果不能从[动作列表]中选出能直接实现目标的动作，比如目标是“数量(火炮炮弹)>0”，实现该目标需要补充弹药的动作，但是动作列表中并不包含“补充弹药”，那么输出’△△’即可。
第三步：分析该动作的作用对象。如火炮发射炮弹是作用于敌方火力点的，以达到摧毁或压制敌方火力点使得步兵班能够抵达A高地。如果该动作没有作用对象也可以，比如“指挥员目视”这个动作，是在目视一片区域，并没有特定的作用对象。
第四步：整合第二步的动作和第三步的对象，形成"动作(对象)"格式的表达式，注意用英文括号。

[任务描述]：
{task_description}

[目标]：
{task}

[动作列表]：
{equipment_components_list}

[注意事项]：
1.在你思考的时候，不允许自我修正。
2.必须严格按照[输出格式]输出最终结果，除此之外不允许输出任何其他内容。
3.你选取的动作能直接地实现装备目标。
4.动作可以为空，动作的作用对象也可以为空，取决于你的推理。

[输出格式]：
△动作(对象)△

[示例]：
任务描述：该潜艇的任务是按照给定航线机动至A军港附近，机动完成后发射1架无人机到预定点位，并通过浮标通信成功获得无人机回传的A军港敌方海军图像。
目标：数量(无人机回传图像)==1
动作列表：["潜艇航行", "水下蜂巢发射无人机", "主动声呐探测", "被动声呐探测", "浮标通信", "水下定位与导航"]
输出：△浮标通信(无人机) △

任务描述：该潜艇的任务是按照给定航线机动至A军港附近，机动完成后发射1架无人机到预定点位，并通过浮标通信成功获得无人机回传的A军港敌方海军图像。
目标：位置(潜艇)==A军港附近
动作列表：["潜艇航行", "水下蜂巢发射无人机", "主动声呐探测", "被动声呐探测", "浮标通信", "水下定位与导航"]
输出：△潜艇航行(指定航线) △

任务描述：该坦克的作战任务是掩护步兵班推进至A高地，推进过程中需向敌方火力点实施火力压制。
目标：位置(步兵班)==A高地
动作列表：["坦克行驶", "火炮发射炮弹", "指挥员目视"]
输出：△火炮发射炮弹(敌方火力点) △

任务描述：该坦克的作战任务是掩护步兵班推进至A高地，推进过程中需向敌方火力点实施火力压制。
目标：数量(火炮炮弹)>0
动作列表：["坦克行驶", "火炮发射炮弹", "指挥员目视"]
输出：△△

任务描述：该坦克的作战任务是掩护步兵班推进至A高地，推进过程中需向敌方火力点实施火力压制。
目标：距离(坦克, 敌方火力点)<=火炮射程
动作列表：["坦克行驶", "火炮发射炮弹", "指挥员目视"]
输出：△火炮行驶(敌方火力点)△
    '''

    prompt_variables_for_action_choose = ""

    llm_choose_action_str = generate(prompt_template_for_action_choose, prompt_variables_for_action_choose, "△", "△")

    e_time = time.time()
    run_time = e_time-s_time
    print(f"-------------------推理时间：{run_time}-------------------")
    return llm_choose_action_str
# 条件生成
def equipment_condition_generate(equi_description, task_description, task, action):
    print("-------------------------条件推理-------------------------")

    s_time = time.time()
    
    prompt_template_for_condition_generate = f'''
你是一个具有武器装备基本常识和严谨逻辑思维的规划师，现在有[装备描述]和[任务描述]分别说明了某个装备的功能参数以及所要完成的任务供你参考，单一布尔表达式[目标]则描述了该装备的一个目标，[动作]则描述该装备为直接实现该目标所要采取的动作及作用对象。你要按照[工作流程]推理出执行[动作]完成[目标]的前提条件。但是在你开始工作之前，必须自我强调一遍[注意事项]。

[工作流程]：
第一步：分析[目标]的含义。如“位置(步兵班)==A高地”表示“步兵班抵达A高地”。
第二步：分析[动作]的含义。如“火炮发射炮弹(敌方火力点)”表示“用火炮向敌方火力点发射炮弹”。
第三步：紧密结合[装备描述]非常审慎地推理装备通过[动作]实现[目标]的前提条件。如装备描述是“一辆坦克最快能以40km/h的速度行驶，它的125mm口径火炮最快每5s装填一发炮弹，火炮射程为1000m。该坦克上有一名指挥员可以目视2000m以内的情况”，目标是“掩护步兵班抵达A高地”，动作是“用火炮向敌方火力点发射炮弹”。要想用火炮向敌方火力点发射炮弹掩护步兵班抵达A高地，首先需要火炮可以装填的炮弹数量最起码大于0（装备描述中提到了火炮装填），还需要坦克知道敌方火力点的位置（装备描述中的指挥员目视），还需要敌方火力点的位置在火炮射程之内（装备描述中提到了火炮射程）。如果该装备执行该[动作]不需要前提，如“指挥员目视()”动作随时可以执行没有条件约束，这一步的结果可以为空。
第四步：紧密结合[任务描述]非常审慎地推理装备实现[目标]的前提条件。如任务描述是“该潜艇的任务是按照给定航线机动至A军港附近，机动完成后发射1架无人机到预定点位，并通过浮标通信成功获得无人机回传的A军港敌方海军图像。”，目标为“获得无人机回传的A军港敌方海军图像”，任务描述中显然指出了这一目标的前提条件是“无人机到达预定点位”。如果任务描述中没体现当前目标的前提条件，这一步的结果可以为空。
第五步：整合第三步和第四步中推理出来的前提条件，将它们分别转换为单一布尔表达式的形式，然后插入到一个列表中。

[装备描述]：
{equi_description}

[任务描述]：
{task_description}

[目标]：
{task}

[动作]：
{action}

[注意事项]：
1.在你思考的时候，不允许自我修正。
2.必须严格按照[输出格式]输出最终结果，除此之外不允许输出任何其他内容。

[输出格式]：
△这里是一个列表，其中的元素是各个条件的单一布尔表达式，也可以为空△

[示例]：
装备描述：一艘潜艇，最大航速32节，有一台水下蜂巢可以在距海平面100m以内的水位发射50架无人机，还有一套主动声呐系统能探测2000m范围内的物体和一套被动声呐系统能探测1000m范围内的物体，一套浮标通信系统，通信范围为10000m，该潜艇还具有水下定位与导航系统。
任务描述：该潜艇的任务是按照给定航线机动至A军港附近，机动完成后发射1架无人机到预定点位，并通过浮标通信成功获得无人机回传的A军港敌方海军图像。
目标：数量(无人机回传图像)==1
动作：浮标通信(无人机)
输出：△["位置(无人机)==预定点位"]△

装备描述：一辆坦克最快能以40km/h的速度行驶，它的125mm口径火炮最快每5s装填一发炮弹，火炮射程为1000m。该坦克上有一名指挥员可以目视2000m以内的情况。
任务描述：该坦克的作战任务是掩护步兵班推进至A高地，推进过程中需向敌方火力点实施火力压制。
目标：位置(步兵班)==A高地
动作：火炮发射炮弹(敌方火力点)
输出：△["数量(炮弹)>0", "距离(坦克, 敌方火力点)<=火炮射程"]△

装备描述：一辆坦克最快能以40km/h的速度行驶，它的125mm口径火炮最快每5s装填一发炮弹，火炮射程为1000m。该坦克上有一名指挥员可以目视2000m以内的情况。
任务描述：该坦克的作战任务是掩护步兵班推进至A高地，推进过程中需向敌方火力点实施火力压制。
目标：距离(坦克, 敌方火力点)<=火炮射程
动作：火炮行驶(敌方火力点)
输出：△["位置(敌方火力点)!=Null"]△

    '''

    prompt_variables_for_condition_generate = ""

    llm_generate_condition_str = generate(prompt_template_for_condition_generate, prompt_variables_for_condition_generate, "△", "△")

    llm_generate_condition_list = literal_eval(llm_generate_condition_str)

    e_time = time.time()
    run_time = e_time-s_time
    print(f"-------------------推理时间：{run_time}-------------------")
    return llm_generate_condition_list
# 条件生成返回的是列表怎么处理
def generate_sequence_element(node_list):
    if len(node_list)>1:
        sequence = ET.Element("Sequence")

        for node in node_list:
            ET.SubElement(sequence, "Condition", name=node)

        return sequence
    elif len(node_list)==1:
        elem = ET.Element("Condition", name=node_list[0])
        return elem
    else:
        return None

# 生成xml格式的PPA子树
def generate_ppa_element(postcondition, precondition, action):
    fallback = ET.Element("Fallback")
    ET.SubElement(fallback, "Condition", name=postcondition)

    #如果条件为空，仅扩展了动作
    if precondition is None:
        ET.SubElement(fallback, "Action", name=action)
    else:
        sequence = ET.SubElement(fallback, "Sequence")
        #生成的条件有多个
        # if type(precondition) == ET.Element:
        sequence.append(precondition)
        #生成的条件仅一个
        # else:
            # ET.SubElement(sequence, "Condition", name=precondition)
        ET.SubElement(sequence, "Action", name=action)
    return fallback
# 比较两个xml.Element元素
def deep_compare(elem1, elem2):
    if elem1.tag != elem2.tag:
        return False
    if elem1.attrib != elem2.attrib:
        return False
    if len(elem1) != len(elem2):
        return False
    return all(deep_compare(c1, c2) for c1, c2 in zip(elem1, elem2))

# 行为树生成
def generate_BT_xml(equipment_desc, equipment_task_desc):

    components_name_list, components_list = equipment_component_parse(equipment_desc)

    final_task = equipment_task_parse(equipment_task_desc)

    action = equipment_action_choose(equipment_task_desc, final_task, components_name_list)

    condition = equipment_condition_generate(equipment_desc, equipment_task_desc, final_task, action)

    initial_tree = generate_ppa_element(final_task, generate_sequence_element(condition), action)

    for i in range(0, 4):#最大扩展10次，意味着树的最大深度10层

        rough_xml = ET.tostring(initial_tree, encoding="unicode")
        parsed = minidom.parseString(rough_xml)
        print(parsed.toprettyxml(indent="    ", encoding="utf-8").decode("utf-8"))

        tmp = deepcopy(initial_tree) #递归复制，不能直接用=赋值，那会使两者指向同一个对象，修改一个另一个也就变了
        for parent in initial_tree.findall(".//Sequence"):
            for idx, child in enumerate(parent):
                if child.tag == "Condition":
                    new_action = equipment_action_choose(equipment_task_desc, child.get("name"), components_name_list)
                    if new_action != "": #能进一步扩展
                        new_condition = equipment_condition_generate(equipment_desc, equipment_task_desc, child.get("name"), new_action)
                        new_fallback = generate_ppa_element(child.get("name"), generate_sequence_element(new_condition), new_action)
                        parent.insert(idx, new_fallback)
                        parent.remove(child)
                    else:
                        continue

        if deep_compare(initial_tree, tmp):
            break
    
    return initial_tree
# # 主函数
# if __name__ == "__main__":

#     equipment_desc = "一台电力巡检机器人配备四驱底盘可移动，携带红外测温仪可以检测电线温度和卫星定位仪可以确定位置坐标，还搭载通信模块支持实时数据传输。"
#     equipment_task_desc = "该机器人的任务是沿输电线路移动至B区段，利用红外测温仪检测B区段线路有无异常升温，使用卫星定位仪确定异常升温点坐标，最终将异常升温点坐标传回控制中心。"

#     start_time = time.time()
#     generate_BT_xml(equipment_desc, equipment_task_desc)
#     end_time = time.time()
#     run_time = end_time-start_time
#     print(f"-------------------运行时间：{run_time}-------------------")
if __name__ == "__main__":
    # 电力巡检机器人示例
    equipment_desc = "一台电力巡检机器人配备四驱底盘可移动，携带红外测温仪可以检测电线温度和卫星定位仪可以确定位置坐标，还搭载通信模块支持实时数据传输。"
    equipment_task_desc = "该机器人的任务是沿输电线路移动至B区段，利用红外测温仪检测B区段线路有无异常升温，使用卫星定位仪确定异常升温点坐标，最终将异常升温点坐标传回控制中心。"
    
    start_time = time.time()
    behavior_tree = generate_BT_xml(equipment_desc, equipment_task_desc)
    end_time = time.time()
    
    # 打印运行时间
    print(f"运行时间：{end_time-start_time}秒")
    
    # 打印最终XML
    rough_xml = ET.tostring(behavior_tree, encoding="unicode")
    parsed = minidom.parseString(rough_xml)
    print(parsed.toprettyxml(indent="    "))
