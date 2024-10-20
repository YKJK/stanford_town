"""
Author: Joon Sung Park (joonspk@stanford.edu)
File: gpt_structure.py
Description: Wrapper functions for calling OpenAI APIs.
"""
import json
import random
import openai
import time 
import os, requests
import textwrap
import logging

from utils import *
from typing import Any, Dict, List, Mapping, Optional

from pydantic import Extra, Field, root_validator
from gpt4all import GPT4All, Embed4All
import datetime
import global_config

# ============================================================================
#                   SECTION 1: Together API STRUCTURE
# ============================================================================
URL = "https://api.together.xyz/inference"
def run_rest_api(question,model_name,max_tokens,temperature):
  prompt_template = global_config.current_prompt_template
  print("---->count_request:"+str(global_config.count_request))
  global_config.count_request += 1

  prompt = question.replace('\n\n', '\n') #clean换行符2->1;这里暂时不用template
  prompt = prompt.strip() #clean去掉前后空格
  prompt = choose_prompt_tag(prompt_template,prompt)

  temperature = choose_temperature(prompt_template,temperature)
  repetition_penalty = choose_repetition_penalty(prompt_template)
  stop = choose_stop_str(prompt_template),

  print("---->model:"+model_name)
  print("---->prompt_template:"+prompt_template)
  print("---->max_tokens:"+str(max_tokens)+
        "---->temperature:"+str(temperature)+
        "---->repetition_penalty:"+str(repetition_penalty))
  payload = {
    "prompt": prompt,
    "model": model_name,
    "max_tokens": max_tokens,
    "stop": stop,
    "temperature": temperature, #0.7
    "top_p": 0.7,
    "top_k": 50,
    "repetition_penalty": repetition_penalty
  }
  headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "Authorization": "Bearer your_together_API_KEY"
  }

  response = requests.post(URL, json=payload, headers=headers)
  result = None 
  if response.status_code == 200:
    result = response.json()['output']['choices'][0]['text']
    result = result.strip() #clean开头结尾空格
    print("---->full prompt start-------------->")
    print(prompt)
    print("---->full prompt end---------------->")
    print("---->result start------------------->")
    print(result)
    print("---->result end--------------------->")
  else:
    print("---->status not 200---->")
    print(response.content)
      
  write_log_file("prompt",prompt)
  write_log_file("result",result)
  return result

def write_log_file(title,text):
  import global_config
  path = global_config.CURRENT_WORK_FOLDER
  file = open(f"{path}/write_llm.log", 'a')
  file.writelines(f"\n---->{title}---->"+str(datetime.datetime.now().strftime("%H:%M:%S.%f"))+"\n")
  file.writelines(text)
  #file.close()


def llm_inference_single(question):
  print("---->llm_inference_single---->")
  result = run_rest_api(
    question = question,
    model_name = "togethercomputer/llama-2-70b-chat",
    max_tokens = 500,
    temperature = 0.5)
  if result == "":
      print("---->change llm due to empty results---->")
      result = run_rest_api(
        question = question,
        model_name = "togethercomputer/mpt-30b-chat",
        max_tokens = 500,
        temperature = 0.5)
  return result

def llm_inference(question,temperature,max_tokens):#用自带的temperature
  print("---->llm_inference_with_temp&tokens---->")
  if max_tokens < 500:
    max_tokens = max_tokens*2
  result = run_rest_api(
    question = question,
    model_name = "togethercomputer/llama-2-70b-chat",
    max_tokens = max_tokens,
    temperature = temperature)
  if result == "":
      print("---->change llm due to empty results---->")
      result = run_rest_api(
        question = question,
        model_name = "togethercomputer/mpt-30b-chat",
        max_tokens = max_tokens,
        temperature = temperature)        
  return result


def choose_stop_str(prompt_template):
  if "decide_to_talk" in prompt_template:
    print("---->stop_str: decide_to_talk")
    stop_str="</s>"
  elif "iterative_convo" in prompt_template:
    print("---->stop_str: iterative_convo")
    stop_str="</s>"
  elif "generate_hourly_schedule" in prompt_template:
    print("---->stop_str: generate_hourly_schedule")
    stop_str="\n"
  elif "relationship" in prompt_template:
    print("---->stop_str: relationship")
    stop_str="\n"
  else:
    print("---->stop_str: none")
    stop_str="\n"
  return stop_str

def choose_prompt_tag(prompt_template, prompt): #需要合成, 所以双参数
  if "decide_to_talk" in prompt_template:
    print("---->prompt_tag: decide_to_talk")
    prompt = "[INST]{"+prompt+"}[/INST]"
  elif "iterative_convo" in prompt_template:
    print("---->prompt_tag: iterative_convo")
    prompt = "[INST]{"+prompt+"}[/INST]"
  else:
    print("---->prompt_tag: none") #加上INST会导致plan异常
  return prompt

def choose_temperature(prompt_template, temperature):#部分自带了temperature
  if "iterative_convo" in prompt_template:
    print("---->choose_temperature: iterative_convo")
    temperature = 1
  else:
    print("---->choose_temperature: default")
  return temperature

def choose_repetition_penalty(prompt_template):
  if "iterative_convo" in prompt_template:
    print("---->repetition_penalty: iterative_convo")
    repetition_penalty = 1.3
  else:
    print("---->repetition_penalty: default")
    repetition_penalty = 1 #对于routine来说, 重复才是正常
  return repetition_penalty

# ============================================================================
#                   SECTION 1: Together API END
# ============================================================================

# 避免API请求过于频繁, 自己的API无所谓
def temp_sleep(seconds=1):
  time.sleep(seconds)

def ChatGPT_single_request(prompt):
  print("---->ChatGPT_single_request--->") 
  temp_sleep()
  try: 
    return llm_inference_single(question=prompt)
  except Exception as e:
    print ("---->error--->\n"+str(e))
    print ("ChatGPT_single_request ERROR")
    return "ChatGPT ERROR"
# ============================================================================
# #####################[SECTION 1: CHATGPT-3 STRUCTURE] ######################
# ============================================================================

def GPT4_request(prompt):
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  print("---->GPT4_request--->") 
  temp_sleep()
  try: 
      return llm_inference_single(question=prompt)
  except Exception as e:
    print ("---->error--->\n"+str(e))
    print ("GPT4_request ERROR")
    return "ChatGPT ERROR"


def ChatGPT_request(prompt): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  print("---->ChatGPT_request--->") 
  temp_sleep()
  try: 
      return llm_inference_single(question=prompt)
  except Exception as e:
    print ("---->error--->\n"+str(e))
    print ("ChatGPT_request ERROR")
    return "ChatGPT ERROR"


def GPT4_safe_generate_response(prompt, 
                                example_output,
                                special_instruction,
                                repeat=3,
                                fail_safe_response="error",
                                func_validate=None,
                                func_clean_up=None,
                                verbose=False): 
  print("---->GPT4_safe_generate_response")
  prompt_new = f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt_new += "Example output json:\n"
  prompt_new += '{"output": "' + str(example_output) + '"}'
  prompt_new = 'Question:\n"""\n' + prompt + '\n"""\n'

  for i in range(repeat): 
    try: 
      curr_gpt_response = GPT4_request(prompt_new).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]
      
      if func_validate(curr_gpt_response, prompt=prompt_new): 
        return func_clean_up(curr_gpt_response, prompt=prompt_new)
      
      if verbose: 
        print ("---->repeat count: \n", i)
    except Exception as e:
      print ("---->GPT4_safe_generate_response error--->\n"+str(e))
      pass
  return False


def ChatGPT_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  print("---->ChatGPT_safe_generate_response")
  prompt_new = f"{special_instruction}\n"
  prompt_new += "Example output:\n"
  prompt_new += f"{example_output}\n"
  prompt_new = 'Question:\n"""\n' + prompt + '\n"""\n'

  for i in range(repeat): #跑3遍就是失败了重试
    try: 
      curr_gpt_response = ChatGPT_request(prompt)
      curr_gpt_response = curr_gpt_response.replace('\n', '') #去除换行符
      curr_gpt_response = '{"output": "'+curr_gpt_response+'"}'#手动改为json格式
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]
      
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      
      if verbose:
        print ("---->repeat count: \n", i)
    except Exception as e:
      print ("---->ChatGPT_safe_generate_response error--->\n"+str(e))
      pass
  return False


def ChatGPT_safe_generate_response_OLD(prompt, 
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  print ("---->ChatGPT_safe_generate_response_OLD--->")
  for i in range(repeat): 
    try: 
      curr_gpt_response = ChatGPT_request(prompt).strip()
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      if verbose: 
        print (f"---->repeat count: {i}")
    except Exception as e:
      print ("---->ChatGPT_safe_generate_response_OLD error--->\n"+str(e))
      pass
  print ("---->ChatGPT_safe_generate_response_OLD:FAIL SAFE TRIGGERED") 
  return fail_safe_response


# ============================================================================
# ###################[SECTION 2: ORIGINAL GPT-3 STRUCTURE] ###################
# ============================================================================

def GPT_request(prompt, gpt_parameter): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  print("---->GPT_request--->")
  temp_sleep()
  try: 
    return llm_inference(
      question=prompt,
      temperature=gpt_parameter["temperature"],
      max_tokens=gpt_parameter["max_tokens"])
  except Exception as e:
    print ("---->GPT_request error--->\n"+str(e))
    print ("TOKEN LIMIT EXCEEDED")
    return "TOKEN LIMIT EXCEEDED"


def generate_prompt(curr_input, prompt_lib_file): 
  import global_config
  global_config.current_prompt_template = prompt_lib_file
  """
  Takes in the current input (e.g. comment that you want to classifiy) and 
  the path to a prompt file. The prompt file contains the raw str prompt that
  will be used, which contains the following substr: !<INPUT>! -- this 
  function replaces this substr with the actual curr_input to produce the 
  final prompt that will be sent to the GPT3 server. 
  ARGS:
    curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                INPUT, THIS CAN BE A LIST.)
    prompt_lib_file: the path to the prompt file. 
  RETURNS: 
    a str prompt that will be sent to OpenAI's GPT server.  
  """
  if type(curr_input) == type("string"): 
    curr_input = [curr_input]
  curr_input = [str(i) for i in curr_input]

  f = open(prompt_lib_file, "r")
  prompt = f.read()
  f.close()
  for count, i in enumerate(curr_input):   
    prompt = prompt.replace(f"!<INPUT {count}>!", i)
  if "<commentblockmarker>###</commentblockmarker>" in prompt: 
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
  return prompt.strip()


def safe_generate_response(prompt, 
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False): 
  print ("---->safe_generate_response---->")
  for i in range(repeat): 
    curr_gpt_response = GPT_request(prompt, gpt_parameter)  
    if func_validate(curr_gpt_response, prompt=prompt): 
      return func_clean_up(curr_gpt_response, prompt=prompt)
    if verbose: 
      print ("---->repeat count: ", i, curr_gpt_response)
  return fail_safe_response


#这里使用GPT4ALL的embeding
def get_embedding(text):
  print ("--->get_embedding---->")
  text = text.replace("\n", " ")
  if not text:
    text = "this is blank"
  embedder = Embed4All()
  embedding = embedder.embed(text)
  return embedding


if __name__ == '__main__':
  gpt_parameter = {"max_tokens": max_tokens,
                     "temperature": temperature, "top_p": 1, "stream": False,
                     "frequency_penalty": 0, "presence_penalty": 0,
                     "stop": ['"']}
  curr_input = ["driving to a friend's house"]
  prompt_lib_file = "prompt_template/test_prompt_July5.txt"
  prompt = generate_prompt(curr_input, prompt_lib_file)

  def __func_validate(gpt_response): 
    if len(gpt_response.strip()) <= 1:
      return False
    if len(gpt_response.strip().split(" ")) > 1: 
      return False
    return True
  def __func_clean_up(gpt_response):
    cleaned_response = gpt_response.strip()
    return cleaned_response

  output = safe_generate_response(prompt, 
                                 gpt_parameter,
                                 5,
                                 "rest",
                                 __func_validate,
                                 __func_clean_up,
                                 True)
  print ("--->output start---->")
  print (output)
  print ("--->output end---->")