"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling OpenAI APIs.
"""
import json
import random
import openai
import time 
import os
from utils import *
import textwrap
import logging
import together
import datetime

from utils import *
from typing import Any, Dict, List, Mapping, Optional

from pydantic import Extra, Field, root_validator
from langchain.memory import ConversationBufferMemory
from gpt4all import GPT4All, Embed4All

# ============================================================================
#                   SECTION 1: Together PYPI STRUCTURE
# ============================================================================
def run_rest_api(question, temperature, max_new_tokens,stop, model_name):
  import global_config
  print("---->"+model_name+"---->count_request:"+str(global_config.count_request))
  global_config.count_request += 1
  
  prompt = question.replace('\n\n', '\n') #clean换行符2->1;这里暂时不用template
  prompt = prompt.strip() #clean去掉前后空格
  
  together.Complete.create(
    model = model_name,
    prompt = prompt,
    max_tokens = max_new_tokens,
    stop = stop,
    temperature = 0.1,
    top_p = 0.8,
    top_k = 90,
    repetition_penalty = 1.1
  )  
  result = output['output']['choices'][0]['text'])
  result = result.strip() #clean开头结尾空格
  print("---->full prompt start-------------->")
  print(prompt)
  print("---->full prompt end---------------->")
  print("---->result start------------------->")
  print(result)
  print("---->result end--------------------->")
      
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
  print("---->single,temperature to 0.1,max_token to 500----->")
  result = run_rest_api(
    question=question,
    temperature=0.1,
    max_new_tokens=500,
    stop="\n",
    model_name = "togethercomputer/llama-2-70b-chat")
  if result == "":
      print("---->change stop to .----->")#只尝试一次
      result = run_rest_api(
        question=question,
        temperature=0.1,
        max_new_tokens=500,
        stop=".",
        model_name = "togethercomputer/mpt-30b-chat")
  return result

def llm_inference(question,temperature,max_tokens):
  #太大了会比较乱, 太小了无法step by step reasoning
  if max_tokens < 1000:
    max_tokens = max_tokens*2
  print("---->temp:"+str(0.01)+",tokens:"+str(max_tokens)+"---->")
  result = run_rest_api(
    question=question,
    temperature=0.01,
    max_new_tokens=max_tokens,
    stop="\n",
    model_name = "togethercomputer/llama-2-70b-chat")
  if result == "":
      print("---->change stop to .----->")
      result = run_rest_api(
        question=question,
        temperature=0.01,
        max_new_tokens=max_tokens,
        stop=".",
        model_name = "togethercomputer/mpt-30b-chat")
  return result
  
# ============================================================================
#                   SECTION 1: Together PYPI END
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
  print ("---->FAIL SAFE TRIGGERED") 
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