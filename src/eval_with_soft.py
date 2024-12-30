import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import numpy as np
import pandas as pd
import time
import re
import torch
import json
from transformers import AutoTokenizer, AutoModel,AutoModelForCausalLM
from tqdm import tqdm
from collections import OrderedDict
import pickle as pkl
from transformers.generation import GenerationConfig

def load_object(path):
    with open(path, 'rb') as f:
        obj = pkl.load(f)
    return obj

def save_object(obj, path):
    with open(path,'wb') as f:
        pkl.dump(obj,f)






tokenizer = AutoTokenizer.from_pretrained("model_chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("model_chat", trust_remote_code=True).half().eval().cuda()

small_device = torch.device("cuda:{}".format(1))
small_tokenizer = AutoTokenizer.from_pretrained("small_model",trust_remote_code=True)
small_model = AutoModelForCausalLM.from_pretrained("small_model",trust_remote_code=True).eval().cuda()



choices = ["A", "B", "C", "D"]

def find_valid_substrings(s):

    pattern = r'[ABCD]{1,4}'
    substrings = re.findall(pattern, s)

    valid_substrings = [substring for substring in substrings if len(substring) == len(set(substring))]
    valid_substrings = "".join(valid_substrings)
    valid_substrings= ''.join(OrderedDict.fromkeys(valid_substrings))
    return valid_substrings

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s




# chatglm
# def plain_chat(prompt,tokenizer,model):

  
#     response,history = model.chat(tokenizer, prompt,history=None)

#     return response
def plain_chat(prompt,tokenizer,model):

    messages = []
    messages.append({"role": "user", "content": prompt})

    config = GenerationConfig(
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            user_token_id=195,
            assistant_token_id=196,
            max_new_tokens=512,
            do_sample=False,
            transformers_version="4.29.2"
        )
    response = model.chat(tokenizer, messages,generation_config=config)

    return response


def eval(args):

    generation_kwargs = {
                    "min_length": 0,
                    "max_new_tokens": 512,
                    'do_sample':False
                }
    prompt = "请阅读以下选择题并根据解析中的法律知识给出正确选项，不要解释原因。请只给出答案的序号。\n"
    logpath = args.save_dir + '/' + args.log_name
    logfile  = open(logpath, 'w', encoding='utf8')

    embeddings = small_model.get_input_embeddings().weight.clone()
    soft_token = load_object('soft_token.pkl').cuda()
    

    path = args.data_dir    
    file = open(path,'r')
    cors = []
    outpath = args.save_dir + '/' + args.output_name
    outfile  = open(outpath, 'w', encoding='utf8')
    #labels = []
    preds = []
    single_cors = []
    num = 0
    for line in tqdm(file.readlines()):
        num = num + 1
        a_dict = json.loads(line)

        small_prompt = '让我们一步步思考并逐项分析，生成与以下问题有关的文档：'
        small_input = small_prompt + a_dict['query']

        prefix="Below is an instruction that describes a task. Write a response that appropriately completes the request."
        small_input=f'### Instruction:\n{small_input}\n\n### Response:\n'
        small_input = prefix + '\n\n' + small_input


        input_ids = small_tokenizer(small_input, return_tensors="pt")['input_ids']
        input_embed = embeddings[input_ids]
        prompt_embedding = soft_token.to(device=input_embed.device, dtype=input_embed.dtype)
        input_embeds = torch.cat((prompt_embedding, input_embed), 1)


        input_text = small_tokenizer.encode(small_input)
        input_ids = torch.LongTensor([input_text]).cuda()
        with torch.no_grad():
            out = small_model.generate(
                        inputs_embeds=input_embeds,**generation_kwargs
                    )
        out_text = small_tokenizer.decode(out[0],skip_special_tokens=True).split('Response:\n')[-1]
        out_text = out_text.split('###Answer:')[0]


        
        query = prompt + '解析:' + out_text +  '\n' + '问题:' + a_dict['query']  + '\n' + '答案:'
        answer = a_dict['answer']

        pred = plain_chat(query,tokenizer,model)
        raw_pred = pred
        pred = pred.split('解析')[0].split('分析')[0]
        pred = pred.replace("、", "").replace(".", "").replace(",", "").replace(";", "").replace("，", "").replace("和", "").replace(", ", "")
        try:                # 识别答案pattern
            pred = find_valid_substrings(pred)
            # pred = set(pred)[0]
            # print(pred)
        except Exception as e:
            pred = "未成功回答"
        
        
        cor = pred == answer


        save_dict = {}
        save_dict['num'] = num
        save_dict['input'] = query
        save_dict['output'] = raw_pred
        save_dict['pred'] = pred
        save_dict['answer'] = answer
        save_dict['final'] = cor
        outline = json.dumps(save_dict,ensure_ascii=False)+'\n'
        outfile.write(outline)
        cors.append(cor)
        if len(answer) == 1:
            single_cors.append(cor)

    acc = np.mean(cors)
    acc_info = "Average accuracy {:.10f}".format(acc)
    print(acc_info)

    logfile.write(f'{acc_info}'+ '\n')

    single_acc = np.mean(single_cors)
    acc_info = "Average accuracy {:.10f}".format(single_acc)
    print(acc_info)

    logfile.write(f'{acc_info}')

def main(args):
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    preds = eval(args)

   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_name", "-o", type=str, default='output.json')
    parser.add_argument("--log_name", "-l", type=str, default='log.txt')
    parser.add_argument("--data_dir", "-d", type=str, default="test_0_all.json")
    parser.add_argument("--save_dir", "-s", type=str, default="save_path")

    args = parser.parse_args()
    main(args)
