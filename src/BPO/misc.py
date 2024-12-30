import torch
import random
import numpy as np
import os
from torch.utils.data import Dataset
from tqdm import tqdm
import json
from transformers import DefaultDataCollator,PreTrainedTokenizer


class InstrutionDataset(Dataset):
    def __init__(self, data_path):
        self.dataset = []
        with open(data_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                sample = json.loads(line.strip())
                self.dataset.append(
                    {"prompt":sample["prompt"],"input": sample["query"],"answer": sample["answer"]})
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):   
        return self.dataset[item]




class Instruction_formatter:
    def __init__(self, tokenizer, max_len, system_prompt,prompt_embedding, embeddings):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.prompt_embedding = prompt_embedding
        self.embeddings = embeddings
    
    def __call__(self, examples):
        max_len = min(self.max_len, max([len(inp['input']+inp['prompt']) for inp in examples]))
        input_embeds = None
        answers = []
        querys = []
        for example in examples:
            prompt = example['prompt']
            query = example['input']
            input_query = prompt + query
            input_text=f'### Instruction:\n{input_query}\n\n### Response:\n'
            input_text = self.system_prompt + '\n\n' + input_text
            # print(input_text)
            
            input_ids = self.tokenizer(input_text, return_tensors="pt")['input_ids']
            input_embed = self.embeddings[input_ids]
            prompt_embedding = self.prompt_embedding.to(device=input_embed.device, dtype=input_embed.dtype)
            input_embed = torch.cat((prompt_embedding, input_embed), 1)
            # print(input_embed.shape)
            if input_embeds == None:
                input_embeds = input_embed
            else:
                input_embeds = torch.cat([input_embeds, input_embed], dim=0)
            answer = example['answer']
            answers.append(answer)
            querys.append(query)
        # print(input_embeds.shape)
        return input_embeds, answers, querys


# class Instruction_formatter:
#     def __init__(self, tokenizer, max_len, system_prompt,prompt_embedding, embeddings):
#         self.max_len = max_len
#         self.tokenizer = tokenizer
#         self.system_prompt = system_prompt
#         self.prompt_embedding = prompt_embedding
#         self.embeddings = embeddings
    
#     def __call__(self, examples):
#         max_len = min(self.max_len, max([len(inp['input']+inp['prompt']) for inp in examples]))
#         input_embeds = None
#         answers = []
#         querys = []
#         for example in examples:
#             prompt = example['prompt']
#             query = example['input']
#             input_query = prompt + query
#             input_text=f'### Instruction:\n{input_query}\n\n### Response:\n'
#             input_text = self.system_prompt + '\n\n' + input_text
#             input_ids = self.tokenizer(input_text, return_tensors="pt")['input_ids']
            
#             # input_ids = self.tokenizer(input_text, return_tensors="pt")['input_ids']
#             input_embed = self.embeddings[input_ids]
#             prompt_embedding = self.prompt_embedding.to(device=input_embed.device, dtype=input_embed.dtype)
#             input_embed = torch.cat((prompt_embedding, input_embed), 1)
#             # print(input_embed.shape)
#             # if input_embeds == None:
#             #     input_embeds = input_embed
#             # else:
#             #     input_embeds = torch.cat([input_embeds, input_embed], dim=0)
#             # answer = example['answer']
#             # answers.append(answer)
#             # querys.append(query)
#         # print(input_embeds.shape)
#         return input_embed, example['answer'], query


def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    return f"Set all the seeds to {seed} successfully!"

def cma_es_concat(starting_point_for_cma, EI, tkwargs):
        if starting_point_for_cma.type() == 'torch.cuda.DoubleTensor':
            starting_point_for_cma = starting_point_for_cma.detach().cpu().squeeze()
        es = cma.CMAEvolutionStrategy(x0=starting_point_for_cma, sigma0=0.8, inopts={'bounds': [-1, 1], "popsize": 50},)
        iter = 1
        while not es.stop():
            iter += 1
            xs = es.ask()
            X = torch.tensor(np.array(xs)).float().unsqueeze(1).to(**tkwargs)
            with torch.no_grad():
                Y = -1 * EI(X)
            es.tell(xs, Y.cpu().numpy())  # return the result to the optimizer
            print("current best")
            print(f"{es.best.f}")
            if (iter > 10):
                break

        return es.best.x, -1 * es.best.f