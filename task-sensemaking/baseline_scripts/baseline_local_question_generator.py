import json
import os
import os.path
import re
import random
import string
from itertools import chain
import numpy as np
import regex
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer, GenerationConfig
from transformers import AutoTokenizer, BitsAndBytesConfig
from udapi.block.corefud.movehead import MoveHead
from udapi.block.read.conllu import Conllu as ConlluReader
from datasets import load_dataset
import argparse
from helpers import *

parser = argparse.ArgumentParser()
parser.add_argument("--token", default=default_access_token, type=str, help="Huggingface token.")


def reshape(enumerable, per=25):
    r = []
    for i, x in enumerate(enumerable):
        if i % per == 0:
            r.append([])
        r[-1].append(x)
    return r


def filter_f(x):
    return x not in string.punctuation


def flatten(x):
    return list(chain.from_iterable(x))


def mask_reponse(x, y):
    return [x if y > 0 else -100 for x, y in zip(x, y)]


def unzip(l):
    assert len(l) > 0
    r = [[] for _ in l[0]]
    for x in l:
        for i in range(len(r)):
            r[i].append(x[i])
    return r

class mock_model:
    def __init__(self):
        self.device = "cuda"

    def forward(self, input_tensor):
        return {
            "logits": torch.tensor([[[0] for y in range(input_tensor.shape[1])] for x in range(input_tensor.shape[0])])}

def make_model(args, mock = False):
    cuda = torch.cuda.is_available()
    locs = ["meta-llama/Llama-3.1-8B",
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-Guard-3-8B-INT8",
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.2-1B-Instruct"]
    loc = locs[-2]
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16) if cuda else None
    model = mock_model() if mock else AutoModelForCausalLM.from_pretrained(loc, token=args.access_token, force_download=False,
                                                 quantization_config=quantization_config, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(loc, token=access_token, force_download=False, add_bos_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    return model, tokenizer

def make_prompt(text, question, answer, tokenizer: PreTrainedTokenizer):
    system_prompt = "Create a question given a text."
    user_prompt = f"Create a question and an example short answer given the following text. {text}"
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": user_prompt}], tokenize=True, return_dict=True, add_generation_prompt=True)
    input_ids = prompt["input_ids"] + (tokenizer.encode(f"\n{question}\n{answer}") if question is not None and answer is not None else [])
    #print(tokenizer.decode(prompt["input_ids"]), tokenizer.decode(input_ids))
    return input_ids, [1 if x > len(prompt["input_ids"]) else 0 for x in range(len(input_ids))]

class Collator:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
    def collate_fn(self, datapoints):
        finishedtokens, masks = unzip([make_prompt(x["context"], x["question"], x["answers"]["text"][0], self.tokenizer) for x in datapoints])
        lens = [len(x) for x in finishedtokens]
        ml = max(lens)
        finishedtokens, masks, attention_masks = unzip(
            [(x + [0] * (ml - len(x)), y + [0] * (ml - len(x)), [1] * len(x) + [0] * (ml - len(x))) for x, y in zip(finishedtokens, masks)])
        return {"concatenated": torch.tensor(finishedtokens),
                "response_mask": torch.tensor(masks),
                "attention_mask": torch.tensor(attention_masks)}

def train(model, tokenizer):
    dataset_path = "rajpurkar/squad"
    train_dataset = load_dataset(dataset_path, split="train", trust_remote_code=True).with_format("torch")
    valid_dataset = load_dataset(dataset_path, split="validation", trust_remote_code=True).with_format("torch")
    train_dataset = torch.utils.data.Subset(train_dataset, np.arange(10))
    valid_dataset = torch.utils.data.Subset(valid_dataset, np.arange(2))
    collator = Collator(tokenizer)
    BS = 4
    train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True, collate_fn=collator.collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=BS, shuffle=False, collate_fn=collator.collate_fn)
    model, tokenizer = lora(model, train_loader, valid_loader, tokenizer, 100, "data/model")
    return model, tokenizer

def make_devset_json(root):
    l = []
    ln = len(root)
    for (root_i, dirs, files) in os.walk(root):
        for x in files:
            if x.endswith(".json"):
                l.append(os.path.join(root_i, x)[ln + 1:])
    d = {}
    for x in l:
        p = os.path.join(root, x)
        d[x] = json.load(open(p))
    json.dump(d, open("devset.json", "w"))


def eval_test(model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    path2 = "/mnt/c/school/mtproject/data/sensemaking-2025-internal-data/devset"
    path = "/mnt/c/school/eloquent-lab.github.io/task-sensemaking/devset"
    make_devset_json(path)
    r = load_questions_and_texts("devset.json", path)
    def get_first_end_with_sentence(x, amount):
        vs = x[:amount]
        i = vs.rfind(".")
        return vs[:i+1]

    prompts = [make_prompt(get_first_end_with_sentence(x[1],1000), None, None, tokenizer)[0] for x in r]
    generationConfig = [GenerationConfig(max_new_tokens=100), GenerationConfig(num_beams=5, max_new_tokens=100),  GenerationConfig(top_k=2 ,max_new_tokens=100,do_sample=True)]
    for x in prompts:
        x = x + tokenizer.encode("""

Here's a question and an example short answer based on the given text:

**Question:** """)
        output = model.generate(torch.tensor([x]).to(model.device), generationConfig[1])
        text = tokenizer.decode(output[0])
        print(text)


def main(args):
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    finetune = False
    model, tokenizer = make_model(args)
    os.makedirs("data", exist_ok=True)
    if finetune:
        model, tokenizer = train(model, tokenizer)
    eval_test(model, tokenizer)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
