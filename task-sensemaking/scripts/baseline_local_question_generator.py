import argparse
import os.path
import random
import string
from itertools import chain

import numpy as np
import torch.utils.data
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer, GenerationConfig
from transformers import AutoTokenizer, BitsAndBytesConfig

from eval_scripts import teacher_evaluator
from eval_scripts.helpers import *

parser = argparse.ArgumentParser()
parser.add_argument("--token", default=default_access_token, type=str, help="Huggingface token.")

locs = ["meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct"]

parser.add_argument("--model_name", default=locs[2], type=str, help="The model and tokenizer to use.")
parser.add_argument("--max_new_tokens", default=200, type=int, help="Max tokens for the model to generate.")
parser.add_argument("--data_path", default="../devset", type=str, help="Path to the folder containing the data.")
parser.add_argument("--words_per", default=200, type=int, help="Number of words to feed the network.")


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


def make_model(args, mock=False):
    cuda = torch.cuda.is_available()
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16) if cuda else None
    model = mock_model() if mock else AutoModelForCausalLM.from_pretrained(args.model_name, token=args.token,
                                                                           force_download=False,
                                                                           quantization_config=quantization_config,
                                                                           device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.token, force_download=False,
                                              add_bos_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    return model, tokenizer


def make_prompt(text, question, answer, tokenizer: PreTrainedTokenizer, question_tag, answer_tag):
    system_prompt = f"You use only English. You are a university teacher's assistant. Create a sequence of question and example answer pairs given a text. Make sure the questions challenge the students but are answerable. Do not describe your output, give the output directly. Mark each question with \"{question_tag}\" and the answer with \"{answer_tag}\"."
    user_prompt = f"Create questions and example short answers given the following text. {text}"
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": user_prompt}], tokenize=True, return_dict=True, add_generation_prompt=True)
    input_ids = prompt["input_ids"] + (
        tokenizer.encode(f"\n{question}\n{answer}") if question is not None and answer is not None else [])
    # print(tokenizer.decode(prompt["input_ids"]), tokenizer.decode(input_ids))
    return input_ids, [1 if x > len(prompt["input_ids"]) else 0 for x in range(len(input_ids))]


class Collator:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def collate_fn(self, datapoints):
        finishedtokens, masks = unzip(
            [make_prompt(x["context"], x["question"], x["answers"]["text"][0], self.tokenizer) for x in datapoints])
        lens = [len(x) for x in finishedtokens]
        ml = max(lens)
        finishedtokens, masks, attention_masks = unzip(
            [(x + [0] * (ml - len(x)), y + [0] * (ml - len(x)), [1] * len(x) + [0] * (ml - len(x))) for x, y in
             zip(finishedtokens, masks)])
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
    json.dump(d, open("devset.json", "w"), indent=4, ensure_ascii=False)


def eval_test(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, args):
    make_devset_json(args.data_path)
    vs = load_questions_and_texts("devset.json", args.data_path, True)
    r = {}

    def get_first_end_with_sentence(x, amount):
        vs = x[:amount]
        i = vs.rfind(".")
        return vs[:i + 1]

    question_tag = "**Question:**"
    answer_tag = "**Answer:**"
    prompts = [
        (make_prompt(get_first_end_with_sentence(text, args.words_per), None, None, tokenizer, question_tag, answer_tag)[0], path)
        for question, text, question_reference, path in vs]
    GenerationConfig(num_beams=5, do_sample=True, top_k=10, top_p=0.9)
    bad_words_ids = [tokenizer.encode("<|eot_id|>", add_special_tokens=False)]
    print(bad_words_ids)
    generationConfig = GenerationConfig(num_beams=10, num_beam_groups=2, max_new_tokens=args.max_new_tokens,
                                        eos_token_id=model.config.eos_token_id[0],
                                        bad_words_ids=bad_words_ids, diversity_penalty=0.1, pad_token_id=model.config.pad_token_id[0])
    for i, (x, path) in enumerate(prompts):
        x = x + tokenizer.encode(f"""\n\n{question_tag}""")
        print("context size", len(x))
        if len(x) > 8192-args.max_new_tokens:
            print("Warning: context size is too large.")

        inp = torch.tensor([x]).to(model.device)
        output = model.generate(inp, generationConfig,attention_mask = torch.ones_like(inp))
        t = tokenizer.decode(output[0])
        print(t)
        lq = len(question_tag)
        la = len(answer_tag)
        questions = [x.strip()[lq:] for x in t.split("\n") if x.strip().startswith(question_tag)]
        answers = [x.strip()[la:] for x in t.split("\n") if x.strip().startswith(answer_tag)]
        questions = questions if len(questions) > 0 else ["Generation failed."]
        r[path] = list(zip(questions, answers)) if len(answers) > 0 else [questions[0]]
    return r


def main(args):
    args.experiment_name = "baseline2"
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    eval_only = False
    finetune = False
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    if not eval_only:
        model, tokenizer = make_model(args)
        if finetune:
            model, tokenizer = train(model, tokenizer)
        r = eval_test(model, tokenizer, args)
        with open(f"outputs/{args.experiment_name}.json", "w") as f:
            json.dump(r, f, indent=4, ensure_ascii=False)
    r_1 = teacher_evaluator.evaluate(json_path=f"outputs/{args.experiment_name}.json", data_path=args.data_path, outs_path="outs_baseline", top_windows = 2)
    for x in r_1:
        for y in r_1[x]:
            for z in r_1[x][y]:
                z.pop("avg_complexity(mocked)")
                z.pop("avg_answerability(mocked)")
    json.dump(r_1, open(f"results/{args.experiment_name}.json", "w"), indent=4, ensure_ascii=False)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
