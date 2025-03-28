import argparse
import os.path
import random
import shutil
from itertools import chain

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig

from .helpers import *

parser = argparse.ArgumentParser()
parser.add_argument("--token", default=default_access_token, type=str, help="Huggingface token.")
parser.add_argument("--json_path", default="devset.json", type=str, help="Path to the json containing the outputs.")
parser.add_argument("--data_path", default="../devset", type=str, help="Path to the folder containing the data.")
parser.add_argument("--outs_path", default="outs", type=str, help="Path to the folder containing the data.")
parser.add_argument("--verbose", default=False, action="store_true")
parser.add_argument("--llm_amount", default=0, type=int)
parser.add_argument("--top_windows", default=0, type=int)
parser.add_argument("--cache", default=False, action="store_true")

SYSTEM_PROMPT_ANSWERABILITY = "You are a model that is given questions and tasked with evaluating whether it can be answered."
SYSTEM_PROMPT_COMPLEXITY = "You are a model that is given questions and tasked with evaluating how hard it is to answer."
RESPONSE_PROMPT = f""

parser.add_argument("--system_prompt_answerability", default=SYSTEM_PROMPT_ANSWERABILITY, type=str)
parser.add_argument("--response_prompt_answerability", default=RESPONSE_PROMPT, type=str)
parser.add_argument("--system_prompt_complexity", default=SYSTEM_PROMPT_COMPLEXITY, type=str)
parser.add_argument("--response_prompt_complexity", default=RESPONSE_PROMPT, type=str)


def run(model, tokenizer, contexts, questions, args, batchsize=10, verbose=False, answerability=False):
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    if verbose:
        print("*" * 80)
        print(f"System prompt: {args.system_prompt}")
        print(f"Response prompt: {args.response_prompt}")
        print("*" * 80)

    numbers = list(range(0, 10))
    numbers_encodings = [tokenizer.encode(str(x)) for x in numbers]
    r = []
    for input_text in questions:
        r.append([])
        for text in contexts:
            torch.cuda.empty_cache()
            if verbose:
                print(f"Context: {text}")
                print(f"Input: {input_text}")
            scale = ["trivial", "very easy", "quite easy", "easy", "somewhat easy", "somewhat challenging",
                     "challenging", "very challenging", "almost impossible", "impossible"]
            scale_string = " ".join([f"{i}: {x}" for i, x in enumerate(scale)])

            def get_prompt_complexity(r1, r2):
                prompt = tokenizer.apply_chat_template(
                    [{"role": "system",
                      "content": f"{args.system_prompt_answerability}. Use the following scale {scale_string}"},
                     {"role": "user",
                      "content": f"{args.response_prompt_answerability}. Given the following context: {text} I would rate the question: {input_text} {r1} out of 9, {scale[r1]} to answer or {r2} out of 9, {scale[r2]} to answer. Which do you think is the correct rating?"}],
                    tokenize=False)

                return prompt + "<|start_header_id|>assistant<|end_header_id|>\nI'd rate it with the number <|eot_id|>"""

            def get_prompt_answerability(r1):
                prompt = tokenizer.apply_chat_template(
                    [{"role": "system",
                      "content": f"{args.system_prompt_answerability}. You answer 1 if you agree with the user and 0 otherwise."},
                     {"role": "user",
                      "content": f"{args.response_prompt_answerability}. Given the following context: {text} I would rate the question \"{input_text}\" as {'not' if r1 == 0 else ''} answerable, do you think this is the correct rating?"}],
                    tokenize=False)
                return prompt + "<|start_header_id|>assistant<|end_header_id|>\nI say <|eot_id|>"

            def eval_answerability(i, response, prompts, values, probs, non_selected):
                logits = response.logits[i, len(prompts[i]) - 2]
                dist = torch.softmax(logits, 0)
                order = torch.argsort(logits, descending=True)
                v = tokenizer.decode(order[0])
                values.append(1 if v == ("0" if r1 == 0 else "1") else 0)
                values.append(0 if v == ("0" if r1 == 0 else "1") else 1)
                probs.append(dist[order[0]].item())
                probs.append(1 - dist[order[0]].item())
                non_selected.append(1 - int(v))

            values = []
            non_selected = []
            probs = []
            ##Run a couple of reference values to select from through the neural network.
            if answerability:
                for r1 in [0, 1]:
                    prompts = [tokenizer.encode(get_prompt_answerability(r1))]
                    ml = max(len(x) for x in prompts)
                    promptsel = [x + [0] * (ml - len(x)) for x in prompts]
                    data = torch.tensor(promptsel, device=model.device)
                    with torch.no_grad():
                        response = model.forward(data)
                    del data
                    for i in range(len(prompts)):
                        eval_answerability(i, response, prompts, values, probs, non_selected)
                    del response
            else:
                r1s = np.arange(10)
                np.random.shuffle(r1s)
                r2s = np.arange(10)
                np.random.shuffle(r2s)
                for r1 in r1s:
                    np.random.shuffle(r2s)
                    r2si = r2s[r2s != r1][:batchsize]
                    prompts = [tokenizer.encode(get_prompt_complexity(r1, r2)) for r2 in r2si]
                    ml = max(len(x) for x in prompts)
                    promptsel = [x + [0] * (ml - len(x)) for x in prompts]
                    data = torch.tensor(promptsel, device=model.device)
                    with torch.no_grad():
                        response = model.forward(data)
                    del data
                    for i in range(len(prompts)):
                        logits = response.logits[i, len(prompts[i]) - 2]
                        dist = torch.softmax(logits, 0)
                        order = torch.argsort(logits, descending=True)
                        v = tokenizer.decode(order[0])
                        values.append(v)
                        probs.append(dist[order[0]].item())
                        non_selected.append(r1 if v.isnumeric() or r1 != int(v) else r2si[i])
                    del response
            mean = np.sum([int(x) * y for x, y in zip(values, probs)]) / np.sum(probs)
            if verbose:
                print([f"{x}-{z}:{y:0.2f}" for x, y, z in zip(values, probs, non_selected)])
                print("---mean", mean)
            r[-1].append(mean)
    return r


def r1f1score(set1, set2):
    il = len(set1.intersection(set2))
    r1prec = il / len(set1)
    r1rec = il / len(set2)
    if r1rec + r1prec == 0:
        return 0
    return 2 * (r1prec * r1rec) / (r1rec + r1prec)


def loadq(path, requiresanswers, forbidabcd):
    with open(path) as f:
        return [x for x in json.loads(f.read()) if (not requiresanswers or len(x) > 1) and (
                not forbidabcd or len(x) == 1 or x[1] not in ["A", "B", "C", "D"])]


def get_windows(list, size, size_is_words_muli, step):
    """
    Generate all windows of given size from the list.

    :param list: Input list.
    :param size: Size of the windows.
    :param size_is_words_muli: If < 0, windows are generated by number of sentences,
    otherwise by a number of words equal to size_is_words_muli*size.
    :param step: Step size in sentences between windows.
    :return: List of windows.
    :return:
    """
    r = []
    for x in range(0, len(list) - size, step):
        if size_is_words_muli < 0:
            r.append(list[x:x + size])
        else:
            sizew = size_is_words_muli * size
            sentances = []
            suml = 0
            i = 0
            while suml < sizew and i < size:
                s = list[x + i]
                sentances.append(s)
                suml += s.count(" ")
                i += 1
            r.append(sentances)
    return r


def get_all_windows(list, minsize=0, maxsize=None, size_is_words_muli=-1, step=1):
    r = []
    for x in range(minsize, len(list) if maxsize is None else maxsize):
        r = r + get_windows(list, x, size_is_words_muli, step)
    return r


def entropy(dist):
    dist = np.array(dist) / np.sum(dist)
    return np.sum(-np.log2(np.where(dist > 0, dist, 1)) * dist)


def kldiv(dist1, dist2):
    additional = 1 / (dist1.shape[0])
    dist1 = dist1 + additional
    dist2 = dist2 + additional
    dist1 = np.array(dist1) / np.sum(dist1)
    dist2 = np.array(dist2) / np.sum(dist2)
    return np.sum(dist1 * np.log2(np.where(dist1 == 0, 0,(dist1 / dist2))))


def r1f1similarity(questions, sentanceset):
    sentanceset_wordset = set(" ".join(sentanceset).split(" "))
    m = ([r1f1score(sentanceset_wordset, set(question.split(" "))) for question in questions])
    return m


class Nnsim:
    """Defines a text section similarity using a sentence embedding model."""

    def __init__(self):
        modelname = "paraphrase-multilingual-mpnet-base-v2" if True else 'all-mpnet-base-v2'
        self.model = SentenceTransformer(modelname, device="cpu")

    def nnsimilarity(self, question, sentanceset):
        embeds = self.model.encode([question] + sentanceset)
        question_embed = embeds[0:1]
        sentanceset_embed = embeds[1:].transpose(1, 0)
        return np.mean(question_embed @ sentanceset_embed)

    def nnembed(self, lists):
        embeds = self.model.encode(list(chain.from_iterable(lists)))
        r = []
        i = 0
        for x in lists:
            r.append([])
            for y in x:
                r[-1].append(embeds[i])
                i += 1
        return r


def expand_and_broadcast(arr, dim, shape):
    return np.broadcast_to(np.expand_dims(arr, dim), shape)


def cosinesimilarity(question_embeds, sentanceset_embeds, args):
    """
    A version of cosine similarity that uses different ways to redistribute the mass of a distribution to make it more peaky.
    :param question_embeds:
    :param sentanceset_embeds:
    :param args: Defines the type of redistribution to use. First half (value) describes the value to use the second part (part) describes how to interpret it.
    If type is quantile remove all values less that the [value] quantile and replace them with epsilon. Distribute the rest among other values.
    If the type is similarity take power to 1/[value] of the vector.
    Otherwise replace all values less than [value] with epsilon.
    :return:
    """
    val = float(args.split(" ")[0])
    type = args.split(" ")[1]
    if question_embeds is None:
        return
    question_embeds = np.array(question_embeds)
    sentanceset_embeds = np.array(sentanceset_embeds)
    question_embeds = question_embeds / expand_and_broadcast(
        np.sum(question_embeds @ question_embeds.transpose(1, 0), 1), 1, question_embeds.shape)
    sentanceset_embeds = sentanceset_embeds / expand_and_broadcast(
        np.sum(sentanceset_embeds @ sentanceset_embeds.transpose(1, 0), 1), 1, sentanceset_embeds.shape)
    m1 = question_embeds @ sentanceset_embeds.transpose(1, 0)
    m = np.mean(m1, 1)
    if type == "quantile":
        q = np.quantile(m, val)
        c = np.where(m > q)[0].shape[0]
        if c != 0:
            to_dist = np.sum(np.where(m > q, 0, np.quantile(m, val / 2)))
            vals = (np.where(m > q, m + to_dist / c, np.finfo(float).eps))
        else:
            vals = m
    elif type == "similarity":
        k = (np.where(m > 0, np.power(m, 1 / val), 0))
        vals = k
    else:
        vals = (np.where(m > val, m, np.finfo(float).eps)).tolist()
    return vals, m1


def identity(x):
    return x


def kldivs(arr):
    """
    Kl-divergence.
    :param arr:
    :return:
    """
    r = []
    one_then_zeros = np.concatenate([[1], np.zeros(arr.shape[0] - 1)])
    maxkldiv = kldiv(one_then_zeros, 1 - one_then_zeros)
    for x in range(arr.shape[1]):
        for y in range(arr.shape[1]):
            if x != y:
                r.append(kldiv(arr[:, x], arr[:, y]) / maxkldiv)
    return r


def normalized_entropy(dist):
    """Return entropy divided by the maximum (entropy of a discrete uniform distribution with the same number of elements)."""
    dist = np.array(dist) / np.sum(dist)
    return entropy(dist) / np.log2(dist.shape[0])


def harmonic_mean(values):
    values = np.asarray(values)
    values = np.maximum(values, 0.0001)
    return values.shape[0] / np.sum(1 / values)


def results(similiarity, prew, preq, args, questionset, windows, **kwargs):
    """
    Calculate a set of results for all combinations of a set of similarity metrics given by args and list of sets of questions working on a set of windows.
    :param similiarity: The similarity function taking the elements of the output of prew and preq.
    :param prew: The function used to preparse windows.
    :param preq: The function used to preparse questions.
    :param args: The arguments parametrizing the set of similarity metrics.
    :param questionset: The list of sets of questions. Each set is a tuple of (name, list of questions).
    :param windows: The set of windows.
    :param global_args:
    :return:
    """
    r = {}
    questionsetpre = preq([[y for y in x[1]] for x in questionset])
    if kwargs["cache"]:
        windowspref = "windowspre"
        if os.path.isfile(windowspref):
            windowspre = np.load(windowspref)
        else:
            windowspre = prew(windows)
            np.save(windowspref, np.array(windowspre))
    else:
        windowspre = prew(windows)
    if kwargs["llm_amount"] > 0:
        access_token = kwargs["access_token"]
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", token=access_token)
        quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                                 bnb_4bit_compute_dtype=torch.bfloat16)
        model2 = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", token=access_token,
                                                      quantization_config=quantization_config, device_map="auto")
    for arg in args:
        r[arg] = []
        for (name, questionstext), questions in zip(questionset, questionsetpre):
            out = [similiarity(questions, x, arg) for x in windowspre]
            scores = np.array([x[0] for x in out])

            def getvals(scores):
                return {"relevance": np.mean(scores) * 1000, "diversity question": np.mean(kldivs(scores)) * 4,
                        "diversity window": np.mean(kldivs(scores.transpose(1, 0))) * 4,
                        "coverage question to window": normalized_entropy(np.mean(scores, 1)),
                        "coverage window to question": normalized_entropy(np.mean(scores, 0))}

            with open((f"{kwargs['outs_path']}/" + name + "-" + arg).replace(".", "-") + "best_texts.txt", "a") as f2:
                sct = scores / np.sum(scores)
                assert sct.shape[0] == len(windows)
                numtop = 15
                avg_complexity = 0
                avg_answerability = 0
                sct = np.transpose(sct, (1, 0))
                toptexts = [sorted(list(zip([x[0] for x in windows], x)), key=lambda x: x[1], reverse=True)[:numtop]
                            for x in sct.tolist()]
                examples = []
                for x in sorted(list(zip(questionstext, toptexts)), key=lambda x: sum([y[1] for y in x[1]]),
                                reverse=True)[:kwargs["top_windows"]]:

                    def writeline2(x):
                        """Log some interesting question answer pairs and some relevant values."""
                        f2.write(str(x))
                        f2.write("\n")
                        f2.flush()

                    if kwargs["llm_amount"] > 0:
                        difficulties = \
                            run(model2, tokenizer, [y[0] for y in x[1]][:kwargs["llm_amount"]], [x[0]], args, 4)[
                                0] + (numtop - kwargs["llm_amount"]) * [0.0]
                        answerability = \
                            run(model2, tokenizer, [y[0] for y in x[1]][:kwargs["llm_amount"]], [x[0]], args, 4,
                                answerability=True)[0] + (numtop - kwargs["llm_amount"]) * [0.0]
                        c = (kwargs["llm_amount"] * kwargs["top_windows"])
                    else:
                        difficulties = numtop * [10.0]
                        answerability = numtop * [1.0]
                        c = numtop
                    avg_complexity += np.sum(difficulties) / c
                    avg_answerability += np.sum(answerability) / c
                    t = ""
                    t += f"{np.sum(difficulties):.05};{np.sum(answerability):.05}: " + x[0] + "\n"
                    t += "-------\n"
                    t += "\n\n".join([f"{y[1]:.05};{z:.05};{a:.05}: {y[0]}" for y, z, a in
                                            zip(x[1], difficulties, answerability)])
                    t += "************************\n"
                    writeline2(t)
                    examples.append(t)

                vals = getvals(scores)
                s = {"name": name, "harmonic_mean": harmonic_mean(
                    list(vals.values()) + [avg_complexity / 9] + [avg_answerability]),
                     "values_from_similarity": vals,
                     f"avg_complexity{'(mocked)' if kwargs['llm_amount'] == 0 else ''}": avg_complexity,
                     f"avg_answerability{'(mocked)' if kwargs['llm_amount'] == 0 else ''}": avg_answerability, "questions": questionstext, "examples": examples}
                r[arg].append(s)
    return r


def evaluate(**kwargs):
    """
    Evaluate the output dictionary of a teacher system.
    :param token: Huggingface token
    :param json_path: Path to the json containing the outputs
    :param data_path: Path to the folder containing the data
    :param verbose:
    :param llm_amount:
    :param top_windows:
    :param outs_path:
    :param system_prompt_answerability:
    :param response_prompt_answerability:
    :param system_prompt_complexity:
    :param response_prompt_complexity:
    :param cache:
    :return:
    """
    default_args = vars(parser.parse_args([]))
    for x in default_args:
        kwargs.setdefault(x, default_args[x])
    d = {}
    for questions, contexts, reference_questions, path in load_questions_and_texts(kwargs["json_path"], kwargs["data_path"]):
        d[path] = evaluate_one(contexts, questions, reference_questions, **kwargs)
    return d

def evaluate_one(text, questions_eval, questions_gold, **kwargs):
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    minsize = 3
    maxsize = 5
    muli = 50
    sentences = text.split(".")
    windows = get_all_windows(sentences, minsize, maxsize, muli, 10)
    join = True
    if join:
        windows = [(".".join(x),) for x in windows]
    if kwargs["verbose"]:
        print("--text--", text)
        print("--questions_gold--", questions_gold)
        print("--questions_eval--", questions_eval)
    manysimilarities = False
    args = (["1 similarity", "0.999 quantile", "0.9 quantile"] + ([
                                                                      "0.5 similarity", "0.1 similarity",
                                                                      "0.5 quantile", "0.7 quantile", "0.1 quantile",
                                                                      "0.001 threshold", "0.002 threshold",
                                                                      "0.0005 threshold", "0.0001 threshold"
                                                                  ] if manysimilarities else []))
    for arg in args:
        cosinesimilarity(None, None, arg)
    questionslist = [("evaluated", questions_eval), ("gold", questions_gold)]
    questionslist = [(x, [z[0] for z in y]) for x, y in questionslist]
    if os.path.isdir(kwargs['outs_path']):
        shutil.rmtree(kwargs['outs_path'])
    os.makedirs(kwargs['outs_path'], exist_ok=True)
    nnsim = Nnsim()
    r = results(cosinesimilarity, nnsim.nnembed, nnsim.nnembed, args, questionslist,
                windows, **kwargs)
    return r


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    d = evaluate(**vars(args))
    print(d)
