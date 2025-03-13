import argparse
from itertools import chain

import numpy as np
import pylcs
from sentence_transformers import SentenceTransformer

from .helpers import *

parser = argparse.ArgumentParser()
parser.add_argument("--json_path", default="devset.json", type=str, help="Path to the json containing the outputs.")
parser.add_argument("--data_path", default="../devset", type=str, help="Path to the json containing the outputs.")


def expand_and_broadcast(arr, dim, shape):
    return np.broadcast_to(np.expand_dims(arr, dim), shape)


def loadq(fol, requiresanswers=True, forbidabcd=True):
    qs = "/questions.json"
    with open(fol + qs) as f:
        return [x for x in json.loads(f.read()) if (not requiresanswers or len(x) > 1) and (
                not forbidabcd or len(x) == 1 or x[1] not in ["A", "B", "C", "D"])]


def cosinesimilarity(question_embed, sentance_embed):
    question_embed = np.array(question_embed)
    sentance_embed = np.array(sentance_embed)
    question_embed = question_embed / np.sum(question_embed * question_embed)
    sentance_embed = sentance_embed / np.sum(sentance_embed * sentance_embed)
    m1 = np.sum(question_embed * sentance_embed)
    return m1


def encode_update_dict(word, dictionary):
    if word not in dictionary:
        dictionary[word] = len(dictionary)
    return dictionary[word]


def ROUGELF1Chars(s1, s2):
    """
    Runs ROGUELF1 on chars.
    :param s1:
    :param s2:
    :return:
    """
    l = pylcs.lcs_sequence_length(s1, s2)
    res = pylcs.lcs_sequence_idx(s1, s2)
    w = ''.join([s2[i] for i in res if i != -1])
    assert len(s2) > 0, f"{w}:{s2}"
    prec = l / len(s1)
    rec = l / len(s2)
    if prec + prec == 0:
        return 0
    return 2 * prec * rec / (prec + rec)


def ROUGELF1Words(s1, s2):
    """
    Runs ROGUELF1 on words. Encodes each unique word as a single consistent char using a dictionary. Then runs the equivalent of ROUGHELF1Chars on it.
    :param s1:
    :param s2:
    :return:
    """
    dictionary = {}
    s1encoded = []
    for x in s1.split(" "):
        s1encoded.append(chr(encode_update_dict(x, dictionary)))
    s2encoded = []
    for x in s2.split(" "):
        s2encoded.append(chr(encode_update_dict(x, dictionary)))
    inverse_dictionary = dict((dictionary[x], x) for x in dictionary)
    s1 = "".join(s1encoded)
    s2 = "".join(s2encoded)
    l = pylcs.lcs_sequence_length(s1, s2)
    res = pylcs.lcs_sequence_idx(s1, s2)
    w = ' '.join([inverse_dictionary[ord(s2[i])] for i in res if i != -1])
    prec = l / len(s1)
    rec = l / len(s2)
    if prec + prec == 0:
        return 0, w
    return 2 * prec * rec / (prec + rec), w


def rate(candidate, references, verbose=False):
    """

    :param candidate: The list of answers to questions.
    :param references: The list of lists of references for each answer.
    :param verbose:
    :return:
    """
    modelname = "paraphrase-multilingual-mpnet-base-v2" if True else 'all-mpnet-base-v2'
    model = SentenceTransformer(modelname, device="cuda")
    all = [candidate] + references
    reflen_presums = [0] + np.cumsum([len(x) for x in references]).tolist()
    embeds = model.encode(list(chain.from_iterable(all)))
    embeds_references = []
    lc = len(candidate)
    for x in range(len(references)):
        embeds_references.append([])
        for y in range(len(references[x])):
            embeds_references[-1].append(embeds[lc + reflen_presums[x] + y])
    averagerougelF1 = 0
    averagesimilarity = 0
    for x in range(lc):
        rouge_scores = [ROUGELF1Words(candidate[x], references[x][y]) for y in range(len(references[x]))]
        besti = max(list(range(len(references[x]))), key=lambda x: rouge_scores[x][0])
        averagerougelF1 += rouge_scores[besti][0] / lc
        similarities = [cosinesimilarity(embeds[x], embeds_references[i]) for i in range(0, len(references[x]))]
        bestisimilaritity = max(list(range(len(references[x]))), key=lambda i: similarities[i])
        averagesimilarity += similarities[bestisimilaritity] / lc
        if verbose:
            print("candidate::::", candidate[x])
            print("best reference::::", references[x][besti])
            print("lcs::::", rouge_scores[besti][1])
            print("best reference semantic similarity::::", references[x][bestisimilaritity])
            print(f"""***rougel::::{rouge_scores[besti][0]}, best semantic similarity::::{
            similarities[bestisimilaritity]}***""")
    return averagerougelF1, averagesimilarity


def evaluate(**kwargs):
    d = {}
    for questions_answers, contexts, reference_questions, path in load_questions_and_texts(kwargs["json_path"],
                                                                                           kwargs["data_path"]):
        if any(len(x) == 1 for x in questions_answers):
            continue
        a = [x[1] for x in questions_answers]
        averagerougelF1, averagesimilarity = rate(a, [x[1:] for x in reference_questions], True)
        d[path] = {"average rouge-L-F1": averagerougelF1, "average semantic similarity": averagesimilarity}
    print(d)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    evaluate(**vars(args))
