from argparse import ArgumentParser
from vllm import LLM, SamplingParams
from datasets import load_dataset
import pandas as pd
import numpy as np
import evaluate
import torch
import json
import re


PROMPT = "Please act as an impartial judge and evaluate if an AI evaluator provides a correct explanation for comparing answers from two other AI assistants to the user instruction. The AI evaluator was asked to choose the assistant that {criterion_description}. You will be given the user instruction, assistant A's answer, assistant B's answer, the AI evaluator's task description, the AI evaluator's decision (one of the four options: assistant A is better than assistant B, assistant B is better than assistant A, both AI responses are equally good, or both AI responses are equally bad), the AI evaluator's explanation, and a human's decision and explanation. Your task is to judge if the AI evaluator's explanation is accurate and aligns with the human explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[Yes]]\" if the AI evaluator's explanation is accurate and similar to the human explanation and \"[[No]]\" otherwise.\n\n[User Instruction]\n{instruction}\n\n[Assistant A's Answer]\n{output_a}\n\n[Assistant B's Answer]\n{output_b}\n\n[AI evaluator's Decision]\n{system_preference}\n\n[AI evaluator's Explanation]\n{system_explanation}\n\n[Human Decision]\n{human_preference}\n\n[Human Explanation]\n{human_explanation}\n\n"

CRITERION_DESCRIPTIONS = {
    "relevance": "more accurately follows the prompt and fulfills the user's request",
    "naturalness": "sounds more human-like and natural",
    "truthfulness": "provides more accurate and factually correct information",
    "safety": "is less harmful",
    "overall_quality": "is overall better",
}

LABEL_MAPPING = {
    "A": "Assistant A",
    "B": "Assistant B",
    "both_good": "Both assistants are equally good (tie)",
    "both_bad": "Both assistants are equally bad (tie)",
    "C": "Both assistants are equally good (tie)",
    "D": "Both assistants are equally bad (tie)",
}


def parse_args():
    """
    Parses the arguments required for the evaluation script.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        required=False,
        type=str,
        default="Eloquent/preference_prediction",
        help="Dataset name",
    )
    parser.add_argument(
        "--dataset_split",
        required=False,
        type=str,
        default="validation",
        help="Dataset split name (validation/test)",
    )
    parser.add_argument(
        "--prediction_fpath",
        required=True,
        type=str,
        help="Path to the prediction file in the form of a tab-separated dataframe",
    )
    parser.add_argument(
        "--judge_model_name",
        required=False,
        type=str,
        default="google/gemma-2-9b-it",
        help="Name of the judge model",
    )
    parser.add_argument(
        "--evaluate_explanations",
        required=False,
        type=bool,
        default=False,
        help="Whether to run the performance evaluation for the second sub-task",
    )
    args = parser.parse_args()
    return args


def compute_rouge(predictions, references, rouge_metric):
    """
    Computes the ROUGE-L score between a list of system and human explanations.
    args:
        predictions (list of str): A list of system-generated explanations.
        references (list of str): A list of human-written explanations.
        rouge_metric (evaluate.metrics.rouge.Rouge): An instance of the `evaluate` ROUGE metric.
    returns:
        float: The ROUGE-L score as a percentage, rounded to three decimal places.
    """
    rougeL = rouge_metric.compute(
        predictions=predictions,
        references=references,
        rouge_types=["rougeL"],
    )["rougeL"]

    rougeL_avg = round(rougeL * 100, 3)
    return rougeL_avg


def compute_bertscore(predictions, references, bertscore_metric):
    """
    Computes the BERTScore between a list of system and human explanations.
    args:
        predictions (list of str): A list of system-generated explanations.
        references (list of str): A list of human-written explanations.
        bertscore_metric (evaluate.metrics.bertscore.BERTScore): An instance of the `evaluate` BERTScore metric.
    returns:
        float: The BERTScore as a percentage, rounded to three decimal places.
    """
    bertscore_f1 = bertscore_metric.compute(
        predictions=predictions,
        references=references,
        lang="en",
    )["f1"]
    bertscore_f1_avg = round(np.mean(bertscore_f1) * 100, 3)
    return bertscore_f1_avg


def create_pairwise_comparison(prediction, reference, criterion):
    """
    Creates an input example for the LLM-as-a-judge evaluator based on the prompt.
    args:
        prediction (dataframe series): A series containing the system's preferences and explanations.
        references (dataframe series): A series from our dataset containing the instruction, outputs A and B, and human preferences and explanations.
        criterion (string): A name of the target criterion.
    returns:
        string: Prompt for the LLM-as-a-judge evaluator.
    """
    return PROMPT.format(
        criterion_description=CRITERION_DESCRIPTIONS[criterion],
        instruction=reference["instruction"],
        output_a=reference["output_a"],
        output_b=reference["output_b"],
        system_preference=LABEL_MAPPING[prediction[f"{criterion}_preference"]],
        system_explanation=prediction[f"{criterion}_explanation"],
        human_preference=LABEL_MAPPING[reference[f"{criterion}_preference"]],
        human_explanation=reference[f"{criterion}_explanation"],
    )


def extract_judge_score(judge_prediction, fallback="[[yes]]"):
    """
    Extracts the LLM-as-a-judge evaluator's decision. If there is no decision in the judge prediction, we favour the participant's system by responding "Yes" by default.
    args:
        judge_prediction (a structured output based on the vLLM framework): An output from the LLM-as-a-judge evaluator.
        fallback (string): A default answer if the evaluator's decision cannot be extracted.
    returns:
        int: 1 if the system's explanation aligns with the human explanation and 0 otherwise.
    """
    judge_score = re.search(
        "\[[^\w\s]*\[\s*(yes|no)\s*\][^\w\s]*\]",
        judge_prediction.outputs[0].text.lower(),
    )
    judge_score = judge_score.group(0) if judge_score is not None else fallback
    return 1 if "yes" in judge_score else 0


def compute_judge_score(predictions, references, judge, criterion):
    """
    Computes the LLM-as-a-judge evaluator's scores for the system's predictions according to the target criterion.
    args:
        predictions (dataframe): A series containing the system's preferences and explanations.
        references (dataframe): A series from our dataset containing the instructions, outputs A and B, and human preferences and explanations.
        judge (vllm.LLM): An LLM-as-a-judge evaluator.
        criterion (string): A name of the target criterion.
    returns:
        float: The percentage of the system's explanations aligned with the human explanations for the target criterion, rounded to three decimal places.
    """
    sampling_params = SamplingParams(max_tokens=256)
    pairwise_comparisons = [
        create_pairwise_comparison(
            prediction=prediction, reference=references.iloc[0], criterion=criterion
        )
        for i, prediction in predictions.iterrows()
    ]
    inputs = list(
        map(
            lambda comparison: judge.llm_engine.tokenizer.tokenizer.apply_chat_template(
                [{"role": "user", "content": comparison}],
                tokenize=False,
            ),
            pairwise_comparisons,
        )
    )
    print(f"Criterion: {criterion}\n\nExample: {inputs[0]}", flush=True)
    judge_scores = list(
        map(extract_judge_score, judge.generate(pairwise_comparisons, sampling_params))
    )
    return round(np.mean(judge_scores) * 100, 3)


def save_results(results, fname):
    """
    Saves the results into a .json file.
    args:
        results (dictionary): A dictionary containing the performance results.
        fname (string): A name of the output file.
    """
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(results, f)


def evaluate_preferences(predictions, references):
    """
    Computes the accuracy scores for all criteria.
    args:
        predictions (dataframe): A series containing the system's preferences and explanations.
        references (dataframe): A series from our dataset containing the instructions, outputs A and B, and human preferences and explanations.
    returns:
        dictionary: A dictionary with the accuracy scores for all criteria.
    """
    preference_results = {}
    for criterion in CRITERION_DESCRIPTIONS:
        criterion_preference_col = f"{criterion}_preference"
        system_preferences = predictions[criterion_preference_col]
        human_preferences = references[criterion_preference_col]
        preference_results[f"{criterion}_accuracy"] = round(
            (system_preferences == human_preferences).mean() * 100, 2
        )
    return preference_results


def evaluate_explanations(predictions, references, judge_model_name):
    """
    Computes the language generation evaluation metric & LLM-as-a-judge evaluator's scores for all criteria.
    args:
        predictions (dataframe): A series containing the system's preferences and explanations.
        references (dataframe): A series from our dataset containing the instructions, outputs A and B, and human preferences and explanations.
    returns:
        dictionary: A dictionary with the performance scores for all criteria.
    """
    explanation_results = {}
    # load the performance metrics
    rouge_metric = evaluate.load("rouge")
    bertscore_metric = evaluate.load("bertscore")
    for criterion in CRITERION_DESCRIPTIONS:
        print(
            f"Computing the language generation evaluation metrics for {criterion}...",
            flush=True,
        )
        criterion_explanation_col = f"{criterion}_explanation"
        system_explanations = predictions[criterion_explanation_col]
        human_explanations = dataset[criterion_explanation_col]
        # compute the rougeL score
        rouge_score = compute_rouge(
            predictions=system_explanations,
            references=human_explanations,
            rouge_metric=rouge_metric,
        )
        explanation_results[f"{criterion}_rougeL"] = rouge_score
        # compute the BERTScore
        bertscore_f1 = compute_bertscore(
            predictions=system_explanations,
            references=human_explanations,
            bertscore_metric=bertscore_metric,
        )
        explanation_results[f"{criterion}_bertscore_f1"] = bertscore_f1
    # another iteration over the criteria to save RAM
    del bertscore_metric
    judge = LLM(judge_model_name, dtype=torch.bfloat16)
    # compute the LLM-as-a-judge score
    for criterion in CRITERION_DESCRIPTIONS:
        print(f"Computing the LLM-as-a-judge scores for {criterion}...", flush=True)
        judge_score = compute_judge_score(
            predictions=predictions,
            references=references,
            judge=judge,
            criterion=criterion,
        )
        explanation_results[f"{criterion}_{judge_model_name}"] = judge_score
    return explanation_results


if __name__ == "__main__":
    args = parse_args()
    dataset = load_dataset(args.dataset_name)[args.dataset_split].to_pandas()
    predictions = pd.read_csv(args.prediction_fpath, sep="\t")
    results = evaluate_preferences(predictions=predictions, references=dataset)
    if args.evaluate_explanations:
        explanation_results = evaluate_explanations(
            predictions=predictions, references=dataset, judge_model_name=args.judge_model_name
        )
        results.update(explanation_results)
    out_fpath = args.prediction_fpath.replace(".tsv", "_results.json")
    print(f"Saving results to {out_fpath}", flush=True)
    save_results(results=results, fname=out_fpath)
