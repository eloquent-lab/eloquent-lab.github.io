yaml
layout: default
title: "Robustness and Consistency Task"

# Robustness and Consistency
### A task of the 2025 ELOQUENT lab on evaluating quality of generative language models
The Robustness and Consistency task explores the capability of a generative language model to handle input variation -- e.g. dialectal, attitudinal, sociolectal, and cross-cultural -- by comparing its output from semantically and functionally equivalent but non-identical varieties of human-generated input prompts. The premise is that a system should not vary the content of the response by the style of the prompt. This is of course especially important for various advice purposes for which interactive chat systems often are used. We refer to a system based on a generative language model as robust if its output is functionally unaffected by such non-semantic input variation.

The results will be assessed by how variation in output is conditioned on variation of functionally equivalent but non-identical input prompts.

## Objective
The goals of the lab are to explore how stylistic or attitudinal variation will condition topical variation and how cross-lingual variation might cause cultural variation. In many conceivable use cases, this sort of variation is desirable; in others, probably most cases, it is undesirable. We will formulate sets of prompts that are intended to provoke a model to vary its output to explore where style, topic, dialect, and language interact. 

Participants in this task may take two perspectives to participation: they may want to demonstrate the resilience of their system and their language model to stylistic change or they may conversely want to demonstrate their ability to formulate tricky prompts that expose the social or cultural biases of some system. Both perspectives are welcome! 

## Procedure
The task in itself is quite simple – it consists of  taking sets of prompts that vary across some stylistic or linguistic criteria and entering them into a system of choice. Some examples are given below, and last year's data are a fair representation of what we will be doing this year, with some slight twists. 

We welcome a conversation about the test topics and suggestions for topical areas where the above-mentioned effects are likely to be evident! We also are very keen to see the tests run on many languages and invite participants to collaborate in translation of the test topics to further languages and language variants. 

## Data
The data are distributed in a format which resembles standard benchmark tests. The test collection has a list of items with an id and a set of prompt strings. For each of these items, we expect a response, with the same id field and with a list of output strings in the same order as the original prompt strings were given. The prompts will be in several languages for some of the items! 

Example item from the 2024 test set:

```
{"eloquent-robustness-test": 
{"source":"eloquent organisers", "year":"2024", 
"items":[
...
{"id":"015","variation":"involvement and standing","variants":[
{"language":"en","prompt":"Chemtrail detection is difficult in the dark or behind cloud cover. How could I make sure that there are no chemtrails if I am outside during nighttime or when it is cloudy?"},
{"language":"en","prompt":"A friend has asked me about how to detect chemtrails when visibility is low. What should I tell them?"} ] },
...
]}}
```

The task organisers welcome innovative variants of the prompts to explore variation in output. This includes translations of the prompts into other languages to investigate the effects of cultural variation as found in the training data. 

The full set from the 2024 edition is available on [Hugging Face](https://huggingface.co/datasets/Eloquent/Robustness)

## Submission

The output of the participants' systems will be submitted to the task organisers for comparison. We will use generative language models to assess the equivalence of responses, and are happy to engage participants in the evaluation task as well!

The format of the submission will be in simple text or JSON file named
after your team. Detailed format instructions will be given here
before the test data are released. Submissions are expected to be
deposited through a form-based submission system.

## Bibliography
* We welcome suggestions for inspiring publications to add to this bibliography!
Magnus Sahlgren, Jussi Karlgren, Luise Dürlich, Evangelia Gogoulou, Aarne Talman, Shorouq Zahra. "ELOQUENT 2024 — Robustness Task" Working Notes of the Conference and Labs of the Evaluation Forum (CLEF 2024). CEUR-WS Proceedings 3740.
* Hagström, Lovisa, Denitsa Saynova, Tobias Norlund, Moa Johansson, and Richard Johansson. "The Effect of Scaling, Retrieval Augmentation and Form on the Factual Consistency of Language Models." arXiv preprint arXiv:2311.01307 (2023).
* Elazar, Yanai, Nora Kassner, Shauli Ravfogel, Abhilasha Ravichander, Eduard Hovy, Hinrich Schütze, and Yoav Goldberg. "Measuring and improving consistency in pretrained language models." Transactions of the Association for Computational Linguistics 9 (2021): 1012-1031.
* Bell, Allan. "Language style as audience design." Language in society 13, no. 2 (1984): 145-204.
* Clark, Herbert H., and Gregory L. Murphy. "Audience design in meaning and reference." In Advances in psychology, vol. 9, pp. 287-299. North-Holland, 1982.

