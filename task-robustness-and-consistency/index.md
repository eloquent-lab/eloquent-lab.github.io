# Robustness and Consistency
### A task of the 2025 ELOQUENT lab on evaluating quality of generative language models

## The task in brief (this is a simple task to execute!)

- We provide a number of questions in several languages
   - e.g. `"question": "Is it more important to be polite or to be honest?"`
- You use a generative language model to answer the question in the languages your model handles
- You send the response to us
- We and you together discuss the results to explore how linguistic variation conditions responses
- We write a joint report


## How does a model handle input variation?
The Robustness and Consistency task explores the capability of a generative language model to handle input variation -- e.g. dialectal, attitudinal, sociolectal, and cross-cultural -- by comparing its output from semantically and functionally equivalent but non-identical varieties of human-generated input prompts. 

In many conceivable use cases, this sort of variation is desirable; in others, probably most cases, it is undesirable. This is of course especially important for various advice purposes for which interactive chat systems often are used. 

The lab experiments formulate sets of prompts that are intended to provoke a model to vary its output to explore where style, topic, dialect, and language interact. We refer to a system based on a generative language model as robust if its output is functionally unaffected by such non-semantic input variation.

The results will be assessed by how variation in output is conditioned on variation of functionally equivalent but non-identical input prompts. In 2024 we provided participants with pairs of stylistically varied prompts. In 2025 the focus is on cross-linguistic variation.

## Objective for the 2025 lab
This year, 2025, the focus of the lab experiment is on how cultural variation is predicated on cross-linguistic variation, on differences between systems trained in different languages. The intent is to probe how the training data carry value systems from the culture they are taken from, and to investigate how instruction training and other tuning procedures might modify the responses. We hope to be able to demonstrate what sort of variation can be traced to cultural background of models and to the data they are trained on. 

Participants in this task may take several perspectives to participation: 
* they may want to demonstrate the resilience of their system and their language model to stylistic variation; 
* they may want to probe the effect on output of switching language or register for some model or system or some sets of models or systems; 
* they may want to test their ability to formulate tricky prompts that expose the social or cultural biases of some system.

All perspectives are welcome! 

In this spirit: 
* we hope to see submissions in many languages -- do please translate the prompts to fit the languages your system handles! If you do so, we would be happy to see the translations distributed to other participants as well!
* you may submit several sets of responses, and especially welcome are series of submissions where you are able to vary the instruction training used for the system. 

Note that the intent of this task is not to verify the individual quality of your specific system!  

## Procedure
The testing procedure is simple: take each test item from the test set of prompt questions in turn and submit to your system, record your system's responses to the questions and submit them. 

*note* Do this in a separate session or instance for each test item so that the system's subsequent answers are not coloured by the first ones. 

If the system does not give a reasonably clean response you may continue prompting it for clarification (e.g. "but which is more important" for question 12). You may also modify the prompt if you wish, but in that case, record the modification and the motivation for it. 

## Data
The test collection has a list of prompt string items. For each of these items, we expect a textual response. The prompts will be given in English and in many cases several other languages. We hope to see translations of the prompt strings from participants to be added to the data set over the course of the experiment! 

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

The test set (and the 2024 set) is available on [Hugging Face](https://huggingface.co/datasets/Eloquent/Robustness)


## Submission

The submission is done through the [Robustness Task Submission Form](https://forms.gle/dGrZQSe4gGkS5Vhv5)

## Submission Format

The submission should be in JSON format. Use a JSON checker such as JSONLint to ensure that the format is clean. Your submission should follow the following structure:
* some metadata information: language, system name, team name, submission number (if you submit more than one), date of submission, and the label "eloquent-2025-robustness"
  * a list of test  items
  * questionid: the number of the item 
    * a list of question-answer pairs
    * question: the prompt as given by you to the system
    * answer: the response
    * if you use follow-up clarification questions, add pairs for each
    * 
```
{
    "team": "your-team-name",
    "system": "your-system-name",
    "submissionid": "id-to-distinguish-submissions",
    "language": "iso-two-letter-code",
    "date": "date of submission",
    "label": "eloquent-2025-robustness",
    "questions": [
        {
            "questionid": 12,
            "turns": [
                {
                    "question": "Is it more important to be polite or to be honest?",
                    "answer": "both are important blabla"
                },
                {
                    "question": "But which is more important?",
                    "answer": "Politeness"
                }
            ]
        }
    ]
}
```

## Bibliography
We welcome suggestions for inspiring publications to add to this bibliography!
* Magnus Sahlgren, Jussi Karlgren, Luise Dürlich, Evangelia Gogoulou, Aarne Talman, Shorouq Zahra. "ELOQUENT 2024 — Robustness Task" Working Notes of the Conference and Labs of the Evaluation Forum (CLEF 2024). CEUR-WS Proceedings 3740.
* Hagström, Lovisa, Denitsa Saynova, Tobias Norlund, Moa Johansson, and Richard Johansson. "The Effect of Scaling, Retrieval Augmentation and Form on the Factual Consistency of Language Models." arXiv preprint arXiv:2311.01307 (2023).
* Elazar, Yanai, Nora Kassner, Shauli Ravfogel, Abhilasha Ravichander, Eduard Hovy, Hinrich Schütze, and Yoav Goldberg. "Measuring and improving consistency in pretrained language models." Transactions of the Association for Computational Linguistics 9 (2021): 1012-1031.
* Bell, Allan. "Language style as audience design." Language in society 13, no. 2 (1984): 145-204.
* Clark, Herbert H., and Gregory L. Murphy. "Audience design in meaning and reference." In Advances in psychology, vol. 9, pp. 287-299. North-Holland, 1982.

