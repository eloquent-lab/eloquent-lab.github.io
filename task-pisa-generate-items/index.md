# Topical PISA Quiz Task --- Generating Test Items
### A task of the 2026 [ELOQUENT lab](https://eloquent-lab.github.io/) on evaluating quality of generative language models

Contact: eloquent-clef2026-organizers@googlegroups.com

## Task overview 

This task focuses on automatic test items generated from a given document, targeting students aged 10 to 15. The objective is to generate assessment items in the form of Question–Answer pairs based on a provided text (the "stimulus").

We provide participants with a ready-to-use question–answer generation prompt as a baseline, along with five automatically generated QA test items and one gold test item, all derived from the same source stimulus. The scope of participation is intentionally open and flexible. Participants may choose to:

* improve or extend the existing prompt,
* experiment with different off-the-shelf or homemade models,
* modify or redesign the prompt to generate new types of questions,
* or explore any other innovative approach to test items generation.

For this edition, English is the selected language.

## Quick Start

## How to participate, in more detail

## Submission instructions

Submission over web form will be opened at release of evaluation data set

## Types of test item
The PISA tests test for a set of cognitive processes related to reading. Test items are expected to probe them, separately or jointly. 

* Access and retrieve information within a text
* Search for and select relevant text
* Represent literal meaning
* Integrate and generate inferences
* Integrate and generate inferences across multiple sources
* Detect and handle conflict
* Reflect on content and form
* Assess quality and credibility


## Quality Criteria
The following quality criteria will in various ways be taken into account. Scoring will be done by OECD readers who currently work with the development of PISA tests. 

An overview of commonly used evaluation metrics for natural language generation is provided by <a href="https://aclanthology.org/2024.inlg-main.44/">Schmidtova et al. (2024)</a>.

* Topical relevance, coherence, and clarity
* Anchoring and coverage of test item set over the stimulus, i.e. that the test items do not address only one section of the stimulus
* Distribution over the above types of item
* Level of difficulty and its variation, to cater for the application of the test in varied cultural areas
* Diversity of set of test items i.e. the distribution of different types of item
* Naturalness and fluency of language

### Additional information on the evaluation criteria

* Item quality: clarity and natural phrasing; appropriateness of text and questions for 15‑year‑olds; cultural and linguistic suitability.
* Alignment: match between question and stimulus; correctness of the declared cognitive process and difficulty level.
* Coverage of cognitive processes, item dimensions and difficulty levels: the full test form should cover the different cognitive processes (search/access, understand/interpret, evaluate/reflect), text dimensions (text sources, literary types, response formats, text length) and difficulty levels elaborated in the PISA reading framework. The exact distribution is less important than showing the ability to generate distinct items on these dimensions.


## Data

### Publicly released PISA items
Publicly released PISA assessment items are available here: <a href="https://github.com/eloquent-lab/eloquent-lab.github.io/tree/pisa-gen-items/task-pisa-generate-items/data">data</a>

### Example training items

Some example items can be found <a href="Examples/">here</a>

### Evaluation data set

Evaluation stimulus items will be released in late March 2026. 

## Models and Tools

Participants can use any models of their choice.

* Open-weight language models can be downloaded from the <a href="https://huggingface.co/models">Hugging Face Hub</a>.
* A tool to estimate the hardware resources required to run different LLMs: <a href="https://huggingface.co/spaces/Vokturz/can-it-run-llm">https://huggingface.co/spaces/Vokturz/can-it-run-llm</a>


## Scoring
The scoring of submissions will be made using expertise from human editors who have worked with putting together previous PISA editions.

## Timeline
* Task launch: First week of March, 2026
* Test data release: TBD, March 2026
* Presentation at European Conference on Information Retrieval (ECIR) in Delft: End of March 2026
* Task submission deadline: May 2026
* Reporting deadline: June 2026
* Task workshop at CLEF Conference in Jena: September 2026
  
## Organisers

* **Université Grenoble Alpes**: Diandra Fabre, Lorraine Goeuriot, Philippe Mulhem, Didier Schwab, Markarit Vartampetian
* **OECD**: Said Ettejjari, Mario Piacentini, Luis Francisco Vargas Madriz, Katherina Thomas
* **AMD Silo AI**: Jussi Karlgren

Contact address for questions or suggestions: eloquent-clef2026-organizers@googlegroups.com

## Bibliography
Some relevant previous work -- feel free to suggest items for this list e.g. by a pull request!

* The PISA website for background on the survey itself: https://www.oecd.org/en/about/programmes/pisa.html
* A survey on automatic question generation: Nikahat Mulla and Prachi Gharpure. 2023. Automatic question generation: a review of methodologies, datasets, evaluation metrics, and applications. _Progress in Artificial Intelligence_ 12. https://doi.org/10.1007/s13748-023-00295-9
* A typology of questions (originally for automatic question answering purposes): Wendy Lehnert. 1977. A Conceptual Theory of Question Answering. _Proceedings of IJCAI_
* A typology of educational goals: Bloom, Benjamin S., Max D. Engelhart, Edward J. Furst, Walker H. Hill, and David R. Krathwohl. Taxonomy of educational objectives: The classification of educational goals. New York: _Longman_, 1956.
* Nikahat Mulla and Prachi Gharpure. 2023. Automatic question generation: a review of
methodologies, datasets, evaluation metrics, and applications. Prog Artificial Intelligence 12.
https://doi.org/10.1007/s13748-023-00295-9
