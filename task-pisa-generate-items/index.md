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

## Example training items

Example items can be found <a href="https://eloquent-lab.github.io/Examples/5ex_QA_generated.jsonl">here</a>

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

## Evaluation and Scoring

<A HREF="prompt_baseline.md">More details about the task, including scoring principles</A>

## Evaluation data set

Evaluation stimulus items will be released in late March 2026. 

## Submission instructions

Submission over web form will be opened at release of evaluation data set

## Quality Criteria
The following quality criteria will in various ways be taken into account. Scoring will be done by OECD readers who currently work with the development of PISA tests. 

* Topical relevance, coherence, and clarity
* Anchoring and coverage of test item set over the stimulus, i.e. that the test items do not address only one section of the stimul
* Distribution over the above types of item
* Level of difficulty and its variation, to cater for the application of the test in varied cultural areas
* Diversity of set of test items i.e. the distribution of different types of item
* Naturalness and fluency of language

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
