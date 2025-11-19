This task is intended to address the general question *"Can text authored by generative language models be distinguished
from text written by human authors?"*


# Motivation
Recent advances in generative language models have made it possible to
automatically generate content for websites, news articles, social media, etc.
The EU has recently suggested to technology companies that labelling such
AI-generated content as such might be a useful tool to combat misinformation
and to protect consumer rights.

# Goals of the task
This task will explore whether automatically-generated text can be
distinguished from human-authored text. Detecting automatically generated
text, with the increased quality of generative AI, is becoming a task quite similar
to human authorship verification and this task will be organised in collaboration
with the PAN lab with years of experience from previous shared tasks on
authorship verification and closely related tasks.
This task will also investigate if models can be self-assessed in a reliable way
with minimal human effort.

# Procedure

1. Organisers pick a number of human-authored texts of about 500 words in
genres such as:
	○ Newswire
	○ Wikipedia intro texts
	○ Fan fiction
	○ Biographies
	○ Weather and stock market reports; sports results
	○ Podcast transcripts
2. Descriptions for those texts are generated automatically
	○ Bullet points, e.g.
	○ Should capture genre and some of stylistic characteristics of the
original
3. Ask participants to use their systems generate a text from those
descriptions
4. Pass the resulting sets to PAN builders to see if
	○ human texts can be distinguished
	○ system characteristics can be tracked across texts, i.e. if the output
of a system is similar enough across texts and genres to be
identifiable
# Data
The data are distributed in a format which resembles standard benchmark tests.
The test collection has a suggested prompt string, and a list of items with a
Content and an optional Genre and Style field. These can be used together
with the suggested prompt string to generate a text. If participants wish to
change the suggested prompt string to something more suitable, this is allowed,
but this must be reported upon submission and in the written report of the
experiment.

[Test data for 2025 and 2024 on Huggingface](https://huggingface.co/datasets/Eloquent/Voight-Kampff)



# Submission format

The submission should be in a plain zip file of a directory named after the team
with the generated texts in plain text form:

OurTeamName/030.txt
...
OurTeamName/062.txt

[Submit using this form](https://forms.gle/tuypEyDCyoUtnoPS9)

# Result scoring
System outputs are scored by how often they fool a classifier into believing the
output was human-authored.

# Sample (from the 2024 data)



```
{
    "voight-kampfftesttopics": {
        "language": "en",
        "date": "2024",
        "type": "example",
        "source": "eloquent organisers",
        "prompt": "Write a text of about 500 words which covers the following items: ",
        "topics": [
{"id": "001", 
 "Genre and Style": "Encyclopedia",
 "Content": ["Uralic languages descended from Proto-Uralic language from 7,000 to 10,000 years ago.",
	"Uralic languages spoken by 25 million people in northeastern Europe, northern Asia, and North America.",
	"Hungarian, Estonian, and Finnish are the most important Uralic languages.",
	"Attempts to trace genealogy of Uralic languages to earlier periods have been hampered by lack of evidence.",
	"Uralic and Indo-European languages are not thought to be related, but speculation exists.",
	"Uralic languages consist of two groups: Finno-Ugric and Samoyedic.",
	"Finno-Ugric and Samoyedic have given rise to divergent subgroups of languages.",
	"Degree of similarity in Finno-Ugric languages is comparable to that between English and Russian.",
	"Finnish and Estonian, closely related members of Finno-Ugric, differ similarly to diverse dialects of the same language."]},
 ...
 ]
 }
 }
```
