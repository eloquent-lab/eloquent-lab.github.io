[ELOQUENT Lab](https://eloquent-lab.github.io/) @ [CLEF 2025](https://clef2025.clef-initiative.eu/)

# _Sensemaking_

**Task 5, first edition 2025**

This task is intended to address the question:

**"Can your language model prep, sit, or rate an exam for you?"**

The goal of the task is to automatically generate a quiz for a given learning material and to answer and evaluate such answers.

## **Motivation**

LLMs store immense amounts of information and easily answer questions about it. Can they _limit_ their knowledge to information provided in given materials?

**Why “sensemaking”:** Here is some input text. Its meaning (i.e. what you should take from it) can be represented as a set of questions and answers to them. Let’s see how reliable LLMs are in extracting this sense (i.e. generating a quiz), “understanding” this sense (i.e. answering the questions) as well as assessing someone else’s understanding (i.e. evaluating the answers).

## **Expected participation types**

“**Teacher**” submissions = systems that, given input materials, generate quizzes.

“**Student**” submissions = systems that, given input materials and quiz questions, provide answers to the questions.

“**Evaluator**” submissions = systems that, given input materials, a question and an answer score the answer.

## **Want to steer the task more?**

“**Director**” participation is also welcome. As a director, you provide input materials to teachers, and you are then required to analyze what “teachers”, “evaluators” and “students” did in that domain.

## **Domains and Data**

The Sensemaking task in 2025 covers 4 source "document domains", three of which will be illustrated in devset:

- **A university lecture on machine translation**: the original videorecording, the transcript of the speech, slides and the text extracted from them; no existing questions will be provided.

- **A Ukrainian textbook**: the text in the original PDF and its plaintext export; 5 sample questions in English without any golden-truth answers.

- **German audio on a popular topic**: the original sound recording, the transcript of the speech; English questions on the content and sample English answers.

All the documents, questions and answers are processed. Participants will be allowed to use the original multimodal data (PDFs, videos and audios, slides), **or our plain text exports** of this.

Across the domains, the inputs will be organized to a uniform hierarchical structure: “source”, “chapter”, “section”. Some sources will have only one chapter. A section is simply a structural unit of the chapter which may be useful for question generation. Some chapters will consist of only one section.

Chapter will be the unit used in evaluation, questions will be expected to be generated for each chapter, across all its sections.

## **Task procedure**

To take part in the task:

1. (by April 25, 2025) Register for our task in [ELOQUENT Labs registration form](https://clef2025.clef-initiative.eu/index.php?page=Pages/registration.html).

2. (by April 25, 2025) Play around with our devset, prepare your system for it. You are free to choose any combination of Teacher, Student, and Evaluator systems.

3. (by April 25, 2025) Register in **our registration form** (link will appear here). This is needed because we need to know your team github account names, so that you can download and upload your submissions.

4. (April 25-April 28) **Teachers** download our test set (link to github will appear here) and upload your questions (and optionally also answers) for each chapter.

5. (April 30-May 4) **Students** download input materials and questions and upload their answers to each question.

6. (May 7-May 10) **Evaluators** download materials, questions and answers and upload the scores.

7. (by May 30) You submit the paper describing your system(s).

8. (June 9) We provide tentative “official” scores for all submission types.

9. (July 7) Camera-ready for all papers.

10. (September 9-12) ELOQUENT Lab day at CLEF in Madrid.

All deadlines are AoE.

This means that participating in this task will involve being available for the above three operations during the two weeks of April 26 till May 10, depending on the exact days and your intended submission type(s).

## **“Teacher” submissions scoring**

For **automatic evaluation**, we will use at least two kinds of different **language models**. The evaluation will primarily concern plain text formulations; assessment of symbolic expressions such as equations is likely to be limited and unreliable.

Subject to funding availability, we will also try to obtain manual **Likert scores**. The properties evaluated will be: **Relevance, Coverage, Answerability, Complexity** and **Diversity** of the questions your model generates.

The total number of questions you provide will also be a factor in our evaluation. Roughly speaking, the fewer questions the better, for the same scores in other criteria.

### Definitions

The following description is our plan, tested on a small sample. We may need to adapt the plans as we go.

We will evaluate the documents **separately**. We take the set of questions _Q_ from the teacher system for a given document _D_.

We work with the document _D_ using its plain text. We split the text into a set of overlapping windows _W_ beginning at sentence boundaries and containing sentences including up to some number of words _K_ ignoring punctuation, i.e. spanning into subsequent sentences. Utilizing the language models we will define _p: WxQ -> R_, estimating the joint probability of a text and question pair appearing in an expert-made quiz. Window-question pairs that are completely unrelated should have a very low probability and vice-versa.

### Summary of measures

|                   |                                                          |                                                              |                |
| :---------------: | :------------------------------------------------------: | :----------------------------------------------------------: | :------------: |
|    **Measure**    |                     **Description**                      |                        **Automatic**                         | **Manual/LLM** |
|   **Relevance**   |           Alignment between questions and text           |                 Arithmetic mean of relevance                 | Likert scores  |
|   **Coverage**    |     Uniformity of question coverage across the text      | Entropy of the p(w). The marginal distribution over windows. | Likert scores  |
|   **Diversity**   |              How diverse are the questions               |   Sum of KL divergences between pairs p(w, q<sub>i</sub>), p(w, q<sub>j</sub>)   | Likert scores  |
| **Answerability** | Whether questions can be answered from the document text |                          ~~-----~~                           | Likert scores  |
|  **Complexity**   |    Difficulty of questions while remaining answerable    |                          ~~-----~~                           | Likert scores  |

### Example probability distribution

To illustrate let us consider the following text.

“Jane likes Bob. Jane hates Stephen. Jane is new in town.”

And the following questions in _Q_:

1. “Who is Jane’s favorite person?”

2. “How long has Jane been living here?”

Using the _K = 6_ we get the following windows in W:

1. “Jane likes Bob Jane hates Stephen”

2. “Jane hates Stephen Jane is new”

3. “Jane is new in town”

The function p could look like this:

|            | Window 1   | Window 2   | Window 3   |
| ---------- | ---------- | ---------- | ---------- |
| Question 1 | p(1,1)=.29 | p(2,1)=.19 | p(3,1)=.01 |
| Question 2 | p(1,2)=.01 | p(2,2)=.01 | p(3,2)=.49 |

As hinted, the individual probabilities are estimated using a language model. The details will be described in the task overview paper. We ensure the distribution has some minimal value for all probabilities so as not to blow up measures which use them in the denominator.

### Measure descriptions

1. **Relevance** measures how much a given set of questions and text **relate to each other**. It is aggregated from the relevance of a question to a window over the elements of WxQ. Not all window question pairs will necessarily be used.  
   In the example above, a simplified calculation could be:  
   _((1+1-0.5)+(-0.5-0.5+1))/6 =  0.25._ Which would indicate a good overall relevance of the question set to the text.  
   The actual computation is more complicated.

2. **Coverage** measures how **uniformly the questions cover** different parts of the text. Formally, we define it as the **entropy of the distribution _p_**_: W -> R._ The marginal distribution over windows.  
   _p(w) = Σ<sub>{q \in Q} </sub>p(w,q)_.  
   Maximizing this is equivalent to making the coverage of the text by the questions as uniform as possible.  
   In the example above, we get _Entropy((.29 + .01, .19 + .01, .01 + .49)) = -log2(.29)\*.29 - log2(.19)\*.19- log2(.49)\*.49 = 1.48_, indicating quite a uniform distribution, which is what we want.

3. **Diversity** estimates how differently **different questions cover different parts** of the text. It will be computed from conditional distributions:  
   _p(w|q<sub>i</sub>): W -> R p(w|q<sub>i</sub>) = p(w,q<sub>i</sub>)/Σ<sub>{w' \in W}</sub> p(w', q<sub>i</sub>)_, for fixed q<sub>i</sub> where q<sub>i</sub> is the i-th question. We take the **sum of the values KL-DIV(p(w|q<sub>i</sub>) || p(w|q<sub>j</sub>))** over all combinations of i and j. High values indicate that the set of distributions _p(q<sub>i</sub>)_ over all i is diverse.  
   In the example above, we get the marginal probabilities:  
   _s1 = p(<sub>1</sub>) = .29+.19+.01_ and  
   _s2 = p(<sub>2</sub>) = .01+.01+.49_.  
   Diversity would be:  
   _KL-DIV((.29/s1, .19/s1, .01/s1) || (.01/s2, .01/s2, .49/s2)) + KL-DIV((.01/s2, .01/s2, .49/s2) || (.29/s1, .19/s1, .01/s1)) = 6.67_. This indicates that the two are very far from each other.  
   This example simplifies the actual computation.

4. **Answerability** will determine whether **questions can be answered from** the document text and will be mainly determined by Likert scores, either obtained automatically using at least two LLMs, or manually, subject to funding availability.

5) **Complexity** will rate **how easy questions are** to answer. Ideally, the questions should be **as difficult as possible while remaining answerable**. It will again be determined mainly by Likert scores, automatic or (hopefully) manual. All of these measures should be **as high as possible**. Preference is given to answerability, but it will not be enough on its own. In the end, the measures for the documents will be aggregated but **also manually reviewed independently**. We will do qualitative manual error analysis to further supplement the automatic quantitative measures.

## **“Student” submissions scoring**

The student submissions will be mostly rated using simple word sequence based algorithms on the confidential test dataset golden **question-answer pairs**. This will be supplemented by manual review and the output of Teacher and Evaluator submissions.

## **“Evaluator” submissions scoring**

We will compare different evaluator systems and **manually select ones with best results**. We will use test dataset golden question-answer pairs and Teacher and Student submissions to base our evaluation of the evaluator on.

## **Submission format**

    If your system cannot produce reliable JSON, you can fix the format manually. We are not testing JSON competence here!
    However, be sure to document all such steps.

You may submit several quizzes, response sets and evaluation sets, e.g. if you experiment with some settings for your system or formulations for your prompts. For expensive (LLM) and esp. manual evaluation, you need to **indicate one output per submission type** as you **primary submission**. Submit quizzes, responses, evaluations as a **json dictionary where the key strings indicate location and the values are lists of lists of values.** You may choose any filename but make sure it begins with “questions”, “answers” or “ratings” and ends with “.json”. Example key “ukrbiology/book01/topic01-Різноманітність тварин/questions1.json”. In a nutshell, **treat the dictionary as you would a file system**.

Example outputs are listed below for each submission type.

    Check the JSON of your submission before uploading – use e.g.
    https://jsonlint.com/
    to validate it!

### Submission 1: Teacher

We expect the participating systems to **accept a document** as described in the Data section and to **output a test quiz**, suitable for submission to student and evaluator systems. The question will be inside its own list. Optionally, example answers may be added to the list as seen below.

Expected output format for a teacher submission:

    {
      "popular/video-22/questions0.json": [
    	[
      	"What three negative effects of vaccination are mentioned on the Internet?",
      	"Allergic reactions, disability, death.",
      	"Allergic reactions, disability."
    	],
    	[
      	"How many children out of 10 million would have an allergic reaction after vaccination against measles? (Percentage or number)",
      	"0.001%",
      	"100"
    	],
    	[
      	"What three types of cells make up the immune system?",
      	"Soldiers, intelligent cells, weapons factories.",
      	"Warriors, commanders, weapons factories."
    	],
    	[
      	"What are the weapons of immunity against infection?",
      	"Antibodies."
    	],
    	[
      	"How long does it take for intelligent cells to prepare weapons against      infection?",
      	"A few days.",
      	"A couple days."
    	]
      ]
       "popular/video-22/questions1.json": [...],
       "popular/video-23/questions0.json": [...]
    }

### Submission 2: Student

We expect the participating systems to **accept a document**, \***\*a number of **test quizzes as above** (without the reference answers) and to **return the questions with responses**, in a format scorable by evaluator systems. When evaluating, **only the first response\*\* is scored but you are allowed to submit multiple answer lists with different locations. Notice that the format is the same as in the task above.

Expected output format for a student submission:

    {
      "popular/video-22/answers0.json": [
    	[
      	"What three negative effects of vaccination are mentioned on the Internet?",
      	"Allergic reactions, disability, death."
    	],
    	[
      	"What three types of cells make up the immune system?",
      	"Soldiers, intelligent cells, weapons factories."
    	],
    	[
      	"What are the weapons of immunity against infection?",
      	"Antibodies."
    	],
    	[
      	"How long does it take for intelligent cells to prepare weapons against infection?",
      	"A few days."
    	]
      ],
       "popular/video-22/answers1.json": [...],
       "popular/video-23/answers0.json": [...]
    }

### Submission 3: Evaluator

We expect the participating systems to **accept a document ,questions and responses** for a test quiz as above and to **return scores for each response**.

Expected output format for an evaluator submission:

    {
      "popular/video-22/ratings0.json": [
    	[
      	0,
      	1
    	],
    	[
      	10,
      	3
    	],
    	[
      	5
    	],
    	[
      	8,
      	8
    	]
      ],
      "popular/video-22/ratings1.json": [...],
      "popular/video-23/ratings0.json": [...]
    }

[Submit your response scores here](https://docs.google.com/forms/d/e/1FAIpQLScrLJrbeBwXnqUBCoZObbNL5Mv5STTkOue68uoDWrz5nCZFZw/viewform?usp=sf_link)!

## **Multilinguality**

The materials are provided in English, German or Ukrainian, depending on the data sources. The **questions and answers generated** by participating Teachers and Students should be provided **exclusively in English**. Student and Evaluator submissions should thus **gracefully handle mismatching languages** of the question and the answer. Note that for some data sources, we have manually created questions in the original language, so e.g. for Ukrainian, both Uk-Uk-En as well as Uk-En-En are possible inputs for Evaluators.
