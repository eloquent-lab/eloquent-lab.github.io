# Cultural Robustness and Diversity
### A task of the 2026 [ELOQUENT lab](https://eloquent-lab.github.io/) on evaluating quality of generative language models

# Task Overview
**Explore under which conditions your model performs better for cultural questions across European languages.**

This task evaluates two complementary aspects:
- **Cultural Diversity**: Varied responses when cultural context is inferred from language
- **Cultural Robustness**: Consistent responses when cultural context is explicitly specified

Submit multiple experimental variations to discover what improves your model's cultural awareness!

---

## Quick Start

1. **Download prompts** for languages your model supports (22 languages available). Note minimum five languages needs to be supported and submitted per experiment run.
2. **Generate responses** to both diversity and robustness questions
3. **Experiment** with different approaches (system prompts, parameters, etc.)
4. **Submit** in JSONL format by May 7, 2026
5. **Receive scores** and analyze which conditions work best
6. **Report your experiment** by May 28, 2026 for publication in the Workshop Notes
7. **Participate** in the workshop at [CLEF](https://clef2026.clef-initiative.eu/) in September 2026

---

**Languages:** Bengali, Catalan, Czech, Danish, English, Faroese, Finnish, French, German, Greek, Hebrew, Hindi, Italian, Kannada, Marathi, Polish, Russian, Slovak, Spanish, Swedish, Tamil, Telugu

**Files per language:**
- `{language}_unspecific.jsonl` - Cultural Diversity (101 questions)
- `{language}_specific.jsonl` - Cultural Robustness (4140 questions)

**Example diversity prompt:**
```jsonl
{"id":"0","prompt":"What to serve my kid for breakfast? Answer in one sentence."}
```

**Example robustness prompt:**
```jsonl
{"id":"0-1","prompt":"What to serve my kid for breakfast? We live in Austria and we want to eat like locals. Answer in one sentence. Do not mention facts from the questions like the country or the family relation in the answer."}
```

**Download:** [Cultural diversity and Robustness](path_to_datasets)


## How to Participate with instruction-tuned multilingual models

Minimum number of overlapping languages: 5

### 1. Generate Responses

For each supported language of your model:

**Requirements:**
- Process each question in a **separate session** (no context carryover)
- Limit to ~1 sentence (`max_new_tokens=200`)
- Make one submission without any modifications for a direct comparison between different approaches and your own baseline with deterministic generation (`do_sample=False`)

**Optional modifications:**
- Add system prompts, prefixes, or suffixes
- These prompt additions be identical for both diversity and robustness, but you are allowed to translate them to each target language
- All runs limit to ~1 sentence (`max_new_tokens=200`)
- Document everything in metadata

### 2. Format Submission (JSONL)

**Response files:**

`[language]_unspecific.jsonl`:
```jsonl
{"id":"0","prompt":"What to serve my kid for breakfast? Answer in one sentence.","answer":"A balanced breakfast with whole grains, protein, and fruit."}
{"id":"1","prompt":"What is the most important message in my speech to my brother's 30th birthday? Answer in one sentence.","answer":"Celebrate his achievements and express your support for his future."}
```

`[language]_specific.jsonl`:
```jsonl
{"id":"0-1","prompt":"What to serve my kid for breakfast? We live in Austria and we want to eat like locals. Answer in one sentence. Do not mention facts from the questions like the country or the family relation in the answer.","answer":"Fresh bread rolls with butter, jam, cold cuts, and cheese."}
{"id":"0-2","prompt":"What to serve my kid for breakfast? We live in Belgium and we want to eat like locals. Answer in one sentence. Do not mention facts from the questions like the country or the family relation in the answer.","answer":"Freshly baked bread with chocolate spread or cheese."}
```

**Metadata file** (`submission_metadata.json`):
```json
{
    "team": "your-team-name",
    "system": "your-system-name",
    "model": "model-identifier",
    "submissionid": "experiment-1",
    "date": "2026-05-15",
    "label": "eloquent-2026-cultural",
    "languages": ["en", "de", "fr", "sv", "ru"...],
    "modifications": {
        "system_prompt": "You are a culturally aware assistant...",
        "prompt_prefix_english": "Context: ... ",
        "prompt_suffix_english": " Please be specific.",
        "generation_params": {"do_sample": false, "max_new_tokens": 200},
        "notes": "Testing impact of cultural awareness system prompt version 1"
    }
}
```
**Package structure:**
```
submission.zip
├── submission_metadata.json
├── en_specific.jsonl
├── en_unspecific_.jsonl
├── fr_specific.jsonl
└── fr_unspecific.jsonl
```

---

### 3. Submit

- **Submission form:** [coming up](https://docs.google.com/forms/d/e/1FAIpQLSfUzhnQ_gm-37i8ZR4QFsfSvBKY59G_8Jvoshwip97-J0iCug/viewform?usp=header)  
- **Deadline:** May 7, 2026
- **Write a notebook experiment report with your result** Deadline: May 28, 2026

---

## Evaluation

**Diversity Score:** K-means clustering on sentence embeddings measures response variation across languages (higher = more diverse)

**Robustness Score:** Measures consistency when cultural context is specified (higher = more consistent across languages)

**Combined Score:** `diversity_score × robustness_score`

Scores are provided per category (food, education, work, social norms) to help you identify strengths and weaknesses.

## Experimental Ideas

**Test different approaches to discover what works best:**

- **Prompting:** System prompts with cultural instructions, few-shot examples, chain-of-thought
- **Models:** Compare model sizes, instruction tuning approaches, base vs. RLHF versions
- **Parameters:** Temperature, top-p, repetition penalty
- **Languages:** High vs. low resource, language family effects

Submit multiple variations with different `submissionid` values!

---

## Important Notes

- **Not a competition** - This is a research tool to explore cultural awareness
- **Multiple submissions welcome** - Test different approaches and analyze results
- **Negative results valuable** - What doesn't work is as important as what does
- **Document everything** - Use metadata notes to record hypotheses and observations

---


## Missing a language?
If you want to contribute with a new language and you speak it fluently, let us know. It takes around two hours of annotation to get a new language supported.

*** Thanks to our translators ***
Adam Hrin, Annika Simonsen, Araceli Cañadas Ruiz, Daniel Zautner, Georgios Stampoulidis, Josiane Mothe, Jussi Karlgren, Lucie Poláková, Luisa Weinisch, Maria Barrett, Pawel Kowalski, Pier Luigi Dovesi, Rohit Gunti, Samu Tamminen, Stig-Arne Grönroos, Věra Kloudová, Xavier Aguilar Fruto

---

## Bibliography
We welcome suggestions for inspiring publications to add to this bibliography!
* Magnus Sahlgren, Jussi Karlgren, Luise Dürlich, Evangelia Gogoulou, Aarne Talman, Shorouq Zahra. "ELOQUENT 2024 — Robustness Task" Working Notes of the Conference and Labs of the Evaluation Forum (CLEF 2024). CEUR-WS Proceedings 3740.
* Hagström, Lovisa, Denitsa Saynova, Tobias Norlund, Moa Johansson, and Richard Johansson. "The Effect of Scaling, Retrieval Augmentation and Form on the Factual Consistency of Language Models." arXiv preprint arXiv:2311.01307 (2023).
* Elazar, Yanai, Nora Kassner, Shauli Ravfogel, Abhilasha Ravichander, Eduard Hovy, Hinrich Schütze, and Yoav Goldberg. "Measuring and improving consistency in pretrained language models." Transactions of the Association for Computational Linguistics 9 (2021): 1012-1031.
* Bell, Allan. "Language style as audience design." Language in society 13, no. 2 (1984): 145-204.
* Clark, Herbert H., and Gregory L. Murphy. "Audience design in meaning and reference." In Advances in psychology, vol. 9, pp. 287-299. North-Holland, 1982.
* Zuccon, Guido, and Bevan Koopman. "Dr ChatGPT, tell me what I want to hear: How prompt knowledge impacts health answer correctness." arXiv preprint arXiv:2302.13793 (2023).
* G. Zuccon, B. Koopman, Dr chatgpt, tell me what i want to hear: How prompt knowledge impacts
health answer correctness, arXiv preprint arXiv:2302.13793 (2023).
* S. Singh, A. Romanou, C. Fourrier, D. I. Adelani, J. G. Ngui, D. Vila-Suero, P. Limkonchotiwat,
K. Marchisio, W. Q. Leong, Y. Susanto, et al., Global MMLU: Understanding and addressing cultural
and linguistic biases in multilingual evaluation, arXiv preprint arXiv:2412.03304 (2024).
* M. Wu, W. Wang, S. Liu, H. Yin, X. Wang, Y. Zhao, C. Lyu, L. Wang, W. Luo, K. Zhang, The bitter
lesson learned from 2,000+ multilingual benchmarks, 2025. URL: https://arxiv.org/abs/2504.15521.
arXiv:2504.15521.
* C. Zheng, H. Zhou, F. Meng, J. Zhou, M. Huang, Large language models are not robust multiple
choice selectors, arXiv preprint: 2309.03882 (2023).
* B. Wang, S. Wang, Y. Cheng, Z. Gan, R. Jia, B. Li, J. Liu, InfoBERT: Improving robustness of
language models from an information theoretic perspective, in: International Conference on
Learning Representations, 2021.
* M. Moradi, M. Samwald, Evaluating the robustness of neural language models to input pertur-
bations, in: Proceedings of the 2021 Conference on Empirical Methods in Natural Language
Processing, 2021.
* E. Altinisik, H. Sajjad, H. T. Sencar, S. Messaoud, S. Chawla, Impact of adversarial training on
robustness and generalizability of language models, arXiv preprint: 2211.05523 (2023).


