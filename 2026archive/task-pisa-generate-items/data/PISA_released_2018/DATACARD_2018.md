# PISA 2018 Reading Literacy — Released Items Dataset

## Overview

This dataset contains the released reading literacy items from the PISA 2018 assessment, serialized into JSONL format. Each line in `PISA_2018.jsonl` represents a single **item** ; i.e. an independently scored response. 
Multi-part questions (e.g., tables, drag-and-drop) are expanded into one item per row or tile.

**Total:** 49 lines across 4 units (28 questions)

| Unit                  | Questions | JSONL Lines |
|-----------------------|-----------|-------------|
| Cow's Milk            | 7         | 12          |
| Rapa Nui              | 7         | 16          |
| Chicken Forum         | 7         | 11          |
| The Galapagos Islands | 7         | 10          |

> One JSONL line = one **item** (an independently scored response). For Multi-part questions such as table-based questions, each table row is a separate line.

## Sources

- **Released Items PDF:** [PISA 2018 Released REA Items](https://www.oecd.org/content/dam/oecd/en/about/programmes/edu/pisa/pisa-test/PISA2018_Released_REA_Items_12112019.pdf)
- **PISA Reading Test Page (includes translations):** [https://www.oecd.org/en/about/programmes/pisa/pisa-test-reading.html](https://www.oecd.org/en/about/programmes/pisa/pisa-test-reading.html)

## Schema

### Field Reference

| Field               | Type            | Required       | Description                                            |
|---------------------|-----------------|----------------|--------------------------------------------------------|
| `item_id`           | string          | Yes            | PISA item code, e.g. `CR557Q03`, `CR557Q07_1`          |
| `unit_title`        | string          | Yes            | Reading unit name, e.g. `"Cow's Milk"`                 |
| `released_item`     | string          | Yes            | Item number within unit, e.g. `"1"`, `"7"`             |
| `cognitive_process` | string          | Yes            | PISA cognitive process (cf. list below)                |
| `response_format`   | string          | Yes            | Item type (cf. Response Formats below)                 |
| `difficulty_score`  | int or null     | Yes            | PISA difficulty score                                  |
| `difficulty_level`  | string or null  | Yes            | Proficiency level: `1b`, `1a`, `2`, `3`, `4`, `5`, `6` |
| `source_requirement`| string          | Yes            | `"Single"` or `"Multiple"` sources needed              |
| `source_tab`        | string or array | Yes            | Tab(s) needed to answer (see conventions below)        |
| `stimulus_text`     | string          | Yes            | Full text of all visible tabs (see conventions below)  |
| `question_stem`     | string          | Complex MCQ    | Shared stem for table-based items                      |
| `question`          | string          | Yes            | The specific question or statement to evaluate         |
| `options`           | object          | MCQ items      | Answer choices keyed by letter: `A`, `B`, `C`, `D`     |
| `answer`            | object          | Yes            | Correct answer and scoring info (see Answer Types)     |
| `item_explanation`  | string          | Yes            | Pedagogical rationale from PISA for the item           |

### Cognitive Processes

The following PISA 2018 reading cognitive processes appear in the dataset:

- Access and retrieve information within a text
- Search for and select relevant text
- Represent literal meaning
- Integrate and generate inferences
- Integrate and generate inferences across multiple sources
- Detect and handle conflict
- Reflect on content and form
- Assess quality and credibility

### Response Formats

| Format                                       | Description                                                                | Applies to      |
|----------------------------------------------|----------------------------------------------------------------------------|-----------------|
| `Simple Multiple Choice`                     | Single correct answer from A/B/C/D options                                 | MCQ items       |
| `Simple Multiple Choice – Computer Scored`   | Same as above, explicitly computer-scored                                  | Some MCQ items  |
| `Complex Multiple Choice`                    | Table-based: each row is a separate JSONL line with shared `question_stem` | Table rows      |
| `Complex Multiple Choice – Computer Scored`  | Same as above, explicitly computer-scored                                  | Some table rows |
| `Open Response – Human Coded`                | Free-text answer scored by human coders                                    | Open items      |

### Answer Types

#### Simple MCQ
```json
{
  "type": "simple",
  "correct": "C"
}
```

#### Open Response — Type 1 (simple free-text)
Student provides a free-text answer.
```json
{
  "type": "open",
  "full_credit": {
    "criteria": "Description of what constitutes a correct answer...",
    "acceptable_answers": [
      {
        "answer": "The canonical acceptable answer",
        "paraphrases": ["An acceptable paraphrase", "Another paraphrase"]
      }
    ]
  }
}
```

#### Open Response — Type 2 (choice + justification)
Student selects a choice from `options`, then provides a free-text justification.
```json
{
  "type": "open",
  "full_credit": {
    "criteria": "Selects one option and gives an appropriate explanation...",
    "acceptable_answers": [
      {
        "choice": "A",
        "acceptable_explanations": ["Valid justification 1", "Valid justification 2"],
        "example_responses": ["Example student answer for choice A"]
      },
      {
        "choice": "B",
        "acceptable_explanations": ["..."],
        "example_responses": ["..."]
      }
    ]
  }
}
```

### Stimulus Text Conventions

- `stimulus_text` always includes **all tabs visible** in the PISA interface, even if the student doesn't need them all. The set of visible tabs may change across questions within the same unit (e.g., Rapa Nui Q1–Q2 show Blog + Book Review; Q4+ show Blog + Book Review + Science News)
- `source_tab` contains only the tab(s) **truly needed** to answer the question (per PISA metadata), NOT all available tabs
- Tab names appear as `[Tab Name]` markers (e.g., `[Blog]`, `[Conservation]`)
- URL of the simulated webpage follows the tab marker when visible in the original interface
- Multiple tab sections are separated by `---`
- Tabs with no accessible content appear as markers only (no URL, no content)
- The stimulus is denormalized (duplicated in each JSONL line that references it)
- No `[Image: ...]` tags — only text content is included
- Text follows the visual order of the original PISA interface

### Serialization Rules

#### Complex Multiple Choice (table-based items)
When the PISA item presents a table where the student evaluates multiple statements (e.g., Yes/No per row), each row becomes its own JSONL line:

- `item_id` uses base code + `_N` suffix: `CR557Q07_1`, `CR557Q07_2`, `CR557Q07_3`
- All sub-items share: `question_stem`, `stimulus_text`, `released_item`, `difficulty_score`, `difficulty_level`, `cognitive_process`, `response_format`, `source_requirement`, `source_tab`, `item_explanation`
- Each sub-item has its own: `item_id`, `question`, `options`, `answer`
- Credit for the original PISA item requires all sub-items correct, but they are stored individually

#### Drag-and-drop items
Drag-and-drop tasks (e.g., CR551Q10 in Rapa Nui) are serialized as Complex MCQ with classification options per tile (e.g., Cause/Effect/None).
