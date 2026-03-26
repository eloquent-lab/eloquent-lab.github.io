# PISA 2000 Reading Literacy — Released Items Dataset

## Overview

This dataset contains the released reading literacy items from the PISA 2000 assessment, serialized into JSONL format. Each line in `PISA_2000.jsonl` represents a single **item** — an independently scored response.
Multi-part questions (e.g., paired open responses, table-based sub-items) are expanded into one item per sub-part.

PISA 2000 was a **paper-based** assessment. Metadata such as difficulty scores, source requirements, or pedagogical explanations are not included in the official released items (unlike PISA 2018). However, these fields are present in the schema for cross-year compatibility but are set to `null`. Cognitive processes and scoring guides (correct answers and rubrics) were sourced separately and are populated for all items.

**Total:** 46 lines across 11 units (43 questions)

| Unit                   | Questions | JSONL Lines |
|------------------------|-----------|-------------|
| Flu                    | 5         | 5           |
| Police                 | 4         | 4           |
| The Gift               | 7         | 8           |
| Graffiti               | 4         | 4           |
| Amanda and the Duchess | 4         | 6           |
| Runners                | 4         | 4           |
| New Rules              | 2         | 2           |
| Personnel              | 2         | 2           |
| Cell Phone Safety      | 4         | 4           |
| The Play's the Thing   | 4         | 4           |
| Telecommuting          | 3         | 3           |

> One JSONL line = one **item** (an independently scored response). For multi-part questions, each sub-part is a separate line.

## Sources

- **Released Items PDF:** [PISA 2000 Reading Items](https://nces.ed.gov/surveys/pisa/pdf/items2_reading.pdf)

## Schema

### Field Reference

| Field               | Type            | Required    | Description                                                   |
|---------------------|-----------------|-------------|---------------------------------------------------------------|
| `item_id`           | string          | Yes         | PISA item code, e.g. `R077Q02`, `R119Q09A`                    |
| `unit_title`        | string          | Yes         | Reading unit name, e.g. `"Flu"`                                |
| `released_item`     | string          | Yes         | Item number within unit, e.g. `"1"`, `"7"`                    |
| `cognitive_process` | string          | Yes         | PISA reading cognitive process (cf. list below)                |
| `response_format`   | string          | Yes         | Item type (cf. Response Formats below)                         |
| `difficulty_score`  | null            | Yes         | Always `null` (not available for PISA 2000 released items)     |
| `difficulty_level`  | null            | Yes         | Always `null` (not available for PISA 2000 released items)     |
| `source_requirement`| null            | Yes         | Always `null` (not available for PISA 2000 released items)     |
| `source_tab`        | null            | Yes         | Always `null` (not available for PISA 2000 released items)     |
| `stimulus_text`     | string          | Yes         | Full text of the reading passage (see conventions below)       |
| `question_stem`     | string          | Sub-items   | Shared stem for multi-part items (e.g., Amanda Q3)             |
| `question`          | string          | Yes         | The specific question or statement to evaluate                 |
| `options`           | object          | MCQ items   | Answer choices keyed by letter (typically `A`–`D`, sometimes `A`–`E`) |
| `answer`            | object          | Yes         | Correct answer and scoring info (see Answer Types below)       |
| `item_explanation`  | null            | Yes         | Always `null` (not available for PISA 2000 released items)     |

### Response Formats

| Format                  | Description                                         | Applies to |
|-------------------------|-----------------------------------------------------|------------|
| `Simple Multiple Choice`| Single correct answer from options                   | MCQ items  |
| `Open Response`         | Free-text answer (no options field)                  | Open items |

### Cognitive Processes

The following PISA 2000 reading cognitive processes appear in the dataset:

- Access and retrieve
- Integrate and interpret
- Interpreting texts
- Reflect and evaluate

### Answer Types

#### Simple MCQ

```json
{
  "type": "simple",
  "correct": "B"
}
```

#### Open Response

Scoring rubrics are verbatim from the PISA 2000 scoring guide. Multi-path criteria are separated by `\n\nOR\n\n`. When the source provides concrete example responses, they are included in the criteria text.

```json
{
  "type": "open",
  "full_credit": {
    "criteria": "Verbatim scoring criteria from PISA...",
    "acceptable_answers": null
  },
  "partial_credit": {
    "criteria": "Verbatim partial credit criteria..."
  },
  "incorrect": {
    "criteria": "Verbatim incorrect response criteria..."
  }
}
```

- `full_credit.criteria` — verbatim from the PISA scoring guide
- `full_credit.acceptable_answers` — always `null` (PISA 2000 does not provide structured example answers as PISA 2018 does; example responses are included inline in the criteria text when provided)
- `partial_credit` — present only for items with partial credit scoring (scoring code `0 1 2 9`)
- `incorrect` — present when the scoring guide provides explicit incorrect response criteria

### Null Fields (Cross-Year Compatibility)

The following fields are always `null` to maintain schema compatibility with PISA 2018, since the released items PDF from 2000 does not include the detailed metadata available for later PISA cycles:

- `difficulty_score` — PISA difficulty score
- `difficulty_level` — PISA proficiency level
- `source_requirement` — single vs. multiple source
- `source_tab` — tab(s) needed to answer
- `item_explanation` — pedagogical rationale

### Stimulus Text Conventions

- `stimulus_text` contains the full reading passage for the unit
- `\n\n` marks paragraph breaks; `\n` marks line breaks within a block (e.g., character name followed by dialogue, numbered list items)
- Two-column PDF layouts are serialized left column first, then right column
- Units with multiple texts use `[Text 1]`/`[Text 2]` or descriptive markers (e.g., `[Are cell phones dangerous?]`/`[If you use a cell phone...]`) with `---` as separator between texts
- Stage directions in play texts use markdown-style italics (`*text*`) to distinguish them from dialogue
- The stimulus is denormalized (duplicated in each JSONL line that references it)
- No `[Image: ...]` tags — only text content is included

### Serialization Rules

#### Multi-part Open Response

When a PISA question requires multiple distinct answers (e.g., The Gift Q1 asks for two separate speaker identifications), each part becomes its own JSONL line:

- `item_id` uses the base PISA code + letter suffix: `R119Q09A`, `R119Q09B`
- Both sub-items share the same `released_item` number
- Each sub-item has its own `question`

#### Table-based sub-items

When a PISA question presents a table where the student evaluates multiple statements (e.g., Amanda Q3), each row becomes its own JSONL line:

- `item_id` uses variations of the base code: `R216Q03A`, `R216Q03B`, `R216Q03C`
- All sub-items share: `question_stem`, `stimulus_text`, `released_item`
- Each sub-item has its own: `item_id`, `question`

#### Skipped items

Some items were not serialized because they depend on visual elements (e.g., Amanda Q4 requires image perception). When an item is skipped, subsequent items preserve their original `released_item` numbering (e.g., Amanda Q5 keeps `"5"` even though Q4 is absent).

### Multi-Text Units

Two units contain multiple distinct texts separated by `---`:

| Unit                   | Text Markers                                                     |
|------------------------|------------------------------------------------------------------|
| Amanda and the Duchess | `[Text 1]` (play extract) and `[Text 2]` (theatrical definitions)|
| Cell Phone Safety      | `[Are cell phones dangerous?]` and `[If you use a cell phone...]`|

### Play Text Formatting

Two units contain play/script extracts with specific formatting conventions:

| Unit                   | Source                              |
|------------------------|-------------------------------------|
| Amanda and the Duchess | Jean Anouilh, *Léocadia* (Scene II) |
| The Play's the Thing   | Ferenc Molnár                       |

- **Character names** appear on their own line in ALL CAPS, followed by `\n` and their dialogue
- **Character annotations** (acting directions after character name) use italics: `AMANDA, *sincerely surprised*`
- **Standalone stage directions** are full italic paragraphs: `*A crossroads in the castle grounds...*`
- **Inline stage directions** within dialogue use italics: `*Stands up.* Good evening.`
- **Parenthetical directions** within dialogue: `(*She has taken her by the arm.*)`
- `\n\n` separates speakers/stage direction blocks
