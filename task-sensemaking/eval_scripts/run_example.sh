#!/bin/sh
python3 teacher_evaluator.py --token "" --json_path devset.json --data_path "../devset" --llm_amount 0 --top_windows 10
python3 student_evaluator.py --json_path devset.json --data_path "../devset"