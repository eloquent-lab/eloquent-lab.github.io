#!/bin/sh
python3 teacher_evaluator.py --token "" --json_path ../baseline_sripts/devset.json --data_path "../devset" --llm_amount 0 --top_windows 10
python3 student_evaluator.py --json_path ../baseline_sripts/devset.json --data_path "../devset"
python3 teacher_evaluator.py --token "" --json_path ../baseline_sripts/baseline.json --data_path "../devset" --llm_amount 0 --top_windows 10
python3 student_evaluator.py --json_path ../baseline_sripts/baseline.json --data_path "../devset"