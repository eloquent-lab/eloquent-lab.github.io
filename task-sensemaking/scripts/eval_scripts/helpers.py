import json
import os

##Fill in you access token here to have it as default
default_access_token = ...

def load_questions_and_texts(json_file, data_path):
    r = []
    with open(json_file) as f:
        questions_answers = json.load(f)
    for path in questions_answers:
        dn = os.path.join(data_path, os.path.dirname(path))
        enpath = os.path.join(dn, "text.en.txt")
        if os.path.isfile(enpath):
            with open(enpath) as f:
                text = f.read()
        else:
            with open(os.path.join(dn, "text.txt")) as f:
                text = f.read()
        qp = os.path.join(dn, "questions.json")
        if os.path.isfile(qp):
            with open(qp) as f:
                question_reference = json.load(f)
        question = questions_answers[path]
        r.append((question, text, question_reference, path))
    return r

##Just a placeholder definition until all the issues are worked out.
def lora(model, train_loader, valid_loader, tokenizer, EPOCHS, MODEL_PATH):
    raise NotImplementedError()