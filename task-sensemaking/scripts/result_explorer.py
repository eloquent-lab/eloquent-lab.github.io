import os
import json
import shutil
from collections import OrderedDict
def topath(x):
    return x.replace("/", "-").replace(" ", "-")

def make(inp = "results", outp = "results_per_text"):
    x = os.listdir(inp)[0]
    p = os.path.join(inp, x)
    with open(p) as f:
        results = json.load(f)
    texts = list(results.keys())
    kinds = results[texts[0]].keys()
    shutil.rmtree(outp)
    os.makedirs(outp, exist_ok=True)
    paths = {}
    for kind in kinds:
        for text in texts:
            all_texts = {}
            first = True
            for x in os.listdir(inp):
                p = os.path.join(inp, x)
                with open(p) as f:
                    results = json.load(f)
                v =results[text][kind]
                examples = v[0]["examples"][0]
                for y in v:
                    y.pop("examples")
                all_texts[x] = v[0]
                if first:
                    all_texts["gold"] = v[1]
                ofol = os.path.join(outp, topath(kind))
                os.makedirs(ofol, exist_ok=True)
                op = os.path.join(ofol, topath(text+"-"+x))
                with open(op+".txt", "w") as f:
                    f.write(examples)
                first = False

            op = os.path.join(ofol, topath(text))
            if text not in paths:
                paths[text] = []
            paths[text].append(op)
            with open(op, "w") as f:
                json.dump(OrderedDict(sorted(all_texts.items(), key=lambda x: x[0])), f, ensure_ascii=False, indent=4)

    #for k, x in paths.items():
    #    fol = "top_windows"
    #    with open(os.path.join(fol,k), "w") as f:
    #        f.write()

if __name__ == "__main__":
    inp = "results"
    outp = "results_per_text"
    interesting = "popular-video-22-questions.json"
    l = []
    if True:
        make(inp, outp)


