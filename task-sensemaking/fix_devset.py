import argparse
import os
import json

if __name__ == "__main__":
    l = []
    root = "devset"
    ln = len(root)
    for (root_i,dirs,files) in os.walk(root):
        for x in files:
            if x.endswith(".json"):
                l.append(os.path.join(root_i, x)[ln+1:])
    for x in l:
        p = os.path.join(root, x)
        with open(p) as f:
            li = json.load(f)
        with open(p, "w") as f:
            d = [{"question":x[0], "reference-answers": x[1:]} for x in li]
            json.dump(d, f, ensure_ascii=False, indent=4)