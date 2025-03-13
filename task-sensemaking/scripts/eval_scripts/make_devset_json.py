import argparse
import os
import json

if __name__ == "__main__":
    l = []
    root = "../devset"
    ln = len(root)
    for (root_i,dirs,files) in os.walk(root):
        for x in files:
            if x.endswith(".json"):
                l.append(os.path.join(root_i, x)[ln+1:])
    d = {}
    for x in l:
        p = os.path.join(root, x)
        d[x] = json.load(open(p))
    json.dump(d, open("devset.json", "w"), indent=4, ensure_ascii=False)