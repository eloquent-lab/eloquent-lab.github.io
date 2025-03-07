import os.path
from translitua import translit, UkrainianSimple
files = ["../devset/ukrbiology/book01/topic01-Різноманітність тварин/text.txt",
"../devset/ukrbiology/book01/topic02-Процеси життєдіяльностітварин/text.txt",
"../devset/ukrbiology/book01/topic03-Поведінка тварин/text.txt"]
for x in files:
    with open(x) as inf:
        t = translit(inf.read(), UkrainianSimple)
    with open(os.path.splitext(x)[0]+".translit.txt", "w") as outf:
        outf.write(t)
