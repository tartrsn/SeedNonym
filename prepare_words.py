import spacy
import pickle

nlp = spacy.load("en_core_web_lg")  # make sure to use larger model!
with open("static/words.txt") as f:
    tokens = nlp(f.read().replace("\n", " "))

similarities = {}

for token1 in tokens:
    for token2 in tokens:
        if token1 == token2:
            continue
        similarities[token1.text] = similarities.get(token1.text, []) + [(token2.text, token1.similarity(token2))]

for token1, smlrts in similarities.items():
    sorted_smlrts = [x for x in sorted(smlrts, key=lambda x: x[1], reverse=True)]
    similarities[token1] = {
        "words": [x[0] for x in sorted_smlrts],
        "similarities": [x[1] for x in sorted_smlrts]
        }

for token, v in similarities.items():
    print(token)
    print(v["words"])
    print(v["similarities"])
    print("+"*50)

pickle.dump(similarities, open("static/similarities.pickle", "wb"))
