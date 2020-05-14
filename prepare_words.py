import spacy
import pickle
import tqdm
import os

LOAD_FROM_CHECKPOIINT = True
CALCULATE_SIMILARITIES = True
FIND_NEIGHBORS = True
CHECKPOINT_STEPS = 1000

similarities = {}

if LOAD_FROM_CHECKPOIINT and os.path.exists("static/similarities_checkpoint.pickle"):
    similarities = pickle.load(open("static/similarities_checkpoint.pickle", "rb"))

if CALCULATE_SIMILARITIES:
    nlp = spacy.load("en_core_web_lg")  # make sure to use larger model!
    with open("static/words.txt") as f:
        tokens = nlp(f.read().replace("\n", " "))

    processed = 0
    for i, token1 in tqdm.tqdm(enumerate(tokens), desc="Calculating similarities for vocabulary"):
        if token1.vector_norm == 0:
            # print(token1.text)
            continue
        processed += 1
        for token2 in tokens[:i]:
            if token2.vector_norm == 0:
                continue
            if token1 == token2:
                continue
            if not (token2.text, token1.similarity(token2)) in similarities.get(token1.text, []):
                similarities[token1.text] = similarities.get(token1.text, []) + [(token2.text, token1.similarity(token2))]
        if processed % CHECKPOINT_STEPS == 0:
            pickle.dump(similarities, open("static/similarities_checkpoint.pickle", "wb"))
            print("saved similarities for {} words".format(processed))


if FIND_NEIGHBORS:
    proccesed_words_f = open("static/processed_words.txt", "w")
    for token1, smlrts in tqdm.tqdm(similarities.items(), desc="Finding neighbors"):
        sorted_smlrts = [x for x in sorted(smlrts, key=lambda x: x[1], reverse=True)]
        similarities[token1] = {
            "words": [x[0] for x in sorted_smlrts],
            "similarities": [x[1] for x in sorted_smlrts]
            }
        proccesed_words_f.write(token1 + "\n")
    proccesed_words_f.close()

for token, v in similarities.items():
    print(token)
    print(v["words"])
    print(v["similarities"])
    print("+"*50)

pickle.dump(similarities, open("static/similarities.pickle", "wb"))
