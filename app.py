import numpy as np
import pickle

from flask import Flask, request

app = Flask(__name__)
words_similarities = pickle.load(open("static/similarities.pickle", "rb"))


def get_k_words_from_n_neighbors(word, n=20, k=5):
    return list(np.random.choice(words_similarities[word]["words"][:n], size=k, replace=False))


@app.route('/synonym', methods=['GET'])
def get_words():
    if request.args.get('word'):
        return "[{}]".format(",".join(get_k_words_from_n_neighbors(request.args.get('word'),
                                                                   int(request.args.get("neighbors", 20)),
                                                                   int(request.args.get("words", 5))
                                                                   )))
    else:
        return 'bad request!', 400


if __name__ == '__main__':
    app.run()
