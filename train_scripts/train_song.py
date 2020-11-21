# adapted from @bact at https://colab.research.google.com/drive/1hdtmwTXHLrqNmDhDqHnTQGpDVy1aJc4t
import pandas as pd
import numpy as np
import re
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import pycrfsuite
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from pythainlp.tokenize import word_tokenize, syllable_tokenize
from pythainlp.tag import pos_tag

token_func = word_tokenize  # syllable_tokenize

import pythainlp

print(f"PyThaiNLP version {pythainlp.__version__}")


def generate_tuples(all_sentences):
    all_tuples = []
    for i in tqdm(range(len(all_sentences))):
        tuples = []
        for s in all_sentences[i].split("|"):
            s_lst = word_tokenize(s)
            for j in range(len(s_lst)):
                lab = "E" if j == len(s_lst) - 1 else "I"
                tuples.append((s_lst[j], lab))
        all_tuples.append(tuples)
    return all_tuples


def generate_tuple_df(all_tuples):
    tuple_dfs = [pd.DataFrame(i) for i in all_tuples]
    tuple_df = pd.concat(tuple_dfs).reset_index(drop=True)
    tuple_df.columns = ["word", "label"]
    tuple_df["next_lab"] = tuple_df.label.shift(-1)
    tuple_df["previous_lab"] = tuple_df.label.shift(1)
    return tuple_df


def extract_features(doc, window=3, max_n_gram=3):
    #     #paddings for POS
    #     doc_pos = ['xxpad' for i in range(window)] + \
    #         [p for (w,p) in pos_tag(doc,engine='perceptorn', corpus='orchid')] + ['xxpad' for i in range(window)]

    # padding for words
    doc = (
        ["xxpad" for i in range(window)]
        + doc
        + ["xxpad" for i in range(window)]
    )

    doc_ender = []
    doc_starter = []
    # add enders
    for i in range(len(doc)):
        if doc[i] in enders:
            doc_ender.append("ender")
        else:
            doc_ender.append("normal")
    # add starters
    for i in range(len(doc)):
        if doc[i] in starters:
            doc_starter.append("starter")
        else:
            doc_starter.append("normal")

    doc_features = []
    # for each word
    for i in range(window, len(doc) - window):
        # bias term
        word_features = ["bias"]

        # ngram features
        for n_gram in range(1, min(max_n_gram + 1, 2 + window * 2)):
            for j in range(i - window, i + window + 2 - n_gram):
                feature_position = f"{n_gram}_{j-i}_{j-i+n_gram}"

                # word
                word_ = f'{"|".join(doc[j:(j+n_gram)])}'
                word_features += [f"word_{feature_position}={word_}"]

                #                #pos
                #                 pos_ =f'{"|".join(doc_pos[j:(j+n_gram)])}'
                #                 word_features += [f'pos_{feature_position}={pos_}']

                # enders
                ender_ = f'{"|".join(doc_ender[j:(j+n_gram)])}'
                word_features += [f"ender_{feature_position}={ender_}"]

                # starters
                starter_ = f'{"|".join(doc_starter[j:(j+n_gram)])}'
                word_features += [f"starter_{feature_position}={starter_}"]

        # number of tokens to the left and right
        nb_left = 0
        for l in range(i)[::-1]:
            if doc[l] == "<space>":
                break
            if True:
                nb_left += 1.0
        nb_right = 0
        for r in range(i + 1, len(doc)):
            if doc[r] == "<space>":
                break
            if True:
                nb_right += 1.0
        word_features += [f"nb_left={nb_left}", f"nb_right={nb_right}"]

        # append to feature per word
        doc_features.append(word_features)
    return doc_features


def generate_xy(all_tuples):
    # target
    y = [[l for (w, l) in t] for t in all_tuples]
    # features
    x_pre = [[w for (w, l) in t] for t in all_tuples]
    x = [extract_features(x_) for x_ in tqdm(x_pre)]
    return x, y


df = pd.read_pickle("data/siamzone_lyrics_line_tokenized.pickle")
print(df.shape)
df["line_tokenized"] = df.line_tokenized.map(lambda x: " |".join(x.split("|")))
df["line_tokenized"] = df.line_tokenized + " "

# Split train and test set at 80/20 proportion
train_lines, test_lines, _, _ = train_test_split(
    df.line_tokenized, df.song_title, test_size=0.2, random_state=1412
)
train_lines = [i for i in train_lines]
test_lines = [i for i in test_lines]

# tuples
train_tuples = generate_tuples(train_lines)
test_tuples = generate_tuples(test_lines)

# tuple df
train_tuple_df = generate_tuple_df(train_tuples)
test_tuple_df = generate_tuple_df(test_tuples)

top_starters = train_tuple_df[
    train_tuple_df.previous_lab == "E"
].word.value_counts()
top_enders = train_tuple_df[train_tuple_df.next_lab == "E"].word.value_counts()

enders = top_enders[:50].reset_index()["index"].tolist()
starters = top_starters[:50].reset_index()["index"].tolist()
print(f"Enders: {enders}")
print(f"Starters: {starters}")

print("generating x, y")
x_train, y_train = generate_xy(train_tuples)
x_test, y_test = generate_xy(test_tuples)

print("start training")
trainer = pycrfsuite.Trainer(verbose=True)

for xseq, yseq in tqdm(zip(x_train, y_train)):
    trainer.append(xseq, yseq)

trainer.set_params(
    {
        "c1": 1.0,
        "c2": 0.0,
        "max_iterations": 1000,
        "feature.possible_transitions": True,
        "feature.minfreq": 2.0,
    }
)

trainer.train("models/song-crf.model")

# Predict (using test set)
tagger = pycrfsuite.Tagger()
tagger.open("models/song-crf.model")
y_pred = []
for xseq in tqdm(x_test, total=len(x_test)):
    y_pred.append(tagger.tag(xseq))

# Evaluate at word-level
labels = {"E": 0, "I": 1}  # classification_report() needs values in 0s and 1s
predictions = np.array([labels[tag] for row in y_pred for tag in row])
words = [tup[0] for li in test_tuples for tup in li]
baselines = [labels["E"] if word == " " else labels["I"] for word in words]
truths = np.array([labels[tag] for row in y_test for tag in row])

print(classification_report(truths, baselines, target_names=["E", "I"]))
print(classification_report(truths, predictions, target_names=["E", "I"]))

# baseline; cut at spaces
#               precision    recall  f1-score   support

#            E       0.53      1.00      0.69    124832
#            I       1.00      0.90      0.95   1087762

#     accuracy                           0.91   1212594
#    macro avg       0.76      0.95      0.82   1212594
# weighted avg       0.95      0.91      0.92   1212594

# model using everything;window=3, starts=50, enders=50, nb_left, nb_right

# window=3, starts=50, enders=50, nb_left, nb_right
# ***** Iteration #306 *****
# Loss: 361670.891636
# Feature norm: 311.532287
# Error norm: 1102.739051
# Active features: 303617
# Line search trials: 1
# Line search step: 1.000000
# Seconds required for this iteration: 10.706

# L-BFGS terminated with the stopping criteria
# Total seconds required for training: 3531.473

# Storing the model
# Number of active features: 303617 (4597874)
# Number of active attributes: 209524 (8375394)
# Number of active labels: 2 (2)
# Writing labels
# Writing attributes
# Writing feature references for transitions
# Writing feature references for attributes
# Seconds required: 7.704
#               precision    recall  f1-score   support

#            E       0.72      0.79      0.76    124832
#            I       0.98      0.97      0.97   1087762

#     accuracy                           0.95   1212594
#    macro avg       0.85      0.88      0.86   1212594
# weighted avg       0.95      0.95      0.95   1212594

# window=2, starts=50, enders=50, nb_left, nb_right
# ***** Iteration #233 *****
# Loss: 412282.127452
# Feature norm: 266.170577
# Error norm: 847.584994
# Active features: 237894
# Line search trials: 1
# Line search step: 1.000000
# Seconds required for this iteration: 7.783

# L-BFGS terminated with the stopping criteria
# Total seconds required for training: 1669.224

# Storing the model
# Number of active features: 237894 (2846580)
# Number of active attributes: 167357 (5203603)
# Number of active labels: 2 (2)
# Writing labels
# Writing attributes
# Writing feature references for transitions
# Writing feature references for attributes
# Seconds required: 5.282
#               precision    recall  f1-score   support

#            E       0.70      0.79      0.75    124832
#            I       0.98      0.96      0.97   1087762

#     accuracy                           0.94   1212594
#    macro avg       0.84      0.88      0.86   1212594
# weighted avg       0.95      0.94      0.95   1212594
