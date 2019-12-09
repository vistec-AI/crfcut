# Sentence Breaking Journal

## What doesn't work

* POS-perceptron
* Larger features than window = 2, max_n_gram = 3
* Number of verbs to the left and right
* Rule-based override
* L2 regularization - also not practical
* POS-artagger - not really too slow

## What to try

* Try with ORCHID to compare performance more fairly

* More data from TNC and ORCHID

## What worked

* Fake "convolutions" of window = 2, max_n_gram = 3
* L1 regularization of 1
* Predict end of sentence (space) instead of beginning of sentence
* Custom POS - only faster convergence