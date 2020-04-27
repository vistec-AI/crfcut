#  CRF-Cut: Sentence Segmentation

The objective of CRF-Cut (Conditional Random Fields - Cut) is to cut sentences and we will able to utilize these sentences.

The process of training is to get sentences and we will tokenize words and assign label for each word `I`: Inside of sentence and `E`: End of sentence.

The result of CRF-Cut is trained by different datasets are as follows:

| dataset_train | dataset_validate | E_f1-score |
|---------------|------------------|------------|
| Ted           | Ted              | 0.72       |
| Orchid        | Orchid           | 0.77       |
| Fake review   | Fake review      | 0.97       |

Google colab: https://colab.research.google.com/drive/12nszk-N5LwpHzitlYvhNWVUDSBj30Z1Y

# Sentence Breaking Journal

## What doesn't work

* POS-perceptron
* Larger features than window = 2, max_n_gram = 3
* Number of verbs to the left and right
* Rule-based override
* L2 regularization - also not practical
* POS-artagger - not really too slow
* ORCHID - different domains get totally different results

## What to try

* TNC

## What worked

* Fake "convolutions" of window = 2, max_n_gram = 3
* L1 regularization of 1
* Predict end of sentence (space) instead of beginning of sentence
* Custom POS - only faster convergence
* Try with ORCHID to compare performance more fairly - 87% vs 95% SOTA

# Requirements

- pythainlp
- python-crfsuite
- pandas
- numpy
- scikit-learn
- tqdm