# Pairwise Online Learning-to-Rank

This is a proof-of-concept of pairwise online learning-to-rank (LTR/OL2R).
We use a truncated version (10,000 documents) of the [CORD-19](https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge) dataset
which contains the metadata and 768-dimensional vector embeddings of research articles around COVID-19.
According to their [paper](https://arxiv.org/pdf/2004.10706.pdf), the embeddings were generated using the [allenai/specter](https://huggingface.co/allenai/specter) language model, which we will use to embed user queries.

## Demo

![Demo](https://user-images.githubusercontent.com/22933507/242871112-09c69e44-db0f-4035-be84-4af59b4aac96.gif)


## How It Works

First, we initialize a feed-forward neural network with 3x768 input neurons, enough for the embedding of a query and one pair of documents.
The output of this model is a single sigmoid neuron since the pairwise approach is an instance of binary classification.

For every unseen query, we do simple semantic search (cosine similarity) on the embeddings, and retrieve an ordered list of the five most relevant documents.

We then bootstrap our model on this ranking.
That is, we generate training data that asserts for all combinations of these five documents (in _query-document-document_ triplets) if one is more relevant than the other.
Furthermore, we symmetrically train on the opposite result.
For instance, we assert that the 2nd document is more relevant than the 1st, but also that the 1st is less relevant than the 2nd.
In our tests, this tweak seemed to yield better results.

The user is now given the results `1, 2, 3, 4, 5`.
If the user selects result `3`, this is taken as a suggestion for `3, 1, 2, 4, 5` as the best ordering,
and the model is re-trained on that.

An important thing to keep in mind with OL2R is that all training must be loose and flexible, so you don't want to overtrain your model on a new "truth".

## Remarks

- A query's result set is eternally biased and restricted to the top-5 results returned by the initial document matching

  → Remedy idea: some measure of exploitation/exploration balancing.
- A single model will be trained on multiple queries. I haven't thought through the dynamics this would have, if they are useful or disruptive, or how they can be exploited.

## Setup

Install dependencies.

```bash
pip install -r requirements.txt
```

Set this environment variable or your console will get spammed with warning messages.

```bash
export TOKENIZERS_PARALLELISM=true
```

Run and enjoy.

```bash
python main.py
```

Model is stored in memory; you restart the app, you start from scratch.
