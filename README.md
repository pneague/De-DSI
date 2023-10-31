# Peer-to-Peer Online Learning-to-Rank

This is a proof-of-concept of a pairwise approach to collaborative online learning-to-rank (LTR/OL2R) in a decentralized p2p network.
We use a truncated version (10,000 documents) of the [CORD-19](https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge) dataset
which contains the metadata and 768-dimensional vector embeddings of research articles around COVID-19.
According to their [paper](https://arxiv.org/pdf/2004.10706.pdf), the embeddings were generated using the [allenai/specter](https://huggingface.co/allenai/specter) language model, which we will use to embed user queries.

## Demo

![Demo](https://github.com/mg98/p2p-ol2r/assets/22933507/003e636d-857c-4089-9955-9ac19d16927a)

## How It Works

First, we initialize a feed-forward neural network with 3x768 input neurons, enough for the embedding of a query and one pair of documents.
The output of this model is a single sigmoid neuron since the pairwise approach is an instance of binary classification.

We do simple semantic search (cosine similarity) on the embeddings to retrieve a list of the five most relevant documents.
The user is given the results `1, 2, 3, 4, 5` (initially, the ordering will be random).
If the user selects result `3`, this is taken as a suggestion for `3, 1, 2, 4, 5` as the best ordering, and the model is loosely trained on that ordering.

That is, we generate training data that asserts for all combinations of these five documents (in _query-document-document_ triplets) if one is more relevant than the other.
Furthermore, we symmetrically train on the opposite result.
For instance, we assert that the 2nd document is more relevant than the 1st, but also that the 1st is less relevant than the 2nd.
In our tests, this tweak seemed to yield better results.

The updated model is broadcasted to all neighbors.
On the receiving end, incoming models are merged with its local model by average aggregation.

## Remarks

- A query's result set is eternally biased and restricted to the top-5 results returned by the initial document matching

  → Remedy idea: some measure of exploitation/exploration balancing.
- A single model will be trained on multiple queries. I haven't thought through the dynamics this would have, if they are useful or disruptive, or how they can be exploited.

## Setup & Usage

Install dependencies.

```bash
pip install -r requirements.txt
```

Set this environment variable or your console will get spammed with warning messages.

```bash
export TOKENIZERS_PARALLELISM=true
```

Launch single peer instances with an arbitrary ID (this will create `certs/ec{id}.pem`).

```bash
python main.py <id>
```

Model is stored in memory; you restart the app, you start from scratch.

### Advanced Options

- Use `-q` to enable quantization of the model.
In our experiments, this yielded a data reduction of 4x when transferring the model,
however, at the cost of performance speed and accuracy.

- Use `-s` to perform a simulation of user clicks on a set query
(e.g., 100 clicks on result #1, 90 clicks on result #2, etc.).
