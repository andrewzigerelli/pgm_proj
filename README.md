Interpretable graphical models for author attribution

Datasets: train.csv, test.csv, val.csv
* Features: text
* Label: author_id

Code:
* bert.py: contains baseline BERT model
* lda.ipynb: contains LDA and supervised LDA models
* bvae.py: contains the beta-vae + shallow nn implementation.
bvae.py contains both a beta-vae implementation on bag of words,
but also contains an implementation using a pretrain word2vec model. this
requires downloading the google word2vec dataset, which is too big to find here,
but it is available on kaggle (word2vec-GoogleNews-vectors-negative300.bin)
