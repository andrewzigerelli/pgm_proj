#!/usr/bin/env python3
import argparse
import csv
import pandas as pd

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.distributions import MultivariateNormal

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
import pickle

from gensim.models import KeyedVectors

import numpy as np


class CustomDataset(Dataset):
    def __init__(self, feat, labels):

        self.features = torch.tensor(feat).float()

        num_classes = labels.nunique()
        label_ten = torch.tensor(labels.values)
        self.labels = nn.functional.one_hot(label_ten, num_classes).float()

        # try to add random noise
        self.features = self.features + .01*torch.randn(self.features.shape)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx].unsqueeze(0), self.labels[idx].unsqueeze(0)

class BetaVAEDoc(nn.Module):
    def __init__(self, beta, latent_dim, num_classes):
        super().__init__()
        self.beta = beta
        self.latent_dim = latent_dim
        self.normal = MultivariateNormal(torch.zeros(latent_dim), torch.eye(latent_dim))
        self.num_classes = num_classes

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 1, 2, 2), #in, out, kern_sz, stride, pad
            nn.ELU(),
            nn.Conv1d(1, 1, 5, 2, 2), #in, out, kern_sz, stride, pad
            nn.ELU(),
            nn.Linear(75, 2*latent_dim) # learn mu and log(var) (isotropic)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 75),
            nn.ELU(),
            nn.ConvTranspose1d(1, 1, 2, 2),
            nn.ELU(),
            nn.ConvTranspose1d(1, 1, 10, 2, 4)
        )

        self.classifier = nn.Sequential(
            nn.Linear(300, 200),
            nn.ELU(),
            nn.Linear(200, 100),
            nn.ELU(),
            nn.Linear(100, self.num_classes),
        )


    def reparam_trick(self, params):
        # we learned log(sigma^2) = 2log(sigma)
        mu = params[:,:, :self.latent_dim]
        std_dev = torch.exp(.5*params[:,:,self.latent_dim:])
        sample = self.normal.sample(std_dev.shape[0:2])
        return mu + std_dev*sample


    def forward(self, x):
        params = self.encoder(x)
        latent = self.reparam_trick(params)
        decoded = self.decoder(latent)
        self.params = params
        logits = self.classifier(decoded)
        return decoded, logits


    def loss(self, x, decoded, logits, labels):

        rl = nn.functional.mse_loss(decoded, x)

        # assuming std normal prior
        mu = self.params[:,:, :self.latent_dim]
        logvar = self.params[:,:, self.latent_dim:]
        kl = -.5 * torch.sum(1 - torch.pow(mu,2) + logvar - torch.exp(logvar))

        # classify loss
        ce = torch.nn.functional.cross_entropy(logits, labels)
        return rl + self.beta*kl + ce


class BetaVAE(nn.Module):
    def __init__(self, beta, latent_dim, num_classes):
        super().__init__()
        self.beta = beta
        self.latent_dim = latent_dim
        self.normal = MultivariateNormal(torch.zeros(latent_dim), torch.eye(latent_dim))
        self.num_classes = num_classes

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 1, 10, 2, 4), #in, out, kern_sz, stride, pad
            nn.ELU(),
            nn.Conv1d(1, 1, 4, 4, 0),
            nn.ELU(),
            nn.Conv1d(1, 1, 2, 4, 0),
            nn.ELU(),
            nn.Conv1d(1, 1, 4, 4, 0),
            nn.ELU(),
            nn.Linear(50, 2*latent_dim) # learn mu and log(var) (isotropic)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 50),
            nn.ELU(),
            nn.ConvTranspose1d(1, 1, 4, 4, 0),
            nn.ELU(),
            nn.ConvTranspose1d(1, 1, 2, 4, 0, 2),
            nn.ELU(),
            nn.ConvTranspose1d(1, 1, 4, 4, 0),
            nn.ELU(),
            nn.ConvTranspose1d(1, 1, 10, 2, 4), 
        )

        self.classifier = nn.Sequential(
            nn.Linear(6400, 3200),
            nn.ELU(),
            nn.Linear(3200, 800),
            nn.ELU(),
            nn.Linear(800, 200),
            nn.ELU(),
            nn.Linear(200, self.num_classes),
        )



    def reparam_trick(self, params):
        # we learned log(sigma^2) = 2log(sigma)
        mu = params[:,:, :self.latent_dim]
        std_dev = torch.exp(.5*params[:,:,self.latent_dim:])
        sample = self.normal.sample(std_dev.shape[0:2])
        return mu + std_dev*sample


    def forward(self, x):
        params = self.encoder(x)
        #print(x.shape)
        #print(x[1,:,:10])
        #print(x[1,:,:10])
        #print(torch.norm(x[1,:,:]-x[2,:,:]))
        #print(params)
        #print(torch.norm(params[1,:,:]-params[2,:,:]))
        latent = self.reparam_trick(params)
        decoded = self.decoder(latent)
        self.params = params
        logits = self.classifier(decoded)
        return decoded, logits


    def loss(self, x, decoded, logits, labels):

        rl = nn.functional.mse_loss(decoded, x)

        # assuming std normal prior
        mu = self.params[:,:, :self.latent_dim]
        logvar = self.params[:,:, self.latent_dim:]
        kl = -.5 * torch.sum(1 - torch.pow(mu,2) + logvar - torch.exp(logvar))

        # classify loss
        ce = torch.nn.functional.cross_entropy(logits, labels)
        return rl + self.beta*kl + ce*self.beta



def train(self, trainset, valset, testset, lr=.01, num_epochs=10):
    this_dict = {}
    tr_loss={}
    val_loss={}
    te_loss={}
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs+1):
        for x_tr, y_tr in DataLoader(trainset, batch_size=4000, shuffle=True, generator=torch.Generator(device='cuda')):
            x_hat_tr, logits_tr = model(x_tr)
            l = model.loss(x_tr, x_hat_tr, logits_tr, y_tr)
            tr_loss[epoch] = l
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            print(f"Epoch: {epoch} loss: {l.item():>7f}")
        with torch.no_grad():
            for x_val, y_val in DataLoader(valset, batch_size=500):
                x_hat_val, logits_val = model(x_val)
                l = model.loss(x_val, x_hat_val, logits_val, y_val)
                print(logits_val.shape)
                probs = torch.softmax(logits_val, dim=2)
                y_hat = torch.zeros_like(y_val)
                arg_max = torch.argmax(probs, dim=2).squeeze()
                y_hat[torch.arange(probs.shape[0]), 0, arg_max] = 1
                correct = torch.sum(torch.all(y_val == y_hat, 2))
                accuracy = correct/500 # we only have 1 batch
                #print(accuracy)
                val_loss[epoch] = l
                print(f"Epoch: {epoch} Val loss: {l.item():>7f}")

            for x_test, y_test in DataLoader(testset, batch_size=500):
                x_hat_test, logits_test = model(x_test)
                l = model.loss(x_test, x_hat_test, logits_test, y_test)
                te_loss[epoch] = l
                print(f"Epoch: {epoch} Test loss: {l.item():>7f}")

        
        #if epoch % 25 == 0:
        #    this_dict[epoch] = {"x_tr": x_tr, "x_hat_tr": x_hat_tr, 
        #                    "y_tr": y_tr, "logits_tr": logits_tr,
        #                    "x_val": x_val, "x_hat_val": x_hat_val,
        #                    "y_val": y_val, "logits_val": logits_val,
        #                    "x_test": x_test, "x_hat_test": x_hat_test,
        #                    "y_test": y_test, "logits_test": logits_test,
        #                    }
        print("="*80)
   # dict_str = f"beta_{model.beta:3.2f}_latentdim_{model.latent_dim}.pkl"
   # with open(dict_str, "wb") as f:
   #     pickle.dump(this_dict, f)
    dict_str = f"beta_{model.beta:3.2f}_latentdim_{model.latent_dim}_loss.pkl"
    losses = {}
    losses['tr_loss'] = tr_loss
    losses['val_loss'] = val_loss
    losses['te_loss'] = te_loss
    with open(dict_str, "wb") as f:
        pickle.dump(losses, f)
    
    


def get_data(train, test, val):
    train_df=pd.read_csv("train.csv", header=0, names=["labels", "text"])
    test_df=pd.read_csv("test.csv", header=0, names=["labels", "text"])
    val_df=pd.read_csv("val.csv", header=0, names=["labels", "text"])
    return train_df, test_df, val_df

def get_tfidf(df, length):
    tfid_mat = TfidfVectorizer(stop_words="english", token_pattern=r"\b[A-z]{3,}\b", max_features=length)
    return tfid_mat.fit_transform(df['text']).toarray()

def get_bow(df, length):
    vectorizer = CountVectorizer(stop_words="english", token_pattern=r"\b[A-z]{3,}\b", max_features=length)
    return vectorizer.fit_transform(df['text']).toarray()

def get_embedding(df, model):
    def doc_vector(words, model):
        wv = [model[word] for word in words if word in model]
        if len(wv) == 0:
            print('hey')
            return np.zeros(model.vector_size)
        return np.mean(wv, axis=0)
    df['tokens'] = df['text'].apply(lambda t: t.split())
    df['vector'] = df['tokens'].apply(lambda tok: doc_vector(tok, model))



if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    parser = argparse.ArgumentParser("Beta VAE Final Project")
    parser.add_argument("train", help="filename of training dataset")
    parser.add_argument("test", help="filename of test dataset")
    parser.add_argument("val", help="filename of eval dataset")
    parser.add_argument("--word2vec_model", help="filename of word2vec bin")

    args = parser.parse_args()
    train_df, test_df, val_df = get_data(args.train, args.test, args.val)

    # load word2vec
    model_filename = args.word2vec_model


    if model_filename:
        w2v_model = KeyedVectors.load_word2vec_format(model_filename, binary=True)
        get_embedding(train_df, w2v_model)
        get_embedding(val_df, w2v_model)
        get_embedding(test_df, w2v_model)

    # get tfidf representation
    #length = 6400
    #train_arr = get_tfidf(train_df, length)
    #test_arr = get_tfidf(test_df, length)
    #val_arr = get_tfidf(val_df, length)

    ## get bow 
    length = 6400
    train_arr = get_bow(train_df, length)
    test_arr = get_bow(test_df, length)
    val_arr = get_bow(val_df, length)
    
    # BetaVAE
    # create datasets
    trainset = CustomDataset(train_arr, train_df['labels'])
    valset = CustomDataset(val_arr, val_df['labels'])
    testset = CustomDataset(test_arr, test_df['labels'])

    
    # BetaVAE DOC
    # create datasets
    #trainset = CustomDataset(train_df['vector'], train_df['labels'])
    #valset = CustomDataset(val_df['vector'], val_df['labels'])
    #testset = CustomDataset(test_df['vector'], test_df['labels'])

    num_classes = test_df['labels'].nunique()
    
    # set up experiment loop
    betas = [1, 2.5, 10, 100]
    latent_dims = [10, 20, 30]
    for beta in betas:
        for latent_dim in latent_dims:
            model = BetaVAE(beta=beta, latent_dim=latent_dim, num_classes=num_classes)
            #model = BetaVAEDoc(beta=beta, latent_dim=latent_dim, num_classes=num_classes)
            train(model, trainset, valset, testset, lr=.01, num_epochs=100)

