import os
import shutil
import subprocess
import csv
import bert_score
import sentence_transformers
import numpy as np
from scipy.spatial.distance import cosine

# TextDiversity pkgs
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from scipy.spatial import distance
import torch
import numpy as np
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import os
import itertools
from multiprocessing import Pool
import spacy

# locals
import metric
from utils import *

class TokenSemanticDiversity(metric.TextDiversity):

    default_config = {
        # TextDiversity configs
        'q': 1,
        'normalize': False,
        'distance_fn': distance.chebyshev, 
        'dim_reducer': PCA,
        'remove_stopwords': False, 
        'scale_dist': "exp", 
        'sq_reg': False, 
        'mean_adj': True,
        'verbose': False,
        # TokenSemanticDiversity configs
        'MODEL_NAME':"bert-large-uncased",
        'batch_size': 16,
        'use_gpu': False,
        'n_components': 'auto' 
    }

    def __init__(self, config={}):
        config = {**self.default_config, **config} 
        super().__init__(config)
        self.model = AutoModel.from_pretrained(config['MODEL_NAME'])
        self.tokenizer = AutoTokenizer.from_pretrained(config['MODEL_NAME'])
        self.undesirable_tokens = [
            self.tokenizer.pad_token_id, 
            self.tokenizer.cls_token_id, 
            self.tokenizer.sep_token_id
        ]
        self.batch_size = config['batch_size']
        self.device = torch.device('cuda' if config['use_gpu'] and torch.cuda.is_available() else 'cpu')
        self.verbose = config['verbose']

        # move model to device
        if isinstance(self.model, torch.nn.Module):
            self.model.to(self.device)

    def encode(self, input_ids, attention_mask):
        self.model.eval()
        with torch.no_grad():
            out = self.model(input_ids, attention_mask=attention_mask)
        emb = out[0]
        return emb

    def get_embeddings(self, corpus):
        inputs = self.tokenizer(corpus, return_tensors='pt', padding=True, truncation=True)
        batches = zip(chunker(inputs.input_ids, self.batch_size), 
                      chunker(inputs.attention_mask, self.batch_size))
        if self.verbose:
            print('getting token embeddings...')
            batches = tqdm(batches, total=int(len(inputs.input_ids)/self.batch_size))

        outputs = []
        for input_ids, attention_mask in batches:
            emb = self.encode(input_ids.to(self.device), 
                       attention_mask.to(self.device))
            outputs.append(emb)
        embeddings = torch.cat(outputs)

        # remove undesirable tokens
        idx = np.isin(inputs['input_ids'],  self.undesirable_tokens, assume_unique=True, invert=True).reshape(-1)
        tok = np.array(self.tokenizer.convert_ids_to_tokens(inputs.input_ids.view(-1)))[idx]
        boe = embeddings.view(-1, embeddings.shape[-1])[idx].detach().cpu()

        # remove stopwords
        if self.config['remove_stopwords']:
            idx = np.isin(tok, stopwords.words('english'), invert=True)
            tok = tok[idx]
            boe = boe[idx]

        # compress embedding to speed up similarity matrix computation
        if self.config['n_components'] == "auto":
            n_components = min(max(2, len(boe) // 10), boe.shape[-1])
            if self.verbose:
                print('Using n_components={}'.format(str(n_components)))

        if type(n_components) == int and n_components > 0 and len(boe) > 1:
            boe = self.config['dim_reducer'](n_components=n_components).fit_transform(boe)

        if len(np.flatnonzero(np.core.defchararray.find(tok,'##')!=-1)) > 0:
            tok, boe = merge_bpe(tok, boe)

        return boe, tok

    def __call__(self, response_set): 
        return super().__call__(response_set)


class SentenceSemanticDiversity(metric.TextDiversity):

    default_config = {
        # TextDiversity configs
        'q': 1,
        'normalize': False,
        'distance_fn': distance.chebyshev, 
        'dim_reducer': PCA,
        'remove_stopwords': False, 
        'scale_dist': "exp", 
        'sq_reg': False, 
        'mean_adj': True,
        'verbose': False,
        # SentenceSemanticDiversity configs
        'MODEL_NAME':"stsb-roberta-large",
        'use_gpu': False,
        'n_components': 'auto' 
    }

    def __init__(self, config={}):
        config = {**self.default_config, **config} 
        super().__init__(config)
        self.device = torch.device('cuda' if config['use_gpu'] and torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(config['MODEL_NAME'], device=self.device)
        self.verbose = config['verbose']

    def get_embeddings(self, corpus):

        boe = np.stack(self.model.encode(corpus))
        
        # compress embedding to speed up similarity matrix computation
        if self.config['n_components'] == "auto":
            n_components = min(max(2, len(boe) // 10), boe.shape[-1])
            if self.verbose:
                print('Using n_components={}'.format(str(n_components)))

        if type(n_components) == int and n_components > 0 and len(boe) > 1:
            boe = self.config['dim_reducer'](n_components=n_components).fit_transform(boe)

        return boe, corpus

    def __call__(self, response_set): 
        return super().__call__(response_set)

class SyntacticDiversity(metric.TextDiversity):

    default_config = {
        # TextDiversity configs
        'q': 1,
        'normalize': False,
        'dim_reducer': PCA,
        'remove_stopwords': False, 
        'sq_reg': False, 
        'mean_adj': False,
        'verbose': False,
        # SentenceSemanticDiversity configs
        'MODEL_NAME': "en_core_web_trf",
        'distance_fn': distance.hamming, 
        'scale_dist': "invert", 
        'part': 'pos_', 
        'part2int': True
    }

    def __init__(self, config={}):
        config = {**self.default_config, **config} 
        super().__init__(config)
        self.model = spacy.load(config['MODEL_NAME'])
        self.verbose = config['verbose']


    def get_embeddings(self, corpus):

        # convert to spacy docs to get parts
        doc_parts = []
        for doc in corpus:
            for sent in sent_tokenize(doc):
                sent_ = []
                for w in self.model(sent):
                    if self.config['remove_stopwords'] and w.text in stopwords.words('english'):
                        continue
                    part_ = getattr(w, self.config['part'])
                    sent_.append(part_)
                doc_parts.append(sent_)

        species = doc_parts

        # pad to max sentence doc length
        pad_to = find_max_list(doc_parts)
        doc_parts = np.array([s + ['NULL']*(pad_to-len(s)) for s in doc_parts])

        # convert doc parts to int
        if self.config['part2int']:
            # build dict of unique doc parts
            part_map = set(itertools.chain(*doc_parts))
            part_map = {tag: i for i, tag in enumerate(part_map)}
            # convert to int for distance comparison
            part2int_fn = np.vectorize(part_map.get)
            doc_parts = part2int_fn(doc_parts)

        return doc_parts, species

    def __call__(self, response_set): 
        return super().__call__(response_set)

if __name__ == '__main__':

    def print_metric(metric, resp_set):
        print('{0}: {1:0.3f}'.format(type(metric).__name__, metric(resp_set)))

    # TEST
    response_set = ['i am going', 'i am going', 'lets go i i']

    config = {'normalize': False}
    print_metric(TokenSemanticDiversity(config), response_set)
    print_metric(SentenceSemanticDiversity(config), response_set)
    print_metric(SyntacticDiversity(config), response_set)

    config = {'normalize': True}
    print_metric(TokenSemanticDiversity(config), response_set)
    print_metric(SentenceSemanticDiversity(config), response_set)
    print_metric(SyntacticDiversity(config), response_set)