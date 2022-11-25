import pickle
import os
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
#from .DGI import DGI
from .train_funcs import load_datasets, X_A_hat, load_state
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import networkx as nx




from nlptoolkit.clustering.models.DGI.trainer import train_and_fit


tqdm.pandas(desc="prog-bar")
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def load_pickle(filename):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data


class class_query_embedding(object):
    def __init__(self, args=None):
        if args is None:
            self.args = load_pickle("args.pkl")
        else:
            self.args = args
        
        args.train_data = "./data/query.csv"
        self.cuda = torch.cuda.is_available()
        #self.embeddings = embeddings

        logger.info("Loading tokenizer and model...")    
        self.G, _ = load_datasets(self.args)
        self.X_A = X_A_hat(self.G)
        print(nx.info(self.G))
        X, A_hat = self.X_A.get_X_A_hat(corrupt=False)
        #self.net = DGI(X.shape[1], self.args, bias=True,\
        #               n_nodes=X.shape[0], A_hat=A_hat, cuda=self.cuda)
        #_, _ = load_state(self.net, None, None, model_no=self.args.model_no, load_best=False)
        
        if self.cuda:
            self.net.cuda()
        
        #self.net.eval()
        logger.info("Done!")
    
    def infer_embeddings_qyery(self, G):
        '''
        gets nodes embeddings from trained model
        '''
        self.net.eval()
        
        X, A_hat = self.X_A.get_X_A_hat(corrupt=False)
        if self.cuda:
            X, A_hat = X.cuda(), A_hat.cuda()
        
        with torch.no_grad():
            self.embeddings = self.net.encoder(X, A_hat)
        
        self.embeddings = self.embeddings.cpu().detach().numpy() if self.cuda else\
                            self.embeddings.detach().numpy()
        
        self.embeddings = self.embeddings[self.document_nodes] # only interested in documents
        return self.embeddings
    