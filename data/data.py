import os, inspect, random, pickle
import numpy as np, scipy.sparse as sp
from tqdm import tqdm
import pickle
import pandas as pd 
import scipy
import itertools 

def load(args):
    """
    parses the dataset
    """
    dataset = parser(args.data, args.dataset).parse()

    current = os.path.abspath(inspect.getfile(inspect.currentframe()))
    Dir, _ = os.path.split(current)
    file = os.path.join(Dir, args.data, args.dataset, "splits", str(args.split) + ".pickle")

    if not os.path.isfile(file): print("split + ", str(args.split), "does not exist")
    with open(file, 'rb') as H: 
        Splits = pickle.load(H)
        train, test = Splits['train'], Splits['test']

    return dataset, train, test



class parser(object):
    """
    an object for parsing data
    """
    
    def __init__(self, data, dataset):
        """
        initialises the data directory 

        arguments:
        data: coauthorship/cocitation
        dataset: cora/dblp/acm for coauthorship and cora/citeseer/pubmed for cocitation
        """
        
        current = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        self.d = os.path.join(current, data, dataset)
        self.data, self.dataset = data, dataset

    

    def parse(self):
        """
        returns a dataset specific function to parse
        """
        
        name = "_load_data"
        function = getattr(self, name, lambda: {})
        return function()



    def _load_data(self):
        """
        loads the coauthorship hypergraph, features, and labels of cora

        assumes the following files to be present in the dataset directory:
        hypergraph.pickle: coauthorship hypergraph
        features.pickle: bag of word features
        labels.pickle: labels of papers

        n: number of hypernodes
        returns: a dictionary with hypergraph, features, and labels as keys
        """
        
        with open(os.path.join(self.d, 'hypergraph.pickle'), 'rb') as handle:
            hypergraph = pickle.load(handle)
            print("number of hyperedges is", len(hypergraph))
        
        """
        df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/GCN_based models/UniGNN/data/Drug_Disease/dis_dr_di_1/dis_sim.csv" , header=None)
        feature_p = df.to_numpy()
        feature_p=feature_p[0:,0:]
        """
        """
        df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/GCN_based models/UniGNN/data/Drug_Disease/drug_dr_di_1/structure_feature_matrix.csv" , header=None)
        feature_s = df.to_numpy()
        feature_s=feature_s[1:,1:]

        df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/GCN_based models/UniGNN/data/Drug_Disease/drug_dr_di_1/target_feature_matrix.csv" , header=None)
        feature_t = df.to_numpy()
        feature_t=feature_t[1:,1:]

        df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/GCN_based models/UniGNN/data/Drug_Disease/drug_dr_di_1/enzyme_feature_matrix.csv" , header=None)
        feature_e = df.to_numpy()
        feature_e=feature_e[1:,1:]

        features = np.concatenate((feature_p,feature_s,feature_t),axis=1)#,feature_t))
        #features=[feature_p,feature_s,feature_t]
        print (len(features[0]))
        """
        
        
        with open(os.path.join(self.d, 'feature_dis_sci.pickle'), 'rb') as handle:
            features = pickle.load(handle)#.todense())
        print (" len len of fea :", len(features))
        
        
        """
        with open(os.path.join(self.d, 'feature_dis_fast.pickle'), 'rb') as handle:
            features = pickle.load(handle)#.todense())
        #features=np.concatenate ([features,features_d])
        """
        
        df = pd.DataFrame(features)
        #scipy.sparse.csr_matrix(features.values) 
        #print (" len len of fea :", len(features))
       # features= [feature_p]
        
        """
        df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/GCN_based models/UniGNN/data_Drug_disease/drug_dis+867vector with dr_dr str similarity.csv" , header=None)
        labels = df.to_numpy()
        """
        with open(os.path.join(self.d, 'labels.pickle'), 'rb') as handle:
            labels = pickle.load(handle)#self._1hot(pickle.load(handle))
        
        return {'hypergraph': hypergraph, 'features': features, 'labels': labels, 'n': features.shape[0]}



    def _1hot(self, labels):
        """
        converts each positive integer (representing a unique class) into ints one-hot form

        Arguments:
        labels: a list of positive integers with eah integer representing a unique label
        """
        
        classes = set(labels)
        onehot = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        return np.array(list(map(onehot.get, labels)), dtype=np.int32)