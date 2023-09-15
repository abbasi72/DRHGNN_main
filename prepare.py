from model import *
import torch, numpy as np, scipy.sparse as sp
import torch.optim as optim, torch.nn.functional as F
import math
import tensorflow as tf
import pickle
import os
from numpy import dot
from numpy.linalg import norm

import random
# generate random integer values
from random import seed
from random import sample

from sklearn import metrics

from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score, precision_recall_curve

drug_num=269
disease_num=598

from math import*
from decimal import Decimal
import heapq 
from operator import itemgetter

def nth_root(value, n_root):
 root_value = 1/float(n_root)
 return round (Decimal(value) ** Decimal(root_value),3)
  
def minkowski_distance(x,y,p_value):
 return nth_root(sum(pow(abs(a-b),p_value) for a,b in zip(x, y)),p_value)

def euclideanDistance(x, y):
    dist = np.linalg.norm(x - y)
    return dist

def accuracy_klarge(Z,Y):
    
   
    Z=Z.detach().numpy()
    Y=Y.detach().numpy()
    
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    #
    i =0
    j  =0
    
    Z_real=Z
    for i in range(len(Z)):
        # N is the k largest value 
        N=70
        res=[]
        res = sorted(range(len(Z[i])), key = lambda sub: Z[i][sub])[-N:]
        
        for j in range (len(Z[0])):
            
            if j in res:
                Z[i][j]=1
            else :
                Z[i][j]=0
            
            if (Y[i][j] == 1) and (Z[i][j] == 1):
                TP += 1
            if (Y[i][j] == 1) and (Z[i][j] == 0):
                FN += 1
            if (Y[i][j] == 0) and (Z[i][j] == 1):
                FP += 1
            if (Y[i][j] == 0) and (Z[i][j] == 0):
                TN += 1
    
    if (TP + TN + FP + FN )==0 :
      acc = 0 
    else :
      acc = (TP + TN)/(TP + TN + FP + FN)
    if (TP+FP)==0 : 
      precision= 0 
    else :
      precision = TP / (TP + FP)
    if (TP+FN)==0:
      recall=0
    else :
      recall = TP / (TP + FN)
    if (2*TP + FP + FN )==0 :
      F1=0
    else :
      F1 = 2*TP /(2*TP + FP + FN )
    
    # Data to plot precision - recall curve
    pre, re, th = precision_recall_curve(Y.reshape(len(Z)*drug_num,1), Z.reshape(len(Z)*drug_num,1))
    # Use AUC function to calculate the area under the curve of precision recall curve
    auc_precision_recall = metrics.auc(re, pre)
    print('test_AUPR : ', auc_precision_recall)
    auc=roc_auc_score(Y.reshape(len(Z)*drug_num,1), Z_real.reshape(len(Z)*drug_num,1))
    print ('test AUC : ',auc)
    print ("TP : ", TP , "FP : ", FP , "TN : ", TN, "FN :",FN)
    return acc , precision , recall , F1


def accuracy_3(Z, Y):

    
    Z=Z.detach().numpy()
    Y=Y.detach().numpy()
    
    arr_zero=np.where(Y==0)
    arr_one=np.where(Y==1)
    
    ar_zero = np.zeros((len(arr_zero[0]), 2), dtype=float)
    ar_one = np.zeros((len(arr_one[0]), 2), dtype=float)
    for i in range (len(arr_zero[0])):
        ar_zero[i,0] = arr_zero[0][i]
        ar_zero[i,1] = arr_zero[1][i]
    for i in range (len(arr_one[0])):
        ar_one[i,0] = arr_one[0][i]
        ar_one[i,1] = arr_one[1][i]
    
    data_balance_0 = np.asarray(random.sample(list(ar_zero),1*len(ar_one)))
    data_balance_1 = np.asarray(list(ar_one))
    test_i= [*data_balance_0[:,0],*data_balance_1[:,0]]#tar_balance_1  + tar_balance_0  #.remove(subset)
    test_j= [*data_balance_0[:,1],*data_balance_1[:,1]]#tar_balance_1  + tar_balance_0  #.remove(subset)
    test_i=np.asarray(test_i)
    test_j=np.asarray(test_j)
    

    TP = 0
    TN = 0
    FN = 0
    FP = 0
    
    i =0
    j  =0
    test_i = test_i.astype(int)
    test_j = test_j.astype(int)
    Y_test , Z_test , Z_real=[] , [] ,[]
    
    Z_real_all=Z
    
    
    for k in range(len(test_i)):
        i=int(test_i[k])
        j=int(test_j[k])
        Z_real.append(Z_real_all[i][j])
        Y_test.append(Y[i][j])
        
        """
        j_val = heapq.nlargest(40, enumerate(Z[i]), key=itemgetter(1))
        j_id=[j for (j, val) in j_val]
        Z[i][:]=0
        for k in j_id :
          Z[i][k]=1
        
        """
        if Z[i][j]>=0.845:
          Z[i][j]=1
        else :
          Z[i][j]=0
        
        Z_test.append(Z[i][j])
        if (Y[i][j] == 1) and (Z[i][j] == 1):
            TP += 1
        if (Y[i][j] == 1) and (Z[i][j] == 0):
            FN += 1
        if (Y[i][j] == 0) and (Z[i][j] == 1):
            FP += 1
        if (Y[i][j] == 0) and (Z[i][j] == 0):
            TN += 1
    
    
    
    
    if (TP + TN + FP + FN )==0 :
      acc = 0 
    else :
      acc = (TP + TN)/(TP + TN + FP + FN)
    if (TP+FP)==0 : 
      precision= 0 
    else :
      precision = TP / (TP + FP)
    if (TP+FN)==0:
      recall=0
    else :
      recall = TP / (TP + FN)
    if (2*TP + FP + FN )==0 :
      F1=0
    else :
      F1 = 2*TP /(2*TP + FP + FN )
    
    # Data to plot precision - recall curve
    pre, re, th = precision_recall_curve(Y_test, Z_test)
    # Use AUC function to calculate the area under the curve of precision recall curve
    auc_precision_recall = metrics.auc(re, pre)
    print('test_AUPR : ', auc_precision_recall)
    auc=roc_auc_score(Y_test, Z_real)
    print ('test AUC : ',auc)
    print ("TP : ", TP , "FP : ", FP , "TN : ", TN, "FN :",FN)
    return acc , precision , recall , F1 , auc_precision_recall



def accuracy_2(Z , Y ):

    i=0
    Z=Z.detach().numpy()
    Y=Y.detach().numpy()
    
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    t = 0
    f=0
    
    for i in range(len(Z)):
      
        
      for j in range(len(Z[0])):
        if Z[i][j]>=0.845:
          Z[i][j]=1
        else :
          Z[i][j]=0
        
        
        if (Y[i][j] == 1) and (Z[i][j] == 1):
            TP += 1
        if (Y[i][j] == 1) and (Z[i][j] == 0):
            FN += 1
        if (Y[i][j] == 0) and (Z[i][j] == 1):
            FP += 1
        if (Y[i][j] == 0) and (Z[i][j] == 0):
            TN += 1
    
    if (TP + TN + FP + FN )==0 :
      acc = 0 
    else :
      acc = (TP + TN)/(TP + TN + FP + FN)
    if (TP+FP)==0 : 
      precision= 0 
    else :
      precision = TP / (TP + FP)
    if (TP+FN)==0:
      recall=0
    else :
      recall = TP / (TP + FN)
    if (2*TP + FP + FN )==0 :
      F1=0
    else :
      F1 = 2*TP /(2*TP + FP + FN )
    
    
    print ("TP : ", TP , "FP : ", FP , "TN : ", TN, "FN :",FN)
    return acc , precision , recall , F1




import torch_sparse

def fetch_data(args):
    from data import data 
    dataset, _, _ = data.load(args)
    args.dataset_dict = dataset 

    X, Y, G = dataset['features'], dataset['labels'], dataset['hypergraph']
    X, G = dataset['features'], dataset['hypergraph']
   
    # node features in sparse representation
    X = sp.csr_matrix(normalise(np.array(X)), dtype=np.float32)#normalise(np.array(X))
    X = torch.FloatTensor(np.array(X.todense()))
    
    # labels
    #Y = np.array(Y)
   # Y = torch.LongTensor(np.where(Y)[1])
    Y = torch.LongTensor(Y)


    #X, Y = X.cuda(), Y.cuda()

    return X, Y, G

def initialise(X, Y, G, args, unseen=None):

    """
    initialises model, optimiser, normalises graph, and features
    
    arguments:
    X, Y, G: the entire dataset (with graph, features, labels)
    args: arguments
    unseen: if not None, remove these nodes from hypergraphs

    returns:
    a tuple with model details (UniGNN, optimiser)    
    """
    
    
    G = G.copy()
    
    if unseen is not None:
        unseen = set(unseen)
        # remove unseen nodes
        for e, vs in G.items():
            G[e] =  list(set(vs) - unseen)
    print("a")
    args.add_self_loop=True
    if args.add_self_loop:
        Vs = set(range(X.shape[0]))

        # only add self-loop to those are orginally un-self-looped
        # TODO:maybe we should remove some repeated self-loops?
        for edge, nodes in G.items():
            if len(nodes) == 1 and nodes[0] in Vs:
                Vs.remove(nodes[0])

        for v in Vs:
            G[f'self-loop-{v}'] = [v]

    N, M = X.shape[0], len(G)
    indptr, indices, data = [0], [], []
    for e, vs in G.items():
        indices += vs 
        data += [1] * len(vs)
        indptr.append(len(indices))
    print("a")

    H = sp.csc_matrix((data, indices, indptr), shape=(N, M), dtype=int).tocsr() # V x E
    print("a")

    degV = torch.from_numpy(H.sum(1)).view(-1, 1).float()
    degE2 = torch.from_numpy(H.sum(0)).view(-1, 1).float()


    (row, col), value = torch_sparse.from_scipy(H)
    V, E = row, col
    from torch_scatter import scatter
    assert args.first_aggregate in ('mean', 'sum'), 'use `mean` or `sum` for first-stage aggregation'
    degE = scatter(degV[V], E, dim=0, reduce=args.first_aggregate)
    degE = degE.pow(-0.5)
    degV = degV.pow(-0.5)
    degV[degV.isinf()] = 1 # when not added self-loop, some nodes might not be connected with any edge
    

    #V, E = V.cuda(), E.cuda()
    args.degV = degV#.cuda()
    args.degE = degE#.cuda()
    args.degE2 = degE2.pow(-1.)#.cuda()


    


    nfeat, nclass = X.shape[1], len(Y.unique())
    print ("nclass ; ", nclass)
    
    nlayer = args.nlayer
    nhid = args.nhid
    nhead = args.nhead

    # UniGNN and optimiser
    if args.model_name == 'UniGCNII':
        model = UniGCNII(args, nfeat, nhid, nclass, nlayer, nhead, V, E)
        optimiser = torch.optim.Adam([
            dict(params=model.reg_params, weight_decay=0.001),
            dict(params=model.non_reg_params, weight_decay=5e-4)
        ], lr=0.01)
    elif args.model_name == 'HyperGCN':
        args.fast = True
        dataset = args.dataset_dict
        model = HyperGCN(args, nfeat, nhid, nclass, nlayer, dataset['n'], dataset['hypergraph'], dataset['features'])
        optimiser = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    else:
        print ("3")
        #nfeat=200
        #nclass = drug_num + disease_num
        model = UniGNN(args, nfeat, nhid, nclass, nlayer, nhead, V, E)
        optimiser = optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)


    model#.cuda()
   
    return model, optimiser



def normalise(M):
    """
    row-normalise sparse matrix

    arguments:
    M: scipy sparse matrix

    returns:
    D^{-1} M  
    where D is the diagonal node-degree matrix 
    """
    
    d = np.array(M.sum(1))
    print (type(d))
    di = np.power(d, -1).flatten()
    di[np.isinf(di)] = 0.
    DI = sp.diags(di)    # D inverse i.e. D^{-1}
    
    return DI.dot(M)