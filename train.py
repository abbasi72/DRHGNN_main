from operator import index
#from tensorflow.python.ops.gen_math_ops import euclidean_norm
import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import optimizer
import os
import sklearn
from sklearn import metrics
from sklearn.metrics import euclidean_distances, roc_auc_score
#from sklearn.metrics import auc
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
import statistics
import keras
import numpy as np
import time
import datetime
import path
import shutil
import math 

import config
import csv
from scipy.spatial.distance import hamming
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import auc#, plot_precision_recall_curve
import matplotlib.pyplot as plt


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import heapq 
from operator import itemgetter

args = config.parse()

def save_result(name,result):
    with open(name + '.csv', "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in result:
            writer.writerow(i)
    return 0

# gpu, seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"        
#os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
#os.environ['PYTHONHASHSEED'] = str(args.seed)


use_norm = 'use-norm' if args.use_norm else 'no-norm'
add_self_loop = 'add-self-loop' if args.add_self_loop else 'no-self-loop'


#### configure output directory

dataname = f'{args.data}_{args.dataset}'
model_name = args.model_name
nlayer = args.nlayer
dirname = f'{datetime.datetime.now()}'.replace(' ', '_').replace(':', '.')
out_dir = path.Path( f'./{args.out_dir}/{model_name}_{nlayer}_{dataname}/seed_{args.seed}' )


if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.makedirs_p()

 

### configure logger 
from logger import get_logger

baselogger = get_logger('base logger', f'{out_dir}/logging.log', not args.nostdout)
resultlogger = get_logger('result logger', f'{out_dir}/result.log', not args.nostdout)
baselogger.info(args)



# load data
from data import data
from prepare import * 


def multiclass_precision_recall_curve(y_true, y_score):
    y_true = y_true.ravel()
    y_score = y_score.ravel()
    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))
    if y_score.ndim == 1:
        y_score = y_score.reshape((-1, 1))
    y_true_c = y_true.take([0], axis=1).ravel()
    y_score_c = y_score.take([0], axis=1).ravel()
    precision, recall, pr_thresholds = precision_recall_curve(y_true_c, y_score_c)
    return (precision, recall, pr_thresholds)


def normalize(Z):
    

        minn= torch.min(Z)
        maxx=torch.max(Z)
        
        Z= (Z-minn)/(maxx-minn)
        return Z


test_accs = []
best_val_accs, best_test_accs = [], []

resultlogger.info(args)

# load data
X, Y, G = fetch_data(args)

for run in range(1, args.n_runs+1):
    run_dir = out_dir / f'{run}'
    run_dir.makedirs_p()

    # load data
    args.split = run
    _, train_idx, test_idx = data.load(args)
    
    
    
    train_idx = torch.LongTensor(train_idx)#index_train[:,0])#train_idx)#.cuda()
    test_idx  = torch.LongTensor(test_idx)#index_test[:,0])#test_idx )#.cuda()
  

    Xseen = X
    Xseen[test_idx]=0

    # model 
    model, optimizer = initialise(Xseen, Y, G, args)


    baselogger.info(f'Run {run}/{args.n_runs}, Total Epochs: {args.epochs}')
    baselogger.info(model)
    baselogger.info( f'total_params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}'  )

    tic_run = time.time()


    from collections import Counter
    
    label_rate = len(train_idx) / X.shape[0]
    baselogger.info(f'label rate: {label_rate}')

    best_test_aupr, t_AUPR, Z = 0, 0, None 
    Z_avg=[]
    Z_avg_eval=[]  
    Z_test=[]
    train_acc=0
    ep=0
    for epoch in range(args.epochs):
        # train
        tic_epoch = time.time()
        model.train()
        optimizer.zero_grad()
        
        Z = model(Xseen)
        
        
        Z = normalize(Z)
        

        Z_avg.append ((torch.sum(Z)/(drug_num*(disease_num-60))).detach().numpy())
        

        
        
        logr = LogisticRegression(solver='lbfgs')
        logr.fit(Z.detach().numpy().reshape(drug_num*disease_num,1), Y.detach().numpy().reshape(drug_num*disease_num,1))

        y_pred = logr.predict_proba(Z.detach().numpy().reshape(drug_num*disease_num,1))[:, 1].ravel()
        loss = log_loss(Y.detach().numpy().reshape(drug_num*disease_num,1), y_pred)
        loss =torch.tensor(loss, requires_grad=True)
        
        
        loss.backward(gradient=torch.ones_like(loss))
        
        optimizer.step()

        train_time = time.time() - tic_epoch 
        
        
        # eval
        model.eval()
        
        Z = model(X)
        
        
        Z = normalize(Z)
        

        Z_avg_eval. append((torch.sum(Z)/(drug_num*disease_num)).detach().numpy())
        
        i=0
        auc=roc_auc_score( Y[test_idx].detach().numpy() , Z[test_idx].detach().numpy() , average='micro')
        
        Z_test.append ((torch.sum(Z[test_idx])/(drug_num*60)).detach().numpy())
        train_acc , precision , recall , F1= accuracy_2(Z[train_idx], Y[train_idx])
        
        test_acc , t_precision , t_recall , t_F1, t_AUPR = accuracy_3(Z[test_idx], Y[test_idx])#,test_idx)

       
        print('=====================  Precision , Recall , F1 ==============================')
        
        print ('precision : ',precision , 'test_precision : ', t_precision)
        print ( 'recal : ', recall, 'test_recall : ', t_recall)
        print ( ' F1 : ', F1 , 'test_F1 : ', t_F1)

        
        # log acc
        best_test_aupr = max(best_test_aupr, t_AUPR)
        if best_test_aupr == t_AUPR :
          ep=epoch
          
        print ("loss = = ", np.mean(loss.detach().numpy()))
        baselogger.info(f'epoch:{epoch}  | train acc:{train_acc:.2f} | test acc:{test_acc:.2f}  | AUC:{auc:.2f}   | time:{train_time*1000:.1f}ms')
    

    
    N=12
    j_val = heapq.nlargest(12, enumerate(Z[0]), key=itemgetter(1))
    #j_id=[j for (j, val) in j_val]
    print (" this is name of drugs suggested " ,  heapq.nlargest(12, enumerate(Z[0]), key=itemgetter(1)))#sorted(range(len(Z[440])), key = lambda sub: Z[0][sub])[-N:])
    print (" this is name of drugs suggested " ,  heapq.nlargest(12, enumerate(Z[7]), key=itemgetter(1)))#sorted(range(len(Z[441])), key = lambda sub: Z[7][sub])[-N:])
    print (" this is name of drugs suggested " ,  heapq.nlargest(12, enumerate(Z[400]), key=itemgetter(1)))#sorted(range(len(Z[442])), key = lambda sub: Z[442][sub])[-N:])
    
    
    
    
    print("Variance of sample set is",np.var(np.array(Z_avg)),np.var(np.array(Z_avg_eval)), np.var(np.array(Z_test)) )
    print ("Z_avg_full   train , eval  , test :::::", np.mean(np.array(Z_avg)), np.mean(np.array(Z_avg_eval)) , np.mean(np.array(Z_test)) )
    Z = Z.detach().numpy()
    save_result("drug_embedding",Z)    
    resultlogger.info(f"Run {run}/{args.n_runs}, best test accuracy: {best_test_acc:.2f}, acc(last): {test_acc:.2f}, total time: {time.time()-tic_run:.2f}s")
    t_AUPR.append(t_AUPR)
    best_test_aupr.append(best_test_aupr)


resultlogger.info(f"Average final test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}")
resultlogger.info(f"Average best test accuracy: {np.mean(best_test_accs)} ± {np.std(best_test_accs)}")