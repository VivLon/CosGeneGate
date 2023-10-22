import pandas as pd
import scanpy as sc
import numpy as np
import sklearn
import scipy.stats
import matplotlib.pyplot as plt
import torch
import pytorch_lightning
import random
import itertools
from stg.stg import STG
from scGeneFit.functions import *
from CosGeneGate import *
from NSForest_default import *
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scib_metrics import nmi_ari_cluster_labels_leiden, silhouette_label
import anndata
import time
import argparse
import pickle
import os
import gc

def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(name):
    with open(name, 'rb') as f:
        obj = pickle.load(f)
    return obj

def change_sparsertiy(adata, keylist):
    for item in keylist:
        if item in list(adata.layers.keys()):
            if scipy.sparse.issparse(adata.layers[item]):
                adata.layers[item] = adata.layers[item].todense()
            else:
                continue
            adata.layers[item] = np.asarray(adata.layers[item])
        else:
            continue
    return adata

def parse_args():
    parser = argparse.ArgumentParser(description='settings of benchmark')

    parser.add_argument('--seed', default=0, type=int, help='random seed')
    return parser.parse_args()

def main(args):
    pytorch_lightning.seed_everything(args.seed)
    #read&process dataset
    adatas = {}
    names = ['3k','68k','ss3','mpal','covid']
    for name in names:
        adatas[name+'_train'] = anndata.read_h5ad("/gpfs/gibbs/pi/zhao/wl545/pbmc/datasets/"+name+"_train2.h5ad")
        adatas[name+'_test'] = anndata.read_h5ad("/gpfs/gibbs/pi/zhao/wl545/pbmc/datasets/"+name+"_test.h5ad")
        adatas[name+'_test'].obs['batch'] = [name]*(adatas[name+'_test'].shape[0])
        change_sparsertiy(adatas[name+'_train'], ['logcounts'])
        change_sparsertiy(adatas[name+'_test'], ['logcounts'])
    
    #marker selection
    final_markers = {}
    for name in names:
        train_name = name+'_train'
        test_name = name+'_test'
        adata_train = adatas[train_name].copy()
        adata_test = adatas[test_name].copy()
        final_markers[name] = {}
        ##CosGeneGate
        ##tune stg_genes=150,200,300
        for stg_genes in [150,200,300]:
            final_markers[name][stg_genes] = {}
            final_markers[name][stg_genes][50] = CosGeneGate(adata_train, stg_genes=stg_genes, final_genes=50, lbm=0.01,
                                              layer_key='logcounts', cluster_header='label', random_state=args.seed)
            for num in range(5,50,5):
                final_markers[name][stg_genes][num] = {}
                for cl in set(adata_train.obs.label):
                    final_markers[name][stg_genes][num][cl] = final_markers[name][stg_genes][50][cl][:num]
        
            
    #save results
    save_obj(final_markers, '/gpfs/gibbs/pi/zhao/wl545/results/pbmc/tune_num/sub/cgg_markers2_'+str(args.seed)+'.pkl')

if __name__=="__main__":
    args = parse_args()
    main(args)





