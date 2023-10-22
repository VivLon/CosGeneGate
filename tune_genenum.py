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
    parser.add_argument('--dataset', default='pbmc', help='input dataset')
    return parser.parse_args()

def main(args):
    pytorch_lightning.seed_everything(args.seed)
    #read&process dataset
    adatas = {}
    if args.dataset=='pbmc':
        names = ['3k','68k','ss3','mpal','covid']
        for name in names:
            adatas[name+'_train'] = anndata.read_h5ad("/gpfs/gibbs/pi/zhao/wl545/pbmc/datasets/"+name+"_train2.h5ad")
            adatas[name+'_test'] = anndata.read_h5ad("/gpfs/gibbs/pi/zhao/wl545/pbmc/datasets/"+name+"_test.h5ad")
            change_sparsertiy(adatas[name+'_train'], ['logcounts'])
            change_sparsertiy(adatas[name+'_test'], ['logcounts'])
    
    elif args.dataset=='pancreas':
        adata = anndata.read_loom("/gpfs/gibbs/pi/zhao/wl545/data/deconvdatasets/Pancrm_raw.loom")
        adata.obs['label'] = adata.obs['celltype']
        adata.layers['raw'] = adata.X
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata.layers['logcounts'] = adata.X
        sc.pp.highly_variable_genes(adata, layer='logcounts', batch_key='batch', n_top_genes=2000)
        adata = adata[:, adata.var.highly_variable]
        names = sorted(list(set(adata.obs.batch)))
        for name in names:
            adatas[name+'_train'] = adata[np.where(adata.obs.batch!=name)].copy()
            adatas[name+'_test'] = adata[np.where(adata.obs.batch==name)].copy()
            change_sparsertiy(adatas[name+'_train'], ['logcounts'])
            change_sparsertiy(adatas[name+'_test'], ['logcounts'])

    elif args.dataset=='heart':
        bdata = anndata.read_h5ad("/gpfs/gibbs/pi/zhao/wl545/data/deconvdatasets/sce_heart.h5ad")
        adata = sc.pp.subsample(bdata, fraction=0.1, random_state=args.seed, copy=True)
        adata.obs['label'] = adata.obs['cell_type']
        sc.pp.highly_variable_genes(adata, layer='logcounts', batch_key='batch', n_top_genes=2000)
        adata = adata[:, adata.var.highly_variable]
        names = sorted(list(set(adata.obs.batch)))
        for name in names:
            adatas[name+'_train'] = adata[np.where(adata.obs.batch!=name)].copy()
            adatas[name+'_test'] = adata[np.where(adata.obs.batch==name)].copy()
            change_sparsertiy(adatas[name+'_train'], ['logcounts'])
            change_sparsertiy(adatas[name+'_test'], ['logcounts'])
            
    elif args.dataset=='lung':
        bdata = anndata.read_h5ad('/gpfs/gibbs/pi/zhao/wl545/data/largedataset/lung/lung_pp.h5ad')
        adata = sc.pp.subsample(bdata, fraction=0.1, random_state=args.seed, copy=True)
        sc.pp.highly_variable_genes(adata, layer='logcounts', batch_key='batch', n_top_genes=2000)
        adata = adata[:, adata.var.highly_variable]
        names = sorted(list(set(adata.obs.batch)))
        for name in names:
            adatas[name+'_train'] = adata[np.where(adata.obs.batch!=name)].copy()
            adatas[name+'_test'] = adata[np.where(adata.obs.batch==name)].copy()
            change_sparsertiy(adatas[name+'_train'], ['logcounts'])
            change_sparsertiy(adatas[name+'_test'], ['logcounts'])
    else:
        print('The wrong dataset setting!')
    
    #marker selection
    final_markers = load_obj('/gpfs/gibbs/pi/zhao/wl545/results/'+args.dataset+'/tune_num/sub/markers'+str(args.seed)+'.pkl')
    methods = ['stg_prob', 'stg_num', 'nsf', 'cosg', 'scgf']
    for name in names:
        train_name = name+'_train'
        test_name = name+'_test'
        adata_train = adatas[train_name].copy()
        adata_test = adatas[test_name].copy()
        '''
        final_markers[name] = {}
        ##STG
        final_markers[name]['stg_prob'], final_markers[name]['stg_num'] = {},{}
        for lbm in [0.01,0.1,1]:
            final_markers[name]['stg_prob'][lbm], final_markers[name]['stg_num'][lbm] = {},{}
            for cl in set(adata_train.obs.label):
                genes = mySTG(adata_train, cl, n_top_genes=None, lbm=lbm, layer_key='logcounts', cluster_header='label',
                              random_state=args.seed).index.tolist()
                final_markers[name]['stg_prob'][lbm][cl] = genes
                final_markers[name]['stg_num'][lbm][cl] = genes[:50]
                
        ##NSForest(default parameter setting)
        final_markers[name]['nsf'] = {}
        adata_train.X = adata_train.layers['logcounts']
        nfresult = NSForest(adata_train, cluster_header='label', seed=args.seed, output_folder="results/NSForest/")
        nfresult.set_index('clusterName', inplace=True)
        for cl in set(adata_train.obs.label):
            genes = nfresult.loc[cl,'NSForest_markers']
            final_markers[name]['nsf'][cl] = genes
    
        ##COSG
        final_markers[name]['cosg'] = {}
        for num in range(5,101,5):
            final_markers[name]['cosg'][num] = {}
        cosg.cosg(adata_train,key_added='cosg',mu=1,n_genes_user=100,groupby='label',use_raw=False,layer='logcounts')
        for cl in set(adata_train.obs.label):
            gene_list = list(adata_train.uns['cosg']['names'][cl])
            for num in range(5,101,5):
                final_markers[name]['cosg'][num][cl] = gene_list[:num]
        '''
                
        ##scGeneFit
        final_markers[name]['scgf'] = {}
        for num in range(5,55,5):
            X = np.asarray(adata_train.layers['logcounts'])
            all_num = len(list(set(adata_train.obs.label)))*num
            predt = get_markers(X, adata_train.obs.label, all_num, method='centers', redundancy=0.25)
            final_markers[name]['scgf'][num] = adata_train.var_names[predt].tolist()
            
        ##CosGeneGate
        final_markers[name]['cgg'] = {}
        for num in range(5,55,5):
            final_markers[name]['cgg'][num] = {}
        final_markers[name]['cgg'][50] = CosGeneGate(adata_train, stg_genes=150, final_genes=50, lbm=0.01,
                                              layer_key='logcounts', cluster_header='label', random_state=args.seed)
        for cl in set(adata_train.obs.label):
            gene_list = final_markers[name]['cgg'][50][cl]
            for num in range(5,50,5):
                final_markers[name]['cgg'][num][cl] = gene_list[:num]
    #save results
    save_obj(final_markers, '/gpfs/gibbs/pi/zhao/wl545/results/'+args.dataset+'/tune_num/sub/markers'+str(args.seed)+'.pkl')
    
if __name__=="__main__":
    args = parse_args()
    main(args)





