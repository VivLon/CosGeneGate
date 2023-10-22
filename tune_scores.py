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
from NSForest import *
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
import utils

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
    parser.add_argument('--dataset', default='pbmc', help='dataset')
    parser.add_argument('--marker_dir', help='the directory to the pickle storing the marker result')
    parser.add_argument('--method', default=None, help='the model used to select markers')
    
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
    
    #generate scores
    annot_scores, dm_scores = {},{}
    for name in names:
        train_name = name+'_train'
        test_name = name+'_test'
        adata_train = adatas[train_name].copy()
        adata_test = adatas[test_name].copy()
        annot_scores[name], dm_scores[name] = {},{}
        #pickle file stores results of all methods
        if args.method==None:
            #pickle file of a specific seed, in order to run marker selection & score generation simultaneously
            final_markers = load_obj(args.marker_dir+str(args.seed)+'.pkl')
            for method in ['cosg','scgf','stg_prob','stg_num','cgg']:
                annot_scores[name][method], dm_scores[name][method] = {},{}
                for setting in final_markers[name][method].keys():
                    marker = final_markers[name][method][setting]
                    if method == 'scgf':#scGeneFit doesn't give cell type specific markers
                        predt = marker
                    else:
                        predt = []
                        for cl in marker.keys():
                            predt.extend(marker[cl])
                    annot_scores[name][method][setting] = utils.calculate_annot_scores(adata_train, adata_test, predt)
                    dm_scores[name][method][setting] = utils.calculate_dm_scores(adata_test, predt)
            for method in ['nsf']:
                annot_scores[name][method], dm_scores[name][method] = {},{}
                marker = final_markers[name][method]
                predt = []
                for cl in marker.keys():
                    predt.extend(marker[cl])
                annot_scores[name][method][setting] = utils.calculate_annot_scores(adata_train, adata_test, predt)
                dm_scores[name][method][setting] = utils.calculate_dm_scores(adata_test, predt)
            
            save_obj(annot_scores, '/gpfs/gibbs/pi/zhao/wl545/results/'+args.dataset+'/tune_num/sub/annot_scores'+
                     str(args.seed)+'.pkl')
            save_obj(dm_scores, '/gpfs/gibbs/pi/zhao/wl545/results/'+args.dataset+'/tune_num/sub/dm_scores'+
                     str(args.seed)+'.pkl')
        
        #pickle file only stores one method    
        if args.method!=None:
            final_markers = load_obj(args.marker_dir+str(args.seed)+'.pkl')
            if args.method == 'cgg': #CosGeneGate has 2 settings: stg_genes and final_genes
                for stg_genes in final_markers[name].keys():
                    annot_scores[name][stg_genes], dm_scores[name][stg_genes] = {},{}
                    for genes in final_markers[name][stg_genes].keys():
                        marker = final_markers[name][stg_genes][genes]
                        predt = []
                        for cl in marker.keys():
                            predt.extend(marker[cl])
                        annot_scores[name][stg_genes][genes] = utils.calculate_annot_scores(adata_train, adata_test, predt)
                        dm_scores[name][stg_genes][genes] = utils.calculate_dm_scores(adata_test, predt)
                        
            elif (args.method!='nsf')&(args.method!='cgg'):
                for setting in final_markers[name].keys():
                    marker = final_markers[name][setting]
                    if method == 'scgf':
                        predt = marker
                    else:
                        predt = []
                        for cl in marker.keys():
                            predt.extend(marker[cl])
                    annot_scores[name][setting] = utils.calculate_annot_scores(adata_train, adata_test, predt)
                    dm_scores[name][setting] = utils.calculate_dm_scores(adata_test, predt)
            
            elif args.method == 'nsf':
                marker = final_markers[name]
                predt = []
                for cl in marker.keys():
                    predt.extend(marker[cl])
                annot_scores[name][setting] = utils.calculate_annot_scores(adata_train, adata_test, predt)
                dm_scores[name][setting] = utils.calculate_dm_scores(adata_test, predt)
            
            save_obj(annot_scores, '/gpfs/gibbs/pi/zhao/wl545/results/'+args.dataset+'/tune_num/sub/'+args.method+'_annot_scores'+
                     str(args.seed)+'.pkl')
            save_obj(dm_scores, '/gpfs/gibbs/pi/zhao/wl545/results/'+args.dataset+'/tune_num/sub/'+args.method+'_dm_scores'+
                     str(args.seed)+'.pkl')
        
            
if __name__=="__main__":
    args = parse_args()
    main(args)

        
