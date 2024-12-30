import pandas as pd
import scanpy as sc
import numpy as np
import sklearn
import scipy.stats
import matplotlib.pyplot as plt 
import torch
import pytorch_lightning
import random
from stg.stg import STG
import cosg

import platform

version = platform.python_version()

if version >= "3.10":
    import collections 
    collections.Sequence = collections.abc.Sequence
    collections.Set = collections.abc.Set
    collections.Mapping = collections.abc.Mapping

    

def mySTG(adata, cl, n_top_genes, lbm=0.01, layer_key='logcounts', cluster_header='label',random_state=0):
    '''
        Input:
            adata: scRNA-seq data in the anndata format.
            cl: The cell-type label used for selection.
            n_top_genes: number of genes selected in the first stage.
            lbm: weight of the regularization for the feature sparsity in STG.
            layer_key: the layer name we used in adata file.
            cluster_header: the name of cell-type labels in the adata file.
        Output:
            marker gene list.
    '''
    df_dummies = pd.get_dummies(adata.obs[cluster_header])
    x_train = adata.layers[layer_key]
    y_train = df_dummies[cl].values

    rf_clf = STG(task_type='classification',input_dim = adata.shape[1], output_dim=2, hidden_dims=[60, 20, 10], activation='tanh',
        optimizer='SGD', learning_rate=0.1, batch_size=adata.shape[0], feature_selection=True, sigma=0.5, lam=lbm, random_state=0, device="cuda") 
    rf_clf.fit(x_train, y_train, nr_epochs=2000, valid_X=x_train, valid_y=y_train, print_interval=1000)

    ## get feature importance and rank/subset top genes
    prob_list = rf_clf.get_gates(mode='prob')
    res = pd.Series(prob_list, index=adata.var_names).sort_values(ascending=False)
    if n_top_genes!=None:
        top_rf_genes = res[:n_top_genes]
    if n_top_genes == None:
        top_rf_genes = res[res>0.5]
    return top_rf_genes

def CosGeneGate(adata, stg_genes=30, final_genes=10, lbm=0.01, layer_key='logcounts', cluster_header='label', random_state=0):
    '''
        Input:
            adata: scRNA-seq data in the anndata format.
            stg_genes: number of genes selected in the first stage.
            final_genes: number of genes selected in the second stage.
            lbm: weight of the regularization for the feature sparsity in STG.
            cluster_header: the name of cell-type labels in the adata file.
            random_state: the root of random numbers. 
        Output:
            marker gene list.
    '''


    marker_stg = []
    for cl in sorted(set(list(adata.obs[cluster_header]))):
        genes = mySTG(adata, cl, n_top_genes=stg_genes, lbm=lbm, layer_key=layer_key, cluster_header=cluster_header, random_state=random_state).index.tolist()
        marker_stg.extend(genes)
    
    final_markers = {}
    adata = adata[:, list(set(marker_stg))]
    if layer_key=='raw':
        cosg.cosg(adata, key_added='CosGeneGate',mu=1,n_genes_user=final_genes,groupby=cluster_header,use_raw=True)
    else:
        cosg.cosg(adata, key_added='CosGeneGate',mu=1,n_genes_user=final_genes,groupby=cluster_header, use_raw=False,layer=layer_key)
    
    for cl in sorted(set(list(adata.obs[cluster_header]))):
        final_markers[cl] = list(adata.uns['CosGeneGate']['names'][cl])
    
    return final_markers
    
    
    
    
    
    
    
