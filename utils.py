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

def calculate_annot_scores(adata_train, adata_test, marker_list):
    scores = {}
    score_table = {'acc':[],'macro_rec':[],'macro_pre':[],'macro_f1':[],'weighted_pre':[],'weighted_rec':[],
                       'weighted_f1':[],'label_ari':[], 'label_nmi':[]}
    trainadata = adata_train[:, list(set(marker_list))].copy()
    testadata = adata_test[:, list(set(marker_list))].copy()
    
    for t in range(5,15):
        model = KNeighborsClassifier(n_neighbors=t)
        model.fit(np.asarray(trainadata.layers['logcounts']), trainadata.obs.label)
        pred = model.predict(np.asarray(testadata.layers['logcounts']))
        reports = classification_report(testadata.obs.label, pred, output_dict = True)
        score_table['acc'].append(reports['accuracy'])
        score_table['macro_rec'].append(reports['macro avg']['recall'])
        score_table['macro_pre'].append(reports['macro avg']['precision'])
        score_table['macro_f1'].append(reports['macro avg']['f1-score'])
        score_table['weighted_rec'].append(reports['weighted avg']['recall'])
        score_table['weighted_pre'].append(reports['weighted avg']['precision'])
        score_table['weighted_f1'].append(reports['weighted avg']['f1-score'])
        score_table['label_ari'].append(adjusted_rand_score(pred, testadata.obs.label))
        score_table['label_nmi'].append(normalized_mutual_info_score(pred, testadata.obs.label))
    for key in list(score_table.keys()):
        scores[key] = np.mean(score_table[key])
    return scores

def calculate_dm_scores(adata_test, marker_list):
    scores = {}
    testadata = adata_test[:, list(set(marker_list))].copy()
    adata = testadata.copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata)
    adata_all = adata_test.copy()
    sc.pp.scale(adata_all, max_value=10)
    sc.tl.pca(adata_all, svd_solver='arpack')
    sc.pp.neighbors(adata_all)
    res = nmi_ari_cluster_labels_leiden(adata.obsp['connectivities'], adata.obs.label, n_jobs=-1)
    scores['cluster_ari'] = res['ari']
    scores['cluster_nmi'] = res['nmi']
    scores['asw'] = silhouette_label(np.array(adata.layers['logcounts']), adata.obs.label)
    sc.tl.paga(adata, 'label')
    sc.tl.paga(adata_all, 'label')
    sim_pre = adata_all.uns['paga']['connectivities'].todense()
    sim_post = adata.uns['paga']['connectivities'].todense()
    scores['paga_sim'] = np.exp(-1*np.abs((sim_pre-sim_post)).mean())
    return scores





