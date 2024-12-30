from .CSCORE_IRLS_new import CSCORE_IRLS
from statsmodels.stats.multitest import multipletests
import pandas as pd
import scanpy as sc
import numpy as np
import anndata
from .utils import *

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

def get_all_markers(markers):
    cluster_list = markers[0].keys()
    seeds = len(list(markers.keys()))
    all_markers = {}
    for cl in cluster_list:
        all_markers[cl] = []
        for i in range(seeds):
            all_markers[cl].extend(markers[i][cl])
        all_markers[cl] = list(set(all_markers[cl]))
    return all_markers
        

def calculate_appear_times(markers):
    cluster_list = markers[0].keys()
    seeds = len(list(markers.keys()))
    appear_times = {}
    all_markers = get_all_markers(markers)
    for cl in cluster_list:
        appear_times[cl] = {}
        for i in range(1,seeds+1):
            appear_times[cl][i] = []
        for m in all_markers[cl]:
            time = 0
            for i in range(seeds):
                if m in markers[i][cl]:
                    time += 1
            appear_times[cl][time].append(m)
    return appear_times

def get_anchor_genes(markers, seeds_num = 1):
    appear_times = calculate_appear_times(markers)
    cluster_list = markers[0].keys()
    seeds = len(list(markers.keys()))
    anchor_dict = {}
    for cl in cluster_list:
        anchor = appear_times[cl][seeds] #initial anchor list
        tmp_seeds = seeds
        while len(anchor)==0:
            tmp_seeds -= 1
            anchor = appear_times[cl][tmp_seeds]
        anchor_dict[cl] = anchor.copy()
        initial_seed = 1
        
        while initial_seed<seeds_num:
            tmp_seeds -= 1
            anchor = appear_times[cl][tmp_seeds]
            while len(anchor)==0:
                tmp_seeds -= 1
                anchor = appear_times[cl][tmp_seeds]
            anchor_dict[cl].extend(anchor)
            initial_seed+=1
                
    return anchor_dict

def run_cscore(adata, save_dir, all_markers):
    adata.obs['n_counts'] = np.array(np.sum(adata.layers['raw'], axis= 1))
    cluster_list = list(set(adata.obs.label))
    change_sparsertiy(adata, ['raw'])
    for cl in cluster_list:
        adata_new = adata[np.where(adata.obs.label == cl)].copy()
        adata_new = adata_new[:, all_markers[cl]]
        counts = adata_new.layers['raw']
        seq_depth = adata_new.obs['n_counts'].values

        result = CSCORE_IRLS(np.array(counts), seq_depth)
        coexp_matrix = result[0]

        #Obtain BH-adjusted p values
        CSCORE_p = result[1]
        p_matrix_BH = np.zeros(CSCORE_p.shape)
        upper_tri_indices = np.triu_indices(CSCORE_p.shape[0], k=1)
        p_values = CSCORE_p[upper_tri_indices]
        rejected, adjusted_p_values, _, _ = multipletests(p_values, method='fdr_bh', returnsorted=False)
        p_matrix_BH[upper_tri_indices] = adjusted_p_values
        p_matrix_BH += p_matrix_BH.T
        coexp_matrix[p_matrix_BH > 0.05] = -100 #set coexp of 'Not Significant' large enough

        coexp_matrix = pd.DataFrame(coexp_matrix, index = all_markers[cl], columns = all_markers[cl])
        np.abs(coexp_matrix).to_csv(save_dir+cl+'_coexp.csv')

def select_credible_set(markers, coexp_dir, anchor_genes, optimal_num):
    appear_times = calculate_appear_times(markers)
    cluster_list = markers[0].keys()
    seeds = len(list(markers.keys()))
    ##initialize credible set with anchor genes
    selected_genes = {}
    anchor_lst,cl_lst = [],[]
    for cl in cluster_list:
        selected_genes[cl] = {}
        anchor = anchor_genes[cl].copy()
        anchor_lst.extend(anchor)
        cl_lst.extend([cl]*len(anchor))
        for m in anchor:
            selected_genes[cl][m] = []
            selected_genes[cl][m].append(m)

    min_coexp = pd.DataFrame(np.zeros((len(anchor_lst),3)), columns=['ct','new_gene','coexp'])
    min_coexp.loc[:,'ct'] = cl_lst
    min_coexp.index = anchor_lst
    for cl in cluster_list:
        anchor = anchor_genes[cl].copy()
        for m in anchor:
            coexp = pd.read_csv(coexp_dir+cl+'_coexp.csv', index_col=0)
            coexp = pd.DataFrame(coexp).drop(m, axis=1) #avoid repeat
            ind = np.where((min_coexp.ct==cl)&(min_coexp.index==m))[0].tolist()[0]
            min_coexp.iloc[ind, 2] = coexp.loc[selected_genes[cl][m],].mean().sort_values()[0]
            min_coexp.iloc[ind, 1] = pd.DataFrame(coexp.loc[selected_genes[cl][m],].mean().sort_values()).index[0]

    ##iteration
    sum_ = len(anchor_lst)
    no_change_count = 0  # Counter for rounds without change in sum
    prev_sum = sum_
    while sum_<(optimal_num)*len(cluster_list):
        #add gene from last iteration
        min_coexp = min_coexp.sort_values(by='coexp')
        cl = min_coexp.ct[0]
        m = min_coexp.index[0]
        selected_genes[cl][m].append(min_coexp.new_gene[0])

        #replace least coexp gene with a new gene
        coexp = pd.read_csv(coexp_dir+cl+"_coexp.csv",index_col=0)
        for g in selected_genes[cl][m]:
            if g not in coexp.columns:
                continue
                #print("All markers of cell type",cl,"has been selected into the credible set. We recommend using more anchor genes (increase 'seeds_num' parameter or self define 'anchor_num' parameter in get_anchor_genes function).")
            else:
                coexp = pd.DataFrame(coexp).drop(g, axis=1) #avoid repeat
        ind = np.where((min_coexp.ct==cl)&(min_coexp.index==m))[0].tolist()[0]
        try:
            min_coexp.iloc[ind, 2] = coexp.loc[selected_genes[cl][m],].mean().sort_values()[0]
            min_coexp.iloc[ind, 1] = pd.DataFrame(coexp.loc[selected_genes[cl][m],].mean().sort_values()).index[0]
        except IndexError:
            min_coexp.iloc[ind, 2] = 100
    
        sum_ = 0
        for celltype in cluster_list:
            anchor = selected_genes[celltype].keys()
            tmp = []
            for m in anchor:
                tmp.extend(selected_genes[celltype][m])
            sum_ += len(list(set(tmp)))
        
        # Check if sum has changed
        if sum_ == prev_sum:
            no_change_count += 1
        else:
            no_change_count = 0  # Reset the counter if sum increases
            prev_sum = sum_  # Update previous sum

        # Exit if no change in sum for 100 consecutive rounds
        if no_change_count >= 100:
            print("No change in sum for 100 rounds. Exiting.")
            break
    return selected_genes
