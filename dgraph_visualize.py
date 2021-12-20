import os
import sys

#sys.path.insert(0, os.path.abspath('env/lib/python3.7/site-packages/'))

import pandas as pd
import numpy as np
import random
import networkx as nx

from tqdm import tqdm
import re
import matplotlib.pyplot as plt


def save_graph(iDG, fig_name):
    # plot graph
    plt.figure(figsize=(10,10))
    pos = nx.random_layout(iDG)
    nx.draw(iDG, with_labels=False,  pos = pos, node_size = 40, alpha = 0.6, width = 0.7)
    #plt.show()
    plt.show(block=False)
    plt.savefig(fig_name, format="PNG")



#from node2vec import Node2Vec

data_dir = "/projets/sig/mullah/nlp/fgpi/"
newretweetuserid_neworiginaluserid_path = os.path.join(data_dir, 'data/processed/20000_UserRetweetID_UserOriginalID_NewID.txt')
figures_path ='figures/'

# load edges (or links)
with open(newretweetuserid_neworiginaluserid_path) as f:
    links = f.read().splitlines()

print (len(links))

# captture nodes in 2 separate lists
node_list_1 = []
node_list_2 = []

for i in tqdm(links):
    node_list_1.append(i.split(' ')[0])
    node_list_2.append(i.split(' ')[1])

links_df = pd.DataFrame({'node_1': node_list_1, 'node_2': node_list_2})

# create graph
DG = nx.from_pandas_edgelist(links_df, "node_1", "node_2", create_using=nx.DiGraph())
#save_graph(DG, figures_path + 'Full-Graph.png')

DG.remove_edges_from(nx.selfloop_edges(DG))
DG_k1 = nx.k_shell(DG, 1)
save_graph(DG_k1, figures_path + 'Dgraph_k1.png')

DG_k2 = nx.k_shell(DG, 2)
save_graph(DG_k2, figures_path + 'Dgraph_k2.png')

DG_k3 = nx.k_shell(DG, 3)
save_graph(DG_k3, figures_path + 'Dgraph_k3.png')

DG_k4 = nx.k_shell(DG, 4)
save_graph(DG_k4, figures_path + 'Dgraph_k4.png')
