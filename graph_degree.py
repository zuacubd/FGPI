import os
import sys

#sys.path.insert(0, os.path.abspath('env/lib/python3.7/site-packages/'))

import collections
import pandas as pd
import numpy as np
import random
import networkx as nx

from tqdm import tqdm
import re
import matplotlib.pyplot as plt


def save_plot(G, deg, cnt, fig_name):
    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color="b")

    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.5 for d in deg])
    ax.set_xticklabels(deg)

    plt.axes([0.4, 0.4, 0.5, 0.5])
    Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    pos = nx.spring_layout(G)
    plt.axis("off")
    nx.draw_networkx_nodes(G, pos, node_size=20)
    nx.draw_networkx_edges(G, pos, alpha=0.4)
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
G = nx.from_pandas_edgelist(links_df, "node_1", "node_2", create_using=nx.Graph())
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())

save_plot(G, deg, cnt, figures_path + 'Graph_degree.png')
