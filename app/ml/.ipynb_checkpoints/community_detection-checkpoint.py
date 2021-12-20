import community
import networkx as nx
import matplotlib.pyplot as plt
import louvain
import igraph as ig
from igraph import *


# def view_karate_club():

# def sample_louvain():
#     G = ig.Graph.Famous('Zachary')
#     print (G)
#     partition = louvain.find_partition(G, louvain.ModularityVertexPartition)
#     print (partition)
#     ig.plot(partition)
#     plt.show()

#graph = ig.Graph.Read_Ncol('Data_processed/20000_UserRetweet_UserOriginal.txt', names=True, directed=False, weights=True) #20000_UserRetweet_UserOriginal.txt
# ig.plot(graph)
# plt.show()
def Louvain(graph, results_dir):
    implementLouvain = louvain.find_partition(graph, louvain.ModularityVertexPartition)#graph.community_multilevel()
    #print(implementLouvain) #print lend(implementLouvain[0] de xem so node trong cluster 0
    f_in = open(os.path.join(results_dir, 'Louvain.txt'),'w')
    for line in implementLouvain:
        f_in.write(str(line))
    outputModularity_Louvain = graph.modularity(implementLouvain)
    print("Modularity Optimal Value of louvain", outputModularity_Louvain)

def Springlass(graph):
    clusters    = graph.clusters()
    giant       = clusters.giant() ## using the biggest component as an example, you can use the others here.
    ImplementSpinglass = giant.community_spinglass()
    #print(ImplementSpinglass)
    outputModularity_Spinglass = giant.modularity(ImplementSpinglass)
    print("Modularity Optimal Value of Spinglass", outputModularity_Spinglass)

# ImplementSpinglass = graph.community_spinglass()
# print(ImplementSpinglass)
# outputModularity_Spinglass = graph.modularity(ImplementSpinglass)
# print("Modularity Optimal Value of Spinglass", outputModularity_Spinglass)

def LeadingEigenvector(graph, results_dir):
    ImplementLeadingEigenvector = graph.community_leading_eigenvector(clusters=None, weights=None,arpack_options=None)
    #print(ImplementLeadingEigenvector)
    outputModularity_LeadingEigenvector = graph.modularity(ImplementLeadingEigenvector)
    print("Modularity Optimal Value of LeadingEigenvector", outputModularity_LeadingEigenvector)


def Label_propagation(graph, results_dir):
    ImplementLabel_propagation = graph.community_label_propagation(weights=None, initial=None, fixed=None)
    #print(ImplementLabel_propagation)
    outputModularity_Label_propagation  = graph.modularity(ImplementLabel_propagation)
    print("Modularity Optimal Value of Label_propagation", outputModularity_Label_propagation)

def Edge_betweenness(graph, results_dir):
    ImplementEdge_betweenness = graph.community_edge_betweenness(  clusters=None, directed=True, weights=None)
    #print(ImplementEdge_betweenness)
    outputModularity_Edge_betweenness  = graph.modularity(ImplementEdge_betweenness)
    print("Modularity Optimal Value of Edge_betweenness", outputModularity_Edge_betweenness)

def Walktrap(graph, results_dir):
    ImplementWalktrap = graph.community_walktrap(steps = 4)
    clust = ImplementWalktrap.as_clustering()
    #print(clust)
    outputModularity_Walktrap = graph.modularity(clust)
    print("Modularity Optimal Value of Spinglass", outputModularity_Walktrap)

def getGraph(data_dir):
    '''
        return the graph for the retweets
    '''
    retweet_data_path = os.path.join(data_dir, 'data/processed/20000_UserRetweet_UserOriginal.txt')
    graph = ig.Graph.Read_Ncol(retweet_data_path, names=True, directed=False, weights=True)
    return graph


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print ("Usage: {} data_directory".format(sys.argv[0]))

    data_dir = sys.argv[1]
    results_dir = os.path.join(data_dir, 'results')

    print ('Loading the graph ..')
    igraph = getGraph(data_dir)
    print ('done.')

    print ('Detecting communities ...')

    print ('louvain ...')
    Louvain(igraph, results_dir)
    print ('done.')

    print ('walktrap ...')
    Walktrap(igraph, results_dir)
    print ('done.')

    print ('label propagation ...')
    Label_propagation(igraph, results_dir)
    print ('done.')

    print ('edge betweenness ...')
    Edge_betweenness(igraph, results_dir)
    print ('done.')

    print ('leading eigen vector ...')
    LeadingEigenvector(igraph, results_dir)
    print ('done.')
