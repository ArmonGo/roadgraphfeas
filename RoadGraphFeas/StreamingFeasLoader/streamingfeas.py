import pandas as pd
import numpy as np 
import networkx as nx
import itertools
import copy 
import networkx.classes.function  as F

class StreamingGraphFea:
    def __init__(self, weight = "all"):
        if weight =="all":
            self.weight = [None, "weight"] 
        elif weight in [None, "weight"]:
            self.weight = weight 
        else:
            raise Exception("weight choice is not available")

    def centrality_extract(self, G, centroid_node, w):
        """extract centrality""" 
        betweeness_ctn = nx.betweenness_centrality(G, weight = w)[centroid_node]
        clossness_ctn = nx.closeness_centrality(G, distance = w)[centroid_node]
        centrality_dict = {"betweeness_ctn_" + str(w) + "_g": betweeness_ctn,
                           "clossness_ctn_" + str(w) + "_g": clossness_ctn
                           }
        return  centrality_dict
    
    def eigenvec_extract(self, G, centroid_node, w = None): # only consider graph without weight
        eigenvec_ctn = nx.eigenvector_centrality(G, max_iter = 10000,weight = w, tol = 1.0e-4)[centroid_node]
        eigenvec_dict = { "eigenvec_ctn_" + str(w) + "_g": eigenvec_ctn}
        return eigenvec_dict
    
    def graph_circle_feas(self, G, centroid_node):
        """extract nearest circles for each target nodes with given range, be careful of node index"""
        circles = nx.cycle_basis(G)
        cir_num = 0
        max_cir_len = 0
        if len(circles)>0:
            cir_num = len(circles)
            max_cir_len = max([len(i) for i in circles])
        circle_dict = {"cir_num"  + "_g":cir_num, 
                        "max_cir_len"  + "_g": max_cir_len}
        return circle_dict
    
    def graph_basic_feas(self, G, centroid_node):
        node_num = F.number_of_nodes(G)
        edge_num = F.number_of_edges(G)
        density = F.density(G)
        basic_feas_dict = {"node_num"+ "_g": node_num,
                           "edge_num"+ "_g": edge_num,
                           "density"+ "_g": density,
                           }
        return basic_feas_dict

    def graph_degree(self, G, centroid_node, w = None):
         de_ls = dict(F.degree(G, weight = w))
         max_degree = max(de_ls.values())
         avg_degree = sum(de_ls.values())/len(de_ls.values())
         degree_dict = {'avg_degree_' +  str(w) + "_g": avg_degree,
                        'max_degree_' +  str(w) + "_g": max_degree
                         }
         return degree_dict

    def node_attr_extract(self, G, centroid_node):
        node_attr = dict(G.nodes(data=True))[centroid_node]
        return node_attr
    
    def streaming_feas_loading(self, G,centroid_node, features_choice = "all", append_fea = None):
        feature_dict = {}
        if not isinstance(self.weight, list): # convert to a list
            self.weight = [self.weight]

        if features_choice == "all":
            func_set = [self.centrality_extract, self.eigenvec_extract, 
                        self.graph_circle_feas, self.graph_degree,
                        self.graph_basic_feas, self.node_attr_extract]
        else:
            raise Exception("sorry no feature is available")
        
        for func in func_set:
            if func != self.centrality_extract and func != self.graph_degree :
                feature_dict.update(func(G, centroid_node))
            else:
                for w in self.weight:
                    feature_dict.update(func(G, centroid_node, w))
        if append_fea is not None:
            assert isinstance(append_fea, dict) # must be dictionary 
            feature_dict.update(append_fea)
        return feature_dict
 