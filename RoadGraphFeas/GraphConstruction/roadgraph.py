import networkx as nx
import numpy as np
import shapely
from scipy.spatial import KDTree
from shapely import ops
from shapely.geometry import LineString, MultiPoint, Point
from tqdm.auto import tqdm

def get_connecting_points(points, roads):
    return MultiPoint([ops.nearest_points(roads, p)[0] for p in points.geoms])


def make_nx_graph(roads, points, features):
    road_points = np.array([coord for line in roads.geoms for coord in line.coords])
    kd_tree = KDTree(road_points)

    def map_to(p, r):
        return np.sort(kd_tree.query_ball_point(p, r=r))[0]

    def dist_pp(p1, p2): # calculate points distance
        return p1.distance(p2)

    def dist_lp(ls, p): 
        return ls.distance(p)

    mapping = {i: map_to(road_points[i], 10) for i in range(len(road_points))}
    midx = len(road_points)
    nodes, edges = {}, {}
    idx = 0

    for line in roads.geoms:
        for i in range(len(line.coords) - 1):
            fm, sm = mapping[idx], mapping[idx + 1] # mapping the neighbor node for road nodes start and end 
            fm, sm = min(fm, sm), max(fm, sm) # get the longest distance 
            idx += 1
            if fm == sm: # if the same pass 
                continue
            dist = dist_pp(Point(line.coords[i]), Point(line.coords[i + 1]))
            if dist <= 1e-10: # if too close pass 
                continue
            nodes[fm] = (
                fm,
                {"type": "road", "target": None, "xy": road_points[sm]},
            )
            nodes[sm] = (
                sm,
                {"type": "road", "target": None, "xy": road_points[sm]},
            )
            edges[(fm, sm)] = (fm, sm, dist)
        idx += 1

    edges = list(edges.values())
    features_col = list(features.columns)
   
   
    for pi, point in enumerate(points.geoms):
        nidx = pi + midx
        eidx = np.argsort(
            [
                dist_lp(
                    LineString(np.stack([nodes[fp][1]["xy"], nodes[sp][1]["xy"]])),
                    point,
                )
                if dist >= 1e-3
                else np.linalg.norm(
                    np.array(nodes[fp][1]["xy"]) - np.array(point.coords[0])
                )
                for (fp, sp, dist) in edges
            ]
        )[0]
        fp, sp, dist = edges[eidx]
        del edges[eidx]
        A = {k : list(features[k])[pi] for k in features_col}
        A.update({"type": "target","xy": np.array(point.coords[0])})
        nodes[nidx] = (nidx, A)
        
        fdist = dist_pp(point, Point(nodes[fp][1]["xy"]))
        sdist = dist_pp(point, Point(nodes[sp][1]["xy"]))
        edges.append((nidx, fp, fdist))
        edges.append((nidx, sp, sdist))
    
    graph = nx.Graph()
    graph.add_weighted_edges_from(edges)
    graph.add_nodes_from(nodes.values())
    # reindex 
    graph = nx.convert_node_labels_to_integers(graph, first_label=0, ordering='default', label_attribute=None)
    return graph


def plot_nx_graph(G, use_pos=True):
    fixed_pos = nx.get_node_attributes(G, "xy")
    pos = nx.spring_layout(G, pos=fixed_pos, fixed=fixed_pos.keys())
    edge_colors = [G[u][v]["weight"] for u, v in G.edges()]
    node_sizes = [
        10 if att == "target" else 0
        for node, att in nx.get_node_attributes(G, "type").items()
    ]
    node_colors = [
        att["target"] if att["type"] == "target" else 0
        for node, att in G.nodes(data=True)
    ]
    nx.draw_networkx(
        G,
        pos=pos if use_pos else None,
        with_labels=False,
        alpha=0.5,
        node_size=node_sizes,
        node_color=node_colors,
        edge_color=edge_colors,
    )


def make_graph_dist_matrix(graph, for_nodes):
    dist_matrix = {node_ix: {} for node_ix in for_nodes}
    for node_ix in for_nodes:
        dist_matrix[node_ix] = nx.shortest_path_length(
            graph, source=node_ix, weight="weight"
        )
    return dist_matrix


def get_neighbor_dists(dist_matrix, node_ix, train_nodes, k=1):
    values = np.array([dist_matrix[node_ix].get(n, np.nan) for n in train_nodes])
    first_k = values[np.argsort(values)][:k]
    return np.nan_to_num(first_k, np.nanmean(first_k))


def get_neighbor_vals(dist_matrix, node_ix, train_nodes, y_train, k=1):
    values = np.array([dist_matrix[node_ix].get(n, np.nan) for n in train_nodes])
    first_k = y_train[np.argsort(values)][:k]
    return np.nan_to_num(first_k, np.nanmean(first_k))

