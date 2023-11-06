# graph 
from GraphConstruction.dataloader import *
from GraphConstruction.overpassloader import OverpassRoadLoader
from GraphConstruction.roadgraph import *
from GraphConstruction.transformer import *

class StreamingGraphLoader:
    def __init__(self, location_centriod, radius, extract_radius,dataloader, data_path, province):
        self.location_centriod = location_centriod # for transformer
        self.radius = radius 
        self.extract_radius = extract_radius # for subgraph construction
        self.data_path = data_path
        self.province = province
        self.dataloader = dataloader
        self._all_nodes_load_() #prepare all nodes

    def _all_nodes_load_(self):
        Local_Transformer = CoordinateTransformer(pyproj.CRS("EPSG:4326"), self.location_centriod,self.radius)
        points, features, target_ix, non_trans_points = self.dataloader(Local_Transformer).load(self.data_path, self.province)
        self.points = points 
        self.features = features
        self.target_ix = target_ix
        self.non_trans_points = non_trans_points
        self.Local_Transformer = Local_Transformer
        
    def load(self, mapload_node, target_node, ix):  
        roads = OverpassRoadLoader(self.Local_Transformer).load_radius(self.extract_radius, mapload_node.x, mapload_node.y)
        target_features  = self.features.iloc[[ix]]
        if len(roads.geoms)>1:
            connecting_points = get_connecting_points(MultiPoint([target_node]), roads)
            G = make_nx_graph(roads, connecting_points, target_features)
            return G
        else:
            return None
        
    def node_ix_finder(self, G):
        """only one target node is included"""
        node_dicts = dict(G.nodes(data=True))
        target_ix = list(map(lambda x : x["type"] == 'target', list(node_dicts.values())))
        a = np.array(list(node_dicts.keys()))[target_ix].item()
        return a 

    def generator(self):
        for ix in self.target_ix:
            G = self.load(self.non_trans_points.geoms[ix], self.points.geoms[ix], ix)
            if G is not None:
                node_ix = self.node_ix_finder(G)
            
            yield G, node_ix, (self.points.geoms[ix].x,self.points.geoms[ix].y)